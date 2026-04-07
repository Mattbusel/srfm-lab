"""
glassnode_client.py -- Glassnode API client with SQLite caching and mock mode.

Provides production-ready access to Glassnode on-chain metrics. All data methods
return pandas Series or DataFrames indexed by UTC date. Mock mode generates
synthetic but statistically realistic data for testing without live API calls.

Rate limiting: 10 requests/minute with exponential backoff on HTTP 429.
Cache TTL: 24 hours per (endpoint, asset, since, until) key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import threading
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics/"
_DEFAULT_CACHE_PATH = Path.home() / ".srfm_cache" / "glassnode.db"
_CACHE_TTL_SECONDS = 86_400  # 24 hours
_STALE_REVALIDATE_SECONDS = 3_600  # serve stale if revalidation fails within 1h
_RATE_LIMIT_RPS = 10.0 / 60.0  # 10 requests per minute
_MAX_BACKOFF_SECONDS = 120.0
_INITIAL_BACKOFF_SECONDS = 2.0

# Glassnode endpoint paths keyed by method name
_ENDPOINT_MAP: Dict[str, str] = {
    "mvrv_z_score": "market/mvrv_z_score",
    "sopr": "indicators/sopr",
    "nupl": "indicators/nupl",
    "exchange_net_flows": "transactions/transfers_volume_exchanges_net",
    "long_term_holder_supply": "supply/lth_sum",
    "short_term_holder_realized_price": "indicators/sth_realized_price",
    "funding_rate": "derivatives/futures_funding_rate_perpetual",
    "open_interest": "derivatives/futures_open_interest_sum",
    "realized_cap": "market/marketcap_realized_usd",
    "thermocap": "mining/thermocap",
}

# Mock data parameters: (mean, std, ar1_coef) for synthetic AR(1) generation
_MOCK_PARAMS: Dict[str, Tuple[float, float, float]] = {
    "mvrv_z_score": (1.5, 1.2, 0.95),
    "sopr": (1.01, 0.03, 0.80),
    "nupl": (0.35, 0.25, 0.92),
    "exchange_net_flows": (0.0, 500.0, 0.60),
    "long_term_holder_supply": (13_500_000.0, 200_000.0, 0.98),
    "short_term_holder_realized_price": (30_000.0, 5_000.0, 0.97),
    "funding_rate": (0.001, 0.008, 0.50),
    "open_interest": (15_000_000_000.0, 2_000_000_000.0, 0.94),
    "realized_cap": (400_000_000_000.0, 40_000_000_000.0, 0.99),
    "thermocap": (25_000_000_000.0, 3_000_000_000.0, 0.97),
}


# ---------------------------------------------------------------------------
# GlassnodeCache
# ---------------------------------------------------------------------------

class GlassnodeCache:
    """
    SQLite-backed response cache with TTL and stale-while-revalidate semantics.

    Entries are keyed by a hash of (endpoint, asset, since, until). The cache
    stores serialized JSON and the insertion timestamp. Reads that find stale
    entries (age > TTL) still return the data alongside a staleness flag so
    callers can attempt background revalidation.
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS glassnode_cache (
            cache_key  TEXT PRIMARY KEY,
            payload    TEXT NOT NULL,
            inserted_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_inserted_at ON glassnode_cache(inserted_at);
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        ttl_seconds: int = _CACHE_TTL_SECONDS,
        stale_revalidate_seconds: int = _STALE_REVALIDATE_SECONDS,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_CACHE_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._stale_revalidate = stale_revalidate_seconds
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection (thread-safe via check_same_thread=False)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(self._CREATE_TABLE_SQL)
            conn.commit()

    @staticmethod
    def _make_key(endpoint: str, asset: str, since: Optional[int], until: Optional[int]) -> str:
        raw = json.dumps({"endpoint": endpoint, "asset": asset, "since": since, "until": until}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, endpoint: str, asset: str, since: Optional[int], until: Optional[int]
    ) -> Tuple[Optional[pd.Series], bool]:
        """
        Retrieve cached data for the given parameters.

        Returns (series, is_stale). series is None on cache miss.
        is_stale is True when the entry exists but exceeds TTL.
        """
        key = self._make_key(endpoint, asset, since, until)
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT payload, inserted_at FROM glassnode_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None, False

        age = now - row["inserted_at"]
        is_stale = age > self._ttl

        # Beyond stale-while-revalidate window: treat as miss
        if age > (self._ttl + self._stale_revalidate):
            return None, False

        try:
            series = pd.read_json(StringIO(row["payload"]), typ="series")
            series.index = pd.to_datetime(series.index, utc=True)
            return series, is_stale
        except Exception as exc:
            logger.warning("Cache deserialize failed for %s: %s", key, exc)
            return None, False

    def set(
        self, endpoint: str, asset: str, since: Optional[int], until: Optional[int], series: pd.Series
    ) -> None:
        """Store a Series in the cache with the current timestamp."""
        key = self._make_key(endpoint, asset, since, until)
        payload = series.to_json(date_format="iso")
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO glassnode_cache (cache_key, payload, inserted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, inserted_at=excluded.inserted_at
                """,
                (key, payload, now),
            )
            conn.commit()

    def invalidate(self, endpoint: str, asset: str, since: Optional[int], until: Optional[int]) -> None:
        """Remove a specific cache entry."""
        key = self._make_key(endpoint, asset, since, until)
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM glassnode_cache WHERE cache_key = ?", (key,))
            conn.commit()

    def purge_expired(self) -> int:
        """Delete all entries older than TTL + stale window. Returns count removed."""
        cutoff = time.time() - (self._ttl + self._stale_revalidate)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM glassnode_cache WHERE inserted_at < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate_per_second: float) -> None:
        self._rate = rate_per_second
        self._tokens = rate_per_second  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_for = (1.0 - self._tokens) / self._rate
            time.sleep(wait_for)


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def _generate_mock_series(
    metric: str,
    asset: str,
    since: Optional[int],
    until: Optional[int],
    seed_salt: int = 0,
) -> pd.Series:
    """
    Generate a synthetic AR(1) time series that resembles real Glassnode data.

    Parameters are defined in _MOCK_PARAMS. The seed is derived from the metric
    name and asset so results are reproducible per (metric, asset) pair.
    """
    params = _MOCK_PARAMS.get(metric, (0.0, 1.0, 0.90))
    mean, std, phi = params

    end_ts = until if until else int(datetime.now(tz=timezone.utc).timestamp())
    start_ts = since if since else end_ts - 365 * 86_400

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    dates = pd.date_range(start=start_dt, end=end_dt, freq="D", tz="UTC")
    n = len(dates)
    if n == 0:
        return pd.Series(dtype=float, name=metric)

    seed = int(hashlib.md5(f"{metric}{asset}{seed_salt}".encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.default_rng(seed)

    noise = rng.normal(0.0, std * np.sqrt(1 - phi**2), size=n)
    values = np.empty(n)
    values[0] = mean + noise[0]
    for i in range(1, n):
        values[i] = mean + phi * (values[i - 1] - mean) + noise[i]

    # Apply metric-specific post-processing to keep values in realistic ranges
    if metric == "sopr":
        values = np.clip(values, 0.85, 1.20)
    elif metric == "nupl":
        values = np.clip(values, -0.5, 1.0)
    elif metric == "mvrv_z_score":
        values = np.clip(values, -2.0, 7.0)
    elif metric == "funding_rate":
        values = np.clip(values, -0.05, 0.05)
    elif metric in ("long_term_holder_supply", "open_interest", "realized_cap", "thermocap"):
        values = np.abs(values)

    return pd.Series(values, index=dates, name=metric, dtype=float)


# ---------------------------------------------------------------------------
# GlassnodeClient
# ---------------------------------------------------------------------------

class GlassnodeClient:
    """
    Client for Glassnode on-chain metrics API.

    Supports real HTTP requests (requires the `requests` package) and mock mode
    for offline testing. All public data methods cache results to SQLite for 24h.

    Usage (live)::

        client = GlassnodeClient(api_key="YOUR_KEY")
        mvrv = client.mvrv_z_score("BTC")

    Usage (mock)::

        client = GlassnodeClient(api_key="", use_mock=True)
        mvrv = client.mvrv_z_score("BTC")
    """

    def __init__(
        self,
        api_key: str,
        use_mock: bool = False,
        cache_db_path: Optional[Path] = None,
        cache_ttl: int = _CACHE_TTL_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._use_mock = use_mock
        self._cache = GlassnodeCache(db_path=cache_db_path, ttl_seconds=cache_ttl)
        self._rate_limiter = _TokenBucket(rate_per_second=_RATE_LIMIT_RPS)

        if not use_mock and not _REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' package is required for live API calls. "
                "Install it or set use_mock=True."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_live(
        self,
        endpoint: str,
        asset: str,
        since: Optional[int],
        until: Optional[int],
    ) -> pd.Series:
        """Make a live HTTP request to Glassnode with retry logic."""
        path = _ENDPOINT_MAP[endpoint]
        params: Dict[str, Any] = {"a": asset, "api_key": self._api_key, "i": "24h"}
        if since is not None:
            params["s"] = since
        if until is not None:
            params["u"] = until

        url = urljoin(_GLASSNODE_BASE_URL, path)
        backoff = _INITIAL_BACKOFF_SECONDS

        for attempt in range(8):
            self._rate_limiter.consume()
            try:
                resp = requests.get(url, params=params, timeout=30)
            except Exception as exc:
                logger.warning("Request error (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(backoff, _MAX_BACKOFF_SECONDS))
                backoff *= 2
                continue

            if resp.status_code == 200:
                return self._parse_response(resp.json(), endpoint)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                logger.warning("Rate limited. Waiting %.1fs (attempt %d).", retry_after, attempt + 1)
                time.sleep(min(retry_after, _MAX_BACKOFF_SECONDS))
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue

            if resp.status_code == 401:
                raise PermissionError(f"Glassnode API key invalid or expired (401): {resp.text}")

            if resp.status_code >= 500:
                logger.warning("Server error %d (attempt %d). Retrying.", resp.status_code, attempt + 1)
                time.sleep(min(backoff, _MAX_BACKOFF_SECONDS))
                backoff *= 2
                continue

            raise RuntimeError(f"Unexpected Glassnode HTTP {resp.status_code}: {resp.text}")

        raise RuntimeError(f"Glassnode request failed after 8 attempts for endpoint '{endpoint}'.")

    @staticmethod
    def _parse_response(data: List[Dict[str, Any]], metric_name: str) -> pd.Series:
        """
        Convert Glassnode JSON response to a pd.Series indexed by UTC date.

        Glassnode format: [{"t": <unix_ts>, "v": <value>}, ...]
        """
        if not data:
            return pd.Series(dtype=float, name=metric_name)

        records = [(r["t"], r["v"]) for r in data if r.get("v") is not None]
        if not records:
            return pd.Series(dtype=float, name=metric_name)

        timestamps, values = zip(*records)
        index = pd.to_datetime(list(timestamps), unit="s", utc=True)
        series = pd.Series(list(values), index=index, name=metric_name, dtype=float)
        return series.sort_index()

    def _get_or_fetch(
        self,
        endpoint: str,
        asset: str,
        since: Optional[int],
        until: Optional[int],
    ) -> pd.Series:
        """Check cache, then fetch if needed (mock or live)."""
        cached, is_stale = self._cache.get(endpoint, asset, since, until)

        if cached is not None and not is_stale:
            logger.debug("Cache hit (fresh): %s %s", endpoint, asset)
            return cached

        if cached is not None and is_stale:
            logger.debug("Cache hit (stale): %s %s -- attempting revalidation", endpoint, asset)

        try:
            if self._use_mock:
                fresh = _generate_mock_series(endpoint, asset, since, until)
            else:
                fresh = self._fetch_live(endpoint, asset, since, until)
            self._cache.set(endpoint, asset, since, until, fresh)
            return fresh
        except Exception as exc:
            if cached is not None:
                logger.warning("Revalidation failed (%s). Serving stale data.", exc)
                return cached
            raise

    @staticmethod
    def _ts(dt: Optional[datetime]) -> Optional[int]:
        """Convert datetime to unix timestamp, or None."""
        if dt is None:
            return None
        return int(dt.replace(tzinfo=timezone.utc).timestamp()) if dt.tzinfo is None else int(dt.timestamp())

    # ------------------------------------------------------------------
    # Public data methods
    # ------------------------------------------------------------------

    def mvrv_z_score(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Market Value to Realized Value Z-Score.

        High values (> 7) historically mark cycle tops; negative values mark bottoms.
        Returns daily pd.Series indexed by UTC timestamp.
        """
        return self._get_or_fetch("mvrv_z_score", asset, self._ts(since), self._ts(until))

    def sopr(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Spent Output Profit Ratio.

        SOPR > 1 means coins moved at a profit; < 1 means coins moved at a loss.
        Values below 1 sustained over multiple days indicate capitulation.
        """
        return self._get_or_fetch("sopr", asset, self._ts(since), self._ts(until))

    def nupl(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Net Unrealized Profit/Loss.

        Ranges roughly from -0.5 (deep capitulation) to 0.75+ (extreme euphoria).
        """
        return self._get_or_fetch("nupl", asset, self._ts(since), self._ts(until))

    def exchange_net_flows(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Net BTC flowing to/from exchanges (positive = inflow, negative = outflow).

        Persistent outflows suggest accumulation/self-custody; inflows suggest
        potential selling pressure.
        """
        return self._get_or_fetch("exchange_net_flows", asset, self._ts(since), self._ts(until))

    def long_term_holder_supply(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Supply held by Long-Term Holders (coins unmoved for 155+ days).

        Rising LTH supply indicates accumulation by conviction holders.
        """
        return self._get_or_fetch("long_term_holder_supply", asset, self._ts(since), self._ts(until))

    def short_term_holder_realized_price(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Short-Term Holder realized price (cost basis).

        When spot price drops below STH-RP, STH holders are underwater -- historically
        a necessary condition for capitulation bottoms.
        """
        return self._get_or_fetch("short_term_holder_realized_price", asset, self._ts(since), self._ts(until))

    def funding_rate(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Perpetual futures funding rate (8h rate, annualized by some aggregators).

        Positive: longs pay shorts (over-leveraged long market).
        Negative: shorts pay longs (over-leveraged short market).
        """
        return self._get_or_fetch("funding_rate", asset, self._ts(since), self._ts(until))

    def open_interest(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Total futures open interest in USD.

        OI rising with price rising suggests real demand; OI rising with price
        falling suggests short pressure building.
        """
        return self._get_or_fetch("open_interest", asset, self._ts(since), self._ts(until))

    def realized_cap(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Realized capitalization -- sum of all coins valued at their last moved price.

        More robust than market cap as a proxy for aggregate cost basis of holders.
        """
        return self._get_or_fetch("realized_cap", asset, self._ts(since), self._ts(until))

    def thermocap(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Thermocap -- cumulative miner revenue in USD.

        Ratio of market cap to thermocap highlights over/undervaluation relative
        to security spend.
        """
        return self._get_or_fetch("thermocap", asset, self._ts(since), self._ts(until))

    # ------------------------------------------------------------------
    # Bulk fetcher
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch all available metrics and return as a combined DataFrame.

        Missing data is forward-filled up to 7 days then left as NaN.
        """
        method_map = {
            "mvrv_z_score": self.mvrv_z_score,
            "sopr": self.sopr,
            "nupl": self.nupl,
            "exchange_net_flows": self.exchange_net_flows,
            "long_term_holder_supply": self.long_term_holder_supply,
            "short_term_holder_realized_price": self.short_term_holder_realized_price,
            "funding_rate": self.funding_rate,
            "open_interest": self.open_interest,
            "realized_cap": self.realized_cap,
            "thermocap": self.thermocap,
        }
        frames: Dict[str, pd.Series] = {}
        for name, method in method_map.items():
            try:
                frames[name] = method(asset=asset, since=since, until=until)
            except Exception as exc:
                logger.warning("Failed to fetch %s for %s: %s", name, asset, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df.ffill(limit=7)
        return df

    def close(self) -> None:
        """Release cache database connection."""
        self._cache.close()

    def __enter__(self) -> "GlassnodeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
