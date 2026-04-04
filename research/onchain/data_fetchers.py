"""
data_fetchers.py — API clients for Glassnode, Dune Analytics, The Graph, Nansen patterns.

Provides async and sync HTTP wrappers, rate-limit handling, response caching,
and structured data models for each upstream source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"
DUNE_BASE = "https://api.dune.com/api/v1"
GRAPH_BASE = "https://api.thegraph.com/subgraphs/name"
NANSEN_BASE = "https://api.nansen.ai/v1"

DEFAULT_TIMEOUT = 30          # seconds
DEFAULT_CACHE_TTL = 300       # 5 minutes
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class DiskCache:
    """Simple JSON disk cache keyed by request fingerprint."""

    def __init__(self, cache_dir: str = ".cache/onchain", ttl: int = DEFAULT_CACHE_TTL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._key_path(key)
        if not p.exists():
            return None
        try:
            with p.open() as fh:
                record = json.load(fh)
            if time.time() - record["ts"] > self.ttl:
                p.unlink(missing_ok=True)
                return None
            return record["data"]
        except Exception:
            return None

    def set(self, key: str, data: Any) -> None:
        p = self._key_path(key)
        try:
            with p.open("w") as fh:
                json.dump({"ts": time.time(), "data": data}, fh)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    def invalidate(self, key: str) -> None:
        self._key_path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)


_global_cache = DiskCache()


# ---------------------------------------------------------------------------
# Session factory with retry logic
# ---------------------------------------------------------------------------

def _build_session(retries: int = MAX_RETRIES, backoff: float = BACKOFF_FACTOR) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self._interval = 60.0 / calls_per_minute
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Glassnode client
# ---------------------------------------------------------------------------

@dataclass
class GlassnodeMetric:
    timestamp: int
    value: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


class GlassnodeClient:
    """
    Wraps the Glassnode REST API.

    Endpoint pattern:
        GET https://api.glassnode.com/v1/metrics/{tier}/{metric}
        Query params: a (asset), s (start), u (end), i (interval), api_key
    """

    def __init__(
        self,
        api_key: str = "",
        cache: DiskCache = _global_cache,
        rate_limit_rpm: int = 60,
    ):
        self.api_key = api_key or _env("GLASSNODE_API_KEY")
        self.session = _build_session()
        self.cache = cache
        self.limiter = RateLimiter(rate_limit_rpm)

    def _get(self, path: str, params: Dict[str, Any]) -> List[Dict]:
        params = dict(params)
        params["api_key"] = self.api_key
        url = f"{GLASSNODE_BASE}/{path}"
        cache_key = f"glassnode:{url}:{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.limiter.wait()
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, data)
        return data

    def get_metric(
        self,
        tier: str,
        metric: str,
        asset: str = "BTC",
        start: Optional[int] = None,
        end: Optional[int] = None,
        interval: str = "24h",
    ) -> List[GlassnodeMetric]:
        params: Dict[str, Any] = {"a": asset, "i": interval}
        if start:
            params["s"] = start
        if end:
            params["u"] = end
        raw = self._get(f"{tier}/{metric}", params)
        return [GlassnodeMetric(r["t"], r["v"]) for r in raw if "t" in r and "v" in r]

    # Convenience wrappers for common metrics

    def active_addresses(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("addresses", "active_count", asset, **kw)

    def nvt_ratio(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("indicators", "nvt", asset, **kw)

    def mvrv_ratio(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("indicators", "mvrv", asset, **kw)

    def sopr(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("indicators", "sopr", asset, **kw)

    def realized_cap(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("market", "realized_cap_usd", asset, **kw)

    def market_cap(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("market", "marketcap_usd", asset, **kw)

    def exchange_inflow(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("transactions", "transfers_to_exchanges_sum", asset, **kw)

    def exchange_outflow(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("transactions", "transfers_from_exchanges_sum", asset, **kw)

    def transaction_volume(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("transactions", "transfers_volume_sum", asset, **kw)

    def transaction_count(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("transactions", "count", asset, **kw)

    def supply_in_profit(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("supply", "profit_sum", asset, **kw)

    def supply_in_loss(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("supply", "loss_sum", asset, **kw)

    def whale_addresses(self, asset: str = "BTC", min_balance: int = 1000, **kw) -> List[GlassnodeMetric]:
        metric = f"min_{min_balance}"
        return self.get_metric("addresses", f"count_balance_greater_{metric}", asset, **kw)

    def funding_rate(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("derivatives", "futures_funding_rate_perpetual", asset, **kw)

    def open_interest(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("derivatives", "futures_open_interest_sum", asset, **kw)

    def long_liquidations(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("derivatives", "futures_liquidated_volume_long_sum", asset, **kw)

    def short_liquidations(self, asset: str = "BTC", **kw) -> List[GlassnodeMetric]:
        return self.get_metric("derivatives", "futures_liquidated_volume_short_sum", asset, **kw)


# ---------------------------------------------------------------------------
# Dune Analytics client
# ---------------------------------------------------------------------------

@dataclass
class DuneQueryResult:
    query_id: int
    execution_id: str
    rows: List[Dict[str, Any]]
    columns: List[str]
    metadata: Dict[str, Any]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DuneClient:
    """
    Wraps Dune Analytics API v1.
    Supports query execution, polling for completion, and result retrieval.
    """

    def __init__(self, api_key: str = "", cache: DiskCache = _global_cache):
        self.api_key = api_key or _env("DUNE_API_KEY")
        self.session = _build_session()
        self.session.headers.update({"X-Dune-API-Key": self.api_key})
        self.cache = cache

    def _post(self, path: str, payload: Dict = None) -> Dict:
        url = f"{DUNE_BASE}/{path}"
        resp = self.session.post(url, json=payload or {}, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Dict = None) -> Dict:
        url = f"{DUNE_BASE}/{path}"
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def execute_query(self, query_id: int, parameters: Dict[str, Any] = None) -> str:
        """Execute a saved Dune query; returns execution_id."""
        payload: Dict[str, Any] = {}
        if parameters:
            payload["query_parameters"] = [
                {"name": k, "type": _infer_dune_type(v), "value": str(v)}
                for k, v in parameters.items()
            ]
        result = self._post(f"query/{query_id}/execute", payload)
        return result["execution_id"]

    def get_status(self, execution_id: str) -> Dict:
        return self._get(f"execution/{execution_id}/status")

    def get_results(self, execution_id: str) -> Dict:
        return self._get(f"execution/{execution_id}/results")

    def wait_for_result(self, execution_id: str, poll_interval: float = 2.0, timeout: float = 300.0) -> Dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_status(execution_id)
            state = status.get("state", "")
            if state == "QUERY_STATE_COMPLETED":
                return self.get_results(execution_id)
            if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                raise RuntimeError(f"Dune query {execution_id} ended with state {state}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Dune query {execution_id} did not complete within {timeout}s")

    def run_query(
        self,
        query_id: int,
        parameters: Dict[str, Any] = None,
        use_cache: bool = True,
    ) -> DuneQueryResult:
        cache_key = f"dune:{query_id}:{json.dumps(parameters or {}, sort_keys=True)}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return DuneQueryResult(**cached)

        exec_id = self.execute_query(query_id, parameters)
        raw = self.wait_for_result(exec_id)

        result_data = raw.get("result", {})
        rows = result_data.get("rows", [])
        meta = result_data.get("metadata", {})
        columns = [c["name"] for c in meta.get("column_names", [])] if "column_names" in meta else (list(rows[0].keys()) if rows else [])

        obj = DuneQueryResult(
            query_id=query_id,
            execution_id=exec_id,
            rows=rows,
            columns=columns,
            metadata=meta,
        )
        if use_cache:
            self.cache.set(cache_key, {
                "query_id": obj.query_id,
                "execution_id": obj.execution_id,
                "rows": obj.rows,
                "columns": obj.columns,
                "metadata": obj.metadata,
            })
        return obj

    # Pre-built query IDs (public Dune dashboards — replace with your own)
    UNISWAP_V3_POOLS_QUERY = 1234567
    WHALE_TRANSFERS_QUERY  = 2345678
    DEX_VOLUME_QUERY       = 3456789

    def get_uniswap_pools(self, min_tvl_usd: float = 1_000_000) -> List[Dict]:
        result = self.run_query(self.UNISWAP_V3_POOLS_QUERY, {"min_tvl": min_tvl_usd})
        return result.rows

    def get_whale_transfers(self, min_usd: float = 100_000, hours: int = 24) -> List[Dict]:
        result = self.run_query(self.WHALE_TRANSFERS_QUERY, {"min_usd": min_usd, "hours": hours})
        return result.rows

    def get_dex_volume(self, days: int = 7) -> List[Dict]:
        result = self.run_query(self.DEX_VOLUME_QUERY, {"days": days})
        return result.rows


def _infer_dune_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    return "text"


# ---------------------------------------------------------------------------
# The Graph client
# ---------------------------------------------------------------------------

@dataclass
class GraphQLResponse:
    data: Dict[str, Any]
    errors: Optional[List[Dict]] = None

    @property
    def ok(self) -> bool:
        return self.errors is None or len(self.errors) == 0


class TheGraphClient:
    """
    Generic GraphQL client for The Graph subgraphs.
    Supports pagination via first/skip or lastID cursor patterns.
    """

    def __init__(
        self,
        subgraph: str = "uniswap/uniswap-v3",
        cache: DiskCache = _global_cache,
        rate_limit_rpm: int = 120,
    ):
        self.endpoint = f"{GRAPH_BASE}/{subgraph}"
        self.session = _build_session()
        self.cache = cache
        self.limiter = RateLimiter(rate_limit_rpm)

    def query(self, gql: str, variables: Dict[str, Any] = None) -> GraphQLResponse:
        payload = {"query": gql, "variables": variables or {}}
        cache_key = f"graph:{self.endpoint}:{json.dumps(payload, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return GraphQLResponse(**cached)

        self.limiter.wait()
        resp = self.session.post(self.endpoint, json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        obj = GraphQLResponse(data=data.get("data", {}), errors=data.get("errors"))
        self.cache.set(cache_key, {"data": obj.data, "errors": obj.errors})
        return obj

    def paginate(
        self,
        gql_template: str,
        entity_key: str,
        page_size: int = 1000,
        max_pages: int = 10,
        extra_vars: Dict = None,
    ) -> Iterator[Dict]:
        """
        Yield individual records using skip-based pagination.
        gql_template must contain {first} and {skip} placeholders.
        """
        skip = 0
        for _ in range(max_pages):
            gql = gql_template.format(first=page_size, skip=skip)
            resp = self.query(gql, extra_vars)
            if not resp.ok:
                logger.error("GraphQL errors: %s", resp.errors)
                break
            records = resp.data.get(entity_key, [])
            if not records:
                break
            yield from records
            if len(records) < page_size:
                break
            skip += page_size

    # ---- Uniswap V3 convenience queries ----

    POOLS_QUERY = """
    {{
      pools(first: {first}, skip: {skip}, orderBy: totalValueLockedUSD, orderDirection: desc) {{
        id
        token0 {{ id symbol decimals }}
        token1 {{ id symbol decimals }}
        feeTier
        liquidity
        sqrtPrice
        tick
        token0Price
        token1Price
        volumeUSD
        totalValueLockedUSD
        poolDayData(first: 7, orderBy: date, orderDirection: desc) {{
          date
          volumeUSD
          feesUSD
          tvlUSD
          high
          low
          open
          close
        }}
      }}
    }}
    """

    POOL_TICKS_QUERY = """
    {{
      ticks(first: {first}, skip: {skip}, where: {{pool: "{pool_id}"}}, orderBy: tickIdx) {{
        tickIdx
        liquidityNet
        liquidityGross
        price0
        price1
      }}
    }}
    """

    SWAPS_QUERY = """
    {{
      swaps(first: {first}, skip: {skip}, orderBy: timestamp, orderDirection: desc,
            where: {{pool: "{pool_id}"}}) {{
        id
        timestamp
        amount0
        amount1
        amountUSD
        tick
        sqrtPriceX96
        sender
        recipient
      }}
    }}
    """

    TOKEN_DAY_DATA_QUERY = """
    {{
      tokenDayDatas(first: {first}, skip: {skip},
                    where: {{token: "{token_id}"}},
                    orderBy: date, orderDirection: desc) {{
        date
        priceUSD
        volume
        volumeUSD
        totalLiquidityUSD
      }}
    }}
    """

    POSITIONS_QUERY = """
    {{
      positions(first: {first}, skip: {skip},
                where: {{owner: "{owner}", liquidity_gt: "0"}}) {{
        id
        owner
        pool {{ id token0 {{ symbol }} token1 {{ symbol }} feeTier }}
        tickLower {{ tickIdx }}
        tickUpper {{ tickIdx }}
        liquidity
        depositedToken0
        depositedToken1
        withdrawnToken0
        withdrawnToken1
        collectedFeesToken0
        collectedFeesToken1
      }}
    }}
    """

    def get_top_pools(self, n: int = 100) -> List[Dict]:
        results = []
        for rec in self.paginate(self.POOLS_QUERY, "pools", page_size=min(n, 1000), max_pages=1):
            results.append(rec)
            if len(results) >= n:
                break
        return results

    def get_pool_ticks(self, pool_id: str) -> List[Dict]:
        template = self.POOL_TICKS_QUERY.replace('"{pool_id}"', f'"{pool_id}"')
        return list(self.paginate(template, "ticks", page_size=1000, max_pages=20))

    def get_pool_swaps(self, pool_id: str, max_swaps: int = 5000) -> List[Dict]:
        template = self.SWAPS_QUERY.replace('"{pool_id}"', f'"{pool_id}"')
        results = []
        for rec in self.paginate(template, "swaps", page_size=1000, max_pages=10):
            results.append(rec)
            if len(results) >= max_swaps:
                break
        return results

    def get_token_history(self, token_id: str, days: int = 90) -> List[Dict]:
        template = self.TOKEN_DAY_DATA_QUERY.replace('"{token_id}"', f'"{token_id}"')
        return list(self.paginate(template, "tokenDayDatas", page_size=days, max_pages=1))

    def get_wallet_positions(self, owner: str) -> List[Dict]:
        template = self.POSITIONS_QUERY.replace('"{owner}"', f'"{owner}"')
        return list(self.paginate(template, "positions", page_size=1000, max_pages=5))


# ---------------------------------------------------------------------------
# Nansen pattern client (simulated / pattern matching layer)
# ---------------------------------------------------------------------------

@dataclass
class NansenLabel:
    address: str
    labels: List[str]
    entity_name: Optional[str]
    entity_type: Optional[str]   # "cex", "dex", "fund", "whale", "miner", etc.
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    tx_count: int
    balance_usd: float


class NansenClient:
    """
    Wraps a Nansen-style wallet-labeling API.
    Falls back to heuristic labeling if API key is absent.
    """

    KNOWN_EXCHANGE_PREFIXES = {
        "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance",
        "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF": "Binance",
        "0x564286362092D8e7936f0549571a803B203aAceD": "Binance",
        "0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43": "Coinbase",
        "0x77696bb39917C91A0c3908D577d5e322095425cA": "Kraken",
        "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken",
    }

    def __init__(self, api_key: str = "", cache: DiskCache = _global_cache):
        self.api_key = api_key or _env("NANSEN_API_KEY")
        self.session = _build_session()
        self.cache = cache

    def get_label(self, address: str) -> NansenLabel:
        cache_key = f"nansen:label:{address.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return NansenLabel(**cached)

        if self.api_key:
            label = self._fetch_label(address)
        else:
            label = self._heuristic_label(address)

        self.cache.set(cache_key, {
            "address": label.address,
            "labels": label.labels,
            "entity_name": label.entity_name,
            "entity_type": label.entity_type,
            "first_seen": label.first_seen.isoformat() if label.first_seen else None,
            "last_seen": label.last_seen.isoformat() if label.last_seen else None,
            "tx_count": label.tx_count,
            "balance_usd": label.balance_usd,
        })
        return label

    def _fetch_label(self, address: str) -> NansenLabel:
        url = f"{NANSEN_BASE}/address/{address}"
        resp = self.session.get(url, headers={"X-API-KEY": self.api_key}, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        d = resp.json()
        return NansenLabel(
            address=address,
            labels=d.get("labels", []),
            entity_name=d.get("entityName"),
            entity_type=d.get("entityType"),
            first_seen=_parse_dt(d.get("firstSeen")),
            last_seen=_parse_dt(d.get("lastSeen")),
            tx_count=d.get("txCount", 0),
            balance_usd=d.get("balanceUsd", 0.0),
        )

    def _heuristic_label(self, address: str) -> NansenLabel:
        """Rule-based labeling from known address list."""
        entity = self.KNOWN_EXCHANGE_PREFIXES.get(address)
        labels = []
        entity_type = None
        if entity:
            labels = ["exchange", "cex"]
            entity_type = "cex"
        return NansenLabel(
            address=address,
            labels=labels,
            entity_name=entity,
            entity_type=entity_type,
            first_seen=None,
            last_seen=None,
            tx_count=0,
            balance_usd=0.0,
        )

    def bulk_label(self, addresses: List[str]) -> Dict[str, NansenLabel]:
        return {addr: self.get_label(addr) for addr in addresses}

    def is_exchange(self, address: str) -> bool:
        label = self.get_label(address)
        return label.entity_type == "cex" or "exchange" in label.labels

    def is_smart_money(self, address: str) -> bool:
        label = self.get_label(address)
        return "smart_money" in label.labels or "fund" in label.entity_type if label.entity_type else False


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Etherscan / EVM RPC client
# ---------------------------------------------------------------------------

@dataclass
class EthTransaction:
    hash: str
    block_number: int
    timestamp: int
    from_addr: str
    to_addr: str
    value_eth: float
    value_usd: float
    gas_used: int
    gas_price_gwei: float
    input_data: str
    is_contract_interaction: bool

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


class EtherscanClient:
    """Wraps Etherscan API for transaction history and address info."""

    BASE = "https://api.etherscan.io/api"

    def __init__(self, api_key: str = "", cache: DiskCache = _global_cache):
        self.api_key = api_key or _env("ETHERSCAN_API_KEY")
        self.session = _build_session()
        self.cache = cache
        self.limiter = RateLimiter(5)   # free tier: 5 req/s

    def _get(self, params: Dict) -> Dict:
        params["apikey"] = self.api_key
        cache_key = f"etherscan:{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        self.limiter.wait()
        resp = self.session.get(self.BASE, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1":
            self.cache.set(cache_key, data)
        return data

    def get_tx_list(self, address: str, start_block: int = 0, end_block: int = 99999999, page: int = 1, offset: int = 1000) -> List[Dict]:
        data = self._get({
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "desc",
        })
        return data.get("result", []) if isinstance(data.get("result"), list) else []

    def get_erc20_transfers(self, address: str, token_address: str = "", start_block: int = 0) -> List[Dict]:
        params: Dict[str, Any] = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": start_block,
            "sort": "desc",
        }
        if token_address:
            params["contractaddress"] = token_address
        data = self._get(params)
        return data.get("result", []) if isinstance(data.get("result"), list) else []

    def get_balance(self, address: str) -> float:
        """Returns ETH balance."""
        data = self._get({"module": "account", "action": "balance", "address": address, "tag": "latest"})
        try:
            return int(data.get("result", 0)) / 1e18
        except (TypeError, ValueError):
            return 0.0

    def get_internal_tx(self, address: str, start_block: int = 0) -> List[Dict]:
        data = self._get({
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": start_block,
            "sort": "desc",
        })
        return data.get("result", []) if isinstance(data.get("result"), list) else []


# ---------------------------------------------------------------------------
# CoinGecko price client (free, no key required)
# ---------------------------------------------------------------------------

class CoinGeckoClient:
    """Simple CoinGecko REST client for price data and market caps."""

    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, cache: DiskCache = _global_cache):
        self.session = _build_session()
        self.cache = cache
        self.limiter = RateLimiter(30)   # free tier

    def _get(self, path: str, params: Dict = None) -> Any:
        url = f"{self.BASE}/{path}"
        cache_key = f"coingecko:{url}:{json.dumps(params or {}, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        self.limiter.wait()
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, data)
        return data

    def get_price(self, coin_ids: List[str], vs_currencies: List[str] = None) -> Dict:
        return self._get("simple/price", {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies or ["usd"]),
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
        })

    def get_market_chart(self, coin_id: str, days: int = 30, interval: str = "daily") -> Dict:
        return self._get(f"coins/{coin_id}/market_chart", {
            "vs_currency": "usd",
            "days": days,
            "interval": interval,
        })

    def get_ohlc(self, coin_id: str, days: int = 30) -> List[List[float]]:
        """Returns list of [timestamp, open, high, low, close]."""
        return self._get(f"coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": days})

    def get_global(self) -> Dict:
        return self._get("global")

    def get_coin_info(self, coin_id: str) -> Dict:
        return self._get(f"coins/{coin_id}", {
            "localization": False,
            "tickers": False,
            "market_data": True,
            "community_data": False,
            "developer_data": False,
        })

    def search_coin(self, query: str) -> List[Dict]:
        result = self._get("search", {"query": query})
        return result.get("coins", [])

    def get_trending(self) -> List[Dict]:
        result = self._get("search/trending")
        return result.get("coins", [])


# ---------------------------------------------------------------------------
# Composite data provider
# ---------------------------------------------------------------------------

class OnChainDataProvider:
    """
    Facade that combines all data sources into a single interface.
    Used by higher-level analytics modules.
    """

    def __init__(
        self,
        glassnode_key: str = "",
        dune_key: str = "",
        etherscan_key: str = "",
        nansen_key: str = "",
        cache_dir: str = ".cache/onchain",
        cache_ttl: int = 300,
    ):
        cache = DiskCache(cache_dir, cache_ttl)
        self.glassnode = GlassnodeClient(glassnode_key, cache)
        self.dune = DuneClient(dune_key, cache)
        self.graph = TheGraphClient(cache=cache)
        self.nansen = NansenClient(nansen_key, cache)
        self.etherscan = EtherscanClient(etherscan_key, cache)
        self.coingecko = CoinGeckoClient(cache)
        self._cache = cache

    def get_btc_metrics_bundle(self, days: int = 90) -> Dict[str, List[GlassnodeMetric]]:
        """Fetch a standard bundle of BTC on-chain metrics."""
        end = int(time.time())
        start = end - days * 86400
        return {
            "active_addresses": self.glassnode.active_addresses(start=start, end=end),
            "nvt":              self.glassnode.nvt_ratio(start=start, end=end),
            "mvrv":             self.glassnode.mvrv_ratio(start=start, end=end),
            "sopr":             self.glassnode.sopr(start=start, end=end),
            "exchange_inflow":  self.glassnode.exchange_inflow(start=start, end=end),
            "exchange_outflow": self.glassnode.exchange_outflow(start=start, end=end),
            "tx_count":         self.glassnode.transaction_count(start=start, end=end),
            "tx_volume":        self.glassnode.transaction_volume(start=start, end=end),
            "open_interest":    self.glassnode.open_interest(start=start, end=end),
            "funding_rate":     self.glassnode.funding_rate(start=start, end=end),
        }

    def get_uniswap_top_pools(self, n: int = 50) -> List[Dict]:
        return self.graph.get_top_pools(n)

    def get_price_usd(self, coin_id: str) -> float:
        data = self.coingecko.get_price([coin_id])
        return data.get(coin_id, {}).get("usd", 0.0)

    def label_addresses(self, addresses: List[str]) -> Dict[str, NansenLabel]:
        return self.nansen.bulk_label(addresses)

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Utility: async batch fetcher
# ---------------------------------------------------------------------------

async def _async_get(session_: Any, url: str, params: Dict) -> Dict:
    """Async HTTP GET — uses aiohttp if available, else falls back to requests."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)) as r:
                return await r.json()
    except ImportError:
        loop = asyncio.get_event_loop()
        sess = _build_session()
        return await loop.run_in_executor(None, lambda: sess.get(url, params=params, timeout=DEFAULT_TIMEOUT).json())


async def async_fetch_glassnode_metrics(
    api_key: str,
    metrics: List[Tuple[str, str]],   # [(tier, metric), ...]
    asset: str = "BTC",
    start: int = 0,
    end: int = 0,
    interval: str = "24h",
) -> Dict[str, List[Dict]]:
    """Fetch multiple Glassnode metrics concurrently."""
    end = end or int(time.time())
    start = start or end - 90 * 86400

    async def _fetch_one(tier: str, metric: str) -> Tuple[str, List[Dict]]:
        params = {"a": asset, "i": interval, "s": start, "u": end, "api_key": api_key}
        url = f"{GLASSNODE_BASE}/{tier}/{metric}"
        data = await _async_get(None, url, params)
        return f"{tier}/{metric}", data

    tasks = [_fetch_one(tier, metric) for tier, metric in metrics]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, List[Dict]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.error("Async fetch error: %s", r)
        else:
            key, data = r
            out[key] = data
    return out


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-chain data fetcher CLI")
    parser.add_argument("--source", choices=["glassnode", "dune", "graph", "coingecko"], default="coingecko")
    parser.add_argument("--metric", default="price")
    parser.add_argument("--asset", default="bitcoin")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    provider = OnChainDataProvider()

    if args.source == "coingecko":
        data = provider.coingecko.get_market_chart(args.asset, args.days)
        prices = data.get("prices", [])
        print(f"Fetched {len(prices)} price points for {args.asset}")
        if prices:
            print(f"Latest: ${prices[-1][1]:,.2f} at {datetime.fromtimestamp(prices[-1][0]/1000, tz=timezone.utc)}")
    elif args.source == "graph":
        pools = provider.get_uniswap_top_pools(10)
        print(f"Top {len(pools)} Uniswap V3 pools:")
        for p in pools:
            print(f"  {p.get('token0',{}).get('symbol')}/{p.get('token1',{}).get('symbol')} "
                  f"TVL=${float(p.get('totalValueLockedUSD',0)):,.0f}")
