"""
whale_tracker.py -- Whale wallet tracking and exchange flow analysis.

Covers:
  - Large wallet movement detection with configurable USD thresholds
  - Exchange deposit / withdrawal classification (bearish / bullish)
  - Address classification: EXCHANGE, WHALE, MINER, DEFI, UNKNOWN
  - Net whale flow z-score signal normalized to [-1, +1]
  - In-memory event store with SQLite persistence cache
  - Cold-storage heuristic: large balance + few transactions
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WHALE_THRESHOLD_USD: float = 1_000_000.0
MEGA_WHALE_THRESHOLD_USD: float = 10_000_000.0

# Approximate breakeven cost basis for mid-tier BTC miners (USD)
MINER_BREAKEVEN_ESTIMATE_USD: float = 20_000.0

# Z-score saturation -- raw z-scores beyond this magnitude clip to +/-1 signal
ZSCORE_CLIP: float = 3.0

# Maximum events held in memory per asset
MAX_EVENTS_PER_ASSET: int = 10_000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    EXCHANGE_DEPOSIT    = "EXCHANGE_DEPOSIT"
    EXCHANGE_WITHDRAWAL = "EXCHANGE_WITHDRAWAL"
    TRANSFER            = "TRANSFER"
    MINT                = "MINT"
    BURN                = "BURN"


class AddressType(str, Enum):
    EXCHANGE = "EXCHANGE"
    WHALE    = "WHALE"
    MINER    = "MINER"
    DEFI     = "DEFI"
    UNKNOWN  = "UNKNOWN"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """Raw on-chain transaction record."""
    tx_hash:               str
    asset:                 str
    from_addr:             str
    to_addr:               str
    amount:                float       # native units (e.g. BTC)
    usd_value:             float       # USD at time of tx
    timestamp:             datetime
    is_exchange_deposit:   bool = False
    is_exchange_withdrawal: bool = False

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f"Transaction amount must be non-negative, got {self.amount}")
        if self.usd_value < 0:
            raise ValueError(f"USD value must be non-negative, got {self.usd_value}")
        # Ensure timestamp is tz-aware
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class WhaleEvent:
    """Processed whale activity event derived from a Transaction."""
    tx_hash:      str
    asset:        str
    event_type:   str               # EventType value
    amount_usd:   float
    significance: float             # z-score vs historical distribution
    is_bullish:   bool              # True = bullish directional bias
    timestamp:    datetime
    from_addr:    str = ""
    to_addr:      str = ""
    from_type:    str = AddressType.UNKNOWN
    to_type:      str = AddressType.UNKNOWN

    @property
    def is_bearish(self) -> bool:
        return not self.is_bullish


# ---------------------------------------------------------------------------
# Known exchange address registry
# ---------------------------------------------------------------------------

# Partial address prefix / exact address sets for well-known exchanges.
# In production these would be loaded from a maintained database.
_KNOWN_EXCHANGE_ADDRESSES_ETH: frozenset[str] = frozenset({
    # Binance hot wallets (partial list)
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
    "0xd551234ae421e3bcba99a0da6d736074f22192ff",
    "0x564286362092d8e7936f0549571a803b203aaced",
    "0x0681d8db095565fe8a346fa0277bffde9c0edbbf",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",
    # Huobi
    "0xab5c66752a9e8167967685f1450532fb96d5d24f",
    "0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b",
})

_KNOWN_EXCHANGE_ADDRESSES_BTC: frozenset[str] = frozenset({
    # Binance cold wallet approximations
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    # Coinbase
    "1FzWLkAahHooV3kzTgyx6qsswXJ6sCXkSR",
    "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",
    # Kraken
    "3E5tu4VgKFGZE3pVLdnEKXqBxMBbHy8sKk",
})

_KNOWN_DEFI_PREFIXES: Tuple[str, ...] = (
    "0x7a250d5630",  # Uniswap v2 router
    "0xe592427a0a",  # Uniswap v3 router
    "0x1111111254",  # 1inch
    "0xd9e1cE17f2",  # SushiSwap
    "0xba12222222",  # Balancer vault
    "0xabea9132eb",  # ZkSync bridge
)


# ---------------------------------------------------------------------------
# Address classifier
# ---------------------------------------------------------------------------

class AddressClassifier:
    """
    Classify blockchain addresses into known categories.

    Uses a combination of:
      - Static address registry (known exchanges, DeFi routers)
      - Heuristics: cold-storage whale detection (large balance, few txs)
      - Miner address patterns (coinbase receivers)
    """

    def __init__(self) -> None:
        # Mutable registry to allow runtime additions
        self._exchange_eth: set[str] = set(_KNOWN_EXCHANGE_ADDRESSES_ETH)
        self._exchange_btc: set[str] = set(_KNOWN_EXCHANGE_ADDRESSES_BTC)
        # address -> (balance_usd, tx_count) for heuristics
        self._address_stats: Dict[str, Tuple[float, int]] = {}
        # Cached classifications
        self._cache: Dict[str, AddressType] = {}

    def register_exchange_address(self, address: str) -> None:
        """Add an address to the known exchange registry."""
        addr_lower = address.lower()
        if addr_lower.startswith("0x"):
            self._exchange_eth.add(addr_lower)
        else:
            self._exchange_btc.add(address)
        self._cache.pop(addr_lower, None)

    def update_address_stats(
        self,
        address: str,
        balance_usd: float,
        tx_count: int,
    ) -> None:
        """Record balance and activity stats for cold-storage heuristic."""
        self._address_stats[address.lower()] = (balance_usd, tx_count)
        self._cache.pop(address.lower(), None)

    def classify(self, address: str) -> AddressType:
        """
        Return the AddressType for a given address.

        Classification priority:
          1. Cache hit
          2. Known exchange registry
          3. Known DeFi router prefix
          4. Cold-storage heuristic (large balance, few txs)
          5. UNKNOWN
        """
        addr_lower = address.lower()

        cached = self._cache.get(addr_lower)
        if cached is not None:
            return cached

        result = self._classify_uncached(addr_lower)
        self._cache[addr_lower] = result
        return result

    def _classify_uncached(self, addr_lower: str) -> AddressType:
        # Exchange registry check
        if addr_lower in self._exchange_eth or addr_lower in self._exchange_btc:
            return AddressType.EXCHANGE

        # DeFi prefix check (Ethereum addresses only)
        if addr_lower.startswith("0x"):
            for prefix in _KNOWN_DEFI_PREFIXES:
                if addr_lower.startswith(prefix.lower()):
                    return AddressType.DEFI

        # Miner heuristic: BTC P2PKH addresses frequently appearing in coinbase
        # In practice this would query a miner address database; here we use
        # a stub that could be extended.
        if self._looks_like_miner(addr_lower):
            return AddressType.MINER

        # Cold-storage whale heuristic
        stats = self._address_stats.get(addr_lower)
        if stats is not None:
            balance_usd, tx_count = stats
            # Large balance with very low transaction frequency
            if balance_usd >= DEFAULT_WHALE_THRESHOLD_USD and tx_count <= 50:
                return AddressType.WHALE

        return AddressType.UNKNOWN

    def _looks_like_miner(self, addr_lower: str) -> bool:
        """
        Stub heuristic: prefix-match known miner pool addresses.
        In production, replace with a proper miner database query.
        """
        _known_miner_prefixes = (
            "1cjpb",   # AntPool legacy prefix sample
            "1hz",     # F2Pool legacy prefix sample
        )
        for prefix in _known_miner_prefixes:
            if addr_lower.startswith(prefix):
                return True
        return False

    def bulk_classify(self, addresses: List[str]) -> Dict[str, AddressType]:
        """Classify a list of addresses, returning a mapping."""
        return {addr: self.classify(addr) for addr in addresses}


# ---------------------------------------------------------------------------
# Whale tracker
# ---------------------------------------------------------------------------

class WhaleTracker:
    """
    Track large wallet movements and produce directional market signals.

    Signal logic:
      - EXCHANGE_DEPOSIT (whale -> exchange): bearish (selling intent)
      - EXCHANGE_WITHDRAWAL (exchange -> whale): bullish (accumulation)
      - Net flow z-score is normalized to [-1, +1] via tanh-like clipping

    Storage:
      - In-memory deque per asset (bounded by MAX_EVENTS_PER_ASSET)
      - Optional SQLite cache for persistence across restarts
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        classifier: Optional[AddressClassifier] = None,
    ) -> None:
        self._thresholds: Dict[str, float] = {}           # asset -> min USD
        self._events: Dict[str, Deque[WhaleEvent]] = defaultdict(
            lambda: deque(maxlen=MAX_EVENTS_PER_ASSET)
        )
        self._classifier = classifier or AddressClassifier()
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Initialize SQLite schema for event persistence."""
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS whale_events (
                    tx_hash      TEXT NOT NULL,
                    asset        TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    amount_usd   REAL NOT NULL,
                    significance REAL NOT NULL,
                    is_bullish   INTEGER NOT NULL,
                    timestamp    TEXT NOT NULL,
                    from_addr    TEXT,
                    to_addr      TEXT,
                    from_type    TEXT,
                    to_type      TEXT,
                    PRIMARY KEY (tx_hash, asset)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_whale_asset_ts "
                "ON whale_events (asset, timestamp)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to initialize whale tracker DB: %s", exc)
            self._conn = None

    def _persist_event(self, evt: WhaleEvent) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO whale_events
                  (tx_hash, asset, event_type, amount_usd, significance,
                   is_bullish, timestamp, from_addr, to_addr, from_type, to_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evt.tx_hash,
                    evt.asset,
                    evt.event_type,
                    evt.amount_usd,
                    evt.significance,
                    int(evt.is_bullish),
                    evt.timestamp.isoformat(),
                    evt.from_addr,
                    evt.to_addr,
                    evt.from_type,
                    evt.to_type,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Failed to persist whale event %s: %s", evt.tx_hash, exc)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def define_whale_threshold(
        self,
        asset: str,
        threshold_usd: float = DEFAULT_WHALE_THRESHOLD_USD,
    ) -> None:
        """Set the minimum USD transfer size considered a whale event for an asset."""
        if threshold_usd <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold_usd}")
        self._thresholds[asset.upper()] = threshold_usd
        logger.debug("Whale threshold for %s set to $%.0f", asset.upper(), threshold_usd)

    def _get_threshold(self, asset: str) -> float:
        return self._thresholds.get(asset.upper(), DEFAULT_WHALE_THRESHOLD_USD)

    # ------------------------------------------------------------------
    # Transaction processing
    # ------------------------------------------------------------------

    def process_transaction(self, tx: Transaction) -> Optional[WhaleEvent]:
        """
        Evaluate a raw transaction and emit a WhaleEvent if it qualifies.

        Returns None if the transaction is below the whale threshold.
        """
        asset = tx.asset.upper()
        threshold = self._get_threshold(asset)

        if tx.usd_value < threshold:
            return None

        from_type = self._classifier.classify(tx.from_addr)
        to_type   = self._classifier.classify(tx.to_addr)

        event_type, is_bullish = self._classify_event(
            from_type, to_type, tx.is_exchange_deposit, tx.is_exchange_withdrawal
        )

        # Compute significance relative to historical distribution for this asset
        significance = self._compute_significance(asset, tx.usd_value)

        evt = WhaleEvent(
            tx_hash=tx.tx_hash,
            asset=asset,
            event_type=event_type,
            amount_usd=tx.usd_value,
            significance=significance,
            is_bullish=is_bullish,
            timestamp=tx.timestamp,
            from_addr=tx.from_addr,
            to_addr=tx.to_addr,
            from_type=from_type.value,
            to_type=to_type.value,
        )

        self._events[asset].append(evt)
        self._persist_event(evt)
        return evt

    @staticmethod
    def _classify_event(
        from_type: AddressType,
        to_type:   AddressType,
        flagged_deposit:    bool,
        flagged_withdrawal: bool,
    ) -> Tuple[str, bool]:
        """
        Determine EventType and bullish/bearish bias.

        Rules:
          - To exchange   -> EXCHANGE_DEPOSIT    -> bearish
          - From exchange -> EXCHANGE_WITHDRAWAL -> bullish
          - Both exchange -> TRANSFER (exchange internal) -> neutral (not bullish)
          - Neither       -> TRANSFER -> neutral
        """
        if flagged_deposit or to_type == AddressType.EXCHANGE:
            return EventType.EXCHANGE_DEPOSIT.value, False
        if flagged_withdrawal or from_type == AddressType.EXCHANGE:
            return EventType.EXCHANGE_WITHDRAWAL.value, True
        if to_type == AddressType.UNKNOWN and from_type == AddressType.WHALE:
            return EventType.TRANSFER.value, False
        return EventType.TRANSFER.value, False

    def _compute_significance(self, asset: str, usd_value: float) -> float:
        """
        Compute z-score of this transaction vs recent whale events for the asset.
        Returns 0.0 if fewer than 2 prior events exist.
        """
        prior_values = [e.amount_usd for e in self._events[asset]]
        if len(prior_values) < 2:
            return 0.0
        mu  = statistics.mean(prior_values)
        std = statistics.stdev(prior_values)
        if std < 1e-9:
            return 0.0
        return (usd_value - mu) / std

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def recent_whale_events(
        self,
        asset: str,
        hours: int = 24,
    ) -> List[WhaleEvent]:
        """Return whale events for the given asset within the last N hours."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        return [
            e for e in self._events[asset.upper()]
            if e.timestamp >= cutoff
        ]

    def net_whale_flow(self, asset: str, hours: int = 24) -> float:
        """
        Compute net USD flow by whales over the last N hours.

        Convention:
          - Exchange withdrawals (accumulation) add positive flow
          - Exchange deposits (selling intent) add negative flow
          - Pure transfers are neutral (zero contribution)

        Returns the net signed USD amount.
        """
        events = self.recent_whale_events(asset, hours)
        net = 0.0
        for evt in events:
            if evt.event_type == EventType.EXCHANGE_WITHDRAWAL.value:
                net += evt.amount_usd
            elif evt.event_type == EventType.EXCHANGE_DEPOSIT.value:
                net -= evt.amount_usd
        return net

    def whale_signal(self, asset: str) -> float:
        """
        Produce a directional signal in [-1, +1] for the asset.

        Method:
          1. Compute net_whale_flow for 24h and 7d windows
          2. Normalize each by the rolling standard deviation of hourly flows
          3. Blend 24h (70%) and 7d (30%) z-scores
          4. Clip to [-ZSCORE_CLIP, +ZSCORE_CLIP] then divide by ZSCORE_CLIP

        Returns 0.0 if insufficient data.
        """
        flow_24h = self.net_whale_flow(asset, hours=24)
        flow_7d  = self.net_whale_flow(asset, hours=168)

        # Compute hourly flow buckets from available events (up to 7d)
        hourly_flows = self._compute_hourly_flows(asset, hours=168)
        if len(hourly_flows) < 4:
            # Not enough history -- return raw sign only
            raw = flow_24h / (abs(flow_24h) + 1e-9) if abs(flow_24h) > 1e-3 else 0.0
            return float(np.clip(raw, -1.0, 1.0))

        std = float(np.std(hourly_flows)) if len(hourly_flows) > 1 else 1.0
        std = max(std, 1.0)  # guard against zero std

        # Convert flows to 24h-equivalent scale
        z_24h = flow_24h / (std * 24.0)
        z_7d  = flow_7d  / (std * 168.0)

        blended = 0.7 * z_24h + 0.3 * z_7d
        clipped = float(np.clip(blended, -ZSCORE_CLIP, ZSCORE_CLIP))
        return clipped / ZSCORE_CLIP

    def _compute_hourly_flows(
        self,
        asset: str,
        hours: int = 168,
    ) -> List[float]:
        """
        Bucket whale events into hourly net-flow bins.
        Returns a list of per-hour net USD flows.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        buckets: Dict[int, float] = defaultdict(float)

        for evt in self._events[asset.upper()]:
            if evt.timestamp < cutoff:
                continue
            # Hour index from cutoff
            delta_hours = int(
                (evt.timestamp - cutoff).total_seconds() // 3600
            )
            if evt.event_type == EventType.EXCHANGE_WITHDRAWAL.value:
                buckets[delta_hours] += evt.amount_usd
            elif evt.event_type == EventType.EXCHANGE_DEPOSIT.value:
                buckets[delta_hours] -= evt.amount_usd

        return list(buckets.values())

    # ------------------------------------------------------------------
    # Bulk loading helpers
    # ------------------------------------------------------------------

    def bulk_process(
        self,
        transactions: List[Transaction],
    ) -> List[WhaleEvent]:
        """Process a batch of transactions and return all emitted events."""
        results: List[WhaleEvent] = []
        for tx in transactions:
            evt = self.process_transaction(tx)
            if evt is not None:
                results.append(evt)
        return results

    def load_from_db(self, asset: str) -> int:
        """
        Reload persisted events from SQLite into memory for the given asset.
        Returns the number of events loaded.
        """
        if self._conn is None:
            return 0
        asset = asset.upper()
        try:
            rows = self._conn.execute(
                """
                SELECT tx_hash, asset, event_type, amount_usd, significance,
                       is_bullish, timestamp, from_addr, to_addr, from_type, to_type
                FROM whale_events
                WHERE asset = ?
                ORDER BY timestamp ASC
                """,
                (asset,),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("DB load failed for asset %s: %s", asset, exc)
            return 0

        count = 0
        for row in rows:
            (
                tx_hash, asset_col, event_type, amount_usd, significance,
                is_bullish, timestamp_str, from_addr, to_addr, from_type, to_type,
            ) = row
            ts = datetime.fromisoformat(timestamp_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            evt = WhaleEvent(
                tx_hash=tx_hash,
                asset=asset_col,
                event_type=event_type,
                amount_usd=amount_usd,
                significance=significance,
                is_bullish=bool(is_bullish),
                timestamp=ts,
                from_addr=from_addr or "",
                to_addr=to_addr or "",
                from_type=from_type or AddressType.UNKNOWN.value,
                to_type=to_type or AddressType.UNKNOWN.value,
            )
            self._events[asset].append(evt)
            count += 1

        logger.info("Loaded %d whale events for %s from DB", count, asset)
        return count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, asset: str) -> Dict[str, object]:
        """Return a human-readable summary dict for the asset."""
        events_24h = self.recent_whale_events(asset, 24)
        deposits     = [e for e in events_24h if e.event_type == EventType.EXCHANGE_DEPOSIT.value]
        withdrawals  = [e for e in events_24h if e.event_type == EventType.EXCHANGE_WITHDRAWAL.value]
        transfers    = [e for e in events_24h if e.event_type == EventType.TRANSFER.value]

        return {
            "asset": asset.upper(),
            "signal": self.whale_signal(asset),
            "net_flow_24h_usd": self.net_whale_flow(asset, 24),
            "events_24h": len(events_24h),
            "deposits_24h":    len(deposits),
            "withdrawals_24h": len(withdrawals),
            "transfers_24h":   len(transfers),
            "deposit_volume_usd":    sum(e.amount_usd for e in deposits),
            "withdrawal_volume_usd": sum(e.amount_usd for e in withdrawals),
        }
