"""
whale_tracker.py — Whale wallet monitoring.

Covers:
  - Large transfer detection (>$100K threshold, configurable)
  - Exchange inflow/outflow classification
  - Accumulation vs distribution pattern detection
  - Wallet clustering (co-spend / common-input heuristic)
  - Cohort analysis (addresses grouped by behavior)
  - Alert generation
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, FrozenSet, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np

from data_fetchers import (
    DiskCache,
    EtherscanClient,
    GlassnodeClient,
    NansenClient,
    NansenLabel,
    OnChainDataProvider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

WHALE_THRESHOLD_USD = 100_000      # minimum transfer to classify as whale move
LARGE_WHALE_THRESHOLD_USD = 1_000_000
MEGA_WHALE_THRESHOLD_USD  = 10_000_000


class FlowDirection(Enum):
    EXCHANGE_INFLOW  = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    WHALE_TO_WHALE   = "whale_to_whale"
    WHALE_TO_UNKNOWN = "whale_to_unknown"
    UNKNOWN_TO_WHALE = "unknown_to_whale"
    INTERNAL         = "internal"


class WalletBehavior(Enum):
    ACCUMULATOR    = "accumulator"
    DISTRIBUTOR    = "distributor"
    TRADER         = "trader"
    DORMANT        = "dormant"
    EXCHANGE       = "exchange"
    MINER          = "miner"
    UNKNOWN        = "unknown"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WhaleTransfer:
    tx_hash: str
    block_number: int
    timestamp: int
    from_addr: str
    to_addr: str
    value_native: float         # ETH / BTC etc.
    value_usd: float
    asset_symbol: str
    flow_direction: FlowDirection
    from_label: Optional[str] = None
    to_label: Optional[str] = None
    is_contract: bool = False

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def is_mega_whale(self) -> bool:
        return self.value_usd >= MEGA_WHALE_THRESHOLD_USD

    @property
    def size_category(self) -> str:
        if self.value_usd >= MEGA_WHALE_THRESHOLD_USD:
            return "mega"
        if self.value_usd >= LARGE_WHALE_THRESHOLD_USD:
            return "large"
        return "whale"


@dataclass
class WalletProfile:
    address: str
    label: Optional[NansenLabel]
    total_received_usd: float
    total_sent_usd: float
    net_flow_usd: float                    # positive = accumulating
    tx_count: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    avg_tx_size_usd: float
    behavior: WalletBehavior
    exchange_inflow_usd: float
    exchange_outflow_usd: float
    cluster_id: Optional[int] = None
    related_addresses: List[str] = field(default_factory=list)

    @property
    def is_accumulating(self) -> bool:
        return self.net_flow_usd > 0

    @property
    def is_distributing(self) -> bool:
        return self.net_flow_usd < 0

    @property
    def exchange_net_flow(self) -> float:
        return self.exchange_outflow_usd - self.exchange_inflow_usd


@dataclass
class AccumulationPattern:
    address: str
    start_date: datetime
    end_date: datetime
    total_accumulated_usd: float
    avg_buy_size_usd: float
    buy_count: int
    sell_count: int
    buy_sell_ratio: float
    dca_score: float          # 0-1 how much it looks like DCA
    stealth_score: float      # 0-1 how much it avoids exchanges
    pattern_type: str         # "dca", "lump_sum", "mixed", "distribution"


@dataclass
class WalletCluster:
    cluster_id: int
    addresses: List[str]
    size: int
    total_balance_usd: float
    dominant_behavior: WalletBehavior
    is_likely_entity: bool       # high confidence same entity
    entity_name: Optional[str]   # if known via Nansen
    common_counterparties: List[str]


@dataclass
class WhaleAlert:
    alert_id: str
    timestamp: datetime
    severity: str                # "critical", "high", "medium", "low"
    alert_type: str              # "large_transfer", "exchange_inflow_spike", etc.
    message: str
    transfers: List[WhaleTransfer]
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exchange address registry
# ---------------------------------------------------------------------------

class ExchangeRegistry:
    """Known exchange hot/cold wallet addresses."""

    ADDRESSES: Dict[str, str] = {
        # Binance
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
        "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance",
        "0x564286362092d8e7936f0549571a803b203aaced": "Binance",
        "0x0681d8db095565fe8a346fa0277bffde9c0edbbf": "Binance",
        "0xfe9e8709d3215310075d67e3ed32a380ccf451c8": "Binance",
        # Coinbase
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "Coinbase",
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
        "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
        # Kraken
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
        "0xae2d4617c862309a3d75a0ffb358c7a5009c673f": "Kraken",
        # Bitfinex
        "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa": "Bitfinex",
        "0xd24400ae8bfebb18ca49be86258a3c749cf46853": "Bitfinex",
        # OKX
        "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
        # Bybit
        "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit",
    }

    def __init__(self):
        self._addr_set: Set[str] = {a.lower() for a in self.ADDRESSES.keys()}
        self._addr_map: Dict[str, str] = {a.lower(): n for a, n in self.ADDRESSES.items()}

    def is_exchange(self, address: str) -> bool:
        return address.lower() in self._addr_set

    def get_exchange_name(self, address: str) -> Optional[str]:
        return self._addr_map.get(address.lower())

    def classify_flow(self, from_addr: str, to_addr: str) -> FlowDirection:
        from_ex = self.is_exchange(from_addr)
        to_ex = self.is_exchange(to_addr)
        if from_ex and to_ex:
            return FlowDirection.INTERNAL
        if to_ex:
            return FlowDirection.EXCHANGE_INFLOW
        if from_ex:
            return FlowDirection.EXCHANGE_OUTFLOW
        return FlowDirection.WHALE_TO_UNKNOWN


_exchange_registry = ExchangeRegistry()


# ---------------------------------------------------------------------------
# Transfer detector
# ---------------------------------------------------------------------------

class WhaleTransferDetector:
    """
    Scans blockchain transaction data for large transfers.
    Supports both real-time polling and historical backfilling.
    """

    def __init__(
        self,
        provider: OnChainDataProvider,
        threshold_usd: float = WHALE_THRESHOLD_USD,
        asset_prices: Dict[str, float] = None,
    ):
        self.provider = provider
        self.threshold_usd = threshold_usd
        self.asset_prices = asset_prices or {}
        self._seen_hashes: Set[str] = set()

    def _price_usd(self, symbol: str) -> float:
        if symbol in self.asset_prices:
            return self.asset_prices[symbol]
        try:
            coin_map = {"ETH": "ethereum", "BTC": "bitcoin", "USDC": "usd-coin", "USDT": "tether"}
            coin_id = coin_map.get(symbol.upper(), symbol.lower())
            price = self.provider.get_price_usd(coin_id)
            self.asset_prices[symbol] = price
            return price
        except Exception:
            return 1.0

    def scan_address(
        self,
        address: str,
        start_block: int = 0,
        max_tx: int = 1000,
    ) -> List[WhaleTransfer]:
        """Scan a single address for whale-size transfers."""
        raw_txs = self.provider.etherscan.get_tx_list(address, start_block=start_block, offset=max_tx)
        eth_price = self._price_usd("ETH")

        results = []
        for tx in raw_txs:
            try:
                value_eth = int(tx.get("value", 0)) / 1e18
                value_usd = value_eth * eth_price
                if value_usd < self.threshold_usd:
                    continue
                tx_hash = tx.get("hash", "")
                if tx_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(tx_hash)

                from_addr = tx.get("from", "")
                to_addr = tx.get("to", "")
                direction = _exchange_registry.classify_flow(from_addr, to_addr)

                results.append(WhaleTransfer(
                    tx_hash=tx_hash,
                    block_number=int(tx.get("blockNumber", 0)),
                    timestamp=int(tx.get("timeStamp", 0)),
                    from_addr=from_addr,
                    to_addr=to_addr,
                    value_native=value_eth,
                    value_usd=value_usd,
                    asset_symbol="ETH",
                    flow_direction=direction,
                    from_label=_exchange_registry.get_exchange_name(from_addr),
                    to_label=_exchange_registry.get_exchange_name(to_addr),
                    is_contract=tx.get("input", "0x") != "0x",
                ))
            except Exception as exc:
                logger.debug("TX parse error: %s", exc)

        return sorted(results, key=lambda t: t.timestamp, reverse=True)

    def scan_erc20_transfers(
        self,
        address: str,
        token_address: str = "",
        token_symbol: str = "USDC",
        start_block: int = 0,
    ) -> List[WhaleTransfer]:
        """Detect large ERC-20 token transfers."""
        raw = self.provider.etherscan.get_erc20_transfers(address, token_address, start_block)
        token_price = self._price_usd(token_symbol)

        results = []
        for tx in raw:
            try:
                decimals = int(tx.get("tokenDecimal", 18))
                value_tokens = int(tx.get("value", 0)) / (10**decimals)
                value_usd = value_tokens * token_price

                if value_usd < self.threshold_usd:
                    continue

                tx_hash = tx.get("hash", "")
                if tx_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(tx_hash)

                from_addr = tx.get("from", "")
                to_addr = tx.get("to", "")
                direction = _exchange_registry.classify_flow(from_addr, to_addr)

                results.append(WhaleTransfer(
                    tx_hash=tx_hash,
                    block_number=int(tx.get("blockNumber", 0)),
                    timestamp=int(tx.get("timeStamp", 0)),
                    from_addr=from_addr,
                    to_addr=to_addr,
                    value_native=value_tokens,
                    value_usd=value_usd,
                    asset_symbol=tx.get("tokenSymbol", token_symbol),
                    flow_direction=direction,
                    from_label=_exchange_registry.get_exchange_name(from_addr),
                    to_label=_exchange_registry.get_exchange_name(to_addr),
                    is_contract=False,
                ))
            except Exception as exc:
                logger.debug("ERC20 parse error: %s", exc)

        return sorted(results, key=lambda t: t.timestamp, reverse=True)

    def scan_glassnode_large_transfers(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> List[Dict]:
        """Use Glassnode's aggregated large-transfer data."""
        end = int(time.time())
        start = end - days * 86400
        inflows = self.provider.glassnode.exchange_inflow(asset, start=start, end=end)
        outflows = self.provider.glassnode.exchange_outflow(asset, start=start, end=end)

        results = []
        for inf, out in zip(inflows, outflows):
            results.append({
                "timestamp": inf.timestamp,
                "dt": inf.dt.isoformat(),
                "exchange_inflow_native": inf.value,
                "exchange_outflow_native": out.value if out else 0.0,
                "net_exchange_flow": (out.value if out else 0.0) - inf.value,
                "asset": asset,
            })
        return results


# ---------------------------------------------------------------------------
# Exchange flow analyzer
# ---------------------------------------------------------------------------

class ExchangeFlowAnalyzer:
    """
    Analyzes exchange inflow/outflow to detect macro accumulation/distribution.
    Positive net outflow = coins leaving exchanges = bullish.
    """

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def get_flow_data(self, asset: str = "BTC", days: int = 90) -> List[Dict]:
        end = int(time.time())
        start = end - days * 86400
        inflows  = self.glassnode.exchange_inflow(asset, start=start, end=end)
        outflows = self.glassnode.exchange_outflow(asset, start=start, end=end)

        ts_map: Dict[int, Dict] = {}
        for m in inflows:
            ts_map.setdefault(m.timestamp, {})["inflow"] = m.value
            ts_map[m.timestamp]["dt"] = m.dt
        for m in outflows:
            ts_map.setdefault(m.timestamp, {})["outflow"] = m.value

        rows = []
        for ts, d in sorted(ts_map.items()):
            inf = d.get("inflow", 0.0)
            out = d.get("outflow", 0.0)
            rows.append({
                "timestamp": ts,
                "dt": d["dt"].isoformat(),
                "inflow": inf,
                "outflow": out,
                "net_outflow": out - inf,
                "flow_ratio": out / inf if inf > 0 else float("inf"),
            })
        return rows

    def net_flow_signal(self, asset: str = "BTC", days: int = 30) -> Dict:
        """Returns a bullish/bearish signal based on exchange net flows."""
        flows = self.get_flow_data(asset, days)
        if not flows:
            return {"signal": "neutral", "confidence": 0.0}

        net_flows = [r["net_outflow"] for r in flows]
        avg_net = float(np.mean(net_flows))
        std_net = float(np.std(net_flows))
        z_score = avg_net / std_net if std_net > 0 else 0.0

        recent = net_flows[-7:]
        trend = (recent[-1] - recent[0]) / abs(recent[0]) if recent[0] != 0 else 0.0

        if z_score > 1.0:
            signal = "bullish"
            confidence = min(1.0, z_score / 3.0)
        elif z_score < -1.0:
            signal = "bearish"
            confidence = min(1.0, abs(z_score) / 3.0)
        else:
            signal = "neutral"
            confidence = 0.3

        return {
            "asset": asset,
            "signal": signal,
            "confidence": confidence,
            "avg_daily_net_outflow": avg_net,
            "z_score": z_score,
            "trend_7d": trend,
            "total_net_outflow": sum(net_flows),
            "lookback_days": days,
        }

    def detect_inflow_spike(
        self,
        asset: str = "BTC",
        days: int = 30,
        spike_multiplier: float = 2.5,
    ) -> List[Dict]:
        """Identify days with exchange inflow spikes (potential distribution events)."""
        flows = self.get_flow_data(asset, days)
        if not flows:
            return []

        inflows = [r["inflow"] for r in flows]
        mean_inf = float(np.mean(inflows))
        std_inf = float(np.std(inflows))
        threshold = mean_inf + spike_multiplier * std_inf

        spikes = []
        for r in flows:
            if r["inflow"] >= threshold:
                spikes.append({
                    **r,
                    "z_score": (r["inflow"] - mean_inf) / std_inf if std_inf > 0 else 0.0,
                    "spike_multiplier": r["inflow"] / mean_inf if mean_inf > 0 else 0.0,
                })
        return spikes


# ---------------------------------------------------------------------------
# Accumulation / distribution detector
# ---------------------------------------------------------------------------

class AccumulationDetector:
    """
    Detects accumulation and distribution patterns at the wallet level.
    Looks for: consistent buying, stealth accumulation (no CEX), DCA patterns.
    """

    def __init__(self, detector: WhaleTransferDetector):
        self.detector = detector

    def analyze_wallet(
        self,
        address: str,
        lookback_days: int = 90,
    ) -> AccumulationPattern:
        end_block = 99999999
        start_block = 0   # would need block-to-timestamp mapping in production
        transfers = self.detector.scan_address(address, start_block=start_block)

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        transfers = [t for t in transfers if t.dt >= cutoff]

        if not transfers:
            return AccumulationPattern(
                address=address,
                start_date=cutoff,
                end_date=datetime.now(timezone.utc),
                total_accumulated_usd=0.0,
                avg_buy_size_usd=0.0,
                buy_count=0,
                sell_count=0,
                buy_sell_ratio=0.0,
                dca_score=0.0,
                stealth_score=0.0,
                pattern_type="unknown",
            )

        # From the perspective of the address: inflows = buys, outflows = sells
        inflows = [t for t in transfers if t.to_addr.lower() == address.lower()]
        outflows = [t for t in transfers if t.from_addr.lower() == address.lower()]

        total_in  = sum(t.value_usd for t in inflows)
        total_out = sum(t.value_usd for t in outflows)
        net = total_in - total_out

        buy_count  = len(inflows)
        sell_count = len(outflows)
        buy_sell_ratio = buy_count / sell_count if sell_count else float("inf")

        # DCA score: are buys evenly spaced in time?
        dca_score = 0.0
        if len(inflows) >= 3:
            times = sorted([t.timestamp for t in inflows])
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
            dca_score = max(0.0, 1.0 - cv)   # lower CV = more regular = higher DCA score

        # Stealth score: fraction of inflows NOT from exchanges
        cex_inflows = sum(1 for t in inflows if t.flow_direction == FlowDirection.EXCHANGE_OUTFLOW)
        stealth_score = 1.0 - (cex_inflows / buy_count) if buy_count > 0 else 0.5

        # Pattern type
        if net > 0 and buy_sell_ratio > 2.0:
            pattern_type = "accumulator"
        elif net < 0 and buy_sell_ratio < 0.5:
            pattern_type = "distribution"
        elif buy_count > 10 and dca_score > 0.6:
            pattern_type = "dca"
        elif buy_count > 0 or sell_count > 0:
            pattern_type = "mixed"
        else:
            pattern_type = "dormant"

        start_date = min(t.dt for t in transfers) if transfers else cutoff

        return AccumulationPattern(
            address=address,
            start_date=start_date,
            end_date=datetime.now(timezone.utc),
            total_accumulated_usd=net,
            avg_buy_size_usd=total_in / buy_count if buy_count else 0.0,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_sell_ratio=buy_sell_ratio,
            dca_score=dca_score,
            stealth_score=stealth_score,
            pattern_type=pattern_type,
        )

    def bulk_analyze(
        self,
        addresses: List[str],
        lookback_days: int = 90,
    ) -> List[AccumulationPattern]:
        results = []
        for addr in addresses:
            try:
                results.append(self.analyze_wallet(addr, lookback_days))
            except Exception as exc:
                logger.warning("Accumulation analysis failed for %s: %s", addr, exc)
        return sorted(results, key=lambda p: abs(p.total_accumulated_usd), reverse=True)


# ---------------------------------------------------------------------------
# Wallet clustering
# ---------------------------------------------------------------------------

class WalletClusterer:
    """
    Groups wallets by behavioral similarity and co-spend patterns.
    Uses union-find for connected-component clustering.
    """

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = defaultdict(int)

    def _find(self, x: str) -> str:
        if self._parent.setdefault(x, x) != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def add_co_spend(self, addresses: Iterable[str]) -> None:
        """Mark a group of addresses as likely controlled by the same entity."""
        addrs = list(addresses)
        for i in range(1, len(addrs)):
            self._union(addrs[0], addrs[i])

    def add_common_counterparty(self, addr_a: str, addr_b: str) -> None:
        """Two addresses that frequently transact together may be the same entity."""
        self._union(addr_a, addr_b)

    def get_clusters(self, addresses: List[str]) -> Dict[str, int]:
        """Returns address -> cluster_id mapping."""
        cluster_roots: Dict[str, int] = {}
        cluster_map: Dict[str, int] = {}
        counter = 0
        for addr in addresses:
            root = self._find(addr)
            if root not in cluster_roots:
                cluster_roots[root] = counter
                counter += 1
            cluster_map[addr] = cluster_roots[root]
        return cluster_map

    def build_clusters(
        self,
        addresses: List[str],
        profiles: Dict[str, WalletProfile],
        nansen: NansenClient = None,
    ) -> List[WalletCluster]:
        """Build full WalletCluster objects from address list."""
        cluster_map = self.get_clusters(addresses)
        cluster_addrs: Dict[int, List[str]] = defaultdict(list)
        for addr, cid in cluster_map.items():
            cluster_addrs[cid].append(addr)

        result = []
        for cid, addrs in cluster_addrs.items():
            if len(addrs) < 2:
                continue  # singleton clusters not interesting

            profiles_in = [profiles[a] for a in addrs if a in profiles]
            total_bal = sum(
                p.total_received_usd - p.total_sent_usd
                for p in profiles_in
            )

            behavior_counts: Dict[WalletBehavior, int] = defaultdict(int)
            for p in profiles_in:
                behavior_counts[p.behavior] += 1
            dominant = max(behavior_counts, key=behavior_counts.get) if behavior_counts else WalletBehavior.UNKNOWN

            # Check if any Nansen label
            entity_name = None
            if nansen:
                for addr in addrs:
                    label = nansen.get_label(addr)
                    if label.entity_name:
                        entity_name = label.entity_name
                        break

            result.append(WalletCluster(
                cluster_id=cid,
                addresses=addrs,
                size=len(addrs),
                total_balance_usd=total_bal,
                dominant_behavior=dominant,
                is_likely_entity=len(addrs) >= 3,
                entity_name=entity_name,
                common_counterparties=[],
            ))

        return sorted(result, key=lambda c: abs(c.total_balance_usd), reverse=True)


# ---------------------------------------------------------------------------
# Whale alert generator
# ---------------------------------------------------------------------------

class WhaleAlertGenerator:
    """Generates structured alerts from whale transfer streams."""

    def __init__(self, threshold_usd: float = WHALE_THRESHOLD_USD):
        self.threshold_usd = threshold_usd
        self._alert_counter = 0

    def _next_id(self) -> str:
        self._alert_counter += 1
        return f"WA-{int(time.time())}-{self._alert_counter:04d}"

    def evaluate(self, transfers: List[WhaleTransfer]) -> List[WhaleAlert]:
        alerts = []
        for t in transfers:
            alert = self._classify_transfer(t)
            if alert:
                alerts.append(alert)
        return alerts

    def _classify_transfer(self, t: WhaleTransfer) -> Optional[WhaleAlert]:
        if t.value_usd < self.threshold_usd:
            return None

        if t.value_usd >= MEGA_WHALE_THRESHOLD_USD:
            severity = "critical"
        elif t.value_usd >= LARGE_WHALE_THRESHOLD_USD:
            severity = "high"
        else:
            severity = "medium"

        if t.flow_direction == FlowDirection.EXCHANGE_INFLOW:
            alert_type = "exchange_inflow"
            msg = (f"🐋 ${t.value_usd:,.0f} {t.asset_symbol} moved TO "
                   f"{t.to_label or t.to_addr[:8]} — potential sell pressure")
        elif t.flow_direction == FlowDirection.EXCHANGE_OUTFLOW:
            alert_type = "exchange_outflow"
            msg = (f"🐋 ${t.value_usd:,.0f} {t.asset_symbol} withdrawn FROM "
                   f"{t.from_label or t.from_addr[:8]} — potential accumulation")
        elif t.is_mega_whale:
            alert_type = "mega_whale_transfer"
            msg = (f"🚨 MEGA WHALE: ${t.value_usd:,.0f} {t.asset_symbol} "
                   f"from {t.from_addr[:8]} to {t.to_addr[:8]}")
        else:
            alert_type = "large_transfer"
            msg = (f"Large transfer: ${t.value_usd:,.0f} {t.asset_symbol} "
                   f"from {t.from_addr[:8]} to {t.to_addr[:8]}")

        return WhaleAlert(
            alert_id=self._next_id(),
            timestamp=t.dt,
            severity=severity,
            alert_type=alert_type,
            message=msg,
            transfers=[t],
            metadata={
                "tx_hash": t.tx_hash,
                "block": t.block_number,
                "value_usd": t.value_usd,
                "asset": t.asset_symbol,
            },
        )

    def detect_coordinated_moves(
        self,
        transfers: List[WhaleTransfer],
        window_minutes: int = 60,
        min_count: int = 3,
    ) -> List[WhaleAlert]:
        """Detect multiple whales moving in the same direction within a time window."""
        window_s = window_minutes * 60
        inflows  = sorted([t for t in transfers if t.flow_direction == FlowDirection.EXCHANGE_INFLOW],  key=lambda t: t.timestamp)
        outflows = sorted([t for t in transfers if t.flow_direction == FlowDirection.EXCHANGE_OUTFLOW], key=lambda t: t.timestamp)

        alerts = []

        def _check_group(group: List[WhaleTransfer], direction: str) -> None:
            for i, lead in enumerate(group):
                window = [t for t in group[i:] if t.timestamp - lead.timestamp <= window_s]
                if len(window) >= min_count:
                    total_usd = sum(t.value_usd for t in window)
                    alerts.append(WhaleAlert(
                        alert_id=self._next_id(),
                        timestamp=lead.dt,
                        severity="high",
                        alert_type=f"coordinated_{direction}",
                        message=(f"Coordinated {direction}: {len(window)} whales "
                                 f"${total_usd:,.0f} total in {window_minutes}min window"),
                        transfers=window,
                        metadata={
                            "count": len(window),
                            "total_usd": total_usd,
                            "window_minutes": window_minutes,
                        },
                    ))

        _check_group(inflows, "inflow")
        _check_group(outflows, "outflow")
        return alerts


# ---------------------------------------------------------------------------
# Main WhaleTracker facade
# ---------------------------------------------------------------------------

class WhaleTracker:
    """
    Unified API for whale wallet monitoring.
    Combines transfer detection, exchange flow analysis, accumulation detection,
    clustering, and alert generation.
    """

    def __init__(self, provider: OnChainDataProvider = None, threshold_usd: float = WHALE_THRESHOLD_USD):
        self.provider = provider or OnChainDataProvider()
        self.threshold = threshold_usd
        self.detector = WhaleTransferDetector(self.provider, threshold_usd)
        self.flow_analyzer = ExchangeFlowAnalyzer(self.provider.glassnode)
        self.acc_detector = AccumulationDetector(self.detector)
        self.clusterer = WalletClusterer()
        self.alert_gen = WhaleAlertGenerator(threshold_usd)

    def monitor_address(
        self,
        address: str,
        lookback_days: int = 30,
    ) -> Dict:
        """Full analysis of a single whale address."""
        transfers = self.detector.scan_address(address)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        transfers = [t for t in transfers if t.dt >= cutoff]

        alerts = self.alert_gen.evaluate(transfers)
        pattern = self.acc_detector.analyze_wallet(address, lookback_days)
        label = self.provider.nansen.get_label(address)
        balance = self.provider.etherscan.get_balance(address)

        inflow_usd  = sum(t.value_usd for t in transfers if t.to_addr.lower() == address.lower())
        outflow_usd = sum(t.value_usd for t in transfers if t.from_addr.lower() == address.lower())

        return {
            "address": address,
            "label": label.entity_name,
            "entity_type": label.entity_type,
            "eth_balance": balance,
            "transfers_count": len(transfers),
            "inflow_usd": inflow_usd,
            "outflow_usd": outflow_usd,
            "net_flow_usd": inflow_usd - outflow_usd,
            "pattern": pattern,
            "alerts": alerts,
            "recent_transfers": transfers[:10],
        }

    def exchange_flow_signal(self, asset: str = "BTC") -> Dict:
        return self.flow_analyzer.net_flow_signal(asset)

    def scan_watchlist(
        self,
        addresses: List[str],
        lookback_days: int = 7,
    ) -> List[Dict]:
        """Monitor a list of whale addresses."""
        results = []
        for addr in addresses:
            try:
                r = self.monitor_address(addr, lookback_days)
                results.append(r)
            except Exception as exc:
                logger.error("Monitor failed for %s: %s", addr, exc)
        return sorted(results, key=lambda r: abs(r["net_flow_usd"]), reverse=True)

    def detect_smart_money_accumulation(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> Dict:
        """
        Combine exchange outflow signal + Glassnode whale address count
        to detect smart money accumulation.
        """
        flow_signal = self.flow_analyzer.net_flow_signal(asset, days)

        end = int(time.time())
        start = end - days * 86400
        whale_counts = self.provider.glassnode.whale_addresses(asset, start=start, end=end)

        if whale_counts:
            recent_whale_count = whale_counts[-1].value
            oldest_whale_count = whale_counts[0].value
            whale_count_change = (recent_whale_count - oldest_whale_count) / oldest_whale_count if oldest_whale_count else 0.0
        else:
            recent_whale_count = 0
            whale_count_change = 0.0

        # Composite accumulation signal
        acc_score = (
            flow_signal["confidence"] * (1 if flow_signal["signal"] == "bullish" else -1) * 0.6 +
            (whale_count_change * 5) * 0.4    # normalize whale count change
        )

        return {
            "asset": asset,
            "accumulation_score": round(acc_score, 3),
            "interpretation": "accumulating" if acc_score > 0.2 else "distributing" if acc_score < -0.2 else "neutral",
            "exchange_flow": flow_signal,
            "whale_address_count": recent_whale_count,
            "whale_count_change_pct": whale_count_change * 100,
            "lookback_days": days,
        }

    def generate_daily_brief(self, asset: str = "BTC") -> str:
        """Generate a human-readable daily whale activity summary."""
        flow = self.flow_analyzer.net_flow_signal(asset, 7)
        acc = self.detect_smart_money_accumulation(asset, 30)
        spikes = self.flow_analyzer.detect_inflow_spike(asset, 30)

        lines = [
            f"=== Whale Tracker Daily Brief: {asset} ===",
            f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "",
            f"Exchange Flow Signal: {flow['signal'].upper()} (confidence: {flow['confidence']:.0%})",
            f"  7-day avg net outflow: {flow['avg_daily_net_outflow']:,.2f} {asset}/day",
            f"  Z-score vs baseline: {flow['z_score']:.2f}",
            "",
            f"Smart Money Accumulation Score: {acc['accumulation_score']:.3f} → {acc['interpretation'].upper()}",
            f"  Whale address count: {acc['whale_address_count']:,.0f}",
            f"  Whale count change (30d): {acc['whale_count_change_pct']:+.1f}%",
            "",
        ]

        if spikes:
            lines.append(f"⚠️  {len(spikes)} exchange inflow spike(s) detected in last 30d:")
            for s in spikes[:3]:
                lines.append(f"  {s['dt']}: inflow={s['inflow']:,.2f}, z={s['z_score']:.1f}x")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whale tracker CLI")
    parser.add_argument("--action", choices=["brief", "flow", "accumulation"], default="brief")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tracker = WhaleTracker()

    if args.action == "brief":
        print(tracker.generate_daily_brief(args.asset))
    elif args.action == "flow":
        signal = tracker.exchange_flow_signal(args.asset)
        for k, v in signal.items():
            print(f"  {k}: {v}")
    elif args.action == "accumulation":
        acc = tracker.detect_smart_money_accumulation(args.asset, args.days)
        for k, v in acc.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
