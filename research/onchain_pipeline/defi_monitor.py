"""
defi_monitor.py -- DeFi protocol monitoring for signal generation.

Three classes:
  DeFiMonitor  -- Total Value Locked tracking, per-protocol and aggregate.
  YieldMonitor -- Staking yields, lending rates, basis and carry signals.
  DEXMonitor   -- Decentralized exchange volume, liquidity depth, MEV proxies.

All classes operate on in-memory state updated via upsert methods. Designed for
integration with a data pipeline that fetches from DefiLlama, Dune Analytics,
on-chain RPCs, or similar providers. No live network calls are made here.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProtocolSnapshot:
    """Single TVL observation for a protocol."""
    protocol: str
    tvl_usd: float
    timestamp: datetime
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class YieldSnapshot:
    """A single yield/rate observation."""
    asset_or_protocol: str
    metric: str  # e.g. "staking_apy", "lending_borrow_rate"
    value: float
    timestamp: datetime


@dataclass
class DEXSnapshot:
    """DEX pair or protocol state."""
    pair_or_protocol: str
    volume_24h_usd: float
    liquidity_usd: float
    timestamp: datetime
    extra: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DeFiMonitor
# ---------------------------------------------------------------------------

class DeFiMonitor:
    """
    Tracks Total Value Locked (TVL) per DeFi protocol over time.

    Data is stored as a rolling deque of ProtocolSnapshots per protocol,
    enabling time-window computations (e.g. 7-day TVL change).

    Typical usage::

        monitor = DeFiMonitor(max_history_days=90)
        monitor.update_protocol("aave", {"tvl_usd": 6.5e9})
        monitor.update_protocol("uniswap", {"tvl_usd": 4.2e9})

        tvl_map = monitor.total_value_locked()
        weekly_change = monitor.tvl_change("aave", window_days=7)
        net_flow = monitor.net_tvl_flows(window_days=7)
    """

    _SECONDS_PER_DAY = 86_400

    def __init__(self, max_history_days: int = 90) -> None:
        self._max_history = max_history_days
        # protocol -> deque of snapshots, oldest first
        self._history: Dict[str, Deque[ProtocolSnapshot]] = defaultdict(
            lambda: deque(maxlen=max_history_days * 24)  # up to hourly granularity
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update_protocol(self, protocol: str, data: Dict[str, float]) -> None:
        """
        Upsert latest TVL data for a protocol.

        Parameters
        ----------
        protocol:
            Protocol identifier, e.g. "aave", "uniswap-v3", "lido".
        data:
            Dict with at least "tvl_usd" key. Additional keys are stored
            in `extra` for custom signal construction.
        """
        if "tvl_usd" not in data:
            raise ValueError(f"Protocol data for '{protocol}' must include 'tvl_usd' key.")

        tvl = float(data["tvl_usd"])
        extra = {k: float(v) for k, v in data.items() if k != "tvl_usd"}
        snap = ProtocolSnapshot(
            protocol=protocol,
            tvl_usd=tvl,
            timestamp=datetime.now(tz=timezone.utc),
            extra=extra,
        )
        self._history[protocol].append(snap)
        logger.debug("Updated %s: TVL=%.2e", protocol, tvl)

    def update_protocol_at(
        self, protocol: str, data: Dict[str, float], timestamp: datetime
    ) -> None:
        """
        Insert a historical TVL snapshot at a specific timestamp.

        Useful for backfilling from external data sources. Note: this does not
        maintain sorted order -- callers should insert in chronological order.
        """
        if "tvl_usd" not in data:
            raise ValueError(f"Protocol data for '{protocol}' must include 'tvl_usd' key.")

        tvl = float(data["tvl_usd"])
        extra = {k: float(v) for k, v in data.items() if k != "tvl_usd"}
        snap = ProtocolSnapshot(protocol=protocol, tvl_usd=tvl, timestamp=timestamp, extra=extra)
        self._history[protocol].append(snap)

    # ------------------------------------------------------------------
    # TVL queries
    # ------------------------------------------------------------------

    def total_value_locked(self) -> Dict[str, float]:
        """
        Return the most recent TVL for each tracked protocol.

        Returns dict of protocol -> TVL in USD. Protocols with no history
        are excluded.
        """
        result: Dict[str, float] = {}
        for protocol, snaps in self._history.items():
            if snaps:
                result[protocol] = snaps[-1].tvl_usd
        return result

    def aggregate_tvl(self) -> float:
        """Sum of most recent TVL across all protocols."""
        return sum(self.total_value_locked().values())

    def tvl_change(self, protocol: str, window_days: int = 7) -> float:
        """
        Percentage change in TVL over the past `window_days` days.

        Returns 0.0 if insufficient history. Raises KeyError if protocol unknown.
        """
        if protocol not in self._history:
            raise KeyError(f"Protocol '{protocol}' not in monitor. Call update_protocol() first.")

        snaps = list(self._history[protocol])
        if len(snaps) < 2:
            return 0.0

        now = snaps[-1].timestamp
        cutoff = now.timestamp() - window_days * self._SECONDS_PER_DAY

        # Find the snapshot closest to (and before) the cutoff
        baseline_snap: Optional[ProtocolSnapshot] = None
        for snap in snaps:
            if snap.timestamp.timestamp() <= cutoff:
                baseline_snap = snap

        if baseline_snap is None:
            # Use oldest available as baseline
            baseline_snap = snaps[0]

        baseline_tvl = baseline_snap.tvl_usd
        current_tvl = snaps[-1].tvl_usd

        if baseline_tvl == 0.0:
            return float("inf") if current_tvl > 0 else 0.0

        return (current_tvl - baseline_tvl) / abs(baseline_tvl) * 100.0

    def net_tvl_flows(self, window_days: int = 7) -> float:
        """
        Aggregate TVL change in absolute USD terms across all protocols
        over the past `window_days` days.

        Positive = net inflows; negative = net outflows.
        """
        total_now = 0.0
        total_baseline = 0.0

        cutoff_offset = window_days * self._SECONDS_PER_DAY

        for protocol, snaps in self._history.items():
            snaps_list = list(snaps)
            if not snaps_list:
                continue
            total_now += snaps_list[-1].tvl_usd

            now_ts = snaps_list[-1].timestamp.timestamp()
            cutoff = now_ts - cutoff_offset

            baseline_snap: Optional[ProtocolSnapshot] = None
            for snap in snaps_list:
                if snap.timestamp.timestamp() <= cutoff:
                    baseline_snap = snap

            if baseline_snap is None:
                baseline_snap = snaps_list[0]

            total_baseline += baseline_snap.tvl_usd

        return total_now - total_baseline

    def protocol_history(self, protocol: str, window_days: int = 30) -> List[Tuple[datetime, float]]:
        """
        Return list of (timestamp, tvl_usd) tuples for a protocol over the window.
        """
        if protocol not in self._history:
            return []

        snaps = list(self._history[protocol])
        if not snaps:
            return []

        cutoff = snaps[-1].timestamp.timestamp() - window_days * self._SECONDS_PER_DAY
        return [
            (s.timestamp, s.tvl_usd)
            for s in snaps
            if s.timestamp.timestamp() >= cutoff
        ]

    def dominance(self, protocol: str) -> float:
        """
        Return this protocol's share of aggregate TVL as a fraction [0, 1].
        Returns 0.0 if aggregate TVL is zero or protocol is unknown.
        """
        tvl_map = self.total_value_locked()
        total = sum(tvl_map.values())
        if total == 0.0 or protocol not in tvl_map:
            return 0.0
        return tvl_map[protocol] / total

    def protocols(self) -> List[str]:
        """Return list of all tracked protocols."""
        return list(self._history.keys())


# ---------------------------------------------------------------------------
# YieldMonitor
# ---------------------------------------------------------------------------

class YieldMonitor:
    """
    Tracks staking yields, lending rates, and basis/carry metrics.

    All stored values are in decimal form (e.g. 0.05 = 5% APY).
    Provides carry_signal which combines funding rate premium with staking
    yield vs spot-futures basis -- useful for positioning in PoS coins.
    """

    def __init__(self) -> None:
        # asset -> list of (timestamp, yield_value) pairs
        self._staking_yields: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        # (protocol, asset) -> list of (timestamp, (borrow_rate, lend_rate)) pairs
        self._lending_rates: Dict[Tuple[str, str], List[Tuple[datetime, Tuple[float, float]]]] = defaultdict(list)
        # asset -> list of (timestamp, basis_pct) pairs
        self._basis_premia: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        # asset -> list of (timestamp, funding_rate) pairs
        self._funding_rates: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update_staking_yield(self, asset: str, apy: float) -> None:
        """Record latest staking APY for an asset (e.g. ETH, SOL)."""
        self._staking_yields[asset].append((datetime.now(tz=timezone.utc), apy))

    def update_lending_rate(
        self, protocol: str, asset: str, borrow_rate: float, lend_rate: float
    ) -> None:
        """Record latest borrow/lend rates from a lending protocol."""
        key = (protocol, asset)
        self._lending_rates[key].append(
            (datetime.now(tz=timezone.utc), (borrow_rate, lend_rate))
        )

    def update_basis(self, asset: str, basis_pct: float) -> None:
        """Record spot-to-futures basis as a percentage (positive = contango)."""
        self._basis_premia[asset].append((datetime.now(tz=timezone.utc), basis_pct))

    def update_funding_rate(self, asset: str, funding: float) -> None:
        """Record perpetual funding rate (8h or annualized -- be consistent)."""
        self._funding_rates[asset].append((datetime.now(tz=timezone.utc), funding))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def staking_yield(self, asset: str) -> float:
        """
        Latest staking APY for a PoS asset.

        Returns 0.0 if no data available.
        """
        history = self._staking_yields.get(asset, [])
        return history[-1][1] if history else 0.0

    def lending_rate(self, protocol: str, asset: str) -> float:
        """
        Latest lending (supply) APY for an asset on a protocol.

        Returns the lend_rate (return to depositors). Returns 0.0 if unavailable.
        """
        key = (protocol, asset)
        history = self._lending_rates.get(key, [])
        if not history:
            return 0.0
        _, (_, lend_rate) = history[-1]
        return lend_rate

    def borrow_rate(self, protocol: str, asset: str) -> float:
        """Latest borrow APY for an asset on a protocol. Returns 0.0 if unavailable."""
        key = (protocol, asset)
        history = self._lending_rates.get(key, [])
        if not history:
            return 0.0
        _, (borrow_rate, _) = history[-1]
        return borrow_rate

    def basis_trade_premium(self, asset: str) -> float:
        """
        Latest spot-vs-futures basis as a percentage.

        Positive = contango (futures above spot). Negative = backwardation.
        Returns 0.0 if unavailable.
        """
        history = self._basis_premia.get(asset, [])
        return history[-1][1] if history else 0.0

    def carry_signal(self, asset: str) -> float:
        """
        Carry signal combining funding rate and staking yield vs basis.

        carry = funding_rate_annualized + staking_yield - basis_annualized

        Positive carry -> holding long position is rewarded -> bullish.
        Negative carry -> holding long position costs money -> bearish.

        The result is not clipped; callers may want to standardize it.
        Assumes funding_rate is in the same units as staking_yield and basis.
        """
        funding = self._funding_rates.get(asset, [])
        latest_funding = funding[-1][1] if funding else 0.0

        staking = self.staking_yield(asset)
        basis = self.basis_trade_premium(asset)

        # carry = what you earn holding the position - what the basis says you should earn
        carry = latest_funding + staking - basis
        return carry

    def best_lending_yield(self, asset: str) -> Tuple[str, float]:
        """
        Find the protocol offering the highest lending yield for an asset.

        Returns (protocol_name, lend_rate). Returns ("", 0.0) if no data.
        """
        best_protocol = ""
        best_rate = 0.0
        for (protocol, a), history in self._lending_rates.items():
            if a != asset or not history:
                continue
            _, (_, lend_rate) = history[-1]
            if lend_rate > best_rate:
                best_rate = lend_rate
                best_protocol = protocol
        return best_protocol, best_rate

    def yield_spread(self, asset: str, protocol_a: str, protocol_b: str) -> float:
        """
        Lending yield spread between two protocols for the same asset.

        Returns rate_a - rate_b. Positive means protocol_a offers higher yield.
        """
        return self.lending_rate(protocol_a, asset) - self.lending_rate(protocol_b, asset)


# ---------------------------------------------------------------------------
# DEXMonitor
# ---------------------------------------------------------------------------

class DEXMonitor:
    """
    Monitors decentralized exchange (DEX) metrics.

    Tracks volume, liquidity depth, and MEV activity across DEX pairs and protocols.
    Useful for signals based on:
      - Volume/TVL efficiency (active capital usage)
      - Sandwich attack rate as a proxy for retail activity and MEV intensity
      - Liquidity depth for execution quality estimation
    """

    def __init__(self) -> None:
        # pair -> deque of DEXSnapshot
        self._pair_history: Dict[str, Deque[DEXSnapshot]] = defaultdict(
            lambda: deque(maxlen=365)
        )
        # protocol -> deque of (timestamp, volume, tvl) for vol/tvl metric
        self._protocol_snapshots: Dict[str, Deque[DEXSnapshot]] = defaultdict(
            lambda: deque(maxlen=365)
        )
        # protocol -> deque of (timestamp, sandwich_count, total_tx_count)
        self._mev_data: Dict[str, Deque[Tuple[datetime, int, int]]] = defaultdict(
            lambda: deque(maxlen=365)
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update_pair(
        self,
        pair: str,
        volume_24h_usd: float,
        liquidity_usd: float,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record latest DEX pair state."""
        snap = DEXSnapshot(
            pair_or_protocol=pair,
            volume_24h_usd=volume_24h_usd,
            liquidity_usd=liquidity_usd,
            timestamp=datetime.now(tz=timezone.utc),
            extra=extra or {},
        )
        self._pair_history[pair].append(snap)

    def update_protocol(
        self,
        protocol: str,
        volume_24h_usd: float,
        tvl_usd: float,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record latest aggregate DEX protocol metrics."""
        snap = DEXSnapshot(
            pair_or_protocol=protocol,
            volume_24h_usd=volume_24h_usd,
            liquidity_usd=tvl_usd,
            timestamp=datetime.now(tz=timezone.utc),
            extra=extra or {},
        )
        self._protocol_snapshots[protocol].append(snap)

    def update_mev(self, protocol: str, sandwich_count: int, total_tx_count: int) -> None:
        """
        Record MEV activity counts for a protocol.

        sandwich_count: number of detected sandwich attacks in the period.
        total_tx_count: total swaps/transactions for rate normalization.
        """
        self._mev_data[protocol].append(
            (datetime.now(tz=timezone.utc), sandwich_count, total_tx_count)
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def volume_24h(self, pair: str) -> float:
        """
        Latest 24h trading volume in USD for a DEX pair.

        Returns 0.0 if no data available.
        """
        history = self._pair_history.get(pair, deque())
        return history[-1].volume_24h_usd if history else 0.0

    def liquidity_depth(self, pair: str, slippage_pct: float = 0.5) -> float:
        """
        Estimate the USD size of a trade that would incur `slippage_pct` slippage.

        Uses a simple constant-product AMM approximation:
          depth = liquidity_usd * slippage_pct / 200

        This is approximate; real depth depends on pool concentrations and ranges.
        Returns 0.0 if no data.
        """
        history = self._pair_history.get(pair, deque())
        if not history:
            return 0.0
        liq = history[-1].liquidity_usd
        # For a CPAMM with equal split: buying X% of pool costs ~X^2/(2(1-X)) extra
        # Simplified: trade_size ~ liq * slippage / 2
        return liq * (slippage_pct / 100.0) / 2.0

    def volume_to_tvl(self, protocol: str) -> float:
        """
        Volume-to-TVL ratio (capital efficiency metric) for a protocol.

        Higher ratios indicate more active use of locked capital.
        Returns 0.0 if no data or TVL is zero.
        """
        snaps = self._protocol_snapshots.get(protocol, deque())
        if not snaps:
            return 0.0
        snap = snaps[-1]
        if snap.liquidity_usd == 0.0:
            return 0.0
        return snap.volume_24h_usd / snap.liquidity_usd

    def sandwich_attack_rate(self, protocol: str) -> float:
        """
        Sandwich attack rate as a fraction of total transactions.

        Returns attacks / total_txs. Returns 0.0 if no MEV data.

        Higher rates suggest elevated MEV activity, which can indicate:
          - High retail participation (more extractable value)
          - Network congestion and elevated gas competition
        """
        mev_history = self._mev_data.get(protocol, deque())
        if not mev_history:
            return 0.0
        _, sandwich_count, total_tx_count = mev_history[-1]
        if total_tx_count == 0:
            return 0.0
        return sandwich_count / total_tx_count

    def volume_zscore(self, pair: str, window: int = 30) -> float:
        """
        Z-score of current 24h volume relative to the trailing `window` days.

        Useful for detecting unusual volume spikes or drops.
        Returns 0.0 if insufficient history.
        """
        history = list(self._pair_history.get(pair, deque()))
        if len(history) < 3:
            return 0.0

        recent = [s.volume_24h_usd for s in history[-window:]]
        if len(recent) < 2:
            return 0.0

        mean_vol = statistics.mean(recent[:-1])
        try:
            std_vol = statistics.stdev(recent[:-1])
        except statistics.StatisticsError:
            return 0.0

        if std_vol < 1e-6:
            return 0.0

        return (recent[-1] - mean_vol) / std_vol

    def top_pairs_by_volume(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the top N pairs by 24h volume.

        Returns list of (pair, volume_usd) sorted descending by volume.
        """
        pairs: List[Tuple[str, float]] = []
        for pair, history in self._pair_history.items():
            if history:
                pairs.append((pair, history[-1].volume_24h_usd))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]

    def aggregate_protocol_volume(self, protocol: str, window_days: int = 7) -> float:
        """
        Sum of daily volumes over `window_days` for a protocol.

        Returns 0.0 if insufficient data.
        """
        snaps = list(self._protocol_snapshots.get(protocol, deque()))
        if not snaps:
            return 0.0

        cutoff = snaps[-1].timestamp.timestamp() - window_days * 86_400
        recent = [s.volume_24h_usd for s in snaps if s.timestamp.timestamp() >= cutoff]
        return sum(recent)

    def liquidity_utilization(self, protocol: str) -> float:
        """
        Volume / TVL over the past 7 days, averaged.

        Returns float in [0, inf). >1 means volume exceeded TVL (high utilization).
        """
        snaps = list(self._protocol_snapshots.get(protocol, deque()))
        if not snaps:
            return 0.0

        recent = snaps[-7:] if len(snaps) >= 7 else snaps
        ratios = [
            s.volume_24h_usd / s.liquidity_usd
            for s in recent
            if s.liquidity_usd > 0
        ]
        return statistics.mean(ratios) if ratios else 0.0

    def protocols(self) -> List[str]:
        """Return list of all tracked DEX protocols."""
        return list(self._protocol_snapshots.keys())

    def pairs(self) -> List[str]:
        """Return list of all tracked DEX pairs."""
        return list(self._pair_history.keys())
