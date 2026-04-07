"""
defi_analytics.py -- DeFi and on-chain derivatives analytics.

Covers:
  - AMM liquidity concentration analysis (Uniswap V3 style)
  - Lending protocol borrow utilization signals (Aave / Compound style)
  - Bridge flow tracker for cross-chain capital movements

All methods accept data as parameters; no live API calls are required.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AMM tick spacing (Uniswap V3 default for 0.3% fee tier)
UNISWAP_V3_TICK_SPACING = 60
UNISWAP_V3_BASE = 1.0001   # Price = 1.0001^tick

# Lending protocol signal thresholds
UTILIZATION_BULLISH_THRESHOLD = 0.85   # >85% utilization -> bullish
UTILIZATION_BEARISH_THRESHOLD = 0.40   # <40% utilization -> bearish

# Bridge flow signal thresholds (net inflow as fraction of 30-day average volume)
BRIDGE_INFLOW_BEARISH_THRESHOLD = 0.15   # net inflow > 15% of avg vol -> bearish
BRIDGE_OUTFLOW_BULLISH_THRESHOLD = 0.10  # net outflow > 10% of avg vol -> bullish

# Saturation level for bridge signal normalization
BRIDGE_SIGNAL_SATURATION = 0.30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickLiquidityBin:
    """Liquidity bin for a Uniswap V3 style AMM pool."""
    tick_lower: int
    tick_upper: int
    liquidity: float            # raw liquidity units
    price_lower: float          # derived from tick
    price_upper: float
    liquidity_usd: float = 0.0  # USD value at current price

    @classmethod
    def from_ticks(
        cls,
        tick_lower: int,
        tick_upper: int,
        liquidity: float,
        liquidity_usd: float = 0.0,
    ) -> "TickLiquidityBin":
        price_lower = UNISWAP_V3_BASE ** tick_lower
        price_upper = UNISWAP_V3_BASE ** tick_upper
        return cls(
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=liquidity,
            price_lower=price_lower,
            price_upper=price_upper,
            liquidity_usd=liquidity_usd,
        )


@dataclass
class BorrowSnapshot:
    """Snapshot of lending protocol state for a single asset."""
    symbol: str
    protocol: str               # e.g. "aave", "compound"
    total_supplied_usd: float
    total_borrowed_usd: float
    borrow_rate_apy: float      # current borrow APY (decimal)
    supply_rate_apy: float      # current supply APY (decimal)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def utilization_rate(self) -> float:
        """Fraction of supplied capital currently borrowed."""
        if self.total_supplied_usd <= 0:
            return 0.0
        return min(1.0, self.total_borrowed_usd / self.total_supplied_usd)


@dataclass
class BridgeFlowEvent:
    """A single bridge transfer event."""
    symbol: str
    bridge_name: str            # e.g. "stargate", "wormhole", "hop"
    direction: str              # "inflow" (to exchange chain) or "outflow"
    amount_usd: float
    timestamp: datetime
    source_chain: str = ""
    dest_chain: str = ""


# ---------------------------------------------------------------------------
# AMMLiquidityAnalyzer
# ---------------------------------------------------------------------------

class AMMLiquidityAnalyzer:
    """
    Analyzes Uniswap V3 style AMM liquidity distributions.

    Provides:
      - Liquidity concentration near current price
      - Detection of liquidity cliffs (gaps) that create slippage zones
      - Price impact estimation for given trade sizes
    """

    def __init__(self) -> None:
        # {symbol -> list of TickLiquidityBin}
        self._liquidity_bins: Dict[str, List[TickLiquidityBin]] = {}
        # {symbol -> current spot price}
        self._spot_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_liquidity(
        self,
        symbol: str,
        bins: List[TickLiquidityBin],
        spot_price: float,
    ) -> None:
        """Replace the stored liquidity distribution for a symbol."""
        self._liquidity_bins[symbol] = sorted(bins, key=lambda b: b.tick_lower)
        self._spot_prices[symbol] = spot_price

    def update_spot(self, symbol: str, spot_price: float) -> None:
        """Update the stored spot price without replacing bins."""
        self._spot_prices[symbol] = spot_price

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def tick_liquidity(
        self,
        symbol: str,
        price_range_pct: float = 0.02,
        bins: Optional[List[TickLiquidityBin]] = None,
        spot_price: Optional[float] = None,
    ) -> float:
        """
        Sum of liquidity (USD) within `price_range_pct` of the current price.

        Parameters
        ----------
        symbol : str
        price_range_pct : float
            Fraction of current price defining the range on each side
            (default 0.02 = +/-2%).
        bins : list of TickLiquidityBin, optional
            Override bins for this call.
        spot_price : float, optional
            Override spot price for this call.

        Returns
        -------
        float
            Total liquidity in USD within the price range.
        """
        current_bins = bins or self._liquidity_bins.get(symbol, [])
        price = spot_price or self._spot_prices.get(symbol, 0.0)

        if not current_bins or price <= 0:
            return 0.0

        price_lo = price * (1.0 - price_range_pct)
        price_hi = price * (1.0 + price_range_pct)

        total = 0.0
        for b in current_bins:
            # Check for overlap between [price_lo, price_hi] and [b.price_lower, b.price_upper]
            overlap_lo = max(price_lo, b.price_lower)
            overlap_hi = min(price_hi, b.price_upper)
            if overlap_lo >= overlap_hi:
                continue
            bin_width = b.price_upper - b.price_lower
            if bin_width <= 0:
                continue
            overlap_frac = (overlap_hi - overlap_lo) / bin_width
            total += b.liquidity_usd * overlap_frac

        return total

    def liquidity_cliff(
        self,
        symbol: str,
        bins: Optional[List[TickLiquidityBin]] = None,
        spot_price: Optional[float] = None,
        min_gap_pct: float = 0.005,
    ) -> float:
        """
        Find the nearest large liquidity gap (cliff) from the current price.

        A liquidity cliff is a price level where liquidity drops sharply,
        creating a zone of high slippage.

        Parameters
        ----------
        symbol : str
        bins : list, optional
        spot_price : float, optional
        min_gap_pct : float
            Minimum price range treated as a significant gap (default 0.5%).

        Returns
        -------
        float
            Distance from current price to nearest cliff as a fraction
            (0.05 = 5% away). Returns float('inf') if no cliff found.
        """
        current_bins = bins or self._liquidity_bins.get(symbol, [])
        price = spot_price or self._spot_prices.get(symbol, 0.0)

        if not current_bins or price <= 0:
            return float("inf")

        # Sort by distance from current price
        nearby = [b for b in current_bins if b.price_lower <= price * 1.5 and b.price_upper >= price * 0.5]
        if not nearby:
            return float("inf")

        # Build a sorted list of (price, liquidity_usd) for nearby bins
        sorted_bins = sorted(nearby, key=lambda b: b.price_lower)
        avg_liq = statistics.mean(b.liquidity_usd for b in sorted_bins) if sorted_bins else 1.0

        min_distance = float("inf")

        for i in range(len(sorted_bins) - 1):
            gap_lo = sorted_bins[i].price_upper
            gap_hi = sorted_bins[i + 1].price_lower
            gap_pct = (gap_hi - gap_lo) / price if price > 0 else 0.0

            if gap_pct < min_gap_pct:
                continue

            # Check if adjacent bins have much lower liquidity
            liq_drop = sorted_bins[i + 1].liquidity_usd / (avg_liq + 1e-10)
            if liq_drop < 0.3:  # 70% drop in liquidity = cliff
                gap_center = (gap_lo + gap_hi) / 2.0
                dist = abs(gap_center - price) / price
                min_distance = min(min_distance, dist)

        return min_distance

    def price_impact_estimate(
        self,
        symbol: str,
        trade_size_usd: float,
        direction: str = "buy",
        bins: Optional[List[TickLiquidityBin]] = None,
        spot_price: Optional[float] = None,
    ) -> float:
        """
        Estimate price impact of a trade of given size in a V3-style pool.

        Uses a simplified constant-product model: walks through the liquidity
        bins from current price outward, consuming liquidity until the trade
        size is filled.

        Parameters
        ----------
        symbol : str
        trade_size_usd : float
            Trade size in USD.
        direction : str
            "buy" (price goes up) or "sell" (price goes down).
        bins : list, optional
        spot_price : float, optional

        Returns
        -------
        float
            Estimated price impact as a decimal fraction (0.01 = 1%).
        """
        current_bins = bins or self._liquidity_bins.get(symbol, [])
        price = spot_price or self._spot_prices.get(symbol, 0.0)

        if not current_bins or price <= 0 or trade_size_usd <= 0:
            return 0.0

        sorted_bins = sorted(current_bins, key=lambda b: b.price_lower)

        # Find the bin containing current price
        current_bin_idx = 0
        for i, b in enumerate(sorted_bins):
            if b.price_lower <= price <= b.price_upper:
                current_bin_idx = i
                break

        if direction == "buy":
            relevant_bins = sorted_bins[current_bin_idx:]
        else:
            relevant_bins = list(reversed(sorted_bins[:current_bin_idx + 1]))

        remaining = trade_size_usd
        final_price = price

        for b in relevant_bins:
            available = b.liquidity_usd
            if available <= 0:
                continue

            consumed = min(remaining, available)
            # Price impact in this bin: consumed / available * bin price range
            bin_range = abs(b.price_upper - b.price_lower)
            if b.price_lower > 0:
                bin_range_pct = bin_range / b.price_lower
            else:
                continue

            price_move = bin_range_pct * (consumed / available)

            if direction == "buy":
                final_price = b.price_upper if consumed >= available else price * (1 + price_move)
            else:
                final_price = b.price_lower if consumed >= available else price * (1 - price_move)

            remaining -= consumed
            if remaining <= 0:
                break

        if price <= 0:
            return 0.0
        return abs(final_price - price) / price

    def concentrated_liquidity_score(
        self,
        symbol: str,
        bins: Optional[List[TickLiquidityBin]] = None,
        spot_price: Optional[float] = None,
    ) -> float:
        """
        Score how concentrated liquidity is around the current price.

        Returns fraction of total pool liquidity that is within 5% of spot.
        Higher score -> better depth, lower slippage, more efficient market.
        """
        current_bins = bins or self._liquidity_bins.get(symbol, [])
        price = spot_price or self._spot_prices.get(symbol, 0.0)

        if not current_bins:
            return 0.0

        total = sum(b.liquidity_usd for b in current_bins)
        if total <= 0:
            return 0.0

        near = self.tick_liquidity(symbol, price_range_pct=0.05, bins=current_bins, spot_price=price)
        return near / total


# ---------------------------------------------------------------------------
# LendingProtocolSignal
# ---------------------------------------------------------------------------

class LendingProtocolSignal:
    """
    Extracts directional signals from lending protocol utilization rates.

    High borrow utilization means there is strong demand for leveraged
    exposure, which is a proxy for bullish sentiment. Low utilization
    indicates reduced demand for leverage.

    Protocols tracked: Aave, Compound (and generic compatible protocols).
    """

    def __init__(self) -> None:
        # {(symbol, protocol) -> deque of BorrowSnapshot}
        self._history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=500)
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_snapshot(self, snapshot: BorrowSnapshot) -> None:
        """Record a lending protocol state snapshot."""
        key = (snapshot.symbol, snapshot.protocol)
        self._history[key].append(snapshot)

    def record_batch(self, snapshots: Sequence[BorrowSnapshot]) -> None:
        """Record a list of snapshots."""
        for s in snapshots:
            self.record_snapshot(s)

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def utilization_rate_signal(
        self,
        symbol: str,
        utilization_rate: Optional[float] = None,
        protocol: Optional[str] = None,
    ) -> float:
        """
        Convert utilization rate to a [-1, 1] directional signal.

        Signal logic:
          utilization > 85%: high demand for borrowing -> bullish signal
          utilization < 40%: low demand -> bearish signal
          Linear interpolation in between.

        Parameters
        ----------
        symbol : str
        utilization_rate : float, optional
            Override: explicit utilization rate (decimal 0-1).
        protocol : str, optional
            If provided, limits history lookup to this protocol.

        Returns
        -------
        float in [-1, 1]
        """
        if utilization_rate is not None:
            rate = utilization_rate
        else:
            rate = self._get_latest_utilization(symbol, protocol)
            if rate is None:
                return 0.0

        return _utilization_to_signal(rate)

    def borrow_rate_trend(
        self,
        symbol: str,
        protocol: str,
        window: int = 7,
    ) -> float:
        """
        Trend in borrow rates over the last `window` days (daily observations).

        Rising borrow rates signal increasing demand for leverage (bullish).
        Falling rates signal deleveraging (bearish).

        Returns float in [-1, 1].
        """
        key = (symbol, protocol)
        history = list(self._history[key])[-window:]
        if len(history) < 2:
            return 0.0

        rates = [s.borrow_rate_apy for s in history]
        slope = _linear_regression_slope(rates)

        # Normalize: 1% per day rate increase = ~0.5 signal
        normalization = 0.02 or 1e-6
        return max(-1.0, min(1.0, slope / normalization))

    def cross_protocol_utilization(
        self,
        symbol: str,
        utilization_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Average utilization rate across all tracked protocols for a symbol.

        Parameters
        ----------
        symbol : str
        utilization_overrides : dict, optional
            {protocol: utilization_rate} -- bypasses history.

        Returns
        -------
        float
            Average utilization rate (decimal). 0.0 if no data.
        """
        if utilization_overrides:
            vals = list(utilization_overrides.values())
            return statistics.mean(vals) if vals else 0.0

        # Gather latest snapshot per protocol
        rates = []
        for (sym, proto), hist in self._history.items():
            if sym == symbol and hist:
                latest = hist[-1]
                rates.append(latest.utilization_rate)

        return statistics.mean(rates) if rates else 0.0

    def composite_lending_signal(
        self,
        symbol: str,
        utilization_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Weighted composite lending signal across all protocols.

        Applies utilization_rate_signal to each protocol and averages the
        resulting signals.

        Returns float in [-1, 1].
        """
        if utilization_overrides:
            signals = [
                _utilization_to_signal(u)
                for u in utilization_overrides.values()
            ]
            return statistics.mean(signals) if signals else 0.0

        signals = []
        for (sym, proto), hist in self._history.items():
            if sym == symbol and hist:
                latest = hist[-1]
                signals.append(_utilization_to_signal(latest.utilization_rate))

        return statistics.mean(signals) if signals else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_latest_utilization(
        self,
        symbol: str,
        protocol: Optional[str] = None,
    ) -> Optional[float]:
        """Get the most recent utilization rate for a symbol."""
        if protocol:
            key = (symbol, protocol)
            hist = self._history.get(key)
            if hist:
                return hist[-1].utilization_rate
            return None

        # Aggregate across protocols
        rates = []
        for (sym, proto), hist in self._history.items():
            if sym == symbol and hist:
                rates.append(hist[-1].utilization_rate)
        return statistics.mean(rates) if rates else None


# ---------------------------------------------------------------------------
# BridgeFlowTracker
# ---------------------------------------------------------------------------

class BridgeFlowTracker:
    """
    Tracks cross-chain bridge flows as a proxy for capital movement intentions.

    Large inflows to exchange-connected chains (e.g. from L2s to Ethereum
    mainnet before sending to Coinbase) suggest potential selling pressure.

    Large outflows (from mainnet to L2s or cold storage chains) suggest
    accumulation and self-custody -- bullish signal.

    Bridge names: "stargate", "wormhole", "hop", "across", "synapse", "cbridge"
    """

    def __init__(self) -> None:
        # {symbol -> deque of BridgeFlowEvent}
        self._events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_event(self, event: BridgeFlowEvent) -> None:
        """Record a bridge flow event."""
        self._events[event.symbol].append(event)

    def record_batch(self, events: Sequence[BridgeFlowEvent]) -> None:
        """Record a list of bridge flow events."""
        for e in events:
            self.record_event(e)

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def bridge_flow_signal(
        self,
        symbol: str,
        window_days: int = 7,
        events: Optional[List[BridgeFlowEvent]] = None,
        reference_volume_usd: Optional[float] = None,
    ) -> float:
        """
        Compute a directional signal from net bridge flows over `window_days`.

        Signal logic:
          Net inflow (to exchange chain) > BRIDGE_INFLOW_BEARISH_THRESHOLD:
            Potential selling pressure -> bearish (negative signal)
          Net outflow > BRIDGE_OUTFLOW_BULLISH_THRESHOLD:
            Accumulation signal -> bullish (positive signal)

        Parameters
        ----------
        symbol : str
        window_days : int
            Lookback window in days.
        events : list of BridgeFlowEvent, optional
            Override: use these events instead of stored history.
        reference_volume_usd : float, optional
            Reference volume for normalization (e.g., 30-day average on-chain
            volume). If not provided, uses the total event volume as reference.

        Returns
        -------
        float in [-1, 1]
        """
        source = events or list(self._events.get(symbol, []))
        if not source:
            return 0.0

        cutoff = _days_ago(window_days)
        recent = [e for e in source if e.timestamp >= cutoff]
        if not recent:
            return 0.0

        inflow = sum(e.amount_usd for e in recent if e.direction == "inflow")
        outflow = sum(e.amount_usd for e in recent if e.direction == "outflow")
        net_flow = inflow - outflow  # positive = net inflow (bearish)

        if reference_volume_usd is not None and reference_volume_usd > 0:
            ref = reference_volume_usd
        else:
            total_flow = inflow + outflow
            if total_flow <= 0:
                return 0.0
            ref = total_flow

        normalized = net_flow / ref  # positive = net inflow

        # Inflow -> bearish (negative signal); outflow -> bullish (positive signal)
        signal = -normalized / BRIDGE_SIGNAL_SATURATION
        return max(-1.0, min(1.0, signal))

    def net_flow_usd(
        self,
        symbol: str,
        window_days: int = 7,
        events: Optional[List[BridgeFlowEvent]] = None,
    ) -> Dict[str, float]:
        """
        Compute net bridge flow statistics over the window.

        Returns dict with: inflow_usd, outflow_usd, net_usd, num_events.
        """
        source = events or list(self._events.get(symbol, []))
        cutoff = _days_ago(window_days)
        recent = [e for e in source if e.timestamp >= cutoff]

        inflow = sum(e.amount_usd for e in recent if e.direction == "inflow")
        outflow = sum(e.amount_usd for e in recent if e.direction == "outflow")

        return {
            "inflow_usd": inflow,
            "outflow_usd": outflow,
            "net_usd": inflow - outflow,
            "num_events": len(recent),
        }

    def bridge_volume_by_protocol(
        self,
        symbol: str,
        window_days: int = 7,
        events: Optional[List[BridgeFlowEvent]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Break down bridge flows by protocol.

        Returns dict: {bridge_name -> {inflow_usd, outflow_usd, net_usd}}.
        """
        source = events or list(self._events.get(symbol, []))
        cutoff = _days_ago(window_days)
        recent = [e for e in source if e.timestamp >= cutoff]

        breakdown: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"inflow_usd": 0.0, "outflow_usd": 0.0, "net_usd": 0.0}
        )

        for e in recent:
            b = breakdown[e.bridge_name]
            if e.direction == "inflow":
                b["inflow_usd"] += e.amount_usd
            else:
                b["outflow_usd"] += e.amount_usd
            b["net_usd"] = b["inflow_usd"] - b["outflow_usd"]

        return dict(breakdown)

    def flow_momentum(
        self,
        symbol: str,
        short_window_days: int = 3,
        long_window_days: int = 14,
        events: Optional[List[BridgeFlowEvent]] = None,
    ) -> float:
        """
        Compare short-term vs long-term net flow to detect accelerating trends.

        Positive momentum: recent outflows accelerating (bullish acceleration)
        Negative momentum: recent inflows accelerating (bearish acceleration)

        Returns float in [-1, 1].
        """
        source = events or list(self._events.get(symbol, []))
        if not source:
            return 0.0

        def net_flow(window: int) -> float:
            cutoff = _days_ago(window)
            recent = [e for e in source if e.timestamp >= cutoff]
            inflow = sum(e.amount_usd for e in recent if e.direction == "inflow")
            outflow = sum(e.amount_usd for e in recent if e.direction == "outflow")
            return outflow - inflow  # positive = net outflow (bullish)

        short_net = net_flow(short_window_days)
        long_net = net_flow(long_window_days)

        if long_net == 0:
            return 0.0 if short_net == 0 else math.copysign(1.0, short_net)

        # Ratio of short to long net flow (annualized-like comparison)
        short_rate = short_net / max(short_window_days, 1)
        long_rate = long_net / max(long_window_days, 1)

        ratio = (short_rate - long_rate) / (abs(long_rate) + 1e-10)
        return max(-1.0, min(1.0, ratio))

    def abnormal_flow_alert(
        self,
        symbol: str,
        events: Optional[List[BridgeFlowEvent]] = None,
        zscore_threshold: float = 2.0,
    ) -> Dict[str, object]:
        """
        Detect abnormally large bridge flows using rolling z-score.

        Compares the last 24h flow to the 30-day rolling distribution.

        Returns dict with: is_abnormal, zscore, direction, amount_usd.
        """
        source = events or list(self._events.get(symbol, []))
        if len(source) < 10:
            return {"is_abnormal": False, "zscore": 0.0}

        cutoff_24h = _days_ago(1)
        cutoff_30d = _days_ago(30)

        # Compute daily net flows for the last 30 days
        daily_flows: Dict[int, float] = defaultdict(float)
        for e in source:
            if e.timestamp < cutoff_30d:
                continue
            day = int(e.timestamp.timestamp()) // 86400
            delta = e.amount_usd if e.direction == "inflow" else -e.amount_usd
            daily_flows[day] += delta

        flows_list = list(daily_flows.values())
        if len(flows_list) < 5:
            return {"is_abnormal": False, "zscore": 0.0}

        # Last day flow
        last_day = int(cutoff_24h.timestamp()) // 86400 + 1
        last_flow = daily_flows.get(last_day, 0.0)

        mean_flow = statistics.mean(flows_list)
        std_flow = statistics.stdev(flows_list) if len(flows_list) > 1 else 0.0

        if std_flow == 0:
            zscore = 0.0
        else:
            zscore = (last_flow - mean_flow) / std_flow

        direction = "inflow" if last_flow > 0 else "outflow"
        return {
            "is_abnormal": abs(zscore) >= zscore_threshold,
            "zscore": zscore,
            "direction": direction,
            "amount_usd": abs(last_flow),
            "mean_daily_usd": mean_flow,
            "std_daily_usd": std_flow,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _utilization_to_signal(rate: float) -> float:
    """
    Map utilization rate [0, 1] to a [-1, 1] signal.

    rate >= UTILIZATION_BULLISH_THRESHOLD (0.85) -> +1.0
    rate <= UTILIZATION_BEARISH_THRESHOLD (0.40) -> -1.0
    Linear interpolation in between.
    """
    if rate >= UTILIZATION_BULLISH_THRESHOLD:
        return 1.0
    if rate <= UTILIZATION_BEARISH_THRESHOLD:
        return -1.0
    midpoint = (UTILIZATION_BULLISH_THRESHOLD + UTILIZATION_BEARISH_THRESHOLD) / 2.0
    half_range = (UTILIZATION_BULLISH_THRESHOLD - UTILIZATION_BEARISH_THRESHOLD) / 2.0
    return (rate - midpoint) / half_range


def _days_ago(days: int) -> datetime:
    """Return a UTC-aware datetime `days` days before now."""
    import datetime as dt_module
    now = datetime.now(timezone.utc)
    return now - dt_module.timedelta(days=days)


def _linear_regression_slope(values: List[float]) -> float:
    """
    Compute the OLS slope of values against their index positions.

    Returns 0.0 for fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0
    return numerator / denominator
