"""
latency_arbitrage.py — Latency arbitrage simulation and detection.

Covers:
  - Co-location advantage model (speed-of-light latency, feed arbitrage)
  - Stale quote detection (quote age, cross-venue staleness)
  - Quote stuffing detection (message rate anomalies, stuffing bursts)
  - Latency arbitrage profitability simulation
  - HFT venue comparison and optimal routing
  - Regulatory metrics (order-to-trade ratio, cancellation rates)
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VenueType(Enum):
    NASDAQ    = "NASDAQ"
    NYSE      = "NYSE"
    BATS      = "BATS"
    EDGX      = "EDGX"
    ARCA      = "ARCA"
    IEX       = "IEX"          # Speed bump: 350 µs
    DARK_POOL = "DARK_POOL"


class ColoTier(Enum):
    """Co-location quality tier at a given exchange."""
    TIER_1 = "tier_1"    # On-premises colo (sub-50 µs round trip)
    TIER_2 = "tier_2"    # Near-site colo (50-200 µs)
    TIER_3 = "tier_3"    # Remote (200-1000 µs)
    RETAIL  = "retail"   # Retail broker (1-50 ms)


class StuffingStatus(Enum):
    NORMAL    = "normal"
    ELEVATED  = "elevated"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class ArbitrageOpportunity(Enum):
    NONE      = "none"
    LATENT    = "latent"       # Would be profitable with better latency
    ACTIVE    = "active"       # Currently exploitable
    EXPIRED   = "expired"      # Already closed


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VenueProfile:
    """Physical and latency characteristics of a trading venue."""
    venue: VenueType
    city: str
    latitude: float
    longitude: float
    # One-way fiber latency to key hubs (microseconds)
    latency_to_ny_us: float
    latency_to_chicago_us: float
    latency_to_london_us: float
    # Exchange-specific speed bump (µs)
    speed_bump_us: float = 0.0
    # Matching engine processing time (µs)
    matching_engine_us: float = 5.0
    # Feed latency (µs) — from trade to SIP/proprietary feed
    sip_feed_lag_us: float = 100.0
    direct_feed_lag_us: float = 5.0
    # Cost per order (USD)
    maker_fee_bps: float = -0.20   # rebate
    taker_fee_bps: float = 0.30


@dataclass
class ColoProfile:
    """Co-location setup and its latency characteristics."""
    venue: VenueType
    tier: ColoTier
    # Round-trip latency to venue matching engine (µs)
    rtt_us: float
    # Monthly cost (USD)
    monthly_cost_usd: float
    # Cross-connect fee (USD/month)
    cross_connect_usd: float
    # Throughput (orders/second)
    max_orders_per_sec: int


@dataclass
class Quote:
    """A single market quote from a specific venue."""
    venue: VenueType
    ticker: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp_ns: int    # nanoseconds since epoch
    sequence: int        # exchange sequence number
    feed_type: str       # "direct" or "sip"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def age_us(self) -> float:
        now_ns = time.time_ns()
        return (now_ns - self.timestamp_ns) / 1_000.0


@dataclass
class QuoteUpdate:
    """A quote update message."""
    venue: VenueType
    ticker: str
    old_bid: float
    old_ask: float
    new_bid: float
    new_ask: float
    timestamp_ns: int
    message_type: str   # "new", "modify", "cancel"


@dataclass
class StaleQuoteAlert:
    """Detected stale quote that may represent an arbitrage opportunity."""
    ticker: str
    slow_venue: VenueType        # venue with stale quote
    fast_venue: VenueType        # venue with fresh quote
    stale_price: float
    current_price: float
    price_discrepancy_bps: float
    quote_age_us: float
    estimated_profit_bps: float
    arb_direction: str           # "buy_slow_sell_fast" or "buy_fast_sell_slow"
    confidence: float            # 0-1
    detected_at_ns: int


@dataclass
class QuoteStuffingEvent:
    """Detected quote stuffing episode."""
    ticker: str
    venue: VenueType
    start_time_ns: int
    end_time_ns: int
    peak_msg_rate: float        # messages per second
    avg_msg_rate_baseline: float
    cancellation_rate: float    # fraction of orders cancelled
    order_to_trade_ratio: float
    duration_ms: float
    status: StuffingStatus
    suspected_actor: Optional[str] = None


@dataclass
class LatencyArbitrageResult:
    """Result of a latency arbitrage simulation."""
    ticker: str
    n_opportunities: int
    n_captured: int
    capture_rate: float
    total_pnl_bps: float
    avg_profit_per_trade_bps: float
    avg_opportunity_duration_us: float
    avg_latency_advantage_us: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown_bps: float


@dataclass
class HFTVenueAnalysis:
    """Analysis of a venue from an HFT perspective."""
    venue: VenueType
    avg_effective_spread_bps: float
    quote_update_rate_per_sec: float
    cancellation_rate: float
    order_to_trade_ratio: float
    toxic_flow_pct: float        # fraction of flow that is informed
    maker_fill_rate: float       # fraction of passive orders filled
    hft_participation_pct: float
    latency_score: float         # 0-100, higher = better for HFT


# ---------------------------------------------------------------------------
# Physical latency model
# ---------------------------------------------------------------------------

# Speed of light in fiber (approx 2/3 c)
FIBER_SPEED_KM_PER_US = 299_792.458 * (2/3) / 1_000_000  # ~0.2 km/µs = 200 km/ms

VENUE_PROFILES: Dict[VenueType, VenueProfile] = {
    VenueType.NASDAQ: VenueProfile(
        venue=VenueType.NASDAQ,
        city="Carteret, NJ",
        latitude=40.5773, longitude=-74.2269,
        latency_to_ny_us=1_200,
        latency_to_chicago_us=7_800,
        latency_to_london_us=38_000,
        speed_bump_us=0.0,
        matching_engine_us=4.5,
        sip_feed_lag_us=120.0,
        direct_feed_lag_us=3.0,
        maker_fee_bps=-0.22,
        taker_fee_bps=0.30,
    ),
    VenueType.NYSE: VenueProfile(
        venue=VenueType.NYSE,
        city="Mahwah, NJ",
        latitude=41.0885, longitude=-74.1480,
        latency_to_ny_us=800,
        latency_to_chicago_us=8_100,
        latency_to_london_us=37_500,
        speed_bump_us=0.0,
        matching_engine_us=6.0,
        sip_feed_lag_us=110.0,
        direct_feed_lag_us=4.0,
        maker_fee_bps=-0.18,
        taker_fee_bps=0.28,
    ),
    VenueType.BATS: VenueProfile(
        venue=VenueType.BATS,
        city="Secaucus, NJ",
        latitude=40.7895, longitude=-74.0565,
        latency_to_ny_us=600,
        latency_to_chicago_us=7_600,
        latency_to_london_us=37_000,
        speed_bump_us=0.0,
        matching_engine_us=3.8,
        sip_feed_lag_us=90.0,
        direct_feed_lag_us=2.5,
        maker_fee_bps=-0.25,
        taker_fee_bps=0.30,
    ),
    VenueType.EDGX: VenueProfile(
        venue=VenueType.EDGX,
        city="Secaucus, NJ",
        latitude=40.7895, longitude=-74.0565,
        latency_to_ny_us=600,
        latency_to_chicago_us=7_600,
        latency_to_london_us=37_000,
        speed_bump_us=0.0,
        matching_engine_us=3.5,
        sip_feed_lag_us=85.0,
        direct_feed_lag_us=2.2,
        maker_fee_bps=-0.30,
        taker_fee_bps=0.32,
    ),
    VenueType.IEX: VenueProfile(
        venue=VenueType.IEX,
        city="Weehawken, NJ",
        latitude=40.7683, longitude=-74.0130,
        latency_to_ny_us=500,
        latency_to_chicago_us=7_400,
        latency_to_london_us=36_800,
        speed_bump_us=350.0,   # Infamous IEX speed bump
        matching_engine_us=5.0,
        sip_feed_lag_us=100.0,
        direct_feed_lag_us=350.0,  # includes speed bump
        maker_fee_bps=0.0,
        taker_fee_bps=0.09,
    ),
}

COLO_PROFILES: Dict[Tuple[VenueType, ColoTier], ColoProfile] = {
    (VenueType.NASDAQ, ColoTier.TIER_1): ColoProfile(
        venue=VenueType.NASDAQ, tier=ColoTier.TIER_1,
        rtt_us=25.0, monthly_cost_usd=25_000, cross_connect_usd=1_500, max_orders_per_sec=500_000,
    ),
    (VenueType.NASDAQ, ColoTier.TIER_2): ColoProfile(
        venue=VenueType.NASDAQ, tier=ColoTier.TIER_2,
        rtt_us=150.0, monthly_cost_usd=8_000, cross_connect_usd=800, max_orders_per_sec=100_000,
    ),
    (VenueType.NASDAQ, ColoTier.RETAIL): ColoProfile(
        venue=VenueType.NASDAQ, tier=ColoTier.RETAIL,
        rtt_us=5_000.0, monthly_cost_usd=0, cross_connect_usd=0, max_orders_per_sec=1_000,
    ),
    (VenueType.BATS, ColoTier.TIER_1): ColoProfile(
        venue=VenueType.BATS, tier=ColoTier.TIER_1,
        rtt_us=20.0, monthly_cost_usd=22_000, cross_connect_usd=1_200, max_orders_per_sec=600_000,
    ),
    (VenueType.IEX, ColoTier.TIER_1): ColoProfile(
        venue=VenueType.IEX, tier=ColoTier.TIER_1,
        rtt_us=380.0,   # speed bump dominates
        monthly_cost_usd=10_000, cross_connect_usd=1_000, max_orders_per_sec=50_000,
    ),
}


class PhysicalLatencyModel:
    """
    Models one-way and round-trip latency between two points based on
    fiber distance and speed of light constraints.
    """

    def haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in km."""
        R = 6_371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def theoretical_one_way_us(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Theoretical minimum one-way latency in microseconds (fiber)."""
        km = self.haversine_km(lat1, lon1, lat2, lon2)
        # Fiber adds ~1.5x routing overhead over straight-line distance
        effective_km = km * 1.5
        return effective_km / FIBER_SPEED_KM_PER_US

    def venue_to_venue_us(self, v1: VenueType, v2: VenueType) -> float:
        """Estimate latency between two venues in microseconds."""
        p1 = VENUE_PROFILES.get(v1)
        p2 = VENUE_PROFILES.get(v2)
        if not p1 or not p2:
            return 500.0  # default
        return self.theoretical_one_way_us(p1.latitude, p1.longitude, p2.latitude, p2.longitude)

    def effective_rtt_us(
        self,
        venue: VenueType,
        colo_tier: ColoTier,
        include_speed_bump: bool = True,
    ) -> float:
        """Effective round-trip time with colo and optional speed bump."""
        key = (venue, colo_tier)
        profile_colo = COLO_PROFILES.get(key)
        if profile_colo:
            rtt = profile_colo.rtt_us
        else:
            tier_latency = {ColoTier.TIER_1: 30, ColoTier.TIER_2: 150, ColoTier.TIER_3: 500, ColoTier.RETAIL: 5000}
            rtt = tier_latency.get(colo_tier, 500)

        venue_profile = VENUE_PROFILES.get(venue)
        if venue_profile and include_speed_bump:
            rtt += venue_profile.speed_bump_us * 2  # applied both ways at IEX

        return rtt

    def latency_advantage_us(
        self,
        venue: VenueType,
        fast_tier: ColoTier,
        slow_tier: ColoTier,
    ) -> float:
        """How many microseconds the fast tier gains over the slow tier."""
        return (
            self.effective_rtt_us(venue, slow_tier)
            - self.effective_rtt_us(venue, fast_tier)
        )

    def report_venue_latencies(self) -> str:
        lines = ["=== Venue Latency Matrix ===", ""]
        lines.append(f"{'Venue':<12} {'City':<20} {'RTT Tier1':>12} {'RTT Retail':>12} {'Speed Bump':>12}")
        lines.append("-" * 72)
        for venue, vp in VENUE_PROFILES.items():
            t1 = COLO_PROFILES.get((venue, ColoTier.TIER_1))
            rtt_t1 = f"{t1.rtt_us:.0f} µs" if t1 else "N/A"
            rtt_retail = "~5,000 µs"
            bump = f"{vp.speed_bump_us:.0f} µs" if vp.speed_bump_us > 0 else "none"
            lines.append(f"{venue.value:<12} {vp.city:<20} {rtt_t1:>12} {rtt_retail:>12} {bump:>12}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stale quote detector
# ---------------------------------------------------------------------------

class StaleQuoteDetector:
    """
    Detects stale quotes by comparing quote timestamps across venues.
    A quote is considered stale if a newer quote at another venue implies
    a different fair value, creating an arbitrage opportunity.

    Theory:
    - Venues receive price-moving news at slightly different times
    - Fast traders can see the update on Venue A before it propagates to Venue B
    - They trade the stale quote on Venue B against the updated Venue B price

    Key parameters:
    - min_discrepancy_bps: minimum price gap to flag (avoids noise)
    - max_staleness_us: beyond this, the arb is likely gone
    - confidence_decay_us: staleness above which confidence drops sharply
    """

    def __init__(
        self,
        min_discrepancy_bps: float = 0.5,
        max_staleness_us: float = 5_000.0,
        confidence_decay_us: float = 500.0,
    ):
        self.min_discrepancy_bps = min_discrepancy_bps
        self.max_staleness_us = max_staleness_us
        self.confidence_decay_us = confidence_decay_us
        self._quote_cache: Dict[Tuple[str, VenueType], Quote] = {}

    def ingest_quote(self, quote: Quote) -> None:
        self._quote_cache[(quote.ticker, quote.venue)] = quote

    def detect_stale_quotes(self, ticker: str) -> List[StaleQuoteAlert]:
        """Cross-venue staleness check for a single ticker."""
        quotes = {
            venue: q for (t, venue), q in self._quote_cache.items()
            if t == ticker
        }
        if len(quotes) < 2:
            return []

        # Find the most recently updated quote (reference)
        sorted_quotes = sorted(quotes.values(), key=lambda q: q.timestamp_ns, reverse=True)
        fast_quote = sorted_quotes[0]

        alerts: List[StaleQuoteAlert] = []
        now_ns = int(time.time() * 1e9)

        for slow_quote in sorted_quotes[1:]:
            age_diff_us = (fast_quote.timestamp_ns - slow_quote.timestamp_ns) / 1_000.0
            if age_diff_us < 0:
                continue
            if age_diff_us > self.max_staleness_us:
                continue

            # Price discrepancy
            fast_mid = fast_quote.mid
            slow_mid = slow_quote.mid
            if fast_mid <= 0:
                continue

            disc_bps = abs(fast_mid - slow_mid) / fast_mid * 10_000
            if disc_bps < self.min_discrepancy_bps:
                continue

            # Direction
            if fast_mid > slow_mid:
                # Fast venue has moved up → buy slow (stale low) and sell fast (current high)
                direction = "buy_slow_sell_fast"
                # We buy at slow ask, sell at fast bid
                raw_profit_bps = (fast_quote.bid - slow_quote.ask) / fast_mid * 10_000
            else:
                direction = "buy_fast_sell_slow"
                raw_profit_bps = (slow_quote.bid - fast_quote.ask) / fast_mid * 10_000

            # Confidence decays with age
            conf = math.exp(-age_diff_us / self.confidence_decay_us)

            estimated_profit_bps = raw_profit_bps * conf

            alerts.append(StaleQuoteAlert(
                ticker=ticker,
                slow_venue=slow_quote.venue,
                fast_venue=fast_quote.venue,
                stale_price=slow_mid,
                current_price=fast_mid,
                price_discrepancy_bps=disc_bps,
                quote_age_us=age_diff_us,
                estimated_profit_bps=estimated_profit_bps,
                arb_direction=direction,
                confidence=conf,
                detected_at_ns=now_ns,
            ))

        return alerts

    def scan_all_tickers(self) -> List[StaleQuoteAlert]:
        tickers = {t for (t, _) in self._quote_cache}
        all_alerts: List[StaleQuoteAlert] = []
        for ticker in tickers:
            all_alerts.extend(self.detect_stale_quotes(ticker))
        # Sort by estimated profit
        return sorted(all_alerts, key=lambda a: a.estimated_profit_bps, reverse=True)


# ---------------------------------------------------------------------------
# Quote stuffing detector
# ---------------------------------------------------------------------------

@dataclass
class MessageRateWindow:
    """Sliding window of message counts for stuffing detection."""
    ticker: str
    venue: VenueType
    window_sec: float
    timestamps_ns: Deque[int] = field(default_factory=deque)
    order_count: int = 0
    trade_count: int = 0
    cancel_count: int = 0

    def add_message(self, ts_ns: int, msg_type: str) -> None:
        self.timestamps_ns.append(ts_ns)
        if msg_type in ("new", "modify"):
            self.order_count += 1
        elif msg_type == "cancel":
            self.cancel_count += 1
        elif msg_type == "trade":
            self.trade_count += 1
        # Evict old messages
        cutoff_ns = ts_ns - int(self.window_sec * 1e9)
        while self.timestamps_ns and self.timestamps_ns[0] < cutoff_ns:
            self.timestamps_ns.popleft()

    @property
    def msg_rate(self) -> float:
        if not self.timestamps_ns:
            return 0.0
        span_s = (self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9
        return len(self.timestamps_ns) / max(span_s, 0.001)

    @property
    def cancellation_rate(self) -> float:
        total = self.order_count + self.cancel_count
        return self.cancel_count / total if total > 0 else 0.0

    @property
    def order_to_trade_ratio(self) -> float:
        return (self.order_count + self.cancel_count) / max(self.trade_count, 1)


class QuoteStuffingDetector:
    """
    Detects quote stuffing — a manipulative strategy where traders flood
    exchanges with rapid order submissions and cancellations to slow down
    competitors' systems and create information asymmetry.

    Key signals:
    1. Message rate spike >> baseline (>10x)
    2. High cancellation rate (>95%)
    3. High order-to-trade ratio (>100:1)
    4. Very short order lifetimes (<1 ms average)
    5. Concentrated in narrow price range (not genuine price discovery)
    """

    # Thresholds
    MSG_RATE_SPIKE_FACTOR    = 8.0    # x above baseline
    MIN_CANCELLATION_RATE    = 0.92   # >92% cancels
    MIN_ORDER_TO_TRADE_RATIO = 50.0   # 50 orders per trade
    STUFFING_DURATION_MS_MIN = 10.0   # at least 10ms burst
    BASELINE_WINDOW_SEC      = 60.0   # baseline computed over 60s
    BURST_WINDOW_SEC         = 1.0    # burst window

    def __init__(self):
        self._windows: Dict[Tuple[str, VenueType], MessageRateWindow] = {}
        self._baseline_rates: Dict[Tuple[str, VenueType], float] = {}
        self._stuffing_events: List[QuoteStuffingEvent] = []
        self._active_stuffing: Dict[Tuple[str, VenueType], int] = {}  # start ts_ns

    def _get_window(self, ticker: str, venue: VenueType) -> MessageRateWindow:
        key = (ticker, venue)
        if key not in self._windows:
            self._windows[key] = MessageRateWindow(ticker, venue, self.BURST_WINDOW_SEC)
        return self._windows[key]

    def ingest_message(self, update: QuoteUpdate) -> Optional[QuoteStuffingEvent]:
        """Process a quote update and return a stuffing event if detected."""
        key = (update.ticker, update.venue)
        window = self._get_window(update.ticker, update.venue)
        window.add_message(update.timestamp_ns, update.message_type)

        current_rate = window.msg_rate
        baseline = self._baseline_rates.get(key, current_rate)
        # Update baseline slowly
        self._baseline_rates[key] = baseline * 0.99 + current_rate * 0.01

        # Detect stuffing
        if baseline < 1.0:
            return None  # too little data

        spike_factor = current_rate / baseline
        status = self._classify_stuffing(
            spike_factor, window.cancellation_rate, window.order_to_trade_ratio
        )

        if status in (StuffingStatus.SUSPECTED, StuffingStatus.CONFIRMED):
            if key not in self._active_stuffing:
                self._active_stuffing[key] = update.timestamp_ns

        elif key in self._active_stuffing:
            # Stuffing burst ended
            start_ns = self._active_stuffing.pop(key)
            duration_ms = (update.timestamp_ns - start_ns) / 1e6

            if duration_ms >= self.STUFFING_DURATION_MS_MIN:
                event = QuoteStuffingEvent(
                    ticker=update.ticker,
                    venue=update.venue,
                    start_time_ns=start_ns,
                    end_time_ns=update.timestamp_ns,
                    peak_msg_rate=current_rate,
                    avg_msg_rate_baseline=baseline,
                    cancellation_rate=window.cancellation_rate,
                    order_to_trade_ratio=window.order_to_trade_ratio,
                    duration_ms=duration_ms,
                    status=status,
                )
                self._stuffing_events.append(event)
                return event

        return None

    def _classify_stuffing(
        self,
        spike_factor: float,
        cancellation_rate: float,
        otr: float,
    ) -> StuffingStatus:
        score = 0
        if spike_factor > self.MSG_RATE_SPIKE_FACTOR:
            score += 3
        elif spike_factor > self.MSG_RATE_SPIKE_FACTOR / 2:
            score += 1

        if cancellation_rate > self.MIN_CANCELLATION_RATE:
            score += 3
        elif cancellation_rate > 0.80:
            score += 1

        if otr > self.MIN_ORDER_TO_TRADE_RATIO:
            score += 2
        elif otr > 20:
            score += 1

        if score >= 6:
            return StuffingStatus.CONFIRMED
        elif score >= 4:
            return StuffingStatus.SUSPECTED
        elif score >= 2:
            return StuffingStatus.ELEVATED
        return StuffingStatus.NORMAL

    def get_stuffing_summary(self) -> Dict:
        if not self._stuffing_events:
            return {"n_events": 0, "venues": {}}

        by_venue: Dict[str, List[QuoteStuffingEvent]] = defaultdict(list)
        for e in self._stuffing_events:
            by_venue[e.venue.value].append(e)

        return {
            "n_events": len(self._stuffing_events),
            "n_confirmed": sum(1 for e in self._stuffing_events if e.status == StuffingStatus.CONFIRMED),
            "n_suspected": sum(1 for e in self._stuffing_events if e.status == StuffingStatus.SUSPECTED),
            "venues": {
                venue: {
                    "count": len(evts),
                    "avg_duration_ms": statistics.mean(e.duration_ms for e in evts),
                    "avg_peak_rate": statistics.mean(e.peak_msg_rate for e in evts),
                    "avg_cancel_rate": statistics.mean(e.cancellation_rate for e in evts),
                    "avg_otr": statistics.mean(e.order_to_trade_ratio for e in evts),
                }
                for venue, evts in by_venue.items()
            },
        }


# ---------------------------------------------------------------------------
# Latency arbitrage simulator
# ---------------------------------------------------------------------------

@dataclass
class SimulatedQuoteState:
    """Internal state for the latency arb simulator."""
    ticker: str
    true_mid: float
    true_spread: float
    last_news_ts_us: float
    last_news_magnitude_bps: float


class LatencyArbitrageSimulator:
    """
    Simulates the economics of latency arbitrage between two venues.

    Model:
    - A "true" price process evolves via a Poisson news process
    - Venue A (fast) receives news at time T
    - Venue B (slow) receives the same news at T + latency_gap
    - A co-located trader can see the Venue A update and trade on Venue B's
      stale quote before it adjusts

    Profitability depends on:
    1. Latency advantage (how much faster vs. next-best competitor)
    2. News frequency and magnitude
    3. Spread (determines if arb is profitable net of costs)
    4. Fill probability (stale quotes may be pulled before fill)
    """

    def __init__(
        self,
        fast_rtt_us: float = 25.0,    # co-located HFT
        slow_rtt_us: float = 5_000.0,  # retail / slower competitor
        news_rate_per_sec: float = 2.0,
        news_magnitude_bps: float = 3.0,
        spread_bps: float = 1.0,
        taker_fee_bps: float = 0.30,
        seed: Optional[int] = None,
    ):
        self.fast_rtt_us = fast_rtt_us
        self.slow_rtt_us = slow_rtt_us
        self.latency_advantage_us = slow_rtt_us - fast_rtt_us
        self.news_rate = news_rate_per_sec
        self.news_magnitude_bps = news_magnitude_bps
        self.spread_bps = spread_bps
        self.taker_fee_bps = taker_fee_bps
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        duration_sec: float = 3600.0,
        initial_price: float = 100.0,
    ) -> LatencyArbitrageResult:
        """Run a full simulation and return performance metrics."""
        dt_sec = 0.001  # 1ms simulation step
        n_steps = int(duration_sec / dt_sec)

        price = initial_price
        news_interval_sec = 1.0 / self.news_rate

        pnl_history: List[float] = []
        opportunity_durations: List[float] = []

        n_opportunities = 0
        n_captured = 0
        n_wins = 0
        total_pnl_bps = 0.0

        # Poisson news process
        next_news_t = self.rng.exponential(news_interval_sec)
        pending_arb: Optional[Tuple[float, float, float]] = None  # (open_t, direction, magnitude)

        for step in range(n_steps):
            t = step * dt_sec

            # News event fires?
            if t >= next_news_t:
                magnitude_bps = self.rng.exponential(self.news_magnitude_bps)
                direction = 1 if self.rng.random() > 0.5 else -1
                next_news_t = t + self.rng.exponential(news_interval_sec)

                # Fast trader sees this update immediately (after fast RTT)
                # Slow venue updates after slow RTT → window = slow_rtt - fast_rtt
                n_opportunities += 1
                window_us = self.latency_advantage_us
                opportunity_durations.append(window_us)

                # Fill probability decreases as window shrinks
                fill_prob = min(1.0, window_us / 200.0)  # 100% at 200µs window

                if self.rng.random() < fill_prob:
                    n_captured += 1

                    # Net profit = move magnitude - half spread (crossing the spread) - fees*2
                    net_bps = magnitude_bps - self.spread_bps / 2 - self.taker_fee_bps * 2

                    # Adverse selection: sometimes quote is already pulled (15%)
                    if self.rng.random() < 0.15:
                        net_bps = -self.taker_fee_bps  # fee only on miss

                    pnl_history.append(net_bps)
                    total_pnl_bps += net_bps
                    if net_bps > 0:
                        n_wins += 1

        # Compute stats
        capture_rate = n_captured / max(n_opportunities, 1)
        win_rate = n_wins / max(n_captured, 1)

        if pnl_history:
            avg_profit = statistics.mean(pnl_history)
            if len(pnl_history) > 1:
                pnl_std = statistics.stdev(pnl_history)
                sharpe = avg_profit / pnl_std * math.sqrt(252 * 6.5 * 3600 / duration_sec) if pnl_std > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown
            cumulative = np.cumsum(pnl_history)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = peak - cumulative
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            avg_profit = 0.0
            sharpe = 0.0
            max_dd = 0.0

        return LatencyArbitrageResult(
            ticker="SIMULATED",
            n_opportunities=n_opportunities,
            n_captured=n_captured,
            capture_rate=capture_rate,
            total_pnl_bps=total_pnl_bps,
            avg_profit_per_trade_bps=avg_profit,
            avg_opportunity_duration_us=statistics.mean(opportunity_durations) if opportunity_durations else 0.0,
            avg_latency_advantage_us=self.latency_advantage_us,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown_bps=max_dd,
        )

    def sensitivity_analysis(self) -> List[Dict]:
        """Run simulations across a range of latency advantages."""
        results = []
        for advantage_us in [10, 25, 50, 100, 200, 500, 1000, 5000]:
            sim = LatencyArbitrageSimulator(
                fast_rtt_us=25.0,
                slow_rtt_us=25.0 + advantage_us,
                news_rate_per_sec=self.news_rate,
                news_magnitude_bps=self.news_magnitude_bps,
                spread_bps=self.spread_bps,
                taker_fee_bps=self.taker_fee_bps,
                seed=42,
            )
            res = sim.simulate(duration_sec=3600.0)
            results.append({
                "latency_advantage_us": advantage_us,
                "capture_rate": res.capture_rate,
                "avg_profit_bps": res.avg_profit_per_trade_bps,
                "total_pnl_bps": res.total_pnl_bps,
                "sharpe": res.sharpe_ratio,
                "win_rate": res.win_rate,
            })
        return results

    def breakeven_analysis(self) -> Dict:
        """Find the minimum latency advantage for profitability."""
        for advantage_us in range(5, 500, 5):
            sim = LatencyArbitrageSimulator(
                fast_rtt_us=25.0,
                slow_rtt_us=25.0 + advantage_us,
                news_rate_per_sec=self.news_rate,
                news_magnitude_bps=self.news_magnitude_bps,
                spread_bps=self.spread_bps,
                taker_fee_bps=self.taker_fee_bps,
                seed=42,
            )
            res = sim.simulate(duration_sec=3600.0)
            if res.total_pnl_bps > 0 and res.win_rate > 0.5:
                return {
                    "breakeven_latency_us": advantage_us,
                    "capture_rate_at_breakeven": res.capture_rate,
                    "win_rate_at_breakeven": res.win_rate,
                    "pnl_at_breakeven": res.total_pnl_bps,
                }
        return {"breakeven_latency_us": 500, "note": "not found in scan range"}


# ---------------------------------------------------------------------------
# HFT venue analyzer
# ---------------------------------------------------------------------------

class HFTVenueAnalyzer:
    """
    Analyzes venues from an HFT perspective, scoring them for:
    - Latency characteristics
    - Fee structure
    - Fill rates for passive orders
    - Toxic flow fraction
    - Quote stuffing prevalence
    """

    def __init__(self):
        self.latency_model = PhysicalLatencyModel()

    def analyze_venue(self, venue: VenueType) -> HFTVenueAnalysis:
        vp = VENUE_PROFILES.get(venue)
        if not vp:
            return HFTVenueAnalysis(
                venue=venue, avg_effective_spread_bps=2.0, quote_update_rate_per_sec=100.0,
                cancellation_rate=0.70, order_to_trade_ratio=10.0, toxic_flow_pct=0.15,
                maker_fill_rate=0.25, hft_participation_pct=0.50, latency_score=50.0,
            )

        # Latency score: lower rtt = higher score
        t1_profile = COLO_PROFILES.get((venue, ColoTier.TIER_1))
        t1_rtt = t1_profile.rtt_us if t1_profile else 100.0
        latency_score = max(0.0, 100.0 - t1_rtt / 5.0)

        # Speed bump penalty
        if vp.speed_bump_us > 0:
            latency_score = max(0.0, latency_score - 30.0)

        # Venue-specific empirical parameters (based on market microstructure literature)
        venue_params = {
            VenueType.NASDAQ: {
                "spread_bps": 1.8, "update_rate": 500_000, "cancel_rate": 0.95,
                "otr": 85, "toxic": 0.18, "fill_rate": 0.22, "hft_pct": 0.52,
            },
            VenueType.NYSE: {
                "spread_bps": 2.1, "update_rate": 350_000, "cancel_rate": 0.91,
                "otr": 60, "toxic": 0.15, "fill_rate": 0.30, "hft_pct": 0.45,
            },
            VenueType.BATS: {
                "spread_bps": 1.5, "update_rate": 600_000, "cancel_rate": 0.97,
                "otr": 120, "toxic": 0.22, "fill_rate": 0.18, "hft_pct": 0.65,
            },
            VenueType.EDGX: {
                "spread_bps": 1.4, "update_rate": 700_000, "cancel_rate": 0.98,
                "otr": 150, "toxic": 0.25, "fill_rate": 0.15, "hft_pct": 0.70,
            },
            VenueType.IEX: {
                "spread_bps": 2.0, "update_rate": 50_000, "cancel_rate": 0.55,
                "otr": 8, "toxic": 0.08, "fill_rate": 0.45, "hft_pct": 0.20,
            },
        }

        params = venue_params.get(venue, {
            "spread_bps": 2.0, "update_rate": 200_000, "cancel_rate": 0.85,
            "otr": 40, "toxic": 0.15, "fill_rate": 0.25, "hft_pct": 0.40,
        })

        return HFTVenueAnalysis(
            venue=venue,
            avg_effective_spread_bps=params["spread_bps"],
            quote_update_rate_per_sec=params["update_rate"],
            cancellation_rate=params["cancel_rate"],
            order_to_trade_ratio=params["otr"],
            toxic_flow_pct=params["toxic"],
            maker_fill_rate=params["fill_rate"],
            hft_participation_pct=params["hft_pct"],
            latency_score=latency_score,
        )

    def compare_venues(self, venues: Optional[List[VenueType]] = None) -> str:
        if venues is None:
            venues = list(VENUE_PROFILES.keys())

        analyses = [(v, self.analyze_venue(v)) for v in venues]
        lines = [
            "=== HFT Venue Comparison ===",
            "",
            f"{'Venue':<10} {'Spread':>8} {'MsgRate':>10} {'Cancel%':>9} {'OTR':>7} "
            f"{'Toxic%':>8} {'FillR%':>8} {'HFT%':>7} {'LatScore':>10}",
            "-" * 80,
        ]
        for venue, a in analyses:
            lines.append(
                f"{venue.value:<10} "
                f"{a.avg_effective_spread_bps:>7.1f}  "
                f"{a.quote_update_rate_per_sec:>9,.0f}  "
                f"{a.cancellation_rate:>8.0%}  "
                f"{a.order_to_trade_ratio:>6.0f}  "
                f"{a.toxic_flow_pct:>7.0%}  "
                f"{a.maker_fill_rate:>7.0%}  "
                f"{a.hft_participation_pct:>6.0%}  "
                f"{a.latency_score:>9.1f}"
            )
        return "\n".join(lines)

    def optimal_venue_for_strategy(self, strategy: str) -> Dict:
        """Recommend best venue for a given HFT strategy type."""
        venues = list(VENUE_PROFILES.keys())
        analyses = {v: self.analyze_venue(v) for v in venues}

        scores: Dict[VenueType, float] = {}

        if strategy == "market_making":
            # Want: high fill rate, low toxic flow, tight spread, rebates
            for v, a in analyses.items():
                vp = VENUE_PROFILES[v]
                scores[v] = (
                    a.maker_fill_rate * 40
                    - a.toxic_flow_pct * 30
                    + a.latency_score * 0.1
                    - a.avg_effective_spread_bps * 5
                    - vp.taker_fee_bps * 10
                )

        elif strategy == "latency_arb":
            # Want: lowest latency, high message rate (signals information)
            for v, a in analyses.items():
                scores[v] = (
                    a.latency_score * 0.6
                    + math.log1p(a.quote_update_rate_per_sec) * 5
                    - a.avg_effective_spread_bps * 10
                )

        elif strategy == "momentum":
            # Want: low toxic flow (to avoid adverse selection), low spread
            for v, a in analyses.items():
                scores[v] = (
                    -a.toxic_flow_pct * 50
                    - a.avg_effective_spread_bps * 8
                    + a.latency_score * 0.2
                )

        else:  # default: balanced
            for v, a in analyses.items():
                scores[v] = a.latency_score - a.avg_effective_spread_bps * 5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "strategy": strategy,
            "recommended_venue": ranked[0][0].value,
            "ranked_venues": [
                {"venue": v.value, "score": round(s, 2)} for v, s in ranked
            ],
        }


# ---------------------------------------------------------------------------
# Order-to-trade ratio analyzer (regulatory metric)
# ---------------------------------------------------------------------------

class OrderToTradeRatioAnalyzer:
    """
    Computes order-to-trade ratios and related regulatory metrics.

    MiFID II requires OTR monitoring and imposes fees on excessive order
    flow in some jurisdictions. High OTR is also a proxy for HFT activity
    and potential market manipulation (stuffing).
    """

    # European regulators flag OTR > 4:1 for investigation; US empirics show HFT at 100:1+
    WARNING_OTR = 50.0
    CRITICAL_OTR = 200.0
    STUFFING_CANCEL_RATE = 0.95

    def __init__(self):
        self._by_participant: Dict[str, Dict] = defaultdict(lambda: {
            "orders": 0, "cancels": 0, "trades": 0, "vol_notional": 0.0
        })

    def record_order(self, participant_id: str, order_type: str, notional: float = 0.0) -> None:
        p = self._by_participant[participant_id]
        if order_type in ("new", "modify"):
            p["orders"] += 1
        elif order_type == "cancel":
            p["cancels"] += 1
        elif order_type == "trade":
            p["trades"] += 1
            p["vol_notional"] += notional

    def analyze_participant(self, participant_id: str) -> Dict:
        p = self._by_participant.get(participant_id, {})
        if not p:
            return {}

        orders = p["orders"]
        cancels = p["cancels"]
        trades = p["trades"]

        otr = (orders + cancels) / max(trades, 1)
        cancel_rate = cancels / max(orders + cancels, 1)

        flag = "normal"
        if otr > self.CRITICAL_OTR or cancel_rate > self.STUFFING_CANCEL_RATE:
            flag = "critical"
        elif otr > self.WARNING_OTR:
            flag = "warning"

        return {
            "participant": participant_id,
            "orders": orders,
            "cancels": cancels,
            "trades": trades,
            "order_to_trade_ratio": otr,
            "cancellation_rate": cancel_rate,
            "notional_traded": p["vol_notional"],
            "flag": flag,
        }

    def top_participants_by_otr(self, top_n: int = 10) -> List[Dict]:
        results = [self.analyze_participant(pid) for pid in self._by_participant]
        return sorted(results, key=lambda x: x.get("order_to_trade_ratio", 0), reverse=True)[:top_n]

    def market_level_stats(self) -> Dict:
        all_p = [self.analyze_participant(pid) for pid in self._by_participant]
        if not all_p:
            return {}
        otrs = [p["order_to_trade_ratio"] for p in all_p]
        return {
            "n_participants": len(all_p),
            "avg_otr": statistics.mean(otrs),
            "median_otr": statistics.median(otrs),
            "max_otr": max(otrs),
            "n_warning": sum(1 for p in all_p if p["flag"] == "warning"),
            "n_critical": sum(1 for p in all_p if p["flag"] == "critical"),
        }


# ---------------------------------------------------------------------------
# Synthetic data generators for testing
# ---------------------------------------------------------------------------

class SyntheticQuoteGenerator:
    """
    Generates realistic multi-venue quote streams for testing detectors.
    Includes:
    - Correlated price processes across venues
    - Venue-specific latency delays
    - Occasional stuffing bursts
    - Stale quote windows after news
    """

    def __init__(
        self,
        venues: Optional[List[VenueType]] = None,
        tickers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.venues = venues or [VenueType.NASDAQ, VenueType.BATS, VenueType.NYSE, VenueType.IEX]
        self.tickers = tickers or ["SPY", "QQQ", "AAPL"]
        self.rng = np.random.default_rng(seed)

        # Venue feed latencies (µs) — direct feed
        self.venue_latency_us = {
            VenueType.NASDAQ: 3.0,
            VenueType.BATS: 2.5,
            VenueType.NYSE: 4.0,
            VenueType.EDGX: 2.2,
            VenueType.IEX: 352.0,  # speed bump
        }

    def generate_quotes(
        self,
        ticker: str,
        n: int = 1000,
        base_price: float = 100.0,
        base_spread: float = 0.02,
    ) -> List[Quote]:
        """Generate n quotes across all venues for a single ticker."""
        quotes: List[Quote] = []
        ts_ns = int(time.time() * 1e9) - n * 1_000_000  # start 1ms per quote ago

        true_mid = base_price
        for i in range(n):
            # True mid evolves with GBM
            dt = 0.001  # 1ms
            vol = 0.20 / math.sqrt(252 * 6.5 * 3600)  # per-second vol → per-ms
            shock = self.rng.normal(0, vol * math.sqrt(dt))
            true_mid *= (1 + shock)

            ts_ns += 1_000_000  # advance 1ms

            # Each venue gets the quote at slightly different times
            for venue in self.venues:
                lag_us = self.venue_latency_us.get(venue, 5.0)
                venue_ts_ns = ts_ns + int(lag_us * 1_000)

                # Add venue-specific spread variation
                spread_multiplier = {
                    VenueType.NASDAQ: 1.0,
                    VenueType.BATS: 0.95,
                    VenueType.NYSE: 1.05,
                    VenueType.IEX: 1.10,
                    VenueType.EDGX: 0.90,
                }.get(venue, 1.0)

                spread = base_spread * spread_multiplier * (1 + abs(self.rng.normal(0, 0.1)))
                bid = true_mid - spread / 2
                ask = true_mid + spread / 2

                quotes.append(Quote(
                    venue=venue,
                    ticker=ticker,
                    bid=round(bid, 2),
                    ask=round(ask, 2),
                    bid_size=int(self.rng.integers(100, 1000)),
                    ask_size=int(self.rng.integers(100, 1000)),
                    timestamp_ns=venue_ts_ns,
                    sequence=i * len(self.venues) + self.venues.index(venue),
                    feed_type="direct",
                ))

        return quotes

    def generate_quote_updates(
        self,
        ticker: str,
        n: int = 500,
        stuffing_burst_at: Optional[int] = None,
        stuffing_n_msgs: int = 100,
    ) -> List[QuoteUpdate]:
        """Generate quote update stream, optionally with stuffing burst."""
        updates: List[QuoteUpdate] = []
        venue = VenueType.NASDAQ
        ts_ns = int(time.time() * 1e9) - n * 2_000_000

        price = 100.0

        for i in range(n):
            ts_ns += 2_000_000  # 2ms apart normally
            dp = self.rng.normal(0, 0.01)
            old_bid = price - 0.01
            old_ask = price + 0.01
            price += dp
            new_bid = price - 0.01
            new_ask = price + 0.01

            updates.append(QuoteUpdate(
                venue=venue,
                ticker=ticker,
                old_bid=old_bid, old_ask=old_ask,
                new_bid=round(new_bid, 2), new_ask=round(new_ask, 2),
                timestamp_ns=ts_ns,
                message_type=self.rng.choice(["new", "modify", "cancel"], p=[0.3, 0.3, 0.4]),
            ))

            # Inject stuffing burst
            if stuffing_burst_at is not None and i == stuffing_burst_at:
                for _ in range(stuffing_n_msgs):
                    ts_ns += 10_000  # 10µs apart
                    updates.append(QuoteUpdate(
                        venue=venue, ticker=ticker,
                        old_bid=new_bid, old_ask=new_ask,
                        new_bid=new_bid, new_ask=new_ask,
                        timestamp_ns=ts_ns,
                        message_type="cancel",  # all cancels
                    ))

        return updates


# ---------------------------------------------------------------------------
# Master latency arbitrage engine
# ---------------------------------------------------------------------------

class LatencyArbitrageEngine:
    """
    Integrates all components into a single analysis engine:
    - Physical latency modeling
    - Stale quote detection
    - Quote stuffing detection
    - Profitability simulation
    - Venue comparison
    """

    def __init__(self):
        self.latency_model = PhysicalLatencyModel()
        self.stale_detector = StaleQuoteDetector()
        self.stuffing_detector = QuoteStuffingDetector()
        self.venue_analyzer = HFTVenueAnalyzer()
        self.otr_analyzer = OrderToTradeRatioAnalyzer()
        self.quote_gen = SyntheticQuoteGenerator(seed=42)

    def run_full_analysis(
        self,
        tickers: Optional[List[str]] = None,
        n_quotes: int = 500,
        simulate_hours: float = 1.0,
    ) -> Dict:
        """Run the complete latency arbitrage analysis pipeline."""
        tickers = tickers or ["SPY", "QQQ", "AAPL"]
        results: Dict = {}

        # 1. Physical latency baseline
        results["latency_report"] = self.latency_model.report_venue_latencies()

        # 2. Stale quote detection
        all_stale_alerts: List[StaleQuoteAlert] = []
        for ticker in tickers:
            quotes = self.quote_gen.generate_quotes(ticker, n=n_quotes)
            for q in quotes:
                self.stale_detector.ingest_quote(q)
            alerts = self.stale_detector.detect_stale_quotes(ticker)
            all_stale_alerts.extend(alerts)

        results["stale_quote_alerts"] = [
            {
                "ticker": a.ticker,
                "slow_venue": a.slow_venue.value,
                "fast_venue": a.fast_venue.value,
                "discrepancy_bps": round(a.price_discrepancy_bps, 3),
                "age_us": round(a.quote_age_us, 1),
                "est_profit_bps": round(a.estimated_profit_bps, 3),
                "direction": a.arb_direction,
                "confidence": round(a.confidence, 3),
            }
            for a in all_stale_alerts[:20]
        ]

        # 3. Quote stuffing detection
        for ticker in tickers[:1]:  # test one ticker
            updates = self.quote_gen.generate_quote_updates(
                ticker, n=300, stuffing_burst_at=150, stuffing_n_msgs=200
            )
            for upd in updates:
                self.stuffing_detector.ingest_message(upd)

        results["stuffing_summary"] = self.stuffing_detector.get_stuffing_summary()

        # 4. Latency arbitrage simulation
        simulator = LatencyArbitrageSimulator(
            fast_rtt_us=25.0,
            slow_rtt_us=5_000.0,
            news_rate_per_sec=2.0,
            news_magnitude_bps=3.0,
            spread_bps=1.0,
            taker_fee_bps=0.30,
            seed=42,
        )
        sim_result = simulator.simulate(duration_sec=simulate_hours * 3600)
        results["arb_simulation"] = {
            "duration_hours": simulate_hours,
            "n_opportunities": sim_result.n_opportunities,
            "n_captured": sim_result.n_captured,
            "capture_rate": round(sim_result.capture_rate, 3),
            "total_pnl_bps": round(sim_result.total_pnl_bps, 2),
            "avg_profit_bps": round(sim_result.avg_profit_per_trade_bps, 3),
            "win_rate": round(sim_result.win_rate, 3),
            "sharpe_ratio": round(sim_result.sharpe_ratio, 2),
            "max_drawdown_bps": round(sim_result.max_drawdown_bps, 2),
        }

        # 5. Venue comparison
        results["venue_comparison"] = self.venue_analyzer.compare_venues()

        # 6. Optimal venue per strategy
        results["venue_recommendations"] = {
            strat: self.venue_analyzer.optimal_venue_for_strategy(strat)
            for strat in ["market_making", "latency_arb", "momentum"]
        }

        # 7. Breakeven analysis
        results["breakeven"] = simulator.breakeven_analysis()

        # 8. Sensitivity table
        sens = simulator.sensitivity_analysis()
        results["sensitivity"] = sens

        return results

    def format_full_report(self, tickers: Optional[List[str]] = None) -> str:
        data = self.run_full_analysis(tickers)

        lines = [
            "=" * 70,
            "        LATENCY ARBITRAGE ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        # Physical latency
        lines.append(data["latency_report"])
        lines.append("")

        # Venue comparison
        lines.append(data["venue_comparison"])
        lines.append("")

        # Simulation results
        sim = data["arb_simulation"]
        lines += [
            "=== Latency Arbitrage Simulation (Tier-1 vs Retail) ===",
            f"Duration:           {sim['duration_hours']:.1f} hours",
            f"Opportunities:      {sim['n_opportunities']:,}",
            f"Captured:           {sim['n_captured']:,} ({sim['capture_rate']:.1%})",
            f"Win rate:           {sim['win_rate']:.1%}",
            f"Total PnL:          {sim['total_pnl_bps']:.1f} bps",
            f"Avg profit/trade:   {sim['avg_profit_bps']:.3f} bps",
            f"Sharpe ratio:       {sim['sharpe_ratio']:.2f}",
            f"Max drawdown:       {sim['max_drawdown_bps']:.1f} bps",
            "",
        ]

        # Breakeven
        be = data["breakeven"]
        lines += [
            "=== Breakeven Analysis ===",
            f"Minimum profitable latency advantage: {be.get('breakeven_latency_us', 'N/A')} µs",
            "",
        ]

        # Sensitivity table
        lines.append("=== Sensitivity to Latency Advantage ===")
        lines.append(f"{'Advantage (µs)':>16} {'Capture%':>10} {'Avg PnL':>10} {'Win%':>8} {'Sharpe':>8}")
        lines.append("-" * 56)
        for row in data["sensitivity"]:
            lines.append(
                f"{row['latency_advantage_us']:>16,} "
                f"{row['capture_rate']:>9.1%}  "
                f"{row['avg_profit_bps']:>9.3f}  "
                f"{row['win_rate']:>7.1%}  "
                f"{row['sharpe']:>7.2f}"
            )
        lines.append("")

        # Stale alerts
        alerts = data["stale_quote_alerts"]
        if alerts:
            lines.append(f"=== Stale Quote Alerts (top {min(5, len(alerts))}) ===")
            for a in alerts[:5]:
                lines.append(
                    f"  {a['ticker']} | {a['slow_venue']} stale {a['age_us']:.0f}µs | "
                    f"disc={a['discrepancy_bps']:.2f}bps | est_pnl={a['est_profit_bps']:.3f}bps | "
                    f"conf={a['confidence']:.2f}"
                )
            lines.append("")

        # Stuffing
        stuffing = data["stuffing_summary"]
        lines += [
            "=== Quote Stuffing Detection ===",
            f"Events detected:    {stuffing.get('n_events', 0)}",
            f"Confirmed:          {stuffing.get('n_confirmed', 0)}",
            f"Suspected:          {stuffing.get('n_suspected', 0)}",
        ]

        # Venue recommendations
        lines.append("")
        lines.append("=== Venue Recommendations by Strategy ===")
        for strat, rec in data["venue_recommendations"].items():
            lines.append(f"  {strat:<20}: {rec['recommended_venue']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latency arbitrage analysis")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--quotes", type=int, default=500)
    args = parser.parse_args()

    engine = LatencyArbitrageEngine()
    print(engine.format_full_report(tickers=args.tickers))
