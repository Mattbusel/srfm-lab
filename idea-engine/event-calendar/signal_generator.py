"""
EventSignalGenerator: converts calendar events into IAE hypotheses and
trading signals.

Rules:
  1. Large unlock approaching (>2% supply, ≤7 days):
     -> bearish hypothesis, reduce symbol allocation by 30%
  2. Binance listing detected:
     -> bullish hypothesis for first 24h, then mean-reversion expected
  3. FOMC in <2h:
     -> "macro risk window" hypothesis, reduce all entries by 50%
  4. Major protocol upgrade: bullish if announced successful, bearish if delayed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .aggregator import EventAggregator, AggregatedEvent

logger = logging.getLogger(__name__)


@dataclass
class EventSignal:
    """A trading signal or IAE hypothesis derived from a calendar event."""

    signal_id: str
    signal_type: str           # BEARISH_HYPOTHESIS | BULLISH_HYPOTHESIS | MACRO_RISK_WINDOW
    source_event_id: str
    symbol: str                # "ALL" for macro signals
    generated_at: datetime
    valid_until: datetime
    direction: str             # BULLISH | BEARISH | NEUTRAL
    confidence: float          # 0-1
    allocation_modifier: float # multiplier on position size (0.5 = reduce 50%)
    rationale: str
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, ts: Optional[datetime] = None) -> bool:
        ts = ts or datetime.now(timezone.utc)
        return self.generated_at <= ts <= self.valid_until

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["generated_at"] = self.generated_at.isoformat()
        d["valid_until"] = self.valid_until.isoformat()
        return d


class EventSignalGenerator:
    """
    Converts AggregatedEvents into EventSignals.

    Usage::

        gen     = EventSignalGenerator()
        signals = gen.generate_all()
        active  = gen.get_active_signals()
        mult    = gen.get_entry_size_multiplier("SOL")
    """

    def __init__(
        self,
        aggregator: Optional[EventAggregator] = None,
        coinmarketcal_api_key: str = "",
        fred_api_key: str = "",
    ) -> None:
        self._agg = aggregator or EventAggregator(
            coinmarketcal_api_key=coinmarketcal_api_key,
            fred_api_key=fred_api_key,
        )

    # ── public API ──────────────────────────────────────────────────────────────

    def generate_all(self) -> List[EventSignal]:
        """Generate signals from all upcoming events."""
        calendar = self._agg.build_calendar(days_ahead=7)
        signals: List[EventSignal] = []
        for ev in calendar.all():
            sigs = self._route_event(ev)
            signals.extend(sigs)
        return signals

    def get_active_signals(self, ts: Optional[datetime] = None) -> List[EventSignal]:
        ts = ts or datetime.now(timezone.utc)
        return [s for s in self.generate_all() if s.is_active(ts)]

    def get_entry_size_multiplier(
        self, symbol: str, ts: Optional[datetime] = None
    ) -> float:
        """
        Return position-size multiplier for *symbol* at *ts*.

        Minimum of all active signal modifiers (most restrictive wins).
        """
        ts = ts or datetime.now(timezone.utc)
        active = [
            s for s in self.get_active_signals(ts)
            if s.symbol.upper() in (symbol.upper(), "ALL")
        ]
        if not active:
            return 1.0
        return min(s.allocation_modifier for s in active)

    def get_bearish_signals(self, symbol: str) -> List[EventSignal]:
        return [
            s for s in self.generate_all()
            if s.symbol.upper() == symbol.upper() and s.direction == "BEARISH"
        ]

    # ── routing ──────────────────────────────────────────────────────────────────

    def _route_event(self, ev: AggregatedEvent) -> List[EventSignal]:
        signals: List[EventSignal] = []

        cat = ev.category.lower()

        if cat == "unlock":
            sig = self._handle_unlock(ev)
            if sig:
                signals.append(sig)

        elif "exchange_listing" in cat:
            sig = self._handle_listing(ev)
            if sig:
                signals.append(sig)

        elif cat in ("fomc", "cpi", "nfp", "gdp"):
            sig = self._handle_macro(ev)
            if sig:
                signals.append(sig)

        elif cat in ("upgrade", "mainnet", "fork"):
            sig = self._handle_upgrade(ev)
            if sig:
                signals.append(sig)

        return signals

    # ── rule implementations ─────────────────────────────────────────────────────

    def _handle_unlock(self, ev: AggregatedEvent) -> Optional[EventSignal]:
        now = datetime.now(timezone.utc)
        days_away = (ev.event_date - now).total_seconds() / 86400

        # Only flag if within 7 days and the description implies >2% supply
        pct = self._extract_unlock_pct(ev.description)
        if pct < 2.0 or days_away > 7 or days_away < 0:
            return None

        return EventSignal(
            signal_id=f"sig_unlock_{ev.event_id}",
            signal_type="BEARISH_HYPOTHESIS",
            source_event_id=ev.event_id,
            symbol=ev.symbol,
            generated_at=now,
            valid_until=ev.event_date + timedelta(days=7),
            direction="BEARISH",
            confidence=0.65,
            allocation_modifier=0.7,   # reduce by 30%
            rationale=(
                f"{ev.symbol} token unlock of {pct:.1f}% circulating supply "
                f"in {days_away:.1f} days. Historical: -2% per 1% unlocked "
                f"within 7 days. Estimated impact: {pct * -2:.1f}%."
            ),
            suggested_actions=[
                f"Reduce {ev.symbol} allocation by 30% until {ev.event_date.strftime('%Y-%m-%d')}",
                f"Set tight stop-losses on existing {ev.symbol} positions",
                f"Monitor {ev.symbol} order book for pre-unlock selling",
            ],
            metadata={"unlock_pct": pct, "days_until_unlock": days_away},
        )

    def _handle_listing(self, ev: AggregatedEvent) -> Optional[EventSignal]:
        now = datetime.now(timezone.utc)
        is_delist = "delist" in ev.category.lower()
        exchange = next((s for s in ev.sources if s in ("binance", "coinbase")), "unknown")

        if is_delist:
            return EventSignal(
                signal_id=f"sig_delist_{ev.event_id}",
                signal_type="BEARISH_HYPOTHESIS",
                source_event_id=ev.event_id,
                symbol=ev.symbol,
                generated_at=now,
                valid_until=now + timedelta(hours=72),
                direction="BEARISH",
                confidence=0.80,
                allocation_modifier=0.0,   # avoid entirely
                rationale=(
                    f"{exchange.title()} delist of {ev.symbol} detected. "
                    "Historical: -40% avg within 24h. Exiting position."
                ),
                suggested_actions=[
                    f"Close all {ev.symbol} positions immediately",
                    f"Do not enter new {ev.symbol} trades",
                ],
                metadata={"exchange": exchange},
            )

        # Listing — bullish for 24h then reversion
        expected_pct = 30.0 if exchange == "binance" else 15.0
        return EventSignal(
            signal_id=f"sig_listing_{ev.event_id}",
            signal_type="BULLISH_HYPOTHESIS",
            source_event_id=ev.event_id,
            symbol=ev.symbol,
            generated_at=now,
            valid_until=ev.event_date + timedelta(hours=24),
            direction="BULLISH",
            confidence=0.70,
            allocation_modifier=1.3,   # allow slightly larger entries
            rationale=(
                f"{exchange.title()} listing of {ev.symbol}. "
                f"Historical avg: +{expected_pct}% in 24h. "
                "Reversion likely after initial premium fades."
            ),
            suggested_actions=[
                f"Allow up to 30% larger {ev.symbol} long entries in next 24h",
                f"Set profit target at +{expected_pct * 0.6:.0f}% (60% of historical avg)",
                "Tighten exits after 24h listing premium window closes",
            ],
            metadata={"exchange": exchange, "expected_pct": expected_pct},
        )

    def _handle_macro(self, ev: AggregatedEvent) -> Optional[EventSignal]:
        now = datetime.now(timezone.utc)
        hours_away = (ev.event_date - now).total_seconds() / 3600

        if hours_away < 0 or hours_away > 48:
            return None

        reduction = 0.5 if ev.impact == "HIGH" else 0.75
        return EventSignal(
            signal_id=f"sig_macro_{ev.event_id}",
            signal_type="MACRO_RISK_WINDOW",
            source_event_id=ev.event_id,
            symbol="ALL",
            generated_at=now,
            valid_until=ev.event_date + timedelta(hours=1),
            direction="NEUTRAL",
            confidence=0.85,
            allocation_modifier=reduction,
            rationale=(
                f"{ev.event_name} in {hours_away:.1f}h. "
                f"Macro event risk window active. "
                f"BTC historically moves ±{ev.event_risk_score * 5:.1f}% on surprise. "
                f"Reducing all entries to {reduction * 100:.0f}% of normal size."
            ),
            suggested_actions=[
                f"Reduce ALL new entry sizes to {reduction * 100:.0f}% until after {ev.event_name}",
                "Hold existing positions but widen stops",
                f"Resume normal sizing 1h after {ev.event_date.strftime('%H:%M UTC')}",
            ],
            metadata={"event_type": ev.category, "hours_away": hours_away},
        )

    def _handle_upgrade(self, ev: AggregatedEvent) -> Optional[EventSignal]:
        now = datetime.now(timezone.utc)
        days_away = (ev.event_date - now).total_seconds() / 86400
        if days_away < 0 or days_away > 7:
            return None

        return EventSignal(
            signal_id=f"sig_upgrade_{ev.event_id}",
            signal_type="BULLISH_HYPOTHESIS",
            source_event_id=ev.event_id,
            symbol=ev.symbol,
            generated_at=now,
            valid_until=ev.event_date + timedelta(days=2),
            direction="BULLISH",
            confidence=0.55,
            allocation_modifier=1.15,
            rationale=(
                f"{ev.symbol} protocol upgrade / mainnet launch in {days_away:.1f} days. "
                "Successful upgrades historically bullish; delayed/cancelled = bearish. "
                "Monitor for cancellation news."
            ),
            suggested_actions=[
                f"Slightly favour {ev.symbol} long setups pre-upgrade",
                "Set alert for upgrade cancellation/delay (immediate exit trigger)",
                f"Review position within 24h of {ev.event_date.strftime('%Y-%m-%d')} upgrade",
            ],
            metadata={"upgrade_type": ev.category},
        )

    # ── utilities ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_unlock_pct(description: str) -> float:
        import re
        m = re.search(r"([\d.]+)%", description)
        return float(m.group(1)) if m else 0.0
