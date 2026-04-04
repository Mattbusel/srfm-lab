"""
News-to-alpha translation.

Converts detected events and sentiment scores into tradeable alpha signals:
- Decay curves by event type
- Cross-asset contagion signals
- Sentiment momentum
- Contrarian signals
- Position sizing hints
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .event_detector import DetectedEvent, EventTypes


# ---------------------------------------------------------------------------
# Alpha signal
# ---------------------------------------------------------------------------

@dataclass
class AlphaSignal:
    """A tradeable alpha signal derived from news/events."""
    ticker: str
    timestamp: datetime
    signal: float              # -1 to +1 (normalized)
    raw_score: float           # pre-normalized score
    confidence: float          # 0-1
    source: str                # "sentiment" | "event" | "momentum" | "contrarian"
    event_type: Optional[str]  # underlying event if applicable
    horizon: str               # "1h" | "4h" | "1d" | "1w"
    decay_halflife_hours: float  # expected signal halflife
    components: Dict[str, float] = field(default_factory=dict)  # breakdown
    metadata: Dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.signal > 0

    @property
    def is_short(self) -> bool:
        return self.signal < 0

    @property
    def strength(self) -> str:
        if abs(self.signal) >= 0.7:
            return "strong"
        elif abs(self.signal) >= 0.4:
            return "medium"
        else:
            return "weak"

    def decayed_signal(self, hours_elapsed: float) -> float:
        """Compute signal value after exponential decay."""
        decay_factor = math.exp(-math.log(2) * hours_elapsed / self.decay_halflife_hours)
        return float(self.signal * decay_factor)


# ---------------------------------------------------------------------------
# Decay curve configurations
# ---------------------------------------------------------------------------

EVENT_DECAY_CONFIG: Dict[str, Dict] = {
    # format: {halflife_hours, signal_peak_delay_hours, curve_type}
    EventTypes.EARNINGS_BEAT:      {"halflife": 8.0,   "peak_delay": 0.5, "curve": "fast_decay"},
    EventTypes.EARNINGS_MISS:      {"halflife": 12.0,  "peak_delay": 0.5, "curve": "fast_decay"},
    EventTypes.GUIDANCE_RAISE:     {"halflife": 24.0,  "peak_delay": 1.0, "curve": "medium_decay"},
    EventTypes.GUIDANCE_CUT:       {"halflife": 24.0,  "peak_delay": 1.0, "curve": "medium_decay"},
    EventTypes.MA_ANNOUNCE:        {"halflife": 168.0, "peak_delay": 0.1, "curve": "sustained"},
    EventTypes.MA_RUMOR:           {"halflife": 12.0,  "peak_delay": 0.5, "curve": "fast_decay"},
    EventTypes.ANALYST_UPGRADE:    {"halflife": 24.0,  "peak_delay": 2.0, "curve": "medium_decay"},
    EventTypes.ANALYST_DOWNGRADE:  {"halflife": 24.0,  "peak_delay": 2.0, "curve": "medium_decay"},
    EventTypes.FED_DECISION:       {"halflife": 48.0,  "peak_delay": 0.1, "curve": "macro_decay"},
    EventTypes.REGULATORY_FINE:    {"halflife": 48.0,  "peak_delay": 0.5, "curve": "medium_decay"},
    EventTypes.REGULATORY_APPROVAL:{"halflife": 96.0,  "peak_delay": 0.1, "curve": "sustained"},
    EventTypes.INSIDER_BUY:        {"halflife": 120.0, "peak_delay": 24.0,"curve": "slow_decay"},
    EventTypes.INSIDER_SELL:       {"halflife": 72.0,  "peak_delay": 24.0,"curve": "slow_decay"},
    EventTypes.DIVIDEND_CHANGE:    {"halflife": 120.0, "peak_delay": 4.0, "curve": "slow_decay"},
    EventTypes.SHARE_REPURCHASE:   {"halflife": 72.0,  "peak_delay": 2.0, "curve": "medium_decay"},
    EventTypes.LITIGATION:         {"halflife": 48.0,  "peak_delay": 1.0, "curve": "medium_decay"},
}


def compute_decay_curve(
    curve_type: str,
    hours: np.ndarray,
    halflife: float,
    peak_delay: float = 0.0,
) -> np.ndarray:
    """
    Compute signal decay over time.

    Curves:
    - fast_decay:   sharp peak, fast exponential decay
    - medium_decay: moderate peak, steady decay
    - slow_decay:   gradual ramp, slow decay
    - sustained:    plateau then slow decay (e.g., M&A)
    - macro_decay:  immediate + lingering effect
    """
    t = hours - peak_delay
    t = np.maximum(t, 0.0)

    if curve_type == "fast_decay":
        # Immediate impact, fast decay
        decay = np.exp(-t * math.log(2) / halflife)
        ramp  = 1.0 - np.exp(-hours * 3)  # quick ramp
        return decay * ramp

    elif curve_type == "medium_decay":
        decay = np.exp(-t * math.log(2) / halflife)
        ramp  = 1.0 - np.exp(-hours * 1.5)
        return decay * ramp

    elif curve_type == "slow_decay":
        decay = np.exp(-t * math.log(2) / halflife)
        ramp  = 1.0 - np.exp(-hours * 0.5)
        return decay * ramp

    elif curve_type == "sustained":
        # Fast to plateau, then very slow decay
        plateau = 0.7  # maintains 70% for a while
        initial = np.exp(-t * math.log(2) / (halflife * 0.1))
        sustained = plateau * np.exp(-t * math.log(2) / halflife)
        return np.maximum(initial, sustained)

    elif curve_type == "macro_decay":
        # Immediate spike, then slow mean reversion
        spike    = np.exp(-t * math.log(2) / (halflife * 0.2))
        lingering = 0.3 * np.exp(-t * math.log(2) / halflife)
        return spike + lingering

    else:
        return np.exp(-t * math.log(2) / halflife)


# ---------------------------------------------------------------------------
# Cross-asset contagion
# ---------------------------------------------------------------------------

SECTOR_CONTAGION: Dict[str, Dict[str, float]] = {
    # For each sector, spillover coefficients to other sectors
    "technology": {
        "technology": 1.0,
        "semiconductors": 0.6,
        "cloud": 0.7,
        "communication": 0.4,
        "financials": 0.2,
    },
    "financials": {
        "financials": 1.0,
        "real_estate": 0.5,
        "insurance": 0.6,
        "banks": 0.8,
        "technology": 0.2,
    },
    "healthcare": {
        "healthcare": 1.0,
        "biotech": 0.8,
        "pharma": 0.7,
        "medtech": 0.5,
    },
    "energy": {
        "energy": 1.0,
        "utilities": 0.4,
        "materials": 0.3,
        "industrials": 0.2,
    },
    "consumer": {
        "consumer": 1.0,
        "retail": 0.7,
        "consumer_staples": 0.5,
        "industrials": 0.2,
    },
    "macro": {
        "technology": 0.6,
        "financials": 0.8,
        "real_estate": 0.7,
        "utilities": 0.5,
        "energy": 0.4,
        "consumer": 0.5,
        "industrials": 0.5,
    },
}

TICKER_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "META": "technology", "NVDA": "semiconductors", "AMD": "semiconductors",
    "TSM": "semiconductors", "INTC": "semiconductors",
    "AMZN": "consumer", "WMT": "consumer", "TGT": "consumer",
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "BRK.B": "financials", "V": "financials", "MA": "financials",
    "JNJ": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "TSLA": "consumer", "GM": "consumer", "F": "consumer",
}


class ContagionModel:
    """
    Models cross-asset contagion from news events.
    Given a signal on one ticker, estimates spillover to related tickers/sectors.
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, str]] = None,
        contagion_map: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.sector_map = sector_map or TICKER_SECTOR_MAP
        self.contagion_map = contagion_map or SECTOR_CONTAGION

    def compute_spillovers(
        self,
        primary_ticker: str,
        primary_signal: float,
        event_type: str,
        known_tickers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute cross-asset spillover signals.
        Returns: {ticker: spillover_signal}
        """
        primary_sector = self.sector_map.get(primary_ticker.upper(), "other")

        # Get contagion coefficients
        contagion = self.contagion_map.get(primary_sector, {})

        # For macro events, use macro sector contagion
        if event_type == EventTypes.FED_DECISION:
            contagion = self.contagion_map.get("macro", {})

        spillovers: Dict[str, float] = {}

        if known_tickers:
            for ticker in known_tickers:
                if ticker == primary_ticker:
                    continue
                t_sector = self.sector_map.get(ticker.upper(), "other")

                # Direct sector contagion
                coef = contagion.get(t_sector, 0.0)

                # Reduce spillover for announcements (less market-wide)
                if event_type in (EventTypes.EARNINGS_BEAT, EventTypes.EARNINGS_MISS):
                    coef *= 0.3
                elif event_type in (EventTypes.MA_ANNOUNCE,):
                    coef *= 0.5
                elif event_type == EventTypes.FED_DECISION:
                    coef *= 0.8

                if abs(coef) > 0.05:
                    spillovers[ticker] = float(primary_signal * coef)

        return spillovers

    def compute_sector_spillovers(
        self,
        primary_sector: str,
        primary_signal: float,
        event_type: str = "other",
    ) -> Dict[str, float]:
        """Return sector-level spillover coefficients."""
        contagion = self.contagion_map.get(primary_sector, {})
        return {
            sector: float(coef * primary_signal)
            for sector, coef in contagion.items()
            if sector != primary_sector and abs(coef) > 0.1
        }


# ---------------------------------------------------------------------------
# Sentiment momentum
# ---------------------------------------------------------------------------

class SentimentMomentum:
    """
    Tracks rolling sentiment scores and detects sentiment momentum signals.
    Positive momentum: improving sentiment trend → long signal
    Negative momentum: deteriorating sentiment trend → short signal
    """

    def __init__(
        self,
        fast_window: int = 5,     # articles/days
        slow_window: int = 20,
        signal_threshold: float = 0.1,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.threshold = signal_threshold
        self._history: Dict[str, List[Tuple[datetime, float]]] = {}

    def update(self, ticker: str, timestamp: datetime, score: float) -> None:
        """Add sentiment score for ticker."""
        self._history.setdefault(ticker, []).append((timestamp, score))
        # Keep only recent history
        cutoff = timestamp - timedelta(days=90)
        self._history[ticker] = [(t, s) for t, s in self._history[ticker] if t >= cutoff]

    def compute_momentum(self, ticker: str) -> Dict[str, float]:
        """
        Compute sentiment momentum for a ticker.
        Returns: {fast_ma, slow_ma, momentum, signal, zscore}
        """
        hist = self._history.get(ticker, [])
        if len(hist) < 2:
            return {"fast_ma": 0.0, "slow_ma": 0.0, "momentum": 0.0, "signal": 0.0, "zscore": 0.0}

        scores = [s for _, s in hist]

        fast = float(np.mean(scores[-self.fast_window:]))
        slow = float(np.mean(scores[-self.slow_window:]))

        momentum = fast - slow

        # Z-score of recent scores
        if len(scores) >= 5:
            recent = scores[-20:]
            zscore = float((scores[-1] - np.mean(recent)) / (np.std(recent) + 1e-8))
        else:
            zscore = 0.0

        # Signal: positive momentum → long, negative → short
        signal = float(np.tanh(momentum / 0.2))  # normalize with tanh

        return {
            "fast_ma": fast,
            "slow_ma": slow,
            "momentum": momentum,
            "signal": signal,
            "zscore": zscore,
            "n_observations": len(scores),
        }

    def get_momentum_signal(self, ticker: str, timestamp: datetime) -> Optional[AlphaSignal]:
        """Generate an AlphaSignal from sentiment momentum."""
        mom = self.compute_momentum(ticker)
        signal = mom.get("signal", 0.0)

        if abs(signal) < self.threshold:
            return None

        return AlphaSignal(
            ticker=ticker,
            timestamp=timestamp,
            signal=signal,
            raw_score=mom["momentum"],
            confidence=min(abs(signal) * 0.7 + 0.2, 0.9),
            source="sentiment_momentum",
            event_type=None,
            horizon="1d",
            decay_halflife_hours=24.0,
            components=mom,
        )


# ---------------------------------------------------------------------------
# Contrarian signal builder
# ---------------------------------------------------------------------------

class ContrarianSignalBuilder:
    """
    Builds contrarian signals from sentiment extremes.
    When sentiment becomes extremely positive/negative,
    generate a contrarian signal for potential mean reversion.
    """

    def __init__(
        self,
        extreme_threshold: float = 0.8,  # absolute sentiment score threshold
        contrarian_window: int = 5,       # consecutive extreme readings
        contrarian_scale: float = 0.5,   # reduce signal strength for contrarian
    ):
        self.threshold = extreme_threshold
        self.window = contrarian_window
        self.scale = contrarian_scale
        self._extreme_counts: Dict[str, int] = {}
        self._last_sentiment: Dict[str, float] = {}

    def update(self, ticker: str, sentiment_score: float) -> None:
        """Update contrarian tracker."""
        self._last_sentiment[ticker] = sentiment_score
        if abs(sentiment_score) >= self.threshold:
            self._extreme_counts[ticker] = self._extreme_counts.get(ticker, 0) + 1
        else:
            self._extreme_counts[ticker] = 0

    def get_contrarian_signal(
        self, ticker: str, timestamp: datetime
    ) -> Optional[AlphaSignal]:
        """
        If extreme sentiment has persisted, generate contrarian signal.
        """
        count = self._extreme_counts.get(ticker, 0)
        if count < self.window:
            return None

        last_sent = self._last_sentiment.get(ticker, 0.0)
        # Contrarian: flip the direction
        contrarian_direction = -float(np.sign(last_sent))
        signal_strength = contrarian_direction * min(abs(last_sent) - self.threshold, 0.2) * 5 * self.scale

        return AlphaSignal(
            ticker=ticker,
            timestamp=timestamp,
            signal=float(signal_strength),
            raw_score=float(last_sent),
            confidence=0.4 + min(count / 10.0, 0.3),  # grows with persistence
            source="contrarian",
            event_type=None,
            horizon="1w",
            decay_halflife_hours=48.0,
            components={"extreme_count": float(count), "last_sentiment": float(last_sent)},
        )


# ---------------------------------------------------------------------------
# Main alpha builder
# ---------------------------------------------------------------------------

class AlphaBuilder:
    """
    Translates news and events into tradeable alpha signals.

    Pipeline:
    1. Convert events to raw signals (direction * magnitude)
    2. Apply decay curves
    3. Add contagion signals for related tickers
    4. Layer in sentiment momentum
    5. Add contrarian signals for sentiment extremes
    6. Combine and normalize
    """

    def __init__(
        self,
        decay_config: Optional[Dict] = None,
        contagion_model: Optional[ContagionModel] = None,
        momentum: Optional[SentimentMomentum] = None,
        contrarian: Optional[ContrarianSignalBuilder] = None,
    ):
        self.decay_config  = decay_config or EVENT_DECAY_CONFIG
        self.contagion     = contagion_model or ContagionModel()
        self.momentum      = momentum or SentimentMomentum()
        self.contrarian    = contrarian or ContrarianSignalBuilder()
        self._signal_history: Dict[str, List[AlphaSignal]] = {}

    def build_event_signal(
        self,
        event: DetectedEvent,
        timestamp: Optional[datetime] = None,
    ) -> AlphaSignal:
        """Convert a single detected event into an AlphaSignal."""
        ts = timestamp or event.detected_at
        cfg = self.decay_config.get(event.event_type, {"halflife": 12.0, "peak_delay": 0.0, "curve": "medium_decay"})

        signal = float(event.direction * event.magnitude * event.confidence)
        signal = float(np.clip(signal, -1.0, 1.0))

        return AlphaSignal(
            ticker=event.ticker,
            timestamp=ts,
            signal=signal,
            raw_score=float(event.direction * event.magnitude),
            confidence=event.confidence,
            source="event",
            event_type=event.event_type,
            horizon=event.horizon,
            decay_halflife_hours=float(cfg["halflife"]),
            components={
                "direction": event.direction,
                "magnitude": event.magnitude,
                "confidence": event.confidence,
            },
            metadata={"extracted_values": event.extracted_values},
        )

    def build_sentiment_signal(
        self,
        ticker: str,
        sentiment_score: float,
        confidence: float,
        timestamp: datetime,
        event_type: Optional[str] = None,
    ) -> AlphaSignal:
        """Convert a sentiment score into an AlphaSignal."""
        # Update momentum and contrarian trackers
        self.momentum.update(ticker, timestamp, sentiment_score)
        self.contrarian.update(ticker, sentiment_score)

        signal = float(np.tanh(sentiment_score * 1.5))  # soften extreme scores

        return AlphaSignal(
            ticker=ticker,
            timestamp=timestamp,
            signal=signal,
            raw_score=sentiment_score,
            confidence=confidence,
            source="sentiment",
            event_type=event_type,
            horizon="1d",
            decay_halflife_hours=8.0,
            components={"raw_sentiment": sentiment_score, "confidence": confidence},
        )

    def build_all_signals(
        self,
        events: List[DetectedEvent],
        sentiment_scores: Dict[str, float],  # ticker -> sentiment
        known_tickers: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, List[AlphaSignal]]:
        """
        Build all alpha signals from events + sentiment.
        Returns: {ticker: [signals]}
        """
        ts = timestamp or datetime.now(timezone.utc)
        ticker_signals: Dict[str, List[AlphaSignal]] = {}

        # Event-based signals
        for event in events:
            if not event.ticker:
                continue
            signal = self.build_event_signal(event, ts)
            ticker_signals.setdefault(event.ticker, []).append(signal)

            # Contagion
            spillovers = self.contagion.compute_spillovers(
                event.ticker, signal.signal, event.event_type, known_tickers
            )
            for spill_ticker, spill_val in spillovers.items():
                spill_sig = AlphaSignal(
                    ticker=spill_ticker,
                    timestamp=ts,
                    signal=spill_val,
                    raw_score=spill_val,
                    confidence=signal.confidence * 0.6,
                    source="contagion",
                    event_type=event.event_type,
                    horizon=event.horizon,
                    decay_halflife_hours=signal.decay_halflife_hours * 0.5,
                    components={"source_ticker": event.ticker, "spill_coef": spill_val / (signal.signal + 1e-8)},
                )
                ticker_signals.setdefault(spill_ticker, []).append(spill_sig)

        # Sentiment signals
        for ticker, score in sentiment_scores.items():
            sent_sig = self.build_sentiment_signal(ticker, score, 0.6, ts)
            ticker_signals.setdefault(ticker, []).append(sent_sig)

            # Momentum
            mom_sig = self.momentum.get_momentum_signal(ticker, ts)
            if mom_sig:
                ticker_signals[ticker].append(mom_sig)

            # Contrarian
            contra_sig = self.contrarian.get_contrarian_signal(ticker, ts)
            if contra_sig:
                ticker_signals[ticker].append(contra_sig)

        return ticker_signals

    def aggregate_signals(
        self,
        signals: List[AlphaSignal],
        method: str = "confidence_weighted",
    ) -> float:
        """
        Aggregate multiple signals for a ticker into one composite signal.
        Methods: "confidence_weighted" | "mean" | "max" | "vote"
        """
        if not signals:
            return 0.0

        sig_vals = np.array([s.signal for s in signals])
        conf_vals = np.array([s.confidence for s in signals])

        if method == "confidence_weighted":
            weights = conf_vals / (conf_vals.sum() + 1e-8)
            return float(np.dot(weights, sig_vals))
        elif method == "mean":
            return float(sig_vals.mean())
        elif method == "max":
            idx = np.argmax(np.abs(sig_vals))
            return float(sig_vals[idx])
        elif method == "vote":
            # Majority vote by direction
            votes = np.sign(sig_vals)
            return float(np.sign(votes.sum()))
        else:
            return float(sig_vals.mean())

    def get_composite_signals(
        self,
        ticker_signals: Dict[str, List[AlphaSignal]],
        method: str = "confidence_weighted",
    ) -> Dict[str, float]:
        """Get one composite signal per ticker."""
        return {
            ticker: self.aggregate_signals(signals, method)
            for ticker, signals in ticker_signals.items()
        }

    def get_position_sizes(
        self,
        composite_signals: Dict[str, float],
        max_position: float = 1.0,
        signal_threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Convert composite signals to position sizes.
        Uses signal as direct position sizing with threshold filter.
        """
        positions = {}
        for ticker, signal in composite_signals.items():
            if abs(signal) < signal_threshold:
                positions[ticker] = 0.0
            else:
                positions[ticker] = float(np.clip(signal * max_position, -max_position, max_position))
        return positions

    def get_decayed_signals(
        self,
        signals: List[AlphaSignal],
        current_time: datetime,
    ) -> List[AlphaSignal]:
        """Apply time decay to a list of signals."""
        decayed = []
        for sig in signals:
            hours_elapsed = (current_time - sig.timestamp).total_seconds() / 3600.0
            if hours_elapsed < 0:
                hours_elapsed = 0.0
            decayed_val = sig.decayed_signal(hours_elapsed)
            new_sig = AlphaSignal(
                ticker=sig.ticker,
                timestamp=current_time,
                signal=decayed_val,
                raw_score=sig.raw_score,
                confidence=sig.confidence * math.exp(-hours_elapsed / (sig.decay_halflife_hours * 2)),
                source=sig.source,
                event_type=sig.event_type,
                horizon=sig.horizon,
                decay_halflife_hours=sig.decay_halflife_hours,
                components=sig.components,
            )
            decayed.append(new_sig)
        return decayed


# ---------------------------------------------------------------------------
# Historical alpha performance tracker
# ---------------------------------------------------------------------------

class AlphaPerformanceTracker:
    """
    Tracks historical alpha signal performance.
    Used for signal quality evaluation and decay calibration.
    """

    def __init__(self):
        self._predictions: List[Dict] = []

    def record_signal(
        self,
        ticker: str,
        signal: float,
        timestamp: datetime,
        event_type: Optional[str],
        horizon: str,
    ) -> None:
        self._predictions.append({
            "ticker": ticker,
            "signal": signal,
            "timestamp": timestamp,
            "event_type": event_type,
            "horizon": horizon,
            "realized_return": None,
        })

    def record_outcome(
        self,
        ticker: str,
        timestamp: datetime,
        realized_return: float,
        horizon: str,
    ) -> None:
        """Record actual return for a predicted signal."""
        for pred in reversed(self._predictions):
            if (pred["ticker"] == ticker
                    and pred["horizon"] == horizon
                    and abs((pred["timestamp"] - timestamp).total_seconds()) < 3600
                    and pred["realized_return"] is None):
                pred["realized_return"] = realized_return
                break

    def compute_ic(
        self, event_type: Optional[str] = None, horizon: str = "1d"
    ) -> float:
        """
        Compute Information Coefficient (rank correlation of signals vs returns).
        """
        preds = [
            p for p in self._predictions
            if p["realized_return"] is not None
            and p["horizon"] == horizon
            and (event_type is None or p["event_type"] == event_type)
        ]
        if len(preds) < 5:
            return float("nan")

        signals = np.array([p["signal"] for p in preds])
        returns = np.array([p["realized_return"] for p in preds])

        # Spearman rank correlation
        from scipy import stats
        try:
            ic, _ = stats.spearmanr(signals, returns)
            return float(ic)
        except Exception:
            return float(np.corrcoef(signals, returns)[0, 1])

    def get_hit_rate(self, event_type: Optional[str] = None) -> float:
        """Percentage of signals with correct direction."""
        preds = [
            p for p in self._predictions
            if p["realized_return"] is not None
            and (event_type is None or p["event_type"] == event_type)
        ]
        if not preds:
            return float("nan")
        correct = sum(
            1 for p in preds
            if np.sign(p["signal"]) == np.sign(p["realized_return"])
        )
        return float(correct / len(preds))

    def summarize(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self._predictions),
            "n_with_outcomes": sum(1 for p in self._predictions if p["realized_return"] is not None),
            "overall_ic": self.compute_ic(),
            "overall_hit_rate": self.get_hit_rate(),
            "by_event_type": {
                et: {"ic": self.compute_ic(et), "hit_rate": self.get_hit_rate(et)}
                for et in set(p["event_type"] for p in self._predictions if p["event_type"])
            },
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .event_detector import DetectedEvent, EventTypes

    print("Testing alpha builder...")

    ts = datetime.now(timezone.utc)

    # Create test events
    events = [
        DetectedEvent(
            event_type=EventTypes.EARNINGS_BEAT,
            subtype="eps_beat",
            ticker="AAPL",
            confidence=0.90,
            direction=+1.0,
            magnitude=0.75,
            detected_at=ts,
            source_text="AAPL beats EPS estimates",
        ),
        DetectedEvent(
            event_type=EventTypes.FED_DECISION,
            subtype="rate_hike",
            ticker="",
            confidence=0.95,
            direction=-0.5,
            magnitude=0.85,
            detected_at=ts,
            source_text="Fed raises rates 25bps",
        ),
    ]

    sentiment_scores = {
        "AAPL": 0.7,
        "MSFT": 0.3,
        "JPM": -0.4,
    }

    builder = AlphaBuilder()
    known_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC"]

    ticker_signals = builder.build_all_signals(events, sentiment_scores, known_tickers, ts)
    composite = builder.get_composite_signals(ticker_signals)
    positions = builder.get_position_sizes(composite)

    print("Composite signals:")
    for ticker, sig in sorted(composite.items()):
        pos = positions.get(ticker, 0.0)
        print(f"  {ticker:6s}: signal={sig:+.3f} position={pos:+.3f}")

    # Test decay curves
    hours = np.linspace(0, 48, 100)
    decay = compute_decay_curve("fast_decay", hours, halflife=8.0, peak_delay=0.5)
    print(f"\nDecay at 2h: {decay[4]:.3f}, 8h: {decay[16]:.3f}, 24h: {decay[50]:.3f}")

    # Test decayed signals
    old_ts = ts - timedelta(hours=12)
    old_signal = AlphaSignal(
        ticker="AAPL", timestamp=old_ts, signal=0.8, raw_score=0.8,
        confidence=0.9, source="event", event_type=EventTypes.EARNINGS_BEAT,
        horizon="1d", decay_halflife_hours=8.0
    )
    decayed = builder.get_decayed_signals([old_signal], ts)
    print(f"Original signal: {old_signal.signal:.3f}, After 12h decay: {decayed[0].signal:.3f}")

    print("Alpha builder self-test passed.")
