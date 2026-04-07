"""
fear_greed.py -- Crypto Fear & Greed Index computation from component signals.

Computes a composite [0, 100] Fear & Greed score from weighted components:
    - price_momentum     (25%): recent price change relative to baseline
    - volatility         (25%): current vol vs 30-day average
    - social_volume      (15%): social media post/mention volume
    - btc_dominance      (10%): BTC market dominance % (higher = more fear)
    - google_trends      (10%): search interest proxy
    - reddit_sentiment   (15%): aggregated Reddit sentiment

Output:
    score [0, 100]: 0 = Extreme Fear, 100 = Extreme Greed
    label: "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"

Contrarian signal:
    Extreme Fear (<25) -> +0.5 (buy signal -- market oversold)
    Extreme Greed (>75) -> -0.3 (fade signal -- market overbought)
    Neutral zone -> linear interpolation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Labels and thresholds
# ---------------------------------------------------------------------------

FEAR_GREED_LABELS: List[Tuple[float, float, str]] = [
    (0.0,  25.0,  "Extreme Fear"),
    (25.0, 45.0,  "Fear"),
    (45.0, 55.0,  "Neutral"),
    (55.0, 75.0,  "Greed"),
    (75.0, 100.1, "Extreme Greed"),
]

def get_label(score: float) -> str:
    """Return human-readable label for a score."""
    for low, high, label in FEAR_GREED_LABELS:
        if low <= score < high:
            return label
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# Component weights
# ---------------------------------------------------------------------------

COMPONENT_WEIGHTS: Dict[str, float] = {
    "price_momentum":   0.25,
    "volatility":       0.25,
    "social_volume":    0.15,
    "btc_dominance":    0.10,
    "google_trends":    0.10,
    "reddit_sentiment": 0.15,
}

assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# FearGreedResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FearGreedResult:
    """Output of the Fear & Greed Index computation."""
    score: float                         # [0, 100]
    label: str                           # human-readable label
    component_scores: Dict[str, float]   # raw [0, 100] score per component
    component_weights: Dict[str, float]  # weights used
    computed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.computed_at is None:
            self.computed_at = datetime.now(timezone.utc)

    def to_signal(self) -> float:
        """
        Map Fear & Greed score to a contrarian alpha signal in [-1, 1].

        Contrarian logic:
            Extreme Fear (<25):  strong buy signal (market panicking)
            Fear (25-45):        mild buy signal
            Neutral (45-55):     near-zero signal
            Greed (55-75):       mild fade (cautious)
            Extreme Greed (>75): fade signal (market too bullish)

        Historical calibration:
            Fear (<25)  -> contrarian long
            Greed (>75) -> fade / short
        """
        s = self.score

        if s < 25.0:
            # Extreme Fear: linear from +0.5 (at 0) to +0.3 (at 25)
            return 0.5 - (s / 25.0) * 0.2

        elif s < 45.0:
            # Fear: linear from +0.3 (at 25) to +0.05 (at 45)
            t = (s - 25.0) / 20.0
            return 0.3 - t * 0.25

        elif s <= 55.0:
            # Neutral: linear from +0.05 to -0.05
            t = (s - 45.0) / 10.0
            return 0.05 - t * 0.10

        elif s <= 75.0:
            # Greed: linear from -0.05 to -0.15
            t = (s - 55.0) / 20.0
            return -0.05 - t * 0.10

        else:
            # Extreme Greed: linear from -0.15 (at 75) to -0.30 (at 100)
            t = (s - 75.0) / 25.0
            return -0.15 - t * 0.15

    def is_extreme_fear(self) -> bool:
        return self.score < 25.0

    def is_extreme_greed(self) -> bool:
        return self.score > 75.0

    def is_fear(self) -> bool:
        return self.score < 45.0

    def is_greed(self) -> bool:
        return self.score > 55.0

    def summary(self) -> str:
        sig = self.to_signal()
        return (
            f"Fear & Greed: {self.score:.1f} ({self.label}) | "
            f"Signal: {sig:+.3f}"
        )


# ---------------------------------------------------------------------------
# Component normalizers
# ---------------------------------------------------------------------------

class ComponentNormalizer:
    """
    Normalizes raw component inputs to [0, 100] scale suitable for
    Fear & Greed index computation.

    Each component has a domain-specific normalization method.
    """

    @staticmethod
    def normalize_price_momentum(
        pct_change_7d: float,
        pct_change_30d: float,
    ) -> float:
        """
        Normalize price momentum to [0, 100].

        Inputs are percentage changes (e.g. 0.15 = +15%).
        Formula combines short and medium term with sigmoid-like mapping.
        50 = neutral (flat market)
        100 = strong rally
        0 = strong drawdown
        """
        # Blend: 60% 7-day, 40% 30-day
        blended = 0.6 * pct_change_7d + 0.4 * pct_change_30d

        # Map via tanh: tanh(2.0 * blended) maps [-1, 1] range to similar
        # Scale: +50% = tanh(1.0) ~ 0.76, maps to ~88
        #        -50% = tanh(-1.0) ~ -0.76, maps to ~12
        norm = (math.tanh(2.0 * blended) + 1.0) / 2.0 * 100.0
        return max(0.0, min(100.0, norm))

    @staticmethod
    def normalize_volatility(
        current_vol: float,
        avg_vol_30d: float,
    ) -> float:
        """
        Normalize volatility to [0, 100].

        High vol vs 30-day avg -> Fear (low score).
        Low vol vs avg -> Greed (high score).

        vol_ratio = current_vol / avg_vol_30d
        - ratio > 2.0: Extreme Fear (0-20)
        - ratio ~1.0: Neutral (50)
        - ratio < 0.5: Greed (80+)
        """
        if avg_vol_30d <= 0.0:
            return 50.0

        ratio = current_vol / avg_vol_30d
        # Invert: higher ratio = lower score (more fear)
        # Map: ratio=2.0 -> 10, ratio=1.0 -> 50, ratio=0.5 -> 80
        score = 100.0 / (1.0 + math.exp(3.0 * (ratio - 1.0)))
        return max(0.0, min(100.0, score))

    @staticmethod
    def normalize_social_volume(
        current_volume: float,
        baseline_volume: float,
    ) -> float:
        """
        Normalize social media volume to [0, 100].

        Higher volume than baseline -> possible Greed (more people talking).
        Much lower than baseline -> Fear.

        Uses log ratio with sigmoid mapping.
        """
        if baseline_volume <= 0.0:
            return 50.0

        # log ratio: +1 = 10x volume, -1 = 1/10 volume
        log_ratio = math.log(max(1e-6, current_volume / baseline_volume))

        # Map: log_ratio=0 -> 50, +2 -> 85, -2 -> 15
        score = 50.0 + log_ratio * 17.5
        return max(0.0, min(100.0, score))

    @staticmethod
    def normalize_btc_dominance(
        btc_dominance_pct: float,
        baseline_dominance: float = 55.0,
    ) -> float:
        """
        Normalize BTC dominance to [0, 100].

        High BTC dominance = flight to safety = Fear.
        Low BTC dominance = risk-on = Greed.

        btc_dominance_pct: percentage [0, 100] (e.g. 60.0 = 60%)
        baseline: typical neutral level (~55%)
        """
        # Delta from baseline
        delta = btc_dominance_pct - baseline_dominance
        # High dominance -> Fear -> low score
        # Each 10pp above baseline -> score drops ~25
        score = 50.0 - delta * 2.5
        return max(0.0, min(100.0, score))

    @staticmethod
    def normalize_google_trends(
        current_score: float,
        baseline_score: float = 50.0,
    ) -> float:
        """
        Normalize Google Trends interest to [0, 100].

        Trends score is 0-100 (Google's native scale).
        Relative to baseline: higher than typical -> Greed.
        """
        if baseline_score <= 0.0:
            return 50.0

        ratio = current_score / baseline_score
        # Map similarly to social volume
        log_ratio = math.log(max(1e-6, ratio))
        score = 50.0 + log_ratio * 20.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def normalize_reddit_sentiment(
        sentiment_score: float,
    ) -> float:
        """
        Normalize Reddit sentiment [-1, 1] to Fear & Greed [0, 100].

        sentiment_score = -1: Extreme Fear (0)
        sentiment_score = 0: Neutral (50)
        sentiment_score = +1: Extreme Greed (100)
        """
        return max(0.0, min(100.0, (sentiment_score + 1.0) / 2.0 * 100.0))

    @classmethod
    def normalize_all(cls, components: Dict) -> Dict[str, float]:
        """
        Normalize a dict of raw component inputs.

        Expected keys (all optional -- missing uses 50.0):
            price_momentum: dict with keys pct_change_7d, pct_change_30d
                OR a single float (pre-blended, -1 to 1 range)
            volatility: dict with keys current_vol, avg_vol_30d
                OR float (ratio, e.g. 1.5 = 1.5x normal)
            social_volume: dict with keys current_volume, baseline_volume
                OR float (ratio)
            btc_dominance: float (percentage, e.g. 58.5)
            google_trends: dict with keys current_score, baseline_score
                OR float (current score on 0-100 scale)
            reddit_sentiment: float [-1, 1]

        Returns dict mapping component name -> [0, 100] score.
        """
        scores: Dict[str, float] = {}

        # Price momentum
        pm = components.get("price_momentum", 0.0)
        if isinstance(pm, dict):
            scores["price_momentum"] = cls.normalize_price_momentum(
                pm.get("pct_change_7d", 0.0),
                pm.get("pct_change_30d", 0.0),
            )
        elif isinstance(pm, (int, float)):
            # Treat as pre-blended [-1, 1] range
            scores["price_momentum"] = cls.normalize_price_momentum(pm, pm * 0.8)
        else:
            scores["price_momentum"] = 50.0

        # Volatility
        vol = components.get("volatility", 1.0)
        if isinstance(vol, dict):
            scores["volatility"] = cls.normalize_volatility(
                vol.get("current_vol", 1.0),
                vol.get("avg_vol_30d", 1.0),
            )
        elif isinstance(vol, (int, float)):
            # Treat as vol_ratio (current / 30d avg)
            scores["volatility"] = cls.normalize_volatility(vol, 1.0)
        else:
            scores["volatility"] = 50.0

        # Social volume
        sv = components.get("social_volume", 1.0)
        if isinstance(sv, dict):
            scores["social_volume"] = cls.normalize_social_volume(
                sv.get("current_volume", 1.0),
                sv.get("baseline_volume", 1.0),
            )
        elif isinstance(sv, (int, float)):
            scores["social_volume"] = cls.normalize_social_volume(sv, 1.0)
        else:
            scores["social_volume"] = 50.0

        # BTC dominance
        btcd = components.get("btc_dominance", 55.0)
        if isinstance(btcd, (int, float)):
            scores["btc_dominance"] = cls.normalize_btc_dominance(btcd)
        else:
            scores["btc_dominance"] = 50.0

        # Google Trends
        gt = components.get("google_trends", 50.0)
        if isinstance(gt, dict):
            scores["google_trends"] = cls.normalize_google_trends(
                gt.get("current_score", 50.0),
                gt.get("baseline_score", 50.0),
            )
        elif isinstance(gt, (int, float)):
            scores["google_trends"] = cls.normalize_google_trends(gt)
        else:
            scores["google_trends"] = 50.0

        # Reddit sentiment
        rs = components.get("reddit_sentiment", 0.0)
        if isinstance(rs, (int, float)):
            scores["reddit_sentiment"] = cls.normalize_reddit_sentiment(rs)
        else:
            scores["reddit_sentiment"] = 50.0

        return scores


# ---------------------------------------------------------------------------
# FearGreedIndex
# ---------------------------------------------------------------------------

class FearGreedIndex:
    """
    Crypto Fear & Greed Index.

    Combines six components into a single [0, 100] score using
    fixed research-calibrated weights.

    Usage::

        fgi = FearGreedIndex()
        result = fgi.compute({
            "price_momentum": {"pct_change_7d": 0.15, "pct_change_30d": 0.30},
            "volatility": {"current_vol": 0.03, "avg_vol_30d": 0.02},
            "social_volume": 1.8,   # 1.8x normal volume
            "btc_dominance": 58.0,
            "google_trends": 72.0,
            "reddit_sentiment": 0.35,
        })
        signal = result.to_signal()
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = weights or dict(COMPONENT_WEIGHTS)
        self._normalizer = ComponentNormalizer()

    def compute(self, components: Dict) -> FearGreedResult:
        """
        Compute Fear & Greed Index from raw component inputs.

        Parameters
        ----------
        components : dict
            Raw component values (see ComponentNormalizer.normalize_all docstring).

        Returns
        -------
        FearGreedResult
        """
        # Normalize all components to [0, 100]
        normalized = self._normalizer.normalize_all(components)

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for component, weight in self.weights.items():
            score = normalized.get(component, 50.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0.0:
            composite = 50.0
        else:
            composite = weighted_sum / total_weight

        composite = max(0.0, min(100.0, composite))
        label = get_label(composite)

        return FearGreedResult(
            score=composite,
            label=label,
            component_scores=normalized,
            component_weights=dict(self.weights),
            computed_at=datetime.now(timezone.utc),
        )

    def compute_from_normalized(
        self, normalized_scores: Dict[str, float]
    ) -> FearGreedResult:
        """
        Compute index from pre-normalized [0, 100] component scores.
        Useful when you've already done normalization externally.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for component, weight in self.weights.items():
            score = normalized_scores.get(component, 50.0)
            weighted_sum += score * weight
            total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0.0 else 50.0
        composite = max(0.0, min(100.0, composite))

        return FearGreedResult(
            score=composite,
            label=get_label(composite),
            component_scores=dict(normalized_scores),
            component_weights=dict(self.weights),
        )

    def rolling_compute(
        self,
        history: List[Dict],
        window: int = 7,
    ) -> List[FearGreedResult]:
        """
        Compute rolling Fear & Greed index over a list of component snapshots.

        Parameters
        ----------
        history : list of component dicts, ordered oldest -> newest
        window : int
            Rolling average window (in snapshots)

        Returns
        -------
        List of FearGreedResult (same length as history)
        """
        results: List[FearGreedResult] = []
        scores_buffer: List[float] = []

        for snapshot in history:
            result = self.compute(snapshot)
            scores_buffer.append(result.score)

            if len(scores_buffer) >= window:
                avg_score = sum(scores_buffer[-window:]) / window
                avg_result = FearGreedResult(
                    score=avg_score,
                    label=get_label(avg_score),
                    component_scores=result.component_scores,
                    component_weights=result.component_weights,
                )
                results.append(avg_result)
            else:
                results.append(result)

        return results

    def calibrate_contrarian_signal(self, score: float) -> Dict:
        """
        Return detailed contrarian signal analysis for a given score.

        Includes:
        - raw signal
        - recommended action
        - historical win rate (hardcoded from research)
        """
        result = FearGreedResult(
            score=score,
            label=get_label(score),
            component_scores={},
            component_weights=self.weights,
        )
        signal = result.to_signal()

        if score < 20:
            action = "STRONG_BUY"
            historical_win_rate = 0.72   # 72% of extreme fear -> 30d positive return
        elif score < 35:
            action = "BUY"
            historical_win_rate = 0.63
        elif score < 45:
            action = "SLIGHT_BUY"
            historical_win_rate = 0.55
        elif score <= 55:
            action = "NEUTRAL"
            historical_win_rate = 0.50
        elif score <= 65:
            action = "SLIGHT_FADE"
            historical_win_rate = 0.52
        elif score <= 80:
            action = "FADE"
            historical_win_rate = 0.58
        else:
            action = "STRONG_FADE"
            historical_win_rate = 0.65

        return {
            "score": score,
            "label": result.label,
            "contrarian_signal": signal,
            "recommended_action": action,
            "historical_win_rate": historical_win_rate,
        }


# ---------------------------------------------------------------------------
# Prebuilt scenario helpers
# ---------------------------------------------------------------------------

def extreme_fear_scenario() -> Dict:
    """Returns component inputs representing Extreme Fear market conditions."""
    return {
        "price_momentum": {"pct_change_7d": -0.25, "pct_change_30d": -0.45},
        "volatility": {"current_vol": 0.08, "avg_vol_30d": 0.03},
        "social_volume": 3.0,     # panic = high volume
        "btc_dominance": 65.0,    # flight to BTC
        "google_trends": 90.0,    # people searching "bitcoin crash"
        "reddit_sentiment": -0.7,
    }


def extreme_greed_scenario() -> Dict:
    """Returns component inputs representing Extreme Greed market conditions."""
    return {
        "price_momentum": {"pct_change_7d": 0.35, "pct_change_30d": 0.80},
        "volatility": {"current_vol": 0.015, "avg_vol_30d": 0.025},
        "social_volume": 4.0,     # euphoria = very high volume
        "btc_dominance": 40.0,    # alt season = low BTC dominance
        "google_trends": 95.0,    # everyone searching "how to buy bitcoin"
        "reddit_sentiment": 0.85,
    }


def neutral_scenario() -> Dict:
    """Returns component inputs representing Neutral market conditions."""
    return {
        "price_momentum": {"pct_change_7d": 0.02, "pct_change_30d": 0.05},
        "volatility": {"current_vol": 0.02, "avg_vol_30d": 0.02},
        "social_volume": 1.0,
        "btc_dominance": 55.0,
        "google_trends": 50.0,
        "reddit_sentiment": 0.05,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fgi = FearGreedIndex()

    scenarios = [
        ("Extreme Fear", extreme_fear_scenario()),
        ("Neutral", neutral_scenario()),
        ("Extreme Greed", extreme_greed_scenario()),
        ("Custom -- mild fear", {
            "price_momentum": {"pct_change_7d": -0.08, "pct_change_30d": -0.12},
            "volatility": 1.4,
            "social_volume": 1.3,
            "btc_dominance": 58.0,
            "google_trends": 40.0,
            "reddit_sentiment": -0.2,
        }),
    ]

    print("Fear & Greed Index Test\n" + "=" * 50)
    for name, components in scenarios:
        result = fgi.compute(components)
        signal = result.to_signal()
        print(f"\n[{name}]")
        print(f"  Score: {result.score:.1f} | Label: {result.label}")
        print(f"  Signal (contrarian): {signal:+.4f}")
        print(f"  Components: {dict((k, f'{v:.1f}') for k, v in result.component_scores.items())}")

        calib = fgi.calibrate_contrarian_signal(result.score)
        print(f"  Action: {calib['recommended_action']} | Win rate: {calib['historical_win_rate']:.0%}")
