"""
debate-system/agents/sentiment_analyst.py

SentimentAnalyst -- evaluates trading hypotheses through a sentiment lens.

Sentiment inputs analysed:
  - News sentiment score (NLP-derived aggregate)
  - Social media volume and tone (Twitter/Reddit/StockTwits)
  - Analyst revision breadth (upgrades vs downgrades ratio)
  - Insider activity (buy/sell ratio, dollar volume)
  - Fund flows (ETF inflows/outflows, mutual fund flows)
  - Retail vs institutional sentiment divergence
  - Put/call ratio as options-implied sentiment
  - AAII bull/bear survey
  - CNN Fear & Greed index proxy

Contrarian logic:
  When sentiment reaches extreme levels (>90th or <10th percentile of
  historical distribution), the analyst triggers a contrarian fade
  recommendation.  Extreme bullishness in a bear regime is interpreted
  as a bull trap; extreme bearishness in a bull regime as capitulation.

Regime-conditional interpretation:
  The meaning of sentiment depends on the prevailing market regime.
  Moderate bullish sentiment in an established uptrend is confirming;
  moderate bullish sentiment during a macro deterioration is suspicious.

Output: SentimentVerdict with direction, conviction, supporting_evidence,
        contrarian_flag, and a cross-asset sentiment divergence score.

Expected market_data keys
-------------------------
news_sentiment          : float  -- aggregate news score [-1, +1]
social_volume           : float  -- normalized social media mention volume
social_tone             : float  -- tone score [-1, +1]
analyst_upgrades        : int    -- number of recent upgrades
analyst_downgrades      : int    -- number of recent downgrades
insider_buy_ratio       : float  -- insider buy $ / total insider $
fund_flow_z             : float  -- fund flow z-score (positive = inflows)
retail_sentiment        : float  -- retail sentiment score [-1, +1]
institutional_sentiment : float  -- institutional sentiment score [-1, +1]
put_call_ratio          : float  -- equity put/call ratio
aaii_bull_pct           : float  -- AAII survey bull percentage (0-100)
aaii_bear_pct           : float  -- AAII survey bear percentage (0-100)
fear_greed_index        : float  -- 0-100 (0 = extreme fear, 100 = extreme greed)
hypothesis_direction    : str    -- "long" | "short" | "neutral"
asset_class             : str    -- "equity" | "crypto" | "fx" | "commodity" | "rates"

Optional historical arrays (for percentile calculations):
hist_news_sentiment     : np.ndarray
hist_put_call           : np.ndarray
hist_fear_greed         : np.ndarray
hist_fund_flow_z        : np.ndarray

Regime context:
current_regime          : str    -- "bull" | "bear" | "transition"

Cross-asset sentiment (optional):
equity_sentiment        : float
credit_sentiment        : float
vol_sentiment           : float  -- inverse VIX sentiment proxy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote
from hypothesis.types import Hypothesis


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SentimentRegime(str, Enum):
    EXTREME_GREED  = "extreme_greed"
    GREED          = "greed"
    NEUTRAL        = "neutral"
    FEAR           = "fear"
    EXTREME_FEAR   = "extreme_fear"


class ContrariaBias(str, Enum):
    """Direction the contrarian signal points."""
    FADE_LONG  = "fade_long"    # sentiment too bullish -> fade to short
    FADE_SHORT = "fade_short"   # sentiment too bearish -> fade to long
    NONE       = "none"


class CrossAssetDivergence(str, Enum):
    ALIGNED   = "aligned"
    MILD      = "mild_divergence"
    SEVERE    = "severe_divergence"


# ---------------------------------------------------------------------------
# SentimentVerdict
# ---------------------------------------------------------------------------

@dataclass
class SentimentVerdict:
    """
    Rich output from SentimentAnalyst.evaluate().

    Contains the composite sentiment score, contrarian flag, supporting
    evidence strings, and cross-asset divergence assessment.
    """
    composite_score: float              # -1 (max bearish) to +1 (max bullish)
    direction: str                      # "long" | "short" | "neutral"
    conviction: float                   # 0-1
    sentiment_regime: SentimentRegime
    contrarian_flag: ContrariaBias
    contrarian_strength: float          # 0-1, 0 = no contrarian signal
    supporting_evidence: list[str]
    warnings: list[str]
    cross_asset_divergence: CrossAssetDivergence
    divergence_detail: str
    factor_scores: dict[str, float]     # per-factor breakdown
    historical_accuracy: float | None   # rolling accuracy of sentiment calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "direction": self.direction,
            "conviction": round(self.conviction, 4),
            "sentiment_regime": self.sentiment_regime.value,
            "contrarian_flag": self.contrarian_flag.value,
            "contrarian_strength": round(self.contrarian_strength, 4),
            "supporting_evidence": self.supporting_evidence,
            "warnings": self.warnings,
            "cross_asset_divergence": self.cross_asset_divergence.value,
            "divergence_detail": self.divergence_detail,
            "factor_scores": {k: round(v, 4) for k, v in self.factor_scores.items()},
            "historical_accuracy": (
                round(self.historical_accuracy, 4)
                if self.historical_accuracy is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Accuracy tracker (in-memory rolling window)
# ---------------------------------------------------------------------------

class _AccuracyTracker:
    """Tracks rolling accuracy of sentiment-based directional calls."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._outcomes: list[bool] = []

    def record(self, was_correct: bool) -> None:
        self._outcomes.append(was_correct)
        if len(self._outcomes) > self._window:
            self._outcomes = self._outcomes[-self._window:]

    @property
    def accuracy(self) -> float | None:
        if not self._outcomes:
            return None
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def n_observations(self) -> int:
        return len(self._outcomes)


# ---------------------------------------------------------------------------
# Factor weights by asset class
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "news":          0.15,
    "social":        0.10,
    "analyst_rev":   0.15,
    "insider":       0.10,
    "fund_flow":     0.15,
    "retail_inst":   0.10,
    "put_call":      0.10,
    "aaii":          0.05,
    "fear_greed":    0.10,
}

_ASSET_CLASS_WEIGHTS: dict[str, dict[str, float]] = {
    "equity": _DEFAULT_WEIGHTS,
    "crypto": {
        "news":        0.15,
        "social":      0.25,    # social media much more relevant for crypto
        "analyst_rev": 0.05,
        "insider":     0.05,
        "fund_flow":   0.15,
        "retail_inst": 0.15,
        "put_call":    0.05,
        "aaii":        0.02,
        "fear_greed":  0.13,
    },
    "fx": {
        "news":        0.20,
        "social":      0.05,
        "analyst_rev": 0.10,
        "insider":     0.00,
        "fund_flow":   0.20,
        "retail_inst": 0.15,
        "put_call":    0.10,
        "aaii":        0.05,
        "fear_greed":  0.15,
    },
    "commodity": {
        "news":        0.20,
        "social":      0.05,
        "analyst_rev": 0.10,
        "insider":     0.05,
        "fund_flow":   0.20,
        "retail_inst": 0.10,
        "put_call":    0.10,
        "aaii":        0.05,
        "fear_greed":  0.15,
    },
    "rates": {
        "news":        0.20,
        "social":      0.03,
        "analyst_rev": 0.15,
        "insider":     0.02,
        "fund_flow":   0.25,
        "retail_inst": 0.10,
        "put_call":    0.05,
        "aaii":        0.05,
        "fear_greed":  0.15,
    },
}

# Contrarian thresholds
_EXTREME_UPPER_PCTILE = 90.0
_EXTREME_LOWER_PCTILE = 10.0
_CONTRARIAN_COMPOSITE_UPPER = 0.70
_CONTRARIAN_COMPOSITE_LOWER = -0.70


# ---------------------------------------------------------------------------
# SentimentAnalyst
# ---------------------------------------------------------------------------

class SentimentAnalyst(BaseAnalyst):
    """
    Multi-factor sentiment analysis agent for the debate system.

    Blends news, social, analyst revision, insider, fund flow, survey,
    and options-implied sentiment into a composite score.  Applies
    contrarian logic at extremes and interprets sentiment through the
    lens of the prevailing market regime.
    """

    def __init__(
        self,
        name: str = "SentimentAnalyst",
        initial_credibility: float = 0.5,
        contrarian_threshold_upper: float = _CONTRARIAN_COMPOSITE_UPPER,
        contrarian_threshold_lower: float = _CONTRARIAN_COMPOSITE_LOWER,
        accuracy_window: int = 100,
        alpha: float = 5.0,
        beta: float = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            specialization="sentiment_analysis",
            initial_credibility=initial_credibility,
            alpha=alpha,
            beta=beta,
        )
        self._contrarian_upper = contrarian_threshold_upper
        self._contrarian_lower = contrarian_threshold_lower
        self._tracker = _AccuracyTracker(window=accuracy_window)

    # ------------------------------------------------------------------
    # Public: full evaluate → SentimentVerdict
    # ------------------------------------------------------------------

    def evaluate(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> SentimentVerdict:
        """
        Run the full sentiment evaluation pipeline and return a rich
        SentimentVerdict.
        """
        asset_class = str(market_data.get("asset_class", "equity"))
        weights = _ASSET_CLASS_WEIGHTS.get(asset_class, _DEFAULT_WEIGHTS)
        current_regime = str(market_data.get("current_regime", "transition"))
        hyp_direction = str(market_data.get("hypothesis_direction", "neutral"))

        # 1. Compute individual factor scores (each [-1, +1])
        factor_scores: dict[str, float] = {}
        evidence: list[str] = []
        warnings: list[str] = []

        factor_scores["news"] = self._score_news(market_data, evidence, warnings)
        factor_scores["social"] = self._score_social(market_data, evidence, warnings)
        factor_scores["analyst_rev"] = self._score_analyst_revisions(
            market_data, evidence, warnings,
        )
        factor_scores["insider"] = self._score_insider(market_data, evidence, warnings)
        factor_scores["fund_flow"] = self._score_fund_flows(
            market_data, evidence, warnings,
        )
        factor_scores["retail_inst"] = self._score_retail_institutional(
            market_data, evidence, warnings,
        )
        factor_scores["put_call"] = self._score_put_call(
            market_data, evidence, warnings,
        )
        factor_scores["aaii"] = self._score_aaii(market_data, evidence, warnings)
        factor_scores["fear_greed"] = self._score_fear_greed(
            market_data, evidence, warnings,
        )

        # 2. Weighted composite
        composite = sum(
            factor_scores[k] * weights.get(k, 0.0)
            for k in factor_scores
        )
        composite = max(-1.0, min(1.0, composite))

        # 3. Sentiment regime classification
        regime = self._classify_sentiment_regime(composite, market_data)

        # 4. Contrarian logic
        contrarian_flag, contrarian_strength = self._evaluate_contrarian(
            composite, regime, current_regime, market_data,
        )
        if contrarian_flag != ContrariaBias.NONE:
            warnings.append(
                f"Contrarian signal active ({contrarian_flag.value}): "
                f"strength={contrarian_strength:.2f}"
            )

        # 5. Regime-conditional adjustment
        adjusted_composite = self._regime_adjust(
            composite, current_regime, contrarian_flag, contrarian_strength,
        )

        # 6. Direction and conviction
        direction = self._derive_direction(adjusted_composite, contrarian_flag)
        conviction = self._derive_conviction(
            adjusted_composite, contrarian_strength, factor_scores, weights,
        )

        # 7. Cross-asset divergence
        ca_div, ca_detail = self._cross_asset_divergence(market_data)

        return SentimentVerdict(
            composite_score=adjusted_composite,
            direction=direction,
            conviction=conviction,
            sentiment_regime=regime,
            contrarian_flag=contrarian_flag,
            contrarian_strength=contrarian_strength,
            supporting_evidence=evidence,
            warnings=warnings,
            cross_asset_divergence=ca_div,
            divergence_detail=ca_detail,
            factor_scores=factor_scores,
            historical_accuracy=self._tracker.accuracy,
        )

    # ------------------------------------------------------------------
    # BaseAnalyst interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """Wrap evaluate() into an AnalystVerdict for the DebateChamber."""
        sv = self.evaluate(hypothesis, market_data)

        hyp_dir = str(market_data.get("hypothesis_direction", "neutral"))
        alignment = self._alignment_score(sv.direction, hyp_dir)

        # Convert alignment + conviction into vote
        if alignment > 0.3:
            vote = Vote.FOR
            confidence = min(1.0, sv.conviction * alignment)
        elif alignment < -0.3:
            vote = Vote.AGAINST
            confidence = min(1.0, sv.conviction * abs(alignment))
        else:
            vote = Vote.ABSTAIN
            confidence = sv.conviction * 0.3

        reasoning_parts = [
            f"Composite sentiment: {sv.composite_score:+.3f}",
            f"Regime: {sv.sentiment_regime.value}",
            f"Recommended direction: {sv.direction}",
        ]
        if sv.contrarian_flag != ContrariaBias.NONE:
            reasoning_parts.append(
                f"CONTRARIAN: {sv.contrarian_flag.value} "
                f"(strength {sv.contrarian_strength:.2f})"
            )
        if sv.cross_asset_divergence != CrossAssetDivergence.ALIGNED:
            reasoning_parts.append(f"Cross-asset: {sv.divergence_detail}")

        return self._make_verdict(
            vote=vote,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            key_concerns=sv.warnings[:5],
        )

    # ------------------------------------------------------------------
    # Outcome recording (called externally after trade resolves)
    # ------------------------------------------------------------------

    def record_outcome(self, was_correct: bool) -> None:
        """Record whether the sentiment call was directionally correct."""
        self._tracker.record(was_correct)
        self.update_credibility(was_correct)

    # ------------------------------------------------------------------
    # Individual factor scoring methods (each returns [-1, +1])
    # ------------------------------------------------------------------

    def _score_news(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        raw = float(md.get("news_sentiment", 0.0))
        score = max(-1.0, min(1.0, raw))
        if abs(score) > 0.6:
            evidence.append(f"News sentiment strongly {'bullish' if score > 0 else 'bearish'}: {score:+.2f}")
        hist = md.get("hist_news_sentiment")
        if hist is not None and len(hist) > 20:
            pctile = float(np.percentile(hist, 50))
            if score > np.percentile(hist, 95):
                warnings.append("News sentiment at historical extreme (>95th pctile)")
            elif score < np.percentile(hist, 5):
                warnings.append("News sentiment at historical extreme (<5th pctile)")
        return score

    def _score_social(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        volume = float(md.get("social_volume", 0.0))
        tone = float(md.get("social_tone", 0.0))
        # High volume amplifies tone signal; low volume mutes it
        volume_multiplier = min(2.0, max(0.3, volume))
        score = max(-1.0, min(1.0, tone * volume_multiplier))
        if volume > 2.0:
            evidence.append(f"Social media volume spike ({volume:.1f}x normal)")
            if abs(tone) > 0.5:
                warnings.append(
                    f"High social volume + extreme tone = potential herding "
                    f"(tone={tone:+.2f})"
                )
        return score

    def _score_analyst_revisions(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        upgrades = int(md.get("analyst_upgrades", 0))
        downgrades = int(md.get("analyst_downgrades", 0))
        total = upgrades + downgrades
        if total == 0:
            return 0.0
        breadth = (upgrades - downgrades) / total  # [-1, +1]
        if abs(breadth) > 0.6:
            direction_str = "upgrade" if breadth > 0 else "downgrade"
            evidence.append(
                f"Analyst revision breadth strongly {direction_str}: "
                f"{upgrades}U/{downgrades}D (breadth={breadth:+.2f})"
            )
        if total < 3:
            warnings.append(f"Low analyst coverage: only {total} revisions")
            breadth *= 0.5  # discount thin coverage
        return max(-1.0, min(1.0, breadth))

    def _score_insider(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        buy_ratio = md.get("insider_buy_ratio")
        if buy_ratio is None:
            return 0.0
        buy_ratio = float(buy_ratio)
        # Normalize: 0.5 = balanced, >0.5 = net buying, <0.5 = net selling
        score = (buy_ratio - 0.5) * 2.0  # maps [0,1] -> [-1, +1]
        if buy_ratio > 0.75:
            evidence.append(f"Heavy insider buying (buy ratio={buy_ratio:.0%})")
        elif buy_ratio < 0.25:
            evidence.append(f"Heavy insider selling (buy ratio={buy_ratio:.0%})")
            warnings.append("Insider selling cluster detected")
        return max(-1.0, min(1.0, score))

    def _score_fund_flows(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        z = float(md.get("fund_flow_z", 0.0))
        # Z-score -> bounded score via tanh
        score = math.tanh(z / 2.0)
        if abs(z) > 2.0:
            direction_str = "inflows" if z > 0 else "outflows"
            evidence.append(f"Extreme fund {direction_str} (z={z:+.2f})")
        hist = md.get("hist_fund_flow_z")
        if hist is not None and len(hist) > 20:
            if z > float(np.percentile(hist, 95)):
                warnings.append("Fund flows at 95th+ percentile — potential crowding")
            elif z < float(np.percentile(hist, 5)):
                warnings.append("Fund flows at 5th- percentile — potential capitulation")
        return score

    def _score_retail_institutional(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        retail = float(md.get("retail_sentiment", 0.0))
        inst = float(md.get("institutional_sentiment", 0.0))
        # Blend: institutions get higher weight
        blended = inst * 0.65 + retail * 0.35
        divergence = abs(retail - inst)
        if divergence > 0.6:
            warnings.append(
                f"Retail-institutional sentiment divergence: "
                f"retail={retail:+.2f}, institutional={inst:+.2f}"
            )
            evidence.append(
                f"Sentiment divergence ({divergence:.2f}): "
                f"{'retail more bullish' if retail > inst else 'institutional more bullish'}"
            )
            # In divergence, lean toward institutional view more heavily
            blended = inst * 0.80 + retail * 0.20
        return max(-1.0, min(1.0, blended))

    def _score_put_call(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        pc = md.get("put_call_ratio")
        if pc is None:
            return 0.0
        pc = float(pc)
        # Typical range: 0.5 (bullish) to 1.5 (bearish)
        # Map to [-1, +1]: lower P/C = bullish, higher = bearish
        midpoint = 0.85
        score = -(pc - midpoint) / 0.5  # invert: high P/C = negative sentiment
        score = max(-1.0, min(1.0, score))
        hist = md.get("hist_put_call")
        if hist is not None and len(hist) > 20:
            if pc > float(np.percentile(hist, 90)):
                evidence.append(f"Put/call ratio extremely elevated ({pc:.2f}): fear signal")
            elif pc < float(np.percentile(hist, 10)):
                evidence.append(f"Put/call ratio extremely low ({pc:.2f}): complacency")
                warnings.append("Low P/C ratio suggests complacency")
        return score

    def _score_aaii(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        bull = float(md.get("aaii_bull_pct", 38.0))
        bear = float(md.get("aaii_bear_pct", 30.0))
        spread = (bull - bear) / 100.0  # normalize to roughly [-1, +1]
        if bull > 55:
            warnings.append(f"AAII bull% extremely high ({bull:.0f}%) — contrarian bearish")
        elif bear > 55:
            warnings.append(f"AAII bear% extremely high ({bear:.0f}%) — contrarian bullish")
        if bull > 50:
            evidence.append(f"Retail survey bullish: {bull:.0f}% bulls vs {bear:.0f}% bears")
        elif bear > 50:
            evidence.append(f"Retail survey bearish: {bear:.0f}% bears vs {bull:.0f}% bulls")
        return max(-1.0, min(1.0, spread))

    def _score_fear_greed(
        self, md: dict, evidence: list[str], warnings: list[str],
    ) -> float:
        fg = float(md.get("fear_greed_index", 50.0))
        # Map 0-100 to [-1, +1]
        score = (fg - 50.0) / 50.0
        if fg > 80:
            evidence.append(f"Fear & Greed at extreme greed ({fg:.0f}/100)")
            warnings.append("Extreme greed — historically precedes pullbacks")
        elif fg < 20:
            evidence.append(f"Fear & Greed at extreme fear ({fg:.0f}/100)")
            warnings.append("Extreme fear — historically precedes bounces")
        hist = md.get("hist_fear_greed")
        if hist is not None and len(hist) > 50:
            if fg > float(np.percentile(hist, 95)):
                warnings.append("Fear & Greed at 95th+ historical percentile")
            elif fg < float(np.percentile(hist, 5)):
                warnings.append("Fear & Greed at 5th- historical percentile")
        return max(-1.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Sentiment regime classification
    # ------------------------------------------------------------------

    def _classify_sentiment_regime(
        self, composite: float, md: dict,
    ) -> SentimentRegime:
        fg = float(md.get("fear_greed_index", 50.0))
        # Blend composite with Fear & Greed for classification
        blended = composite * 0.6 + ((fg - 50) / 50) * 0.4
        if blended > 0.6:
            return SentimentRegime.EXTREME_GREED
        elif blended > 0.2:
            return SentimentRegime.GREED
        elif blended > -0.2:
            return SentimentRegime.NEUTRAL
        elif blended > -0.6:
            return SentimentRegime.FEAR
        else:
            return SentimentRegime.EXTREME_FEAR

    # ------------------------------------------------------------------
    # Contrarian logic
    # ------------------------------------------------------------------

    def _evaluate_contrarian(
        self,
        composite: float,
        sentiment_regime: SentimentRegime,
        market_regime: str,
        md: dict,
    ) -> tuple[ContrariaBias, float]:
        """
        Check whether sentiment is extreme enough to trigger a contrarian
        fade signal.  Returns the contrarian bias direction and its strength.
        """
        if composite > self._contrarian_upper:
            base_strength = (composite - self._contrarian_upper) / (
                1.0 - self._contrarian_upper
            )
            # Amplify if bullish sentiment in bear market (bull trap)
            if market_regime == "bear":
                base_strength = min(1.0, base_strength * 1.5)
            return ContrariaBias.FADE_LONG, min(1.0, base_strength)

        if composite < self._contrarian_lower:
            base_strength = (self._contrarian_lower - composite) / (
                1.0 - abs(self._contrarian_lower)
            )
            # Amplify if bearish sentiment in bull market (capitulation buy)
            if market_regime == "bull":
                base_strength = min(1.0, base_strength * 1.5)
            return ContrariaBias.FADE_SHORT, min(1.0, base_strength)

        return ContrariaBias.NONE, 0.0

    # ------------------------------------------------------------------
    # Regime-conditional adjustment
    # ------------------------------------------------------------------

    def _regime_adjust(
        self,
        composite: float,
        market_regime: str,
        contrarian_flag: ContrariaBias,
        contrarian_strength: float,
    ) -> float:
        """
        Adjust composite sentiment based on market regime.

        - Bull market + bearish sentiment -> slight bullish offset (mean reversion)
        - Bear market + bullish sentiment -> slight bearish offset (trap detection)
        - Contrarian signal further adjusts the composite
        """
        adjusted = composite

        if market_regime == "bull" and composite < 0:
            # Bearish sentiment in bull market is less meaningful
            adjusted *= 0.7
        elif market_regime == "bear" and composite > 0:
            # Bullish sentiment in bear market is suspicious
            adjusted *= 0.6

        # Apply contrarian override at extremes
        if contrarian_flag == ContrariaBias.FADE_LONG:
            adjusted -= contrarian_strength * 0.4
        elif contrarian_flag == ContrariaBias.FADE_SHORT:
            adjusted += contrarian_strength * 0.4

        return max(-1.0, min(1.0, adjusted))

    # ------------------------------------------------------------------
    # Direction & conviction derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_direction(
        adjusted_composite: float,
        contrarian_flag: ContrariaBias,
    ) -> str:
        if contrarian_flag == ContrariaBias.FADE_LONG:
            return "short"
        if contrarian_flag == ContrariaBias.FADE_SHORT:
            return "long"
        if adjusted_composite > 0.15:
            return "long"
        if adjusted_composite < -0.15:
            return "short"
        return "neutral"

    @staticmethod
    def _derive_conviction(
        adjusted_composite: float,
        contrarian_strength: float,
        factor_scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """
        Conviction is higher when:
        - |adjusted_composite| is large
        - Factors are aligned (low dispersion)
        - Contrarian signal is strong (high conviction fade)
        """
        base = abs(adjusted_composite)

        # Factor alignment: std of weighted factor scores
        vals = [
            factor_scores[k] * weights.get(k, 0.0)
            for k in factor_scores
            if weights.get(k, 0.0) > 0
        ]
        if len(vals) > 1:
            dispersion = float(np.std(vals))
            alignment_bonus = max(0.0, 0.15 - dispersion) / 0.15 * 0.2
        else:
            alignment_bonus = 0.0

        # Contrarian boost
        contrarian_bonus = contrarian_strength * 0.15

        conviction = min(1.0, base + alignment_bonus + contrarian_bonus)
        return conviction

    # ------------------------------------------------------------------
    # Cross-asset sentiment divergence
    # ------------------------------------------------------------------

    def _cross_asset_divergence(
        self, md: dict,
    ) -> tuple[CrossAssetDivergence, str]:
        """
        Compare equity vs credit vs vol sentiment.
        Severe divergence (e.g., equity bullish but credit bearish) is
        a warning that the market is sending mixed signals.
        """
        eq = md.get("equity_sentiment")
        cr = md.get("credit_sentiment")
        vol = md.get("vol_sentiment")

        available = {}
        if eq is not None:
            available["equity"] = float(eq)
        if cr is not None:
            available["credit"] = float(cr)
        if vol is not None:
            available["vol"] = float(vol)

        if len(available) < 2:
            return CrossAssetDivergence.ALIGNED, "Insufficient cross-asset data"

        values = list(available.values())
        max_spread = max(values) - min(values)

        if max_spread > 1.0:
            detail_parts = [f"{k}={v:+.2f}" for k, v in available.items()]
            return (
                CrossAssetDivergence.SEVERE,
                f"Severe cross-asset divergence: {', '.join(detail_parts)}",
            )
        elif max_spread > 0.5:
            detail_parts = [f"{k}={v:+.2f}" for k, v in available.items()]
            return (
                CrossAssetDivergence.MILD,
                f"Mild cross-asset divergence: {', '.join(detail_parts)}",
            )
        else:
            return CrossAssetDivergence.ALIGNED, "Cross-asset sentiment aligned"

    # ------------------------------------------------------------------
    # Alignment helper
    # ------------------------------------------------------------------

    @staticmethod
    def _alignment_score(sentiment_dir: str, hypothesis_dir: str) -> float:
        """
        +1 if directions match, -1 if they oppose, 0 if either is neutral.
        """
        direction_map = {"long": 1.0, "short": -1.0, "neutral": 0.0}
        s = direction_map.get(sentiment_dir, 0.0)
        h = direction_map.get(hypothesis_dir, 0.0)
        if s == 0.0 or h == 0.0:
            return 0.0
        return s * h  # +1 if same sign, -1 if opposite
