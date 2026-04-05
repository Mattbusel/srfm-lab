"""
inference/signal_bridge.py
===========================
Convert ML ensemble predictions into IAE (Idea/Alpha Engine) hypothesis format.

Financial rationale
-------------------
The IAE (Idea-Alpha Engine) is the SRFM system that evaluates and ranks
competing trading ideas (hypotheses) before committing capital.  Each
hypothesis is a structured claim of the form:

    "Signal X is [strongly/weakly] [bullish/bearish] on instrument Y
     with confidence Z for horizon H bars."

The SignalBridge translates the raw ensemble_score (a float in [-1,+1])
into IAE-compatible hypothesis objects, enriched with:

* Model attribution: which base models agree with the direction?
* Regime context: what regime does the Transformer believe we are in?
* Strength labelling: STRONGLY/WEAKLY based on |score| thresholds.
* Hypothesis ID: deterministic hash for deduplication in the IAE queue.

Threshold convention (calibrated on historical IC data):
    |score| >= 0.6  →  STRONGLY BULLISH / BEARISH
    |score| >= 0.3  →  WEAKLY BULLISH / BEARISH
    |score| <  0.3  →  NO SIGNAL (hypothesis not generated)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRONG_THRESHOLD = 0.6
WEAK_THRESHOLD   = 0.3


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLHypothesis:
    """An IAE-compatible hypothesis generated from ML ensemble output.

    Parameters
    ----------
    hypothesis_id : str
        Deterministic hash for deduplication.
    instrument : str
        Trading pair (e.g. 'BTC-USDT').
    direction : str
        'BULLISH' or 'BEARISH'.
    strength : str
        'STRONGLY' or 'WEAKLY'.
    ensemble_score : float
        Raw ensemble score [-1, +1].
    confidence : float
        Cross-model agreement [0, 1].
    regime : str
        Current regime label from Transformer (BULL/BEAR/CHOPPY/CRISIS).
    agreeing_models : list[str]
        Base models that agree with the direction.
    horizon_bars : int
        Suggested holding horizon in bars.
    timestamp : float
        Unix timestamp of signal generation.
    meta : dict
        Arbitrary extra metadata.
    text : str
        Human-readable hypothesis string for IAE queue.
    """

    hypothesis_id:   str
    instrument:      str
    direction:       str
    strength:        str
    ensemble_score:  float
    confidence:      float
    regime:          str
    agreeing_models: List[str]
    horizon_bars:    int
    timestamp:       float
    text:            str
    meta:            Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# SignalBridge
# ---------------------------------------------------------------------------

class SignalBridge:
    """Convert ML ensemble scores to IAE hypothesis objects.

    Parameters
    ----------
    strong_threshold : float
        |score| above this → STRONGLY labelled.
    weak_threshold : float
        |score| above this (but below strong) → WEAKLY labelled.
    default_horizon : int
        Default holding horizon in bars when none is provided.
    """

    def __init__(
        self,
        strong_threshold: float = STRONG_THRESHOLD,
        weak_threshold:   float = WEAK_THRESHOLD,
        default_horizon:  int   = 1,
    ) -> None:
        self.strong_threshold = strong_threshold
        self.weak_threshold   = weak_threshold
        self.default_horizon  = default_horizon
        self._hypothesis_log: List[MLHypothesis] = []

    # ------------------------------------------------------------------
    # Main conversion method
    # ------------------------------------------------------------------

    def convert(
        self,
        instrument:     str,
        ensemble_score: float,
        confidence:     float,
        base_scores:    Optional[Dict[str, float]] = None,
        regime:         str   = "UNKNOWN",
        horizon_bars:   Optional[int] = None,
        extra_meta:     Optional[Dict] = None,
    ) -> Optional[MLHypothesis]:
        """Convert an ensemble score to an IAE hypothesis.

        Parameters
        ----------
        instrument : str
        ensemble_score : float   in [-1, +1]
        confidence : float       in [0, 1]
        base_scores : dict, optional
            {'lstm': score, 'transformer': score, 'xgboost': score, 'bh': score}
        regime : str
            Regime label from TransformerSignal.predict_regime().
        horizon_bars : int, optional
        extra_meta : dict, optional

        Returns
        -------
        MLHypothesis or None (if |score| < weak_threshold)
        """
        abs_score = abs(ensemble_score)
        if abs_score < self.weak_threshold:
            return None

        direction = "BULLISH" if ensemble_score > 0 else "BEARISH"
        strength  = "STRONGLY" if abs_score >= self.strong_threshold else "WEAKLY"

        # Model attribution
        agreeing_models = []
        if base_scores:
            for model_name, score in base_scores.items():
                if np.sign(score) == np.sign(ensemble_score) and abs(score) > 0.1:
                    agreeing_models.append(model_name)

        horizon = horizon_bars or self.default_horizon

        # Build human-readable text
        text = self._build_text(
            instrument, direction, strength, ensemble_score, confidence,
            regime, agreeing_models, horizon,
        )

        # Deterministic hypothesis ID
        h_id = self._make_id(instrument, direction, strength, round(ensemble_score, 2))

        hyp = MLHypothesis(
            hypothesis_id   = h_id,
            instrument      = instrument,
            direction       = direction,
            strength        = strength,
            ensemble_score  = float(ensemble_score),
            confidence      = float(confidence),
            regime          = regime,
            agreeing_models = agreeing_models,
            horizon_bars    = horizon,
            timestamp       = time.time(),
            text            = text,
            meta            = extra_meta or {},
        )
        self._hypothesis_log.append(hyp)
        return hyp

    def batch_convert(
        self,
        scores:       Dict[str, float],
        confidences:  Dict[str, float],
        base_scores:  Optional[Dict[str, Dict[str, float]]] = None,
        regimes:      Optional[Dict[str, str]] = None,
        horizon_bars: Optional[int] = None,
    ) -> List[MLHypothesis]:
        """Convert scores for multiple instruments at once.

        Parameters
        ----------
        scores       : {instrument: ensemble_score}
        confidences  : {instrument: confidence}
        base_scores  : {instrument: {model_name: score}}
        regimes      : {instrument: regime_label}

        Returns
        -------
        List of non-None MLHypothesis objects, sorted by |score| descending.
        """
        hypotheses = []
        for instr, score in scores.items():
            conf     = confidences.get(instr, 0.5)
            bs       = (base_scores or {}).get(instr)
            regime   = (regimes or {}).get(instr, "UNKNOWN")
            hyp      = self.convert(instr, score, conf, bs, regime, horizon_bars)
            if hyp is not None:
                hypotheses.append(hyp)

        hypotheses.sort(key=lambda h: abs(h.ensemble_score), reverse=True)
        return hypotheses

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text(
        instrument:      str,
        direction:       str,
        strength:        str,
        score:           float,
        confidence:      float,
        regime:          str,
        agreeing_models: List[str],
        horizon:         int,
    ) -> str:
        agreement_str = (
            f"Models in agreement: {', '.join(agreeing_models)}."
            if agreeing_models
            else "No individual model agreement above threshold."
        )
        return (
            f"ML ensemble is {strength} {direction} on {instrument}. "
            f"Score={score:+.3f}, confidence={confidence:.2f}. "
            f"Current regime: {regime}. "
            f"Suggested horizon: {horizon} bar(s). "
            f"{agreement_str}"
        )

    @staticmethod
    def _make_id(
        instrument: str, direction: str, strength: str, score: float
    ) -> str:
        raw = f"{instrument}:{direction}:{strength}:{score}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------

    def recent_hypotheses(self, n: int = 10) -> List[MLHypothesis]:
        """Return the n most recently generated hypotheses."""
        return list(reversed(self._hypothesis_log[-n:]))

    def filter_by_instrument(self, instrument: str) -> List[MLHypothesis]:
        return [h for h in self._hypothesis_log if h.instrument == instrument]

    def filter_by_strength(self, strength: str) -> List[MLHypothesis]:
        """strength: 'STRONGLY' or 'WEAKLY'."""
        return [h for h in self._hypothesis_log if h.strength == strength]

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        if not self._hypothesis_log:
            return pd.DataFrame()
        rows = [h.to_dict() for h in self._hypothesis_log]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df
