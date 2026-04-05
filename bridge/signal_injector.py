"""
bridge/signal_injector.py

SignalInjector: inject external IAE signals into the live trader.

Translates macro, on-chain, and sentiment signals into concrete multipliers
and trading overrides. Writes signal_overrides.json; the live trader polls
this file every bar to adjust its behaviour.

Override format:
  {
    "version": N,
    "generated_at": ISO8601,
    "expires_at": ISO8601,
    "multipliers": {"BTC": 1.2, "ETH": 1.1},
    "blocked_hours": [1, 13],
    "sizing_override": 0.8,
    "notes": ["CRISIS macro regime — sizing reduced 20%"]
  }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[1]
_OVERRIDES_FILE = _REPO_ROOT / "config" / "signal_overrides.json"
_DEFAULT_EXPIRY_HOURS = 2   # overrides expire after 2 hours unless refreshed

# Sentiment thresholds
_EXTREME_FEAR_THRESHOLD = 20     # Fear & Greed score
_GREED_THRESHOLD = 75
_SOPR_CAPITULATION = 0.97        # SOPR below this suggests capitulation

# Multiplier bounds
_MIN_MULTIPLIER = 0.3
_MAX_MULTIPLIER = 1.5


class SignalInjector:
    """
    Translate IAE signal layers into live trader overrides.

    Processes:
      - Macro regime: CRISIS → reduce sizing
      - On-chain SOPR: capitulation → allow larger entries
      - Fear & Greed: extreme fear → contrarian boost
      - Derivatives: extreme OI → caution
    """

    def __init__(self, overrides_file: Path | str | None = None) -> None:
        self.overrides_file = Path(overrides_file) if overrides_file else _OVERRIDES_FILE
        self.overrides_file.parent.mkdir(parents=True, exist_ok=True)
        self._version = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, system_signal) -> dict[str, Any]:
        """
        Compute and write overrides from a SystemSignal.
        Returns the override dict that was written.
        """
        overrides = self._compute_overrides(system_signal)
        self._write(overrides)
        logger.info(
            "SignalInjector: wrote overrides v%d — sizing=%.2f, blocked_hours=%s, notes=%s",
            overrides["version"],
            overrides.get("sizing_override", 1.0),
            overrides.get("blocked_hours", []),
            overrides.get("notes", []),
        )
        return overrides

    def clear(self) -> None:
        """Write a neutral (no-override) file."""
        neutral = self._neutral_override()
        self._write(neutral)
        logger.info("SignalInjector: cleared all overrides.")

    def read_current(self) -> dict[str, Any]:
        if not self.overrides_file.exists():
            return self._neutral_override()
        try:
            data = json.loads(self.overrides_file.read_text())
            # Check expiry
            expires = data.get("expires_at")
            if expires:
                exp_dt = datetime.fromisoformat(expires)
                if datetime.now(timezone.utc) > exp_dt:
                    logger.info("SignalInjector: overrides expired — returning neutral.")
                    return self._neutral_override()
            return data
        except Exception as exc:
            logger.warning("SignalInjector: could not read overrides: %s", exc)
            return self._neutral_override()

    # ------------------------------------------------------------------
    # Override computation
    # ------------------------------------------------------------------

    def _compute_overrides(self, signal) -> dict[str, Any]:
        """
        Build override dict from a SystemSignal (or compatible object with attributes).
        """
        multipliers: dict[str, float] = {}
        blocked_hours: list[int] = []
        sizing_override = 1.0
        notes: list[str] = []

        macro_regime = getattr(signal, "macro_regime", "RISK_NEUTRAL")
        onchain_score = float(getattr(signal, "onchain_score", 0.0))
        sentiment_score = float(getattr(signal, "sentiment_score", 0.0))
        derivatives_signal = getattr(signal, "derivatives_signal", "NEUTRAL")
        liquidation_risk = float(getattr(signal, "liquidation_risk", 0.0))

        # 1. Macro regime override
        if macro_regime == "CRISIS":
            sizing_override *= 0.5
            blocked_hours.extend([1, 2, 3, 13, 14])   # avoid low-liquidity hours in crisis
            notes.append("CRISIS macro regime — sizing halved, overnight hours blocked")
        elif macro_regime == "RISK_OFF":
            sizing_override *= 0.75
            notes.append("RISK_OFF macro regime — sizing reduced 25%")
        elif macro_regime == "RISK_ON":
            sizing_override = min(sizing_override * 1.1, _MAX_MULTIPLIER)
            notes.append("RISK_ON macro regime — sizing +10%")

        # 2. On-chain SOPR capitulation
        # onchain_score < -0.4 is a proxy for SOPR capitulation
        if onchain_score < -0.4:
            for sym in ["BTC", "ETH"]:
                multipliers[sym] = multipliers.get(sym, 1.0) * 1.15
            notes.append(f"On-chain bearish (score={onchain_score:.2f}) — contrarian entry boost +15%")

        # 3. Fear & Greed extreme fear → contrarian boost
        # sentiment_score is normalised: [-1, 1], extreme fear ≈ < -0.6 (raw < 20)
        if sentiment_score < -0.6:
            for sym in ["BTC", "ETH", "SOL"]:
                multipliers[sym] = multipliers.get(sym, 1.0) * 1.20
            notes.append(
                f"Extreme fear (score={sentiment_score:.2f}) — contrarian sizing +20%"
            )
        elif sentiment_score > 0.5:
            # Extreme greed → reduce size to protect against fomo reversals
            sizing_override *= 0.85
            notes.append(f"Extreme greed (score={sentiment_score:.2f}) — sizing -15%")

        # 4. Derivatives signal
        if derivatives_signal == "BEARISH":
            sizing_override *= 0.80
            notes.append("Derivatives bearish — sizing -20%")
        elif derivatives_signal == "BULLISH":
            sizing_override = min(sizing_override * 1.05, _MAX_MULTIPLIER)
            notes.append("Derivatives bullish — sizing +5%")

        # 5. High liquidation risk
        if liquidation_risk > 0.7:
            sizing_override *= 0.60
            blocked_hours.extend([0, 1, 2, 23])  # avoid lowest-liquidity hours
            notes.append(f"High liquidation risk ({liquidation_risk:.2f}) — sizing -40%, night hours blocked")

        # Clamp multipliers
        for sym in list(multipliers.keys()):
            multipliers[sym] = max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, multipliers[sym]))
        sizing_override = max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, sizing_override))

        # Deduplicate blocked hours
        blocked_hours = sorted(set(blocked_hours))

        return {
            "version": self._version + 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=_DEFAULT_EXPIRY_HOURS)
            ).isoformat(),
            "multipliers": multipliers,
            "blocked_hours": blocked_hours,
            "sizing_override": round(sizing_override, 4),
            "macro_regime": macro_regime,
            "composite_score": float(getattr(signal, "composite_score", 0.0)),
            "notes": notes,
        }

    def _neutral_override(self) -> dict[str, Any]:
        return {
            "version": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=_DEFAULT_EXPIRY_HOURS)
            ).isoformat(),
            "multipliers": {},
            "blocked_hours": [],
            "sizing_override": 1.0,
            "macro_regime": "RISK_NEUTRAL",
            "composite_score": 0.0,
            "notes": ["neutral — no overrides active"],
        }

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.overrides_file.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.overrides_file)
            self._version = data.get("version", self._version)
        except Exception as exc:
            logger.error("SignalInjector: write failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
