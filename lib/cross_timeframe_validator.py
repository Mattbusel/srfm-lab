"""
lib/cross_timeframe_validator.py
=================================
LARSA v18 -- Cross-timeframe signal confluence validator.

Validates that signals on the 15m, 1h, and 4h timeframes are aligned
before allowing entry. Misalignment -- especially on the higher timeframes
-- suppresses position sizing.

Classes:
  TimeframeSignal        -- Dataclass holding the signal snapshot for one TF
  CrossTimeframeValidator -- Computes alignment score and entry/size gates

Alignment scoring:
  All 3 TFs BH-active AND same direction  -> 1.0 (maximum confluence)
  2/3 TFs aligned                         -> 0.6
  Conflicting or only 1 TF active         -> 0.2

Weighted score:
  4h : 0.50
  1h : 0.30
  15m: 0.20

Position multiplier:
  alignment == 1.0  -> 1.0x size
  alignment == 0.6  -> 0.5x size
  linear interpolation; clamped to [0.0, 1.0]

Entry gate:
  Allow entry only when alignment >= ALIGNMENT_MIN (0.6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

TIMEFRAME_15M = "15m"
TIMEFRAME_1H  = "1h"
TIMEFRAME_4H  = "4h"

_ALL_TFS: tuple[str, ...] = (TIMEFRAME_15M, TIMEFRAME_1H, TIMEFRAME_4H)

# Weight per timeframe -- must sum to 1.0
_TF_WEIGHT: dict[str, float] = {
    TIMEFRAME_4H:  0.50,
    TIMEFRAME_1H:  0.30,
    TIMEFRAME_15M: 0.20,
}

# ---------------------------------------------------------------------------
# TimeframeSignal
# ---------------------------------------------------------------------------

@dataclass
class TimeframeSignal:
    """
    Snapshot of all relevant signal state for a single timeframe.

    Attributes
    ----------
    timeframe:
        One of '15m', '1h', '4h'.
    bh_active:
        True when the Black-Hole is currently formed (tf score >= threshold).
    bh_mass:
        Current BH mass value (raw).
    hurst_h:
        Hurst exponent for the bar sequence on this timeframe.
        H > 0.5 -> trending; H < 0.5 -> mean-reverting.
    cf_cross_direction:
        Direction of the most recent Correlation-Function cross:
          +1 -> bullish cross (cf rose above threshold)
          -1 -> bearish cross (cf fell below threshold)
           0 -> no recent cross
    nav_omega:
        QuatNav angular velocity (from quaternion navigation overlay).
        High omega -> rapid state change -> uncertainty.
    direction:
        Overall signal direction (+1 long, -1 short, 0 flat).
        Derived automatically from cf_cross_direction when not set explicitly.
    """
    timeframe:          str
    bh_active:          bool
    bh_mass:            float
    hurst_h:            float   = 0.5
    cf_cross_direction: int     = 0
    nav_omega:          float   = 0.0
    direction:          int     = 0

    def __post_init__(self) -> None:
        if self.timeframe not in _ALL_TFS:
            raise ValueError(
                f"timeframe must be one of {_ALL_TFS}, got {self.timeframe!r}"
            )
        # Auto-derive direction from cf_cross if not explicitly set
        if self.direction == 0 and self.cf_cross_direction != 0:
            self.direction = self.cf_cross_direction

    @property
    def is_trending(self) -> bool:
        """True when Hurst exponent indicates trending regime (H > 0.55)."""
        return self.hurst_h > 0.55

    @property
    def is_mean_reverting(self) -> bool:
        """True when Hurst exponent indicates mean-reverting regime (H < 0.45)."""
        return self.hurst_h < 0.45

    @property
    def nav_high_uncertainty(self) -> bool:
        """
        True when nav_omega is unusually high (> 2.0 as a rough threshold).
        High angular velocity in quaternion navigation implies the system is
        traversing state space rapidly -- signal reliability drops.
        """
        return self.nav_omega > 2.0

    def __str__(self) -> str:
        dir_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.direction, "?")
        return (
            f"TimeframeSignal({self.timeframe}, "
            f"bh={'ON' if self.bh_active else 'off'}, "
            f"mass={self.bh_mass:.3f}, "
            f"dir={dir_str}, "
            f"H={self.hurst_h:.3f})"
        )


# ---------------------------------------------------------------------------
# CrossTimeframeValidator
# ---------------------------------------------------------------------------

class CrossTimeframeValidator:
    """
    Validates signal confluence across the 15m, 1h, and 4h timeframes.

    The validator uses a weighted alignment model:

      Step 1: Determine direction agreement for each active TF.
      Step 2: Compute weighted agreement score.
      Step 3: Apply bonus if all 3 TFs agree.
      Step 4: Gate entry and scale position multiplier.

    Usage::

        validator = CrossTimeframeValidator()
        signals = {
            '15m': TimeframeSignal('15m', bh_active=True,  bh_mass=2.1, direction=1),
            '1h':  TimeframeSignal('1h',  bh_active=True,  bh_mass=1.8, direction=1),
            '4h':  TimeframeSignal('4h',  bh_active=False, bh_mass=0.4, direction=0),
        }
        if validator.entry_gate(signals):
            size_scale = validator.position_multiplier(signals)
    """

    # Alignment level thresholds
    ALIGNMENT_FULL:    float = 1.0
    ALIGNMENT_PARTIAL: float = 0.6
    ALIGNMENT_LOW:     float = 0.2

    # Minimum alignment to allow entry
    ALIGNMENT_MIN: float = 0.6

    # Position multiplier at each alignment level
    SIZE_AT_FULL_ALIGNMENT:    float = 1.0
    SIZE_AT_PARTIAL_ALIGNMENT: float = 0.5
    SIZE_MINIMUM:              float = 0.0

    def __init__(self) -> None:
        self._last_alignment: float = 0.0
        self._last_dominant_direction: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_alignment(self, signals: dict[str, TimeframeSignal]) -> float:
        """
        Compute the cross-timeframe alignment score.

        Parameters
        ----------
        signals:
            Dict mapping timeframe string ('15m', '1h', '4h') to
            TimeframeSignal. Missing timeframes are treated as inactive.

        Returns
        -------
        float in [0.0, 1.0] -- higher is better.

        Scoring detail:
          Each active TF contributes its weight if its direction matches the
          dominant direction. The dominant direction is the plurality direction
          among all active BH signals.

          Bonus: if all 3 timeframes are active AND directionally aligned,
          the score is set to ALIGNMENT_FULL (1.0) regardless of weights.

          Penalty: if a higher-weight TF actively contradicts the dominant
          direction, the score is reduced by 0.2 (clamped >= 0).
        """
        if not signals:
            self._last_alignment = self.ALIGNMENT_LOW
            return self.ALIGNMENT_LOW

        dominant = self._dominant_direction(signals)
        self._last_dominant_direction = dominant

        # Case: dominant direction is flat (no signal) -> minimal alignment
        if dominant == 0:
            self._last_alignment = self.ALIGNMENT_LOW
            return self.ALIGNMENT_LOW

        # Weighted agreement computation
        weighted_agree   = 0.0
        weighted_disagree = 0.0
        active_count = 0
        all_aligned  = True
        all_active   = True

        for tf in _ALL_TFS:
            sig = signals.get(tf)
            weight = _TF_WEIGHT[tf]
            if sig is None or not sig.bh_active:
                all_active = False
                all_aligned = False
                continue
            active_count += 1
            if sig.direction == dominant:
                weighted_agree += weight
            elif sig.direction != 0 and sig.direction != dominant:
                weighted_disagree += weight
                all_aligned = False
            else:
                # direction == 0 -- neutral/flat -- doesn't break alignment
                all_aligned = False

        # All 3 active AND all aligned -> maximum confluence
        if active_count == 3 and all_active and all_aligned:
            self._last_alignment = self.ALIGNMENT_FULL
            return self.ALIGNMENT_FULL

        # Compute raw weighted score
        raw_score = weighted_agree

        # Penalty for high-weight TF contradiction
        if weighted_disagree > 0.0:
            raw_score = max(0.0, raw_score - 0.20)

        # Map to alignment levels
        if raw_score >= 0.75:
            alignment = self.ALIGNMENT_FULL
        elif raw_score >= 0.45:
            alignment = self.ALIGNMENT_PARTIAL
        else:
            alignment = self.ALIGNMENT_LOW

        self._last_alignment = alignment
        return alignment

    def entry_gate(self, signals: dict[str, TimeframeSignal]) -> bool:
        """
        Return True only if alignment is sufficient to allow entry.

        Minimum required alignment: ALIGNMENT_MIN (0.6).

        Additional veto conditions:
          - 4h signal actively opposes the dominant direction -> veto
          - More than one TF has high nav_omega uncertainty    -> veto
        """
        alignment = self.compute_alignment(signals)
        if alignment < self.ALIGNMENT_MIN:
            return False

        # Veto: 4h opposition
        sig_4h = signals.get(TIMEFRAME_4H)
        dominant = self._last_dominant_direction
        if (
            sig_4h is not None
            and sig_4h.bh_active
            and sig_4h.direction != 0
            and sig_4h.direction != dominant
        ):
            return False

        # Veto: multiple high-uncertainty timeframes
        high_uncertainty_count = sum(
            1 for tf in _ALL_TFS
            if signals.get(tf) is not None and signals[tf].nav_high_uncertainty
        )
        if high_uncertainty_count >= 2:
            return False

        return True

    def position_multiplier(self, signals: dict[str, TimeframeSignal]) -> float:
        """
        Return a position size multiplier in [0.0, 1.0].

          alignment == 1.0  -> 1.0
          alignment == 0.6  -> 0.5
          alignment < 0.6   -> 0.0 (should not enter anyway -- gate prevents this)
          linear interpolation between 0.6 and 1.0

        Hurst exponent modifier:
          If the dominant-direction TFs are trending (H > 0.55), add +0.1
          to the multiplier (capped at 1.0).
          If mean-reverting (H < 0.45), subtract 0.1.
        """
        alignment = self.compute_alignment(signals)
        if alignment < self.ALIGNMENT_MIN:
            return self.SIZE_MINIMUM

        # Linear interpolation: alignment 0.6 -> 0.5, alignment 1.0 -> 1.0
        base = _lerp(
            alignment,
            self.ALIGNMENT_PARTIAL,  # 0.6
            self.ALIGNMENT_FULL,     # 1.0
            self.SIZE_AT_PARTIAL_ALIGNMENT,  # 0.5
            self.SIZE_AT_FULL_ALIGNMENT,     # 1.0
        )

        # Hurst modifier
        base += self._hurst_modifier(signals)

        return _clamp(base, 0.0, 1.0)

    def alignment_detail(self, signals: dict[str, TimeframeSignal]) -> dict:
        """
        Return a detailed dict for logging / diagnostics.

        Keys:
          alignment, dominant_direction, entry_allowed, position_multiplier,
          per_tf (dict of tf -> {active, direction, agrees, weight})
        """
        alignment = self.compute_alignment(signals)
        dominant  = self._last_dominant_direction

        per_tf = {}
        for tf in _ALL_TFS:
            sig = signals.get(tf)
            if sig is None:
                per_tf[tf] = {"active": False, "direction": 0, "agrees": False, "weight": _TF_WEIGHT[tf]}
            else:
                per_tf[tf] = {
                    "active":    sig.bh_active,
                    "direction": sig.direction,
                    "agrees":    sig.bh_active and sig.direction == dominant,
                    "weight":    _TF_WEIGHT[tf],
                    "hurst_h":   sig.hurst_h,
                    "nav_omega": sig.nav_omega,
                }

        return {
            "alignment":           alignment,
            "dominant_direction":  dominant,
            "entry_allowed":       self.entry_gate(signals),
            "position_multiplier": self.position_multiplier(signals),
            "per_tf":              per_tf,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_direction(signals: dict[str, TimeframeSignal]) -> int:
        """
        Determine the dominant direction among active BH signals.

        Returns +1, -1, or 0 (no clear dominant direction).
        Uses weight-adjusted vote: each active TF casts a vote equal to its
        weight * direction.
        """
        weighted_vote = 0.0
        for tf in _ALL_TFS:
            sig = signals.get(tf)
            if sig is not None and sig.bh_active:
                weighted_vote += _TF_WEIGHT[tf] * sig.direction

        if weighted_vote > 0.05:
            return 1
        elif weighted_vote < -0.05:
            return -1
        return 0

    @staticmethod
    def _hurst_modifier(signals: dict[str, TimeframeSignal]) -> float:
        """
        Compute a Hurst-based size modifier.

        Trending across multiple TFs boosts size by up to +0.1.
        Mean-reverting regime reduces size by up to -0.1.
        """
        trend_score = 0.0
        for tf in _ALL_TFS:
            sig = signals.get(tf)
            if sig is None or not sig.bh_active:
                continue
            h = sig.hurst_h
            if h > 0.55:
                trend_score += _TF_WEIGHT[tf] * 0.10
            elif h < 0.45:
                trend_score -= _TF_WEIGHT[tf] * 0.10
        return trend_score

    def __repr__(self) -> str:
        return (
            f"CrossTimeframeValidator("
            f"last_alignment={self._last_alignment:.2f}, "
            f"dominant={self._last_dominant_direction})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation, clamped."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_signal(
    timeframe: str,
    bh_active: bool,
    bh_mass: float,
    direction: int = 0,
    hurst_h: float = 0.5,
    cf_cross_direction: int = 0,
    nav_omega: float = 0.0,
) -> TimeframeSignal:
    """
    Convenience constructor for TimeframeSignal with sensible defaults.

    If direction is not provided but cf_cross_direction is, the
    TimeframeSignal.__post_init__ will derive direction automatically.
    """
    return TimeframeSignal(
        timeframe          = timeframe,
        bh_active          = bh_active,
        bh_mass            = bh_mass,
        hurst_h            = hurst_h,
        cf_cross_direction = cf_cross_direction,
        nav_omega          = nav_omega,
        direction          = direction,
    )


def full_alignment_signals(direction: int = 1) -> dict[str, TimeframeSignal]:
    """
    Return a fully-aligned signal dict (all 3 TFs active, same direction).
    Useful for testing and simulation.
    """
    return {
        TIMEFRAME_15M: make_signal(TIMEFRAME_15M, bh_active=True,  bh_mass=2.1, direction=direction, hurst_h=0.60),
        TIMEFRAME_1H:  make_signal(TIMEFRAME_1H,  bh_active=True,  bh_mass=1.9, direction=direction, hurst_h=0.58),
        TIMEFRAME_4H:  make_signal(TIMEFRAME_4H,  bh_active=True,  bh_mass=1.8, direction=direction, hurst_h=0.62),
    }


def conflicting_signals() -> dict[str, TimeframeSignal]:
    """
    Return a conflicting signal dict (4h bearish, 15m bullish, 1h inactive).
    Useful for testing.
    """
    return {
        TIMEFRAME_15M: make_signal(TIMEFRAME_15M, bh_active=True,  bh_mass=2.0, direction=1),
        TIMEFRAME_1H:  make_signal(TIMEFRAME_1H,  bh_active=False, bh_mass=0.3, direction=0),
        TIMEFRAME_4H:  make_signal(TIMEFRAME_4H,  bh_active=True,  bh_mass=1.7, direction=-1),
    }


# ---------------------------------------------------------------------------
# Stand-alone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validator = CrossTimeframeValidator()

    print("=== Full Alignment (bull) ===")
    sigs = full_alignment_signals(direction=1)
    detail = validator.alignment_detail(sigs)
    print(f"  alignment={detail['alignment']:.2f}, "
          f"entry={detail['entry_allowed']}, "
          f"size_mult={detail['position_multiplier']:.3f}")

    print("\n=== Conflicting Signals ===")
    sigs = conflicting_signals()
    detail = validator.alignment_detail(sigs)
    print(f"  alignment={detail['alignment']:.2f}, "
          f"entry={detail['entry_allowed']}, "
          f"size_mult={detail['position_multiplier']:.3f}")

    print("\n=== Partial Alignment (1h + 15m bull, 4h inactive) ===")
    sigs = {
        TIMEFRAME_15M: make_signal(TIMEFRAME_15M, bh_active=True,  bh_mass=2.1, direction=1),
        TIMEFRAME_1H:  make_signal(TIMEFRAME_1H,  bh_active=True,  bh_mass=1.8, direction=1),
        TIMEFRAME_4H:  make_signal(TIMEFRAME_4H,  bh_active=False, bh_mass=0.2, direction=0),
    }
    detail = validator.alignment_detail(sigs)
    print(f"  alignment={detail['alignment']:.2f}, "
          f"entry={detail['entry_allowed']}, "
          f"size_mult={detail['position_multiplier']:.3f}")
