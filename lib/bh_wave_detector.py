"""
lib/bh_wave_detector.py
========================
LARSA v18 -- BH wave detector and simplified Elliott Wave adapter.

Detects recurring "BH wave" events -- sequences of Black-Hole mass
accumulation and release that exhibit impulse / corrective / consolidation
character -- and maps them onto a simplified Elliott Wave count to guide
entry and exit timing.

Classes:
  WaveType          -- Enum of wave character
  BHWave            -- Dataclass for a single detected wave
  BHWaveDetector    -- Stateful detector that builds BHWave objects
  ElliottWaveAdapter -- Maps BHWave history onto Elliott Wave counts

Detection logic:
  IMPULSE       : BH mass rises from < 0.5 to > 1.92 in < 20 bars with a
                  strong directional price move (|return| > IMPULSE_RETURN_MIN).
  CORRECTIVE    : BH mass rises after a recent IMPULSE wave but peak mass
                  stays below CORRECTIVE_PEAK_MAX (< 1.5 by default) and
                  price move is smaller than the preceding impulse.
  CONSOLIDATION : BH mass oscillates within [CONSOL_MASS_LO, CONSOL_MASS_HI]
                  for > CONSOL_MIN_BARS bars without triggering a new IMPULSE.

Elliott Wave mapping (simplified 5-3 structure):
  Wave 1: first IMPULSE
  Wave 2: first CORRECTIVE after wave 1
  Wave 3: second IMPULSE (typically the strongest)
  Wave 4: second CORRECTIVE
  Wave 5: third IMPULSE (exhaustion)
  Wave A: first CORRECTIVE after wave 5 (bear correction)
  Wave B: temporary rally (IMPULSE-like but weaker)
  Wave C: final CORRECTIVE (completes full 8-wave cycle)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional


# ---------------------------------------------------------------------------
# WaveType enum
# ---------------------------------------------------------------------------

class WaveType(str, Enum):
    """Character classification of a BH wave."""
    IMPULSE       = "IMPULSE"
    CORRECTIVE    = "CORRECTIVE"
    CONSOLIDATION = "CONSOLIDATION"
    UNKNOWN       = "UNKNOWN"


# ---------------------------------------------------------------------------
# BHWave dataclass
# ---------------------------------------------------------------------------

@dataclass
class BHWave:
    """
    Represents a single BH wave event.

    Attributes
    ----------
    start_bar:
        Bar index at which the wave began (mass first started rising).
    peak_mass_bar:
        Bar index at which BH mass reached its peak.
    peak_mass_value:
        Maximum BH mass value observed during the wave.
    duration_bars:
        Total bar count from start_bar to the bar after peak (or resolution).
    price_return:
        Cumulative price return (close[-1] / close[start_bar] - 1) over the
        wave's duration. Positive = bullish wave, negative = bearish.
    resolved:
        True once the wave has ended (mass has subsequently decayed or a new
        wave has started). False while still in progress.
    wave_type:
        IMPULSE, CORRECTIVE, or CONSOLIDATION.
    direction:
        +1 for bullish, -1 for bearish, 0 for neutral/consolidation.
    """
    start_bar:       int
    peak_mass_bar:   int
    peak_mass_value: float
    duration_bars:   int
    price_return:    float
    resolved:        bool
    wave_type:       WaveType
    direction:       int = 0

    def __post_init__(self) -> None:
        if self.price_return > 0:
            self.direction = 1
        elif self.price_return < 0:
            self.direction = -1
        else:
            self.direction = 0


# ---------------------------------------------------------------------------
# Internal bar snapshot
# ---------------------------------------------------------------------------

@dataclass
class _BarSnap:
    """Minimal per-bar record used by BHWaveDetector."""
    bar_idx: int
    bh_mass: float
    price:   float


# ---------------------------------------------------------------------------
# BHWaveDetector
# ---------------------------------------------------------------------------

class BHWaveDetector:
    """
    Stateful detector that classifies incoming BH mass + price data into
    BH waves and maintains a wave history.

    Parameters
    ----------
    symbol:
        Instrument symbol (for identification).
    history_maxlen:
        Maximum number of completed waves to retain. Default 50.

    Detection thresholds (class-level, tunable at instantiation):
      IMPULSE_MASS_START  : mass must be below this to start an impulse
      IMPULSE_MASS_PEAK   : mass must exceed this to qualify as impulse
      IMPULSE_MAX_BARS    : impulse must complete within this many bars
      IMPULSE_RETURN_MIN  : minimum |price_return| for impulse classification
      CORRECTIVE_PEAK_MAX : peak mass must be below this for corrective
      CONSOL_MASS_LO      : lower bound for consolidation mass range
      CONSOL_MASS_HI      : upper bound for consolidation mass range
      CONSOL_MIN_BARS     : minimum bars in range to declare consolidation
    """

    # Impulse thresholds
    IMPULSE_MASS_START: float = 0.50   # mass must start below this
    IMPULSE_MASS_PEAK:  float = 1.92   # mass must exceed this
    IMPULSE_MAX_BARS:   int   = 20     # within this many bars
    IMPULSE_RETURN_MIN: float = 0.002  # |return| >= 0.2% over the wave

    # Corrective thresholds
    CORRECTIVE_PEAK_MAX: float = 1.50  # mass peak must be below this

    # Consolidation thresholds
    CONSOL_MASS_LO:  float = 0.80
    CONSOL_MASS_HI:  float = 1.60
    CONSOL_MIN_BARS: int   = 30

    # Mass decay threshold -- wave considered resolved when mass drops here
    RESOLVE_MASS_THRESHOLD: float = 0.40

    def __init__(self, symbol: str, history_maxlen: int = 50) -> None:
        self.symbol = symbol
        self._bar_idx: int = 0

        # Ring buffer of all bar snapshots (kept for wave calculation)
        self._bars: Deque[_BarSnap] = deque(maxlen=200)

        # Completed waves
        self._wave_history: Deque[BHWave] = deque(maxlen=history_maxlen)

        # Current in-progress wave (None when idle)
        self._active_wave_start_idx: Optional[int] = None
        self._active_wave_peak_mass: float = 0.0
        self._active_wave_peak_bar:  int   = 0
        self._active_wave_start_price: float = 0.0
        self._active_wave_bars_in_rise: int  = 0

        # Consolidation tracking
        self._consol_bars_in_range: int  = 0
        self._consol_start_bar:     int  = 0
        self._consol_start_price:   float = 0.0
        self._in_consolidation:     bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bh_mass: float, price: float) -> None:
        """
        Process one new bar.

        Parameters
        ----------
        bh_mass:
            Black-Hole mass value for this bar (raw, not normalised).
        price:
            Bar closing price.
        """
        snap = _BarSnap(bar_idx=self._bar_idx, bh_mass=bh_mass, price=price)
        self._bars.append(snap)

        self._check_impulse(snap)
        self._check_consolidation(snap)

        self._bar_idx += 1

    def get_current_wave(self) -> Optional[BHWave]:
        """
        Return the current in-progress wave (unresolved), or None.

        The returned BHWave has resolved=False and duration_bars is
        approximate (bars elapsed since wave start).
        """
        if self._active_wave_start_idx is None:
            return None

        start_price = self._active_wave_start_price
        curr_price  = self._bars[-1].price if self._bars else start_price
        price_ret   = (curr_price / start_price - 1.0) if start_price > 0 else 0.0
        duration    = self._bar_idx - self._active_wave_start_idx

        wave_type = self._classify_active_wave(price_ret)

        return BHWave(
            start_bar       = self._active_wave_start_idx,
            peak_mass_bar   = self._active_wave_peak_bar,
            peak_mass_value = self._active_wave_peak_mass,
            duration_bars   = duration,
            price_return    = price_ret,
            resolved        = False,
            wave_type       = wave_type,
        )

    def get_wave_history(self, n: int = 10) -> List[BHWave]:
        """
        Return up to n most recent completed waves, newest first.

        Parameters
        ----------
        n:
            Number of waves to return. Defaults to 10.
        """
        history = list(self._wave_history)
        return list(reversed(history))[:n]

    def predict_next_wave_type(self) -> str:
        """
        Predict the next wave type based on Elliott Wave sequencing.

        The pattern follows: IMPULSE -> CORRECTIVE -> IMPULSE -> ...
        with CONSOLIDATION interspersed during indecisive periods.

        Returns the string name of the predicted wave type.
        """
        recent = self.get_wave_history(n=3)
        if not recent:
            return WaveType.IMPULSE.value

        last = recent[0]  # most recent completed wave

        if last.wave_type == WaveType.IMPULSE:
            return WaveType.CORRECTIVE.value
        elif last.wave_type == WaveType.CORRECTIVE:
            return WaveType.IMPULSE.value
        elif last.wave_type == WaveType.CONSOLIDATION:
            # After consolidation, typically an impulse breaks out
            return WaveType.IMPULSE.value
        return WaveType.IMPULSE.value

    def is_favorable_entry(self) -> bool:
        """
        Return True when current wave type historically yields high IC.

        Favorable entry conditions:
          1. Currently in an IMPULSE wave (mass rising strongly) -- ride it
          2. A CORRECTIVE wave just completed and the next predicted wave
             is IMPULSE -- position before the next leg
          3. No active consolidation suppressing signal quality
        """
        current = self.get_current_wave()
        history = self.get_wave_history(n=5)

        # If mass is currently rising in an IMPULSE -- favorable
        if current is not None and current.wave_type == WaveType.IMPULSE:
            return True

        # If last resolved wave was CORRECTIVE, next is likely IMPULSE -- favorable
        if history and history[0].wave_type == WaveType.CORRECTIVE:
            # Only favorable if the corrective return was modest (not a crash)
            if abs(history[0].price_return) < 0.08:
                return True

        # In consolidation with no imminent impulse signal -- not favorable
        if self._in_consolidation:
            return False

        # Default -- check if we're in a wave-3 equivalent position
        elliott_pos = self._estimate_elliott_position(history)
        return elliott_pos in (3, 4)  # wave 3 setup is the best entry

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_impulse(self, snap: _BarSnap) -> None:
        """
        Check for impulse wave start, peak update, and resolution.

        State machine:
          IDLE   -> watch for mass dropping below IMPULSE_MASS_START; once seen,
                    record the low and wait for a meaningful rise.
          RISING -> track peak mass; resolve when mass decays back below
                    RESOLVE_MASS_THRESHOLD (after having risen above it).
                    If mass never surpassed a minimum meaningful level
                    (WAVE_MIN_PEAK) before decaying, discard silently.
        """
        bm = snap.bh_mass

        if self._active_wave_start_idx is not None:
            # --- RISING state ---
            # Update peak
            if bm > self._active_wave_peak_mass:
                self._active_wave_peak_mass = bm
                self._active_wave_peak_bar  = snap.bar_idx

            bars_elapsed = snap.bar_idx - self._active_wave_start_idx

            # Only resolve when:
            #   (a) mass has risen above IMPULSE_MASS_START at some point
            #       (i.e., the wave left the baseline), AND
            #   (b) mass has now decayed back below RESOLVE_MASS_THRESHOLD
            wave_left_baseline = self._active_wave_peak_mass >= self.IMPULSE_MASS_START
            decayed = bm <= self.RESOLVE_MASS_THRESHOLD

            if wave_left_baseline and decayed:
                self._resolve_active_wave(snap)
                return

            # Hard timeout: resolve regardless (prevents stuck-wave state)
            if bars_elapsed > 80:
                if wave_left_baseline:
                    self._resolve_active_wave(snap)
                else:
                    # Mass never rose meaningfully -- discard candidate silently
                    self._active_wave_start_idx    = None
                    self._active_wave_peak_mass    = 0.0
                    self._active_wave_peak_bar     = 0
                    self._active_wave_start_price  = 0.0
                    self._active_wave_bars_in_rise = 0
                return

            self._active_wave_bars_in_rise = bars_elapsed

        else:
            # --- IDLE state ---
            # Start a candidate wave when mass is at the baseline (< IMPULSE_MASS_START)
            # and has not yet been tracked.  We record the low entry point so that
            # the wave duration and price return are measured from the correct start.
            if bm < self.IMPULSE_MASS_START:
                self._active_wave_start_idx    = snap.bar_idx
                self._active_wave_peak_mass    = bm
                self._active_wave_peak_bar     = snap.bar_idx
                self._active_wave_start_price  = snap.price
                self._active_wave_bars_in_rise = 0

    def _check_consolidation(self, snap: _BarSnap) -> None:
        """Check for consolidation range conditions."""
        bm = snap.bh_mass
        in_range = self.CONSOL_MASS_LO <= bm <= self.CONSOL_MASS_HI

        if in_range:
            if not self._in_consolidation:
                self._consol_bars_in_range += 1
                if self._consol_bars_in_range == 1:
                    self._consol_start_bar   = snap.bar_idx
                    self._consol_start_price = snap.price
                if self._consol_bars_in_range >= self.CONSOL_MIN_BARS:
                    self._in_consolidation = True
                    # Emit a CONSOLIDATION wave if not already active
                    if self._active_wave_start_idx is None:
                        duration = snap.bar_idx - self._consol_start_bar
                        sp = self._consol_start_price
                        ret = (snap.price / sp - 1.0) if sp > 0 else 0.0
                        wave = BHWave(
                            start_bar       = self._consol_start_bar,
                            peak_mass_bar   = snap.bar_idx,
                            peak_mass_value = bm,
                            duration_bars   = duration,
                            price_return    = ret,
                            resolved        = True,
                            wave_type       = WaveType.CONSOLIDATION,
                        )
                        self._wave_history.append(wave)
        else:
            self._consol_bars_in_range = 0
            self._in_consolidation     = False

    def _resolve_active_wave(self, snap: _BarSnap) -> None:
        """Finalize the current active wave and append to history."""
        if self._active_wave_start_idx is None:
            return

        sp = self._active_wave_start_price
        price_ret = (snap.price / sp - 1.0) if sp > 0 else 0.0
        duration  = snap.bar_idx - self._active_wave_start_idx

        wave_type = self._classify_wave_by_params(
            peak_mass   = self._active_wave_peak_mass,
            price_ret   = price_ret,
            duration    = duration,
        )

        wave = BHWave(
            start_bar       = self._active_wave_start_idx,
            peak_mass_bar   = self._active_wave_peak_bar,
            peak_mass_value = self._active_wave_peak_mass,
            duration_bars   = duration,
            price_return    = price_ret,
            resolved        = True,
            wave_type       = wave_type,
        )
        self._wave_history.append(wave)

        # Reset active wave state
        self._active_wave_start_idx    = None
        self._active_wave_peak_mass    = 0.0
        self._active_wave_peak_bar     = 0
        self._active_wave_start_price  = 0.0
        self._active_wave_bars_in_rise = 0

    def _classify_wave_by_params(
        self,
        peak_mass: float,
        price_ret: float,
        duration:  int,
    ) -> WaveType:
        """
        Determine WaveType from peak mass, price return, and duration.

        Logic:
          IMPULSE: mass exceeded BH_FORM threshold, decent price move,
                   completed quickly
          CORRECTIVE: mass rose but stayed below CORRECTIVE_PEAK_MAX
          CONSOLIDATION: slow / low-magnitude, treated as consolidation
        """
        is_impulse_mass   = peak_mass >= self.IMPULSE_MASS_PEAK
        is_impulse_return = abs(price_ret) >= self.IMPULSE_RETURN_MIN
        # Duration is the full wave lifetime (rise + decay). The IMPULSE_MAX_BARS
        # constraint applies only to the rise phase, tracked separately. Here we
        # use a relaxed total-duration cap (3x the rise cap) to avoid mis-classifying
        # genuine impulses that have a slow decay tail.
        is_fast = duration <= self.IMPULSE_MAX_BARS * 3

        # Strong impulse: mass exceeded BH_FORM threshold AND price moved AND
        # the total wave was not excessively drawn out.
        if is_impulse_mass and is_impulse_return:
            return WaveType.IMPULSE

        # Mass exceeded threshold but price move was small -- still impulse
        # (can occur in low-volatility assets or very short moves).
        if is_impulse_mass and is_fast:
            return WaveType.IMPULSE

        if peak_mass < self.CORRECTIVE_PEAK_MAX and abs(price_ret) < self.IMPULSE_RETURN_MIN * 3:
            return WaveType.CORRECTIVE

        if duration > self.CONSOL_MIN_BARS and not is_impulse_mass:
            return WaveType.CONSOLIDATION

        # Fallback -- treat as corrective if mass was meaningful but not impulse
        if peak_mass > 0.8:
            return WaveType.CORRECTIVE
        return WaveType.UNKNOWN

    def _classify_active_wave(self, price_ret: float) -> WaveType:
        """Best-guess classification of the currently active (unresolved) wave."""
        return self._classify_wave_by_params(
            peak_mass = self._active_wave_peak_mass,
            price_ret = price_ret,
            duration  = self._active_wave_bars_in_rise,
        )

    @staticmethod
    def _estimate_elliott_position(history: List[BHWave]) -> int:
        """
        Estimate current position in the 1-8 Elliott Wave sequence
        from recent wave history.

        Returns an integer 1-8. Returns 0 if insufficient history.
        """
        if not history:
            return 0
        # Count alternating IMPULSE / CORRECTIVE waves from oldest to newest
        # (history[0] = most recent, so reverse for counting)
        seq = list(reversed(history))
        pos = 0
        for wave in seq:
            if wave.wave_type == WaveType.IMPULSE:
                pos += 1
            elif wave.wave_type == WaveType.CORRECTIVE:
                pos += 1
            # consolidation does not advance the count
        return min(pos, 8)

    def __repr__(self) -> str:
        return (
            f"BHWaveDetector(symbol={self.symbol!r}, "
            f"bar={self._bar_idx}, "
            f"waves_completed={len(self._wave_history)}, "
            f"active={'yes' if self._active_wave_start_idx is not None else 'no'})"
        )


# ---------------------------------------------------------------------------
# ElliottWaveAdapter
# ---------------------------------------------------------------------------

class ElliottWaveAdapter:
    """
    Maps LARSA BH wave history onto a simplified Elliott Wave count.

    Elliott Wave structure:
      Impulse phase (bull): waves 1, 2, 3, 4, 5
      Corrective phase:     waves A (6), B (7), C (8)
      Total 8-wave cycle repeats.

    Wave count significance:
      Wave 1 : Initial impulse -- enter cautiously (+0.3)
      Wave 2 : First pullback -- potential add or hold (0.0)
      Wave 3 : Strongest impulse -- STRONG entry signal (+1.0)
      Wave 4 : Second pullback -- hold / no new entry (-0.3)
      Wave 5 : Exhaustion impulse -- start reducing (-0.7 at peak)
      Wave A : First corrective leg -- exit long (-1.0)
      Wave B : Bear rally -- potential short (0.0 to -0.5)
      Wave C : Final corrective -- cover shorts, look for new cycle (+0.2)

    The adapter tracks an internal count [1..8] and cycles back to 1 after
    completing wave C (8).
    """

    # Entry signal by wave number (1-indexed)
    _WAVE_SIGNALS: dict[int, float] = {
        1: 0.3,
        2: 0.0,
        3: 1.0,
        4: -0.3,
        5: -0.7,
        6: -1.0,   # Wave A
        7: 0.0,    # Wave B
        8: 0.2,    # Wave C
    }

    # Wave names for display
    _WAVE_NAMES: dict[int, str] = {
        1: "Wave-1 (Impulse)", 2: "Wave-2 (Corrective)",
        3: "Wave-3 (Impulse)", 4: "Wave-4 (Corrective)",
        5: "Wave-5 (Impulse)", 6: "Wave-A (Corrective)",
        7: "Wave-B (Rally)",   8: "Wave-C (Corrective)",
    }

    def __init__(self, detector: BHWaveDetector) -> None:
        """
        Parameters
        ----------
        detector:
            A BHWaveDetector instance whose wave_history drives the count.
        """
        self._detector = detector
        self._wave_count: int = 0
        self._last_history_len: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wave_count(self) -> int:
        """
        Return current position in the 1-8 Elliott Wave sequence.

        Updates the count from fresh wave history on each call.
        Returns 0 if no waves have been detected yet.
        """
        self._sync_count()
        return self._wave_count

    def entry_signal(self) -> float:
        """
        Return an entry signal in [-1.0, 1.0] based on wave count.

          Wave 3 (count == 3) : +1.0 (strongest entry -- ride the power wave)
          Wave 5 peak         : -1.0 (exit -- exhaustion)
          Other counts        : see _WAVE_SIGNALS table

        The current wave (unresolved) also modulates the signal:
          If the detector has an active IMPULSE wave, signal is boosted +0.2
          (capped at +1.0).
        """
        self._sync_count()
        count = self._wave_count
        if count == 0:
            return 0.0

        base_signal = self._WAVE_SIGNALS.get(count, 0.0)

        # Boost if currently in an active impulse
        current = self._detector.get_current_wave()
        if current is not None and current.wave_type == WaveType.IMPULSE:
            base_signal = min(1.0, base_signal + 0.2)

        return _clamp(base_signal)

    def wave_name(self) -> str:
        """Human-readable name of the current wave count."""
        self._sync_count()
        return self._WAVE_NAMES.get(self._wave_count, "Unknown")

    def is_in_power_wave(self) -> bool:
        """Return True when in wave 3 (the 'power wave' with highest IC)."""
        return self.get_wave_count() == 3

    def is_at_exhaustion(self) -> bool:
        """Return True when at or past wave 5 (expect corrective to follow)."""
        return self.get_wave_count() in (5, 6, 7)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_count(self) -> None:
        """
        Re-derive the wave count from the detector's history if it has grown.
        """
        history = self._detector.get_wave_history(n=50)
        hist_len = len(history)
        if hist_len == self._last_history_len:
            return

        # Count alternating impulse/corrective waves (oldest first)
        seq = list(reversed(history))  # oldest -> newest
        count = 0
        for wave in seq:
            if wave.wave_type in (WaveType.IMPULSE, WaveType.CORRECTIVE):
                count += 1
            # CONSOLIDATION doesn't advance the Elliott count

        # Map to 1-8 cycle (modulo 8, but start at 1)
        if count == 0:
            self._wave_count = 0
        else:
            self._wave_count = ((count - 1) % 8) + 1

        self._last_history_len = hist_len

    def __repr__(self) -> str:
        return (
            f"ElliottWaveAdapter("
            f"wave={self.wave_name()}, "
            f"signal={self.entry_signal():.2f})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_wave_system(symbol: str, history_maxlen: int = 50) -> tuple[BHWaveDetector, ElliottWaveAdapter]:
    """
    Build a linked BHWaveDetector + ElliottWaveAdapter pair.

    Returns
    -------
    (detector, adapter) -- update detector.update() on each bar; read
    adapter.entry_signal() for trade signals.
    """
    detector = BHWaveDetector(symbol=symbol, history_maxlen=history_maxlen)
    adapter  = ElliottWaveAdapter(detector=detector)
    return detector, adapter


# ---------------------------------------------------------------------------
# Internal helpers (shared with other modules)
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Stand-alone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _random
    _random.seed(42)

    detector, adapter = build_wave_system("BTC_TEST")

    # Simulate a synthetic BH mass sequence with an impulse
    prices = [50000.0]
    for i in range(120):
        prices.append(prices[-1] * (1 + _random.gauss(0.0002, 0.003)))

    # Impulse sequence: mass rises from 0.2 to 2.1 over 15 bars
    mass_seq = [0.2] * 30
    for i in range(15):
        mass_seq.append(0.2 + (2.1 - 0.2) * i / 14)
    mass_seq += [2.1] + [2.1 * 0.95**i for i in range(1, 30)]
    # pad / trim
    while len(mass_seq) < len(prices):
        mass_seq.append(0.3)
    mass_seq = mass_seq[:len(prices)]

    for bm, px in zip(mass_seq, prices):
        detector.update(bh_mass=bm, price=px)

    print("Completed waves:", len(detector.get_wave_history()))
    for w in detector.get_wave_history(n=5):
        print(" ", w)
    print("Current wave:", detector.get_current_wave())
    print("Predicted next:", detector.predict_next_wave_type())
    print("Favorable entry:", detector.is_favorable_entry())
    print("Elliott count:", adapter.get_wave_count(), adapter.wave_name())
    print("Elliott signal:", adapter.entry_signal())
