"""
wave4_strategy.py # Elliott Wave 4 retracement strategy for SRFM.

Theory: In a 5-wave impulse sequence, Wave 4 offers a pullback entry
before the final Wave 5 thrust.  This module uses BH mass accumulation
as a proxy for impulse energy and Fibonacci retracements to time entries.

Key thresholds (from system config):
  BH_MASS_THRESH  = 1.92  # impulse confirmation
  HURST_WINDOW    = 100
  MIN_HOLD_BARS   = 3
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

# ---------------------------------------------------------------------------
# System-level constants (mirrors live_trader_alpaca config)
# ---------------------------------------------------------------------------

BH_MASS_THRESH       = 1.92   # impulse confirmation threshold
BH_MASS_RETRACE_LO   = 0.5    # below this after impulse => potential wave 4
IMPULSE_BARS_MIN     = 2      # minimum consecutive positive-mass bars = impulse
IMPULSE_BARS_MAX     = 20     # window to look back for impulse peak
HURST_TRENDING_MIN   = 0.55   # minimum Hurst for wave4 trades
MIN_HOLD_BARS        = 3

# Fibonacci retracement levels for Wave 4
FIB_382 = 0.382
FIB_500 = 0.500
FIB_618 = 0.618

# Fibonacci extension for Wave 5 target
FIB_EXT_618  = 1.618
FIB_EXT_1000 = 1.000


# ---------------------------------------------------------------------------
# WaveLabel enum
# ---------------------------------------------------------------------------

class WaveLabel(str, Enum):
    """Elliott Wave position label."""
    WAVE_1     = "WAVE_1"
    WAVE_2     = "WAVE_2"
    WAVE_3     = "WAVE_3"
    WAVE_4     = "WAVE_4"
    WAVE_5     = "WAVE_5"
    CORRECTION = "CORRECTION"
    UNKNOWN    = "UNKNOWN"


# ---------------------------------------------------------------------------
# Wave4Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Wave4Signal:
    """
    Signal produced when a Wave 4 retracement is detected.

    Attributes
    #--------
    wave_number:
        Should always be 4 for entries emitted by Wave4Detector.
    retracement_pct:
        Fibonacci level where price is retracing # 0.382, 0.500, or 0.618.
    entry_price:
        Recommended entry price (current close at detection time).
    stop_price:
        Hard stop below Wave 1 start (bullish) or above Wave 1 start (bearish).
    target_price:
        1.618 extension of the Wave 1-to-3 range projected from Wave 4 low.
    confidence:
        0-1 score based on Fibonacci confluence and BH mass alignment.
    bullish:
        True for long entry, False for short entry.
    impulse_range:
        (low_price, high_price) of the impulse move W1-W3.
    bar_index:
        Bar counter at signal emission.
    """
    wave_number:    int
    retracement_pct: float
    entry_price:    float
    stop_price:     float
    target_price:   float
    confidence:     float
    bullish:        bool
    impulse_range:  Tuple[float, float]
    bar_index:      int = 0


# ---------------------------------------------------------------------------
# Internal state for wave tracking
# ---------------------------------------------------------------------------

@dataclass
class _ImpulseRecord:
    """Tracks one completed impulse leg."""
    start_price: float
    end_price:   float
    start_bar:   int
    end_bar:     int
    peak_mass:   float

    @property
    def bullish(self) -> bool:
        return self.end_price > self.start_price

    @property
    def range_size(self) -> float:
        return abs(self.end_price - self.start_price)


# ---------------------------------------------------------------------------
# Wave4Detector
# ---------------------------------------------------------------------------

class Wave4Detector:
    """
    Detects potential Elliott Wave 4 retracement setups.

    Feed one bar at a time via update().  A bar is a dict with keys:
      close, open, high, low, volume, bh_mass (float)

    Detection pipeline:
      1. Track BH mass to identify impulse bars (mass > BH_MASS_THRESH).
      2. When mass drops below BH_MASS_RETRACE_LO after impulse, measure
         retracement depth against prior impulse range.
      3. If depth lands near 38.2%, 50%, or 61.8% Fib level emit Wave4Signal.
      4. classify_wave() provides coarser label for the current price series.
    """

    def __init__(
        self,
        mass_thresh:    float = BH_MASS_THRESH,
        retrace_lo:     float = BH_MASS_RETRACE_LO,
        impulse_bars:   int   = IMPULSE_BARS_MIN,
        fib_tolerance:  float = 0.04,   # within 4% of fib level = hit
    ):
        self._mass_thresh   = mass_thresh
        self._retrace_lo    = retrace_lo
        self._impulse_bars  = impulse_bars
        self._fib_tol       = fib_tolerance

        self._bar_index: int = 0

        # Rolling price + mass history (200 bars)
        self._prices: Deque[float] = deque(maxlen=200)
        self._masses: Deque[float] = deque(maxlen=200)
        self._highs:  Deque[float] = deque(maxlen=200)
        self._lows:   Deque[float] = deque(maxlen=200)

        # State machine
        self._in_impulse:        bool  = False
        self._consec_impulse:    int   = 0
        self._impulse_start_px:  float = 0.0
        self._impulse_start_bar: int   = 0
        self._impulse_peak_mass: float = 0.0

        # Completed impulse legs (keep last 3)
        self._impulse_records: List[_ImpulseRecord] = []

        # Retracement tracking
        self._in_retrace:    bool  = False
        self._retrace_start_px: float = 0.0

        # Suppress repeated signals: bar index of last emit
        self._last_signal_bar: int = -50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bar: dict) -> Optional[Wave4Signal]:
        """
        Process one bar.  Returns Wave4Signal if a Wave 4 setup is detected,
        otherwise returns None.

        bar must contain: close, high, low, bh_mass
        """
        close = float(bar.get("close", 0.0))
        high  = float(bar.get("high",  close))
        low   = float(bar.get("low",   close))
        mass  = float(bar.get("bh_mass", 0.0))

        self._prices.append(close)
        self._masses.append(mass)
        self._highs.append(high)
        self._lows.append(low)
        self._bar_index += 1

        if len(self._prices) < 10:
            return None

        signal = self._update_state_machine(close, high, low, mass)
        return signal

    def classify_wave(self, prices: List[float]) -> WaveLabel:
        """
        Coarse Elliott Wave label for the supplied price series.

        Uses pivot detection (local extrema) to identify wave structure.
        Requires at least 10 prices.
        """
        if len(prices) < 10:
            return WaveLabel.UNKNOWN

        arr = np.array(prices, dtype=float)
        pivots = self._find_pivots(arr)

        if len(pivots) < 3:
            return WaveLabel.UNKNOWN

        return self._label_from_pivots(arr, pivots)

    # ------------------------------------------------------------------
    # Internal: state machine
    # ------------------------------------------------------------------

    def _update_state_machine(
        self, close: float, high: float, low: float, mass: float
    ) -> Optional[Wave4Signal]:

        # # Phase 1: detect impulse accumulation --
        if mass >= self._mass_thresh:
            if not self._in_impulse:
                # Starting a new impulse leg
                self._in_impulse        = True
                self._consec_impulse    = 1
                self._impulse_start_px  = close
                self._impulse_start_bar = self._bar_index
                self._impulse_peak_mass = mass
                self._in_retrace        = False
            else:
                self._consec_impulse   += 1
                self._impulse_peak_mass = max(self._impulse_peak_mass, mass)
        else:
            # Mass has dropped below threshold
            if self._in_impulse and self._consec_impulse >= self._impulse_bars:
                # Lock in the completed impulse
                rec = _ImpulseRecord(
                    start_price = self._impulse_start_px,
                    end_price   = close,
                    start_bar   = self._impulse_start_bar,
                    end_bar     = self._bar_index,
                    peak_mass   = self._impulse_peak_mass,
                )
                self._impulse_records.append(rec)
                if len(self._impulse_records) > 3:
                    self._impulse_records.pop(0)

            self._in_impulse     = False
            self._consec_impulse = 0

        # Phase 2: detect retracement after impulse
        if not self._impulse_records:
            return None

        last_impulse = self._impulse_records[-1]

        if mass < self._retrace_lo and not self._in_retrace:
            self._in_retrace        = True
            self._retrace_start_px  = close

        if not self._in_retrace:
            return None

        # Phase 3: measure retracement depth and check Fib levels
        signal = self._check_fib_levels(close, last_impulse)
        return signal

    def _check_fib_levels(
        self, close: float, impulse: _ImpulseRecord
    ) -> Optional[Wave4Signal]:
        """
        Given a live close price and the most recent completed impulse,
        check whether close is near a key Fibonacci retracement level.
        Returns a Wave4Signal if a confluence hit is found.
        """
        # Minimum bars since impulse ended before accepting retrace signal
        bars_since = self._bar_index - impulse.end_bar
        if bars_since < 2:
            return None

        # Suppress if signal was emitted recently
        if (self._bar_index - self._last_signal_bar) < MIN_HOLD_BARS:
            return None

        i_high = max(impulse.start_price, impulse.end_price)
        i_low  = min(impulse.start_price, impulse.end_price)
        i_range = i_high - i_low

        if i_range < 1e-8:
            return None

        # Retracement depth from impulse end
        if impulse.bullish:
            # Price fell from i_high toward i_low
            retrace_depth = (i_high - close) / i_range
        else:
            # Price rose from i_low toward i_high
            retrace_depth = (close - i_low) / i_range

        if retrace_depth <= 0 or retrace_depth > 0.80:
            return None

        # Check proximity to each Fibonacci level
        fib_hit, best_fib = self._nearest_fib(retrace_depth)
        if not fib_hit:
            return None

        # Build the signal
        confidence = self._compute_confidence(retrace_depth, best_fib, impulse)
        if confidence < 0.35:
            return None

        # Stop placement: beyond Wave 1 start (opposite side of impulse)
        stop_buffer = i_range * 0.10
        if impulse.bullish:
            stop_price  = i_low  - stop_buffer
            target_price = i_high + (i_range * FIB_EXT_618)
        else:
            stop_price  = i_high + stop_buffer
            target_price = i_low  - (i_range * FIB_EXT_618)

        self._last_signal_bar = self._bar_index

        return Wave4Signal(
            wave_number     = 4,
            retracement_pct = best_fib,
            entry_price     = close,
            stop_price      = stop_price,
            target_price    = target_price,
            confidence      = confidence,
            bullish         = impulse.bullish,
            impulse_range   = (i_low, i_high),
            bar_index       = self._bar_index,
        )

    # ------------------------------------------------------------------
    # Internal: Fibonacci helpers
    # ------------------------------------------------------------------

    def _nearest_fib(self, depth: float) -> Tuple[bool, float]:
        """
        Return (True, fib_level) if depth is within self._fib_tol of
        a canonical Fibonacci level, else (False, 0.0).
        Prefers the closest level.
        """
        candidates = [FIB_382, FIB_500, FIB_618]
        best_dist = float("inf")
        best_fib  = 0.0

        for fib in candidates:
            dist = abs(depth - fib)
            if dist < best_dist:
                best_dist = dist
                best_fib  = fib

        if best_dist <= self._fib_tol:
            return True, best_fib
        return False, 0.0

    def _compute_confidence(
        self,
        retrace_depth: float,
        best_fib:      float,
        impulse:       _ImpulseRecord,
    ) -> float:
        """
        Score 0-1 based on:
          - Fibonacci proximity (40%)
          - BH mass at impulse peak relative to BH_MASS_THRESH (30%)
          - Number of completed impulse records (trend depth) (30%)
        """
        # Fib proximity score: perfect hit = 1, tolerance boundary = 0
        fib_dist = abs(retrace_depth - best_fib)
        fib_score = max(0.0, 1.0 - (fib_dist / self._fib_tol))

        # Mass score: higher peak mass = more energy = more confident
        mass_score = min(1.0, (impulse.peak_mass - self._mass_thresh) / 2.0 + 0.5)
        mass_score = max(0.0, mass_score)

        # Depth score: having at least 2 prior impulses increases confidence
        depth_score = min(1.0, len(self._impulse_records) / 3.0)

        confidence = 0.40 * fib_score + 0.30 * mass_score + 0.30 * depth_score
        return round(confidence, 4)

    # ------------------------------------------------------------------
    # Internal: pivot detection for classify_wave
    # ------------------------------------------------------------------

    def _find_pivots(self, arr: np.ndarray, order: int = 3) -> List[Tuple[int, float, str]]:
        """
        Find local maxima and minima using a simple window comparison.
        Returns list of (index, price, 'HIGH'|'LOW').
        """
        pivots: List[Tuple[int, float, str]] = []
        n = len(arr)
        for i in range(order, n - order):
            window = arr[i - order: i + order + 1]
            if arr[i] == window.max() and arr[i] > window.mean():
                pivots.append((i, float(arr[i]), "HIGH"))
            elif arr[i] == window.min() and arr[i] < window.mean():
                pivots.append((i, float(arr[i]), "LOW"))
        return pivots

    def _label_from_pivots(
        self,
        arr: np.ndarray,
        pivots: List[Tuple[int, float, str]],
    ) -> WaveLabel:
        """
        Map pivot structure onto Elliott Wave label.
        Simplified: count alternating H/L pivots to determine wave position.
        """
        # Deduplicate adjacent same-type pivots, keep highest/lowest
        deduped: List[Tuple[int, float, str]] = []
        for piv in pivots:
            if deduped and deduped[-1][2] == piv[2]:
                if piv[2] == "HIGH" and piv[1] > deduped[-1][1]:
                    deduped[-1] = piv
                elif piv[2] == "LOW" and piv[1] < deduped[-1][1]:
                    deduped[-1] = piv
            else:
                deduped.append(piv)

        n_pivots = len(deduped)
        if n_pivots < 2:
            return WaveLabel.UNKNOWN

        last = deduped[-1]
        second_last = deduped[-2]

        # Determine dominant trend from first to last pivot
        first_px = arr[0]
        last_px  = float(arr[-1])
        trend_up = last_px > first_px

        # Count pivots to estimate wave position
        if n_pivots == 2:
            return WaveLabel.WAVE_1 if trend_up else WaveLabel.CORRECTION
        elif n_pivots == 3:
            return WaveLabel.WAVE_2
        elif n_pivots == 4:
            return WaveLabel.WAVE_3
        elif n_pivots == 5:
            # Most interesting: possible Wave 4 in progress
            if last[2] == "LOW" and trend_up:
                return WaveLabel.WAVE_4
            elif last[2] == "HIGH" and not trend_up:
                return WaveLabel.WAVE_4
            return WaveLabel.WAVE_5
        elif n_pivots >= 6:
            return WaveLabel.CORRECTION
        else:
            return WaveLabel.UNKNOWN


# ---------------------------------------------------------------------------
# Wave4StrategyAdapter
# ---------------------------------------------------------------------------

class Wave4StrategyAdapter:
    """
    Wraps Wave4Detector for integration with the SRFM live trader loop.

    compute_signal() returns a normalized float in [-1, 1]:
      - Positive: bullish Wave 4 long setup
      - Negative: bearish Wave 4 short setup
      - Zero: no signal or Hurst filter blocks trade

    The Hurst filter requires H > HURST_TRENDING_MIN (0.55) because
    Wave 4 trades should only be taken in trending markets.
    """

    def __init__(
        self,
        hurst_min:      float = HURST_TRENDING_MIN,
        fib_tolerance:  float = 0.04,
        mass_thresh:    float = BH_MASS_THRESH,
    ):
        self._hurst_min  = hurst_min
        self._detector   = Wave4Detector(
            mass_thresh   = mass_thresh,
            fib_tolerance = fib_tolerance,
        )
        self._last_signal:   Optional[Wave4Signal] = None
        self._signal_age:    int   = 0   # bars since last signal
        self._current_hurst: float = 0.5

    # ------------------------------------------------------------------

    def update_hurst(self, hurst: float) -> None:
        """Update current Hurst exponent estimate."""
        self._current_hurst = float(hurst)

    def compute_signal(self, bars: List[dict]) -> float:
        """
        Process latest bar from bars list and return normalized signal.

        Parameters
        #--------
        bars:
            List of bar dicts (newest last).  Must contain close, high, low,
            bh_mass keys.  At least 1 bar required.

        Returns
        #-----
        float in [-1, 1]: positive = long, negative = short, 0 = no signal.
        """
        if not bars:
            return 0.0

        # Hurst filter: Wave 4 only valid in trending market
        if self._current_hurst < self._hurst_min:
            self._signal_age += 1
            return 0.0

        bar = bars[-1]
        signal = self._detector.update(bar)
        self._signal_age += 1

        if signal is not None:
            self._last_signal = signal
            self._signal_age  = 0

        # Use most recent signal if it is fresh enough (within 5 bars)
        if self._last_signal is not None and self._signal_age <= 5:
            direction  = 1.0 if self._last_signal.bullish else -1.0
            magnitude  = self._last_signal.confidence   # 0-1
            return round(direction * magnitude, 4)

        return 0.0

    def get_last_signal(self) -> Optional[Wave4Signal]:
        """Return the most recently emitted Wave4Signal."""
        return self._last_signal

    def reset(self) -> None:
        """Reset detector and adapter state."""
        self._detector       = Wave4Detector()
        self._last_signal    = None
        self._signal_age     = 0
        self._current_hurst  = 0.5

    # ------------------------------------------------------------------
    # Utility: build synthetic bar dict from price + mass
    # ------------------------------------------------------------------

    @staticmethod
    def make_bar(
        close:   float,
        bh_mass: float,
        high:    Optional[float] = None,
        low:     Optional[float] = None,
    ) -> dict:
        """Convenience factory for bar dicts used in testing."""
        return {
            "close":   close,
            "high":    high if high is not None else close * 1.001,
            "low":     low  if low  is not None else close * 0.999,
            "bh_mass": bh_mass,
        }


# ---------------------------------------------------------------------------
# Hurst Exponent helper (R/S method)
# Used by Wave4StrategyAdapter when caller doesn't supply Hurst externally.
# ---------------------------------------------------------------------------

def hurst_rs(prices: np.ndarray, min_window: int = 10) -> float:
    """
    Estimate Hurst exponent via the rescaled range (R/S) method.

    H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk.
    Returns float in [0, 1].  Requires at least 20 prices.
    """
    n = len(prices)
    if n < 20:
        return 0.5

    returns = np.diff(np.log(np.maximum(prices, 1e-9)))
    rs_vals  = []
    log_ns   = []

    # Use windows from min_window to n//2
    windows = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(len(returns)),
            num=10,
            dtype=int,
        )
    )

    for w in windows:
        if w < min_window or w > len(returns):
            continue
        segments = len(returns) // w
        if segments < 1:
            continue
        rs_segment = []
        for s in range(segments):
            chunk = returns[s * w: (s + 1) * w]
            mean_c = chunk.mean()
            dev    = np.cumsum(chunk - mean_c)
            r      = dev.max() - dev.min()
            std_c  = chunk.std(ddof=1)
            if std_c > 1e-10:
                rs_segment.append(r / std_c)
        if rs_segment:
            rs_vals.append(np.mean(rs_segment))
            log_ns.append(np.log(w))

    if len(rs_vals) < 3:
        return 0.5

    log_rs = np.log(np.maximum(rs_vals, 1e-12))
    try:
        slope, _ = np.polyfit(log_ns, log_rs, 1)
        return float(np.clip(slope, 0.01, 0.99))
    except Exception:
        return 0.5
