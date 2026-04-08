"""
Gravitational Wave Detector (T3-3)
Detects coordinated multi-instrument BH formation events.

In SRFM physics: when a BH forms on one instrument, gravitational waves propagate
to correlated instruments with a delay proportional to financial 'spacetime distance'.
Coordinated BH formations (within the causal window) are higher-conviction events.

Signal:
  isolated BH:           1.0x sizing multiplier
  2-instrument wave:     1.5x sizing multiplier
  3+ instrument wave:    2.0x sizing multiplier
"""
import time
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

log = logging.getLogger(__name__)

@dataclass
class GravWaveConfig:
    causal_window_bars: int = 8  # max propagation lag (8 bars = 2 hours)
    min_correlation: float = 0.40  # minimum correlation for sympathetic BH
    size_multipliers: tuple = (1.0, 1.5, 2.0)  # for 1, 2, 3+ coherent BHs
    max_multiplier: float = 2.0

@dataclass
class BHFormationEvent:
    symbol: str
    bar_idx: int
    mass: float
    intensity: float  # normalized BH formation strength

class GravitationalWaveDetector:
    """
    Tracks BH formation events across instruments and detects coherent waves.

    Usage:
        detector = GravitationalWaveDetector(corr_matrix)
        detector.record_bh_formation(sym, bar_idx, mass, intensity)
        multiplier = detector.get_sizing_multiplier(sym, bar_idx)
    """

    def __init__(self, cfg: GravWaveConfig = None):
        self.cfg = cfg or GravWaveConfig()
        self._events: deque[BHFormationEvent] = deque(maxlen=500)
        self._corr_matrix: dict[tuple[str, str], float] = {}
        self._bar_idx: int = 0

    def set_correlation_matrix(self, corr: dict[tuple[str, str], float]):
        """Update the correlation matrix. corr[(sym_a, sym_b)] = correlation."""
        self._corr_matrix = corr

    def tick(self):
        """Advance bar counter."""
        self._bar_idx += 1

    def record_bh_formation(self, sym: str, mass: float, intensity: float):
        """Record a BH formation event for a symbol at current bar."""
        event = BHFormationEvent(
            symbol=sym,
            bar_idx=self._bar_idx,
            mass=mass,
            intensity=intensity,
        )
        self._events.append(event)
        log.debug("GravWave: BH formation recorded %s mass=%.3f intensity=%.3f", sym, mass, intensity)

    def get_sizing_multiplier(self, sym: str) -> float:
        """
        Returns sizing multiplier for sym based on coherent BH wave detection.
        Only call this when a BH is forming for sym.
        """
        window = self.cfg.causal_window_bars

        # Find coherent BH formations within the causal window (excluding sym itself)
        coherent_syms: list[str] = []
        for event in reversed(self._events):
            if self._bar_idx - event.bar_idx > window:
                break
            if event.symbol == sym:
                continue
            corr = self._get_correlation(sym, event.symbol)
            if corr >= self.cfg.min_correlation:
                coherent_syms.append(event.symbol)

        coherent_count = 1 + len(set(coherent_syms))  # include current sym

        if coherent_count >= 3:
            multiplier = self.cfg.size_multipliers[2]
        elif coherent_count == 2:
            multiplier = self.cfg.size_multipliers[1]
        else:
            multiplier = self.cfg.size_multipliers[0]

        if coherent_count > 1:
            log.info("GravWave: %d-instrument coherent wave detected for %s → %.1fx sizing",
                     coherent_count, sym, multiplier)

        return min(multiplier, self.cfg.max_multiplier)

    def _get_correlation(self, sym_a: str, sym_b: str) -> float:
        key1 = (sym_a, sym_b)
        key2 = (sym_b, sym_a)
        return self._corr_matrix.get(key1, self._corr_matrix.get(key2, 0.0))
