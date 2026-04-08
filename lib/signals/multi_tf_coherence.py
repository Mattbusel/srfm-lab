"""
Multi-Timeframe BH Coherence Engine (T3-7)
Computes BH physics stack across multiple timeframes and scores coherence.

Coherence filtering: entry only when 3+ timeframes show aligned BH formation.
Higher coherence → larger position size scaling.
Divergence between fast and slow TF → reduce or skip.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

log = logging.getLogger(__name__)

# Timeframe bar counts relative to 15m (the base TF)
TF_RATIOS = {
    "5m":  0.33,   # 1/3 of a 15m bar
    "15m": 1,
    "1h":  4,
    "4h":  16,
    "1d":  96,
}

@dataclass
class TFBHState:
    """BH state for a single timeframe."""
    tf: str
    mass: float = 0.0
    active: bool = False
    ds2_ema: float = 0.0
    cf_ema: float = 0.0
    bar_count: int = 0
    # Aggregation buffer for higher TFs
    _agg_high: float = 0.0
    _agg_low: float = 0.0
    _agg_close: float = 0.0
    _agg_open: float = 0.0
    _agg_count: int = 0

@dataclass
class MTFCoherenceConfig:
    timeframes: list = field(default_factory=lambda: ["15m", "1h", "4h"])
    min_coherent_tfs: int = 2   # minimum TFs required for entry
    full_coherence_tfs: int = 3  # TFs for full sizing
    coherence_scale_min: float = 0.6  # size scale when min_coherent_tfs met
    coherence_scale_full: float = 1.3  # size scale when full_coherence_tfs met
    bh_form_threshold: float = 1.92   # BH formation mass threshold
    cf_threshold_15m: float = 0.010   # default CF threshold (per-instrument overridable)
    mass_alpha: float = 0.15
    cf_alpha: float = 0.10
    mass_decay: float = 0.924  # BH_DECAY constant

class MTFCoherenceEngine:
    """
    Multi-timeframe BH coherence engine for a single instrument.

    Feed 15m bars; higher-TF bars are synthesized by aggregation.

    Usage:
        engine = MTFCoherenceEngine(cf_thresholds={"15m": 0.010, "1h": 0.030, "4h": 0.016})
        result = engine.update_15m(open_, high, low, close, volume)
        if result['enter']:
            size_scale = result['size_scale']
    """

    def __init__(self, cfg: MTFCoherenceConfig = None, cf_thresholds: dict = None):
        self.cfg = cfg or MTFCoherenceConfig()
        self._cf_thresholds = cf_thresholds or {
            "15m": self.cfg.cf_threshold_15m,
            "1h":  self.cfg.cf_threshold_15m * 3,
            "4h":  self.cfg.cf_threshold_15m * 1.6,
        }
        self._tf_states: dict[str, TFBHState] = {
            tf: TFBHState(tf=tf) for tf in self.cfg.timeframes
        }
        self._bar_count_15m: int = 0
        self._prev_close: Optional[float] = None

    def update_15m(self, open_: float, high: float, low: float, close: float, volume: float) -> dict:
        """Process one 15m bar. Returns coherence analysis."""
        self._bar_count_15m += 1

        # Update 15m TF
        if "15m" in self._tf_states:
            self._update_tf("15m", open_, high, low, close, volume)

        # Aggregate into higher TFs
        agg_ratio_map = {"1h": 4, "4h": 16, "1d": 96}
        for tf, ratio in agg_ratio_map.items():
            if tf not in self._tf_states:
                continue
            st = self._tf_states[tf]
            # Accumulate into the aggregation buffer
            if st._agg_count == 0:
                st._agg_open = open_
                st._agg_high = high
                st._agg_low = low
            else:
                st._agg_high = max(st._agg_high, high)
                st._agg_low = min(st._agg_low, low)
            st._agg_close = close
            st._agg_count += 1

            if st._agg_count >= ratio:
                # Emit a completed higher-TF bar
                self._update_tf(tf, st._agg_open, st._agg_high, st._agg_low, st._agg_close, volume)
                st._agg_count = 0

        self._prev_close = close
        return self._compute_coherence()

    def _update_tf(self, tf: str, open_: float, high: float, low: float, close: float, volume: float):
        """Run BH physics for one bar of a given timeframe."""
        st = self._tf_states[tf]
        st.bar_count += 1
        cf_thresh = self._cf_thresholds.get(tf, self.cfg.cf_threshold_15m)

        # Minkowski interval
        if self._prev_close and self._prev_close > 0:
            dp_frac = (close - self._prev_close) / self._prev_close
        else:
            dp_frac = 0.0

        cf = (high - low) / (self._prev_close + 1e-10) if self._prev_close else 0.0
        beta = abs(dp_frac) / (cf + 1e-10)
        ds2 = cf * cf - dp_frac * dp_frac
        timelike = ds2 > 0

        st.ds2_ema = cf * ds2 if st.ds2_ema == 0 else (0.85 * st.ds2_ema + 0.15 * abs(ds2))
        st.cf_ema = 0.90 * st.cf_ema + 0.10 * cf

        # Mass update
        if timelike and cf > cf_thresh:
            # TIMELIKE bar above CF threshold: mass accumulates
            delta_mass = cf * (1.0 - beta)
            slingshot = 1.0 + 0.5 * (st.cf_ema / (cf_thresh + 1e-10) - 1.0)
            slingshot = max(1.0, min(2.5, slingshot))
            st.mass += delta_mass * slingshot * self.cfg.mass_alpha
        else:
            st.mass *= self.cfg.mass_decay

        # BH activation
        st.active = st.mass >= self.cfg.bh_form_threshold

    def _compute_coherence(self) -> dict:
        """Compute coherence score across all tracked timeframes."""
        active_tfs = [tf for tf, st in self._tf_states.items() if st.active]
        active_count = len(active_tfs)

        # Coherence requires minimum number of aligned TFs
        enter = active_count >= self.cfg.min_coherent_tfs

        # Size scale: interpolate between min and full coherence
        if active_count >= self.cfg.full_coherence_tfs:
            size_scale = self.cfg.coherence_scale_full
        elif active_count >= self.cfg.min_coherent_tfs:
            frac = (active_count - self.cfg.min_coherent_tfs) / max(
                self.cfg.full_coherence_tfs - self.cfg.min_coherent_tfs, 1)
            size_scale = self.cfg.coherence_scale_min + frac * (
                self.cfg.coherence_scale_full - self.cfg.coherence_scale_min)
        else:
            size_scale = 0.0

        # Divergence warning: fast TF active but slow TF not
        fast_active = self._tf_states.get("15m", TFBHState("15m")).active
        slow_active = self._tf_states.get("4h", TFBHState("4h")).active
        divergence = fast_active and not slow_active

        if divergence:
            size_scale *= 0.5  # halve on divergence

        masses = {tf: st.mass for tf, st in self._tf_states.items()}

        return {
            "enter": enter,
            "size_scale": size_scale,
            "active_tfs": active_tfs,
            "active_count": active_count,
            "divergence": divergence,
            "masses": masses,
        }

    @property
    def coherence_score(self) -> float:
        """Simple scalar coherence score in [0, 1]."""
        n = len(self.cfg.timeframes)
        active = sum(1 for st in self._tf_states.values() if st.active)
        return active / n if n > 0 else 0.0
