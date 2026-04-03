"""
srfm_core.py — SRFM physics extracted from LARSA.

All formulas are taken directly from the production 274% strategy.
No LEAN dependency — pure Python + numpy.

Class map (source location in LARSA):
  MinkowskiClassifier  ← _process_instrument: beta, bit, tl_confirm, proper_time
  BlackHoleDetector    ← update_bh()
  GeodesicAnalyzer     ← _process_instrument: 20-bar log-linear regression block
  GravitationalLens    ← _process_instrument: mu computation block
"""

from __future__ import annotations
import math
import numpy as np
from collections import deque
from enum import IntEnum
from typing import Optional, List, Tuple


class MarketRegime(IntEnum):
    BULL           = 0
    BEAR           = 1
    SIDEWAYS       = 2
    HIGH_VOLATILITY = 3


# ─────────────────────────────────────────────────────────────────────────────
# MinkowskiClassifier
# ─────────────────────────────────────────────────────────────────────────────

class MinkowskiClassifier:
    """
    Classify each bar as TIMELIKE or SPACELIKE via the Minkowski metric.

    beta = |close_t - close_{t-1}| / close_{t-1} / cf
    TIMELIKE  ↔ beta < 1   (sub-luminal: ordered, causal move)
    SPACELIKE ↔ beta >= 1  (super-luminal: anomalous velocity event)

    Also tracks:
      tl_confirm  : consecutive TIMELIKE bars (capped at 3)
      proper_time : cumulative proper time τ = Σ 1/γ where γ = 1/√(1−v²)
                    v = min(0.99, |Δclose/close| / max_vol)
    """

    def __init__(self, cf: float = 0.001, max_vol: float = 0.01):
        self.cf      = cf
        self.max_vol = max_vol

        self.beta:        float = 0.0
        self.bit:         str   = "UNKNOWN"   # "TIMELIKE" | "SPACELIKE"
        self.tl_confirm:  int   = 0
        self.proper_time: float = 0.0

        self._prev_close: Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, close: float) -> str:
        """Feed one bar's close price. Returns "TIMELIKE" or "SPACELIKE"."""
        if self._prev_close is None or self._prev_close <= 0:
            self._prev_close = close
            return self.bit

        price_diff = abs(close - self._prev_close)
        self.beta  = price_diff / (self._prev_close + 1e-9) / self.cf

        self.bit = "TIMELIKE" if self.beta < 1.0 else "SPACELIKE"

        if self.bit == "TIMELIKE":
            self.tl_confirm = min(self.tl_confirm + 1, 3)
        else:
            self.tl_confirm = 0

        # Proper time: τ += 1/γ  (γ = Lorentz factor)
        hv = price_diff / (self._prev_close + 1e-9)
        v  = min(0.99, hv / self.max_vol)
        gamma = 1.0 / math.sqrt(max(1e-9, 1.0 - v * v))
        self.proper_time += 1.0 / gamma

        self._prev_close = close
        return self.bit

    def reset_proper_time(self):
        self.proper_time = 0.0

    @property
    def is_timelike(self) -> bool:
        return self.bit == "TIMELIKE"


# ─────────────────────────────────────────────────────────────────────────────
# BlackHoleDetector
# ─────────────────────────────────────────────────────────────────────────────

class BlackHoleDetector:
    """
    Detect and track gravitational wells (black holes) in price momentum.

    Mass dynamics (from update_bh):
      TIMELIKE bar  : ctl += 1; sb = min(2.0, 1+ctl*0.1)
                      bh_mass = bh_mass*bh_decay + |br|*100*sb
      SPACELIKE bar : ctl = 0; bh_mass *= 0.7

    Formation  : bh_mass > bh_form  AND ctl >= 5
    Active     : bh_mass > bh_collapse AND ctl >= 5
    Collapse   : bh_mass <= bh_collapse OR ctl < 5

    Reform memory : if 0 < reform_bars < 15, the next formation adds
                    0.5 * prev_bh_mass to help re-form faster.
    """

    def __init__(
        self,
        bh_form:    float = 1.5,
        bh_collapse: float = 1.0,
        bh_decay:   float = 0.95,
    ):
        self.bh_form     = bh_form
        self.bh_collapse = bh_collapse
        self.bh_decay    = bh_decay

        self.bh_mass:   float = 0.0
        self.bh_active: bool  = False
        self.bh_dir:    int   = 0     # +1 long well, -1 short well
        self.ctl:       int   = 0     # consecutive TIMELIKE count
        self.cum_disp:  float = 0.0   # cumulative bar return (direction tracker)
        self.prev_bh_mass: float = 0.0
        self.reform_bars:  int   = 0

        # Diagnostics
        self.well_events: List[dict] = []

    # ------------------------------------------------------------------
    def update(self, bit: str, close: float, prev_close: float) -> bool:
        """
        Feed one bar. Returns True if a BH is active after this bar.

        Parameters
        ----------
        bit        : "TIMELIKE" or "SPACELIKE"
        close      : current bar close
        prev_close : previous bar close
        """
        br = (close - prev_close) / (prev_close + 1e-9)

        if bit == "TIMELIKE":
            self.ctl      += 1
            self.cum_disp += br
            sb            = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass  = self.bh_mass * self.bh_decay + abs(br) * 100 * sb
        else:
            self.ctl       = 0
            self.bh_mass  *= 0.7

        # Direction from cumulative displacement
        if   self.cum_disp > 0: self.bh_dir = 1
        elif self.cum_disp < 0: self.bh_dir = -1

        prev_active = self.bh_active

        if not prev_active:
            # Reform boost: if we recently collapsed, add half of previous peak mass
            if 0 < self.reform_bars < 15:
                self.bh_mass       += self.prev_bh_mass * 0.5
                self.prev_bh_mass   = 0.0
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= 5
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= 5

        if self.bh_active and not prev_active:
            # New well formed
            self.reform_bars = 0
            self.well_events.append({
                "event": "formed",
                "mass": self.bh_mass,
                "dir": self.bh_dir,
                "ctl": self.ctl,
            })
        elif not self.bh_active and prev_active:
            # Well collapsed
            self.prev_bh_mass = self.bh_mass
            self.reform_bars  = 1
            self.well_events.append({
                "event": "collapsed",
                "mass": self.bh_mass,
            })

        if not self.bh_active:
            if self.reform_bars > 0:
                self.reform_bars += 1
            if self.ctl == 0:
                self.cum_disp = 0.0

        return self.bh_active

    def reset(self):
        self.bh_mass = 0.0; self.bh_active = False; self.bh_dir = 0
        self.ctl = 0; self.cum_disp = 0.0; self.prev_bh_mass = 0.0
        self.reform_bars = 0


# ─────────────────────────────────────────────────────────────────────────────
# GeodesicAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class GeodesicAnalyzer:
    """
    20-bar log-linear regression + Minkowski causal fraction + rapidity.

    From LARSA _process_instrument (the 20-bar block):

    geo_dev   = tanh((price - geo_projected) / atr)
    geo_slope = slope * 100

    causal_frac: fraction of (price, price[k]) pairs whose spacetime interval
                 ds² = -(cf*k)² + (|Δp/p|)² < 0   (i.e., TIMELIKE interval)

    rapidity: tanh(0.5 * log((E+px)/(E-px)))   where E = px = 19-bar return
              (note: LARSA sets px=E so denom = 1e-9, giving rapidity ≈ 0
               whenever the 19-bar return is symmetric — this is the production
               formula exactly as shipped)
    """

    def __init__(self, cf: float = 0.001, window: int = 20):
        self.cf     = cf
        self.window = window
        self._closes: deque = deque(maxlen=window)

    # ------------------------------------------------------------------
    def update(self, close: float, atr: float) -> Tuple[float, float, float, float]:
        """
        Returns (geo_dev, geo_slope, causal_frac, rapidity).
        Returns (0,0,1,0) if fewer than `window` bars available.
        """
        self._closes.appendleft(close)   # index 0 = most recent (matches LARSA cw[0])

        if len(self._closes) < self.window:
            return 0.0, 0.0, 1.0, 0.0

        # 20-bar log-linear regression  (matches LARSA exactly)
        prices = list(reversed(list(self._closes)))   # oldest → newest
        lp     = np.log(np.array(prices) + 1e-9)
        x      = np.arange(self.window, dtype=float)
        n      = float(self.window)
        sx     = x.sum()
        slp    = lp.sum()
        sxx    = np.dot(x, x)
        sxlp   = np.dot(x, lp)
        slope  = (n * sxlp - sx * slp) / (n * sxx - sx * sx + 1e-9)
        intercept = (slp - slope * sx) / n
        geo_p     = float(np.exp(slope * (self.window - 1) + intercept))
        geo_dev   = float(np.tanh((close - geo_p) / (atr + 1e-9)))
        geo_slope = float(slope * 100)

        # Causal fraction
        cp = close; cc = 0
        closes_list = list(self._closes)   # index 0 = newest
        for k in range(1, self.window):
            pp = closes_list[k]
            dp = abs(cp - pp) / (pp + 1e-9)
            ds2 = -(self.cf * k) ** 2 + dp * dp
            if ds2 < 0:
                cc += 1
        causal_frac = cc / (self.window - 1)

        # Rapidity (LARSA production formula)
        E  = (close - closes_list[self.window - 1]) / (closes_list[self.window - 1] + 1e-9)
        px = E
        denom = E - px + 1e-9
        num   = E + px
        if abs(denom) > 1e-9 and num / denom > 0:
            rapidity = float(np.tanh(0.5 * np.log(num / denom + 1e-9)))
        else:
            rapidity = 0.0

        return geo_dev, geo_slope, causal_frac, rapidity

    @property
    def ready(self) -> bool:
        return len(self._closes) >= self.window


# ─────────────────────────────────────────────────────────────────────────────
# GravitationalLens
# ─────────────────────────────────────────────────────────────────────────────

class GravitationalLens:
    """
    Compute lensing amplification μ from TIMELIKE-VWAP distance.

    From LARSA _process_instrument (the mu block):
      M   = ctl + (1 if TIMELIKE else 0)
      R_E = sqrt(M)
      VWAP: volume-weighted average of closes on TIMELIKE bars only
      r   = |price - VWAP| / atr
      mu  = 1 + R_E / (r + R_E)

    When M < 2: mu = max(0.3, M/3.0)
    """

    def __init__(self):
        self._closes_tl: List[float] = []   # TIMELIKE bar closes
        self._vols_tl:   List[float] = []   # corresponding volumes
        self.mu: float = 1.0

    # ------------------------------------------------------------------
    def update(
        self,
        close:   float,
        volume:  float,
        bit:     str,
        ctl:     int,
        atr:     float,
    ) -> float:
        """Returns current mu."""
        if bit == "TIMELIKE":
            self._closes_tl.append(close)
            self._vols_tl.append(volume)
        else:
            # Reset on SPACELIKE (LARSA rebuilds window each bar using cw/tlw)
            self._closes_tl = [close] if bit == "TIMELIKE" else []
            self._vols_tl   = [volume] if bit == "TIMELIKE" else []

        M   = float(ctl + (1 if bit == "TIMELIKE" else 0))
        R_E = math.sqrt(M + 1e-9)

        if M < 2.0:
            self.mu = max(0.3, M / 3.0)
        elif self._closes_tl:
            sv   = sum(self._vols_tl)
            if sv > 0:
                vwap = sum(p * v for p, v in zip(self._closes_tl, self._vols_tl)) / sv
            else:
                vwap = sum(self._closes_tl) / len(self._closes_tl)
            r       = abs(close - vwap) / (atr + 1e-9)
            self.mu = float(1.0 + R_E / (r + R_E))
        else:
            self.mu = 1.0

        return self.mu

    def update_from_window(
        self,
        closes_window:  List[float],   # index 0 = newest, matches cw
        vols_window:    List[float],
        tl_window:      List[float],   # 1.0=TIMELIKE, 0.0=SPACELIKE
        close:          float,
        bit:            str,
        ctl:            int,
        atr:            float,
    ) -> float:
        """
        Exact reproduction of LARSA's window-scan approach.
        Use this when you have rolling window data available.
        """
        ni = min(len(closes_window), len(tl_window), len(vols_window))
        lp, lv = [], []
        for i in range(ni):
            if tl_window[i] == 1.0:
                lp.append(closes_window[i])
                lv.append(vols_window[i])

        M   = float(ctl + (1 if bit == "TIMELIKE" else 0))
        R_E = math.sqrt(M + 1e-9)

        if M < 2.0:
            self.mu = max(0.3, M / 3.0)
        elif lp:
            sv   = sum(lv)
            vwap = sum(p * v for p, v in zip(lp, lv)) / sv if sv > 0 else sum(lp) / len(lp)
            r    = abs(close - vwap) / (atr + 1e-9)
            self.mu = float(1.0 + R_E / (r + R_E))
        else:
            self.mu = 1.0

        return self.mu


# ─────────────────────────────────────────────────────────────────────────────
# HawkingMonitor  (Bollinger-band Z² proxy for Hawking temperature)
# ─────────────────────────────────────────────────────────────────────────────

class HawkingMonitor:
    """
    Tracks the Hawking-temperature proxy used in LARSA.

    From _process_instrument:
      z  = (close - bb_middle) / std
      ht = z * (z - prev_z)   (second-order Z² derivative)

    ht > 1.8  → hot well → reduce/avoid long entry
    ht < -1.5 → inverted → add to signal
    """

    def __init__(self):
        self.ht:  float = 0.0
        self._pz: float = 0.0

    def update(self, close: float, bb_middle: float, std: float) -> float:
        if std <= 0:
            return self.ht
        z       = (close - bb_middle) / std
        self.ht = z * (z - self._pz)
        self._pz = z
        return self.ht

    @property
    def is_hot(self) -> bool:
        return self.ht > 1.8

    @property
    def is_inverted(self) -> bool:
        return self.ht < -1.5
