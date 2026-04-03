"""
risk.py — Portfolio risk management extracted from LARSA.

Two classes:
  PortfolioRiskManager : portfolio-level drawdown + daily loss + trailing stop
  KillConditions       : per-instrument kill checks on each candidate position

Both return scalar multipliers / booleans so they can be tested in isolation.
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioRiskManager
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioRiskManager:
    """
    Exact reproduction of LARSA's portfolio-level risk logic.

    _portfolio_risk() → returns 0.0, 0.5, or 1.0:
      0.0 : circuit breaker (dd >= maxdd) or daily loss >= dlim
      0.5 : caution zone (dd >= 70% of maxdd)
      1.0 : normal

    Trailing stop (from on_data):
      Peak-gain-scaled trail percentages:
        gain >= 1.50 → 10%
        gain >= 1.00 → 12%
        gain >= 0.50 → 15%
        gain >= 0.20 → 18%
        else         → 20%

      When trail fires: flat for 5 bars, then ramp back 5 bars (cap pos at 0.5).

    Profit floor: initial_equity * (1 + peak_gain * 0.50)
      Fires when pv < floor and gain > 100% and no grace period.

    HWM hard stop: if dd >= 12% anytime → 3-bar cooldown.

    Parameters
    ----------
    initial_equity : float  Starting portfolio value (default 1_000_000)
    maxdd          : float  Hard max drawdown (default 0.12)
    dlim           : float  Daily loss limit (default 0.02)
    """

    def __init__(
        self,
        initial_equity: float = 1_000_000.0,
        maxdd:          float = 0.12,
        dlim:           float = 0.02,
    ):
        self.initial_equity  = initial_equity
        self.maxdd           = maxdd
        self.dlim            = dlim

        self.peak:            float = initial_equity
        self.daystart:        float = initial_equity
        self.cb:              bool  = False         # circuit breaker

        self.hwm_cooldown:    int   = 0
        self.trail_flat_bars: int   = 0
        self.ramp_back:       int   = 0
        self.grace_period:    int   = 0

    # ------------------------------------------------------------------
    def daily_reset(self, equity: float):
        """Call at start of each trading day."""
        self.daystart = equity
        self.cb       = False

    def portfolio_risk(self, equity: float) -> float:
        """
        Returns risk multiplier: 0.0 | 0.5 | 1.0.
        Exact reproduction of LARSA._portfolio_risk().
        """
        if equity > self.peak:
            self.peak = equity

        dd = (self.peak - equity) / (self.peak + 1e-9)

        if dd >= self.maxdd:
            if not self.cb:
                self.cb = True
            return 0.0

        daily_loss = (equity - self.daystart) / (self.daystart + 1e-9)
        if daily_loss <= -self.dlim:
            return 0.0

        if dd >= self.maxdd * 0.70:
            return 0.50

        return 1.0

    # ------------------------------------------------------------------
    def on_bar(self, equity: float) -> str:
        """
        Full on_data risk logic. Call once per bar.

        Returns one of:
          "hwm_cooldown"   : HWM hard stop active, skip bar
          "profit_floor"   : profit floor hit, liquidate
          "trail_stop"     : trailing stop hit, liquidate
          "flat"           : still flat after stop (trail_flat_bars > 0)
          "ramp"           : ramp-back active (cap positions)
          "ok"             : normal
        """
        if equity > self.peak:
            self.peak = equity

        # HWM hard stop check
        if self.hwm_cooldown > 0:
            self.hwm_cooldown -= 1
            if self.hwm_cooldown == 0:
                self.peak = equity
            return "hwm_cooldown"

        dd = (self.peak - equity) / (self.peak + 1e-9)
        if dd >= 0.12:
            self.hwm_cooldown = 3
            self.ramp_back    = 5
            self.grace_period = 50
            return "hwm_cooldown"

        peak_gain   = (self.peak - self.initial_equity) / self.initial_equity
        profit_floor = self.initial_equity * (1.0 + peak_gain * 0.50)

        # Profit floor check
        if equity < profit_floor and peak_gain > 1.00 and self.grace_period == 0:
            self.trail_flat_bars = 5
            self.ramp_back       = 5
            self.grace_period    = 50
            return "profit_floor"

        # Trailing stop flat period
        if self.trail_flat_bars > 0:
            self.trail_flat_bars -= 1
            if self.trail_flat_bars == 0:
                self.peak = equity
            return "flat"

        # Trailing stop trigger
        trail_pct = self._trail_pct(peak_gain)
        if self.grace_period == 0 and dd >= trail_pct:
            self.trail_flat_bars = 5
            self.ramp_back       = 5
            self.grace_period    = 50
            return "trail_stop"

        if self.grace_period > 0:
            self.grace_period -= 1
        if self.ramp_back > 0:
            self.ramp_back -= 1
            return "ramp"

        return "ok"

    @staticmethod
    def _trail_pct(peak_gain: float) -> float:
        """Scale trail percentage with accumulated gain."""
        if peak_gain >= 1.50: return 0.10
        if peak_gain >= 1.00: return 0.12
        if peak_gain >= 0.50: return 0.15
        if peak_gain >= 0.20: return 0.18
        return 0.20

    @property
    def drawdown(self) -> float:
        return (self.peak - self.peak) / (self.peak + 1e-9)  # always 0 at peak

    def current_drawdown(self, equity: float) -> float:
        return (self.peak - equity) / (self.peak + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# KillConditions
# ─────────────────────────────────────────────────────────────────────────────

class KillConditions:
    """
    Per-instrument kill conditions from LARSA _process_instrument().

    Usage:
        kc     = KillConditions()
        killed, tgt = kc.apply(tgt, geo_dev, bc, tl_confirm, bit, regime, ...)
    """

    GEO_KILL_THRESHOLD = 2.0    # arctanh(|geo_dev|) > 2.0 → kill
    GEO_FLOOR_RESET    = 1.5    # arctanh > 1.5 → reset pos_floor
    MIN_POSITION_SIZE  = 0.02   # smaller than this → zero
    MIN_SIGNAL_SIZE    = 0.03   # below this after clamping → zero
    WEAK_BAR_THRESHOLD = 0.30   # tgt below this increments weak_bars counter

    def __init__(self):
        self.weak_bars: int   = 0
        self.pos_floor: float = 0.0

    def apply(
        self,
        tgt:          float,
        geo_dev:      float,
        bc:           int,       # bar count (bc < 120 → kill)
        tl_confirm:   int,
        bit:          str,
        regime:       "MarketRegime",
        ctl:          int,       # for pos_floor update
        last_target:  float,
        ramp_back:    int,
    ) -> tuple:
        """
        Returns (killed: bool, final_tgt: float).

        Applies kill conditions in LARSA order:
          1. geo_raw > 2.0
          2. bc < 120
          3. tl_confirm < tl_req
          4. SPACELIKE penalty
          5. < 0.03 → zero
          6. weak_bars accumulation
          7. pos_floor ratchet
          8. geo_floor reset
          9. ramp_back cap
        """
        from srfm_core import MarketRegime as MR
        import numpy as np

        killed   = False
        geo_raw  = float(np.arctanh(np.clip(abs(geo_dev), 0.0, 0.9999)))

        # 1. Extreme geodesic deviation
        if geo_raw > self.GEO_KILL_THRESHOLD:
            tgt    = 0.0
            killed = True

        # 2. Warmup period
        if bc < 120:
            tgt    = 0.0
            killed = True

        # 3. tl_confirm gate
        tl_req = 1 if regime == MR.HIGH_VOLATILITY else 3
        if tl_confirm < tl_req:
            tgt    = 0.0
            killed = True
        elif bit == "SPACELIKE":
            # Spacelike penalty (not a full kill)
            tgt *= 0.50 if regime == MR.HIGH_VOLATILITY else 0.15

        # 4. Minimum signal size
        if 0.0 < abs(tgt) < self.MIN_SIGNAL_SIZE:
            tgt = 0.0

        # 5. Weak bar accumulation
        wb_thresh = 6 if regime == MR.HIGH_VOLATILITY else 3
        if abs(tgt) < self.WEAK_BAR_THRESHOLD:
            self.weak_bars += 1
            if self.weak_bars >= wb_thresh:
                tgt    = 0.0
                killed = True
        else:
            self.weak_bars = 0

        # 6. pos_floor ratchet: lock in 90% of strong signal
        if not killed and abs(tgt) > 0.5 and ctl >= 3:
            self.pos_floor = max(self.pos_floor, 0.90 * abs(tgt))

        if not killed and self.pos_floor > 0.0 and last_target != 0.0:
            tgt = float(np.sign(last_target) * max(abs(tgt), self.pos_floor))

        # 7. Reset pos_floor on crisis
        if geo_raw > self.GEO_FLOOR_RESET or killed:
            self.pos_floor = 0.0

        # 8. Ramp-back cap
        if ramp_back > 0:
            tgt = float(np.clip(tgt, -0.5, 0.5))

        # 9. Final minimum
        if abs(tgt) < self.MIN_POSITION_SIZE:
            return killed, (0.0 if abs(last_target) <= self.MIN_POSITION_SIZE else 0.0)

        return killed, tgt
