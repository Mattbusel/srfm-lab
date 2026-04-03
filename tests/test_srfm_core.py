"""
Tests for lib/srfm_core.py — MinkowskiClassifier, BlackHoleDetector,
GeodesicAnalyzer, GravitationalLens, HawkingMonitor.

All formulas verified against LARSA production code.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import math
import numpy as np
import pytest
from srfm_core import (
    MinkowskiClassifier, BlackHoleDetector,
    GeodesicAnalyzer, GravitationalLens, HawkingMonitor,
    MarketRegime,
)


# ─────────────────────────────────────────────────────────────────────────────
# MinkowskiClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestMinkowskiClassifier:

    def test_initial_state(self):
        mc = MinkowskiClassifier(cf=0.001)
        assert mc.bit == "UNKNOWN"
        assert mc.tl_confirm == 0
        assert mc.proper_time == 0.0

    def test_first_bar_no_classification(self):
        mc  = MinkowskiClassifier(cf=0.001)
        bit = mc.update(5000.0)
        assert bit == "UNKNOWN"          # no prev close yet

    def test_timelike_small_move(self):
        """A 0.05% move with cf=0.001 → beta=0.5 → TIMELIKE."""
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        bit = mc.update(5002.5)         # 0.05% move → beta = 0.5
        assert bit == "TIMELIKE"
        assert mc.beta == pytest.approx(0.5, abs=1e-6)

    def test_spacelike_large_move(self):
        """A 0.2% move with cf=0.001 → beta=2.0 → SPACELIKE."""
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        bit = mc.update(5010.0)         # 0.2% move → beta = 2.0
        assert bit == "SPACELIKE"
        assert mc.beta == pytest.approx(2.0, abs=1e-6)

    def test_tl_confirm_increments(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        mc.update(5002.0)   # TIMELIKE
        assert mc.tl_confirm == 1
        mc.update(5004.0)   # TIMELIKE
        assert mc.tl_confirm == 2
        mc.update(5006.0)   # TIMELIKE
        assert mc.tl_confirm == 3

    def test_tl_confirm_caps_at_3(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        for _ in range(10):
            mc.update(mc._prev_close + 1.0)    # small TIMELIKE moves
        assert mc.tl_confirm == 3

    def test_tl_confirm_resets_on_spacelike(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        mc.update(5002.0)   # TL → confirm=1
        mc.update(5004.0)   # TL → confirm=2
        mc.update(5100.0)   # BIG move → SPACELIKE → confirm=0
        assert mc.tl_confirm == 0

    def test_proper_time_increases(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0)
        mc.update(5002.0)
        pt1 = mc.proper_time
        mc.update(5004.0)
        pt2 = mc.proper_time
        assert pt2 > pt1

    def test_proper_time_smaller_for_faster_move(self):
        """Faster bars (higher v) → lower 1/γ → proper_time grows slower."""
        mc_slow  = MinkowskiClassifier(cf=0.001, max_vol=0.01)
        mc_fast  = MinkowskiClassifier(cf=0.001, max_vol=0.01)
        mc_slow.update(5000.0); mc_slow.update(5001.0)   # tiny move
        mc_fast.update(5000.0); mc_fast.update(5005.0)   # larger move
        # faster bar → larger v → larger γ → smaller 1/γ
        assert mc_slow.proper_time > mc_fast.proper_time

    def test_proper_time_reset(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0); mc.update(5002.0)
        assert mc.proper_time > 0
        mc.reset_proper_time()
        assert mc.proper_time == 0.0

    def test_is_timelike_property(self):
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(5000.0); mc.update(5001.0)
        assert mc.is_timelike is True

    def test_nan_guard_prev_zero(self):
        """If prev_close is 0, should not divide by zero."""
        mc = MinkowskiClassifier(cf=0.001)
        mc.update(0.0)
        bit = mc.update(5000.0)
        assert bit in ("TIMELIKE", "SPACELIKE", "UNKNOWN")

    def test_cf_sensitivity(self):
        """Higher cf → easier to be TIMELIKE (same move is sub-luminal)."""
        mc_low  = MinkowskiClassifier(cf=0.0005)
        mc_high = MinkowskiClassifier(cf=0.002)
        price   = 5000.0
        move    = 5001.0   # 0.02% move
        mc_low.update(price);  bit_low  = mc_low.update(move)
        mc_high.update(price); bit_high = mc_high.update(move)
        # low cf → small threshold → same move more likely SPACELIKE
        # high cf → large threshold → same move more likely TIMELIKE
        assert mc_low.beta > mc_high.beta


# ─────────────────────────────────────────────────────────────────────────────
# BlackHoleDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackHoleDetector:

    def _build_active_bh(self) -> tuple:
        """Helper: create a BH in ACTIVE state via 6 TIMELIKE bars."""
        bh = BlackHoleDetector(bh_form=1.5, bh_collapse=1.0, bh_decay=0.95)
        prices = [5000.0]
        for i in range(1, 8):
            prices.append(prices[-1] + 5.0)    # small steady gains → TIMELIKE
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        return bh, prices

    def test_initial_state(self):
        bh = BlackHoleDetector()
        assert not bh.bh_active
        assert bh.bh_mass == 0.0
        assert bh.ctl == 0

    def test_timelike_accretes_mass(self):
        bh = BlackHoleDetector()
        bh.update("TIMELIKE", 5010.0, 5000.0)
        assert bh.bh_mass > 0.0
        assert bh.ctl == 1

    def test_spacelike_decays_mass(self):
        bh = BlackHoleDetector()
        # First add some mass
        bh.update("TIMELIKE", 5010.0, 5000.0)
        m_before = bh.bh_mass
        bh.update("SPACELIKE", 5200.0, 5010.0)
        assert bh.bh_mass < m_before * 0.71    # 0.7 decay
        assert bh.ctl == 0

    def test_bh_forms_after_threshold(self):
        """After enough TIMELIKE bars, BH should become active."""
        bh = BlackHoleDetector(bh_form=0.5, bh_collapse=0.2, bh_decay=0.95)
        prices = [5000.0 + i * 5 for i in range(10)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.bh_active

    def test_bh_requires_ctl_gte_5(self):
        """BH won't form until ctl >= 5 even if mass is high."""
        bh = BlackHoleDetector(bh_form=0.001, bh_collapse=0.0, bh_decay=0.99)
        # Only 4 TIMELIKE bars
        prices = [5000.0 + i * 50 for i in range(5)]
        for i in range(1, 5):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.ctl == 4
        assert not bh.bh_active

    def test_direction_positive_from_cumulative(self):
        bh = BlackHoleDetector(bh_form=0.5, bh_collapse=0.2, bh_decay=0.95)
        prices = [5000.0 + i * 10 for i in range(10)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.bh_dir == 1

    def test_direction_negative(self):
        bh = BlackHoleDetector(bh_form=0.5, bh_collapse=0.2, bh_decay=0.95)
        prices = [5000.0 - i * 10 for i in range(10)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.bh_dir == -1

    def test_collapse_when_mass_drops(self):
        bh = BlackHoleDetector(bh_form=1.0, bh_collapse=0.5, bh_decay=0.95)
        # Build up mass
        prices = [5000.0 + i * 10 for i in range(15)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.bh_active
        # Now decay with spacelike
        for _ in range(20):
            bh.update("SPACELIKE", prices[-1], prices[-1])
        assert not bh.bh_active

    def test_reform_memory_boost(self):
        """After collapse, prev_bh_mass is saved on the first SPACELIKE."""
        bh = BlackHoleDetector(bh_form=1.0, bh_collapse=0.5, bh_decay=0.95)
        prices = [5000.0 + i * 10 for i in range(15)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        assert bh.bh_active, "BH should be active after sustained TIMELIKE run"
        # First SPACELIKE: ctl→0 → bh collapses → prev_bh_mass saved
        bh.update("SPACELIKE", prices[-1], prices[-1])
        assert not bh.bh_active
        # prev_bh_mass is set on the transition bar itself
        assert bh.prev_bh_mass > 0.0
        # reform_bars is set to 1 then immediately incremented to 2 within the same update
        assert bh.reform_bars > 0

    def test_sb_scaling(self):
        """Sustained TIMELIKE run → sb grows → mass accretes faster."""
        bh1 = BlackHoleDetector()
        bh2 = BlackHoleDetector()
        # 1 TIMELIKE bar
        bh1.update("TIMELIKE", 5010.0, 5000.0)
        # 10 TIMELIKE bars consecutively
        prices = [5000.0 + i * 10 for i in range(11)]
        for i in range(1, len(prices)):
            bh2.update("TIMELIKE", prices[i], prices[i-1])
        # bh2 should have much higher mass relative to its return
        assert bh2.bh_mass > bh1.bh_mass

    def test_reset(self):
        bh = BlackHoleDetector(bh_form=0.5)
        prices = [5000.0 + i * 10 for i in range(10)]
        for i in range(1, len(prices)):
            bh.update("TIMELIKE", prices[i], prices[i-1])
        bh.reset()
        assert bh.bh_mass == 0.0
        assert not bh.bh_active
        assert bh.ctl == 0


# ─────────────────────────────────────────────────────────────────────────────
# GeodesicAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestGeodesicAnalyzer:

    def _feed_trend(self, ga, start=5000.0, step=10.0, n=25):
        """Feed a clean uptrend into the analyzer."""
        prices = [start + i * step for i in range(n)]
        last = (0.0, 0.0, 1.0, 0.0)
        for p in prices:
            last = ga.update(p, atr=10.0)
        return last

    def test_not_ready_below_window(self):
        ga = GeodesicAnalyzer(window=20)
        for i in range(19):
            result = ga.update(5000.0 + i, atr=10.0)
        assert result == (0.0, 0.0, 1.0, 0.0)

    def test_ready_at_window(self):
        ga = GeodesicAnalyzer(window=20)
        assert not ga.ready
        for i in range(20):
            ga.update(5000.0 + i, atr=10.0)
        assert ga.ready

    def test_positive_slope_uptrend(self):
        ga = GeodesicAnalyzer(window=20)
        _, slope, _, _ = self._feed_trend(ga, step=10.0)
        assert slope > 0

    def test_negative_slope_downtrend(self):
        ga = GeodesicAnalyzer(window=20)
        _, slope, _, _ = self._feed_trend(ga, step=-10.0)
        assert slope < 0

    def test_geo_dev_bounded(self):
        """geo_dev = tanh(x) so must be in (-1, 1)."""
        ga = GeodesicAnalyzer(window=20)
        dev, _, _, _ = self._feed_trend(ga)
        assert -1.0 < dev < 1.0

    def test_causal_frac_bounded(self):
        ga = GeodesicAnalyzer(window=20, cf=0.001)
        _, _, cf, _ = self._feed_trend(ga)
        assert 0.0 <= cf <= 1.0

    def test_causal_frac_high_for_slow_trend(self):
        """Very slow trend → small price moves → mostly TIMELIKE intervals."""
        ga = GeodesicAnalyzer(window=20, cf=0.001)
        _, _, cf, _ = self._feed_trend(ga, step=0.5)    # tiny step → sub-luminal
        assert cf > 0.5

    def test_causal_frac_low_for_fast_trend(self):
        """Very fast price moves → SPACELIKE intervals → low causal fraction."""
        ga = GeodesicAnalyzer(window=20, cf=0.001)
        _, _, cf, _ = self._feed_trend(ga, step=100.0)  # huge step → super-luminal
        assert cf < 0.5

    def test_flat_price_zero_slope(self):
        ga = GeodesicAnalyzer(window=20)
        for _ in range(25):
            ga.update(5000.0, atr=10.0)
        _, slope, _, _ = ga.update(5000.0, atr=10.0)
        assert abs(slope) < 1e-6

    def test_atr_zero_guard(self):
        """atr=0 should not raise ZeroDivisionError."""
        ga = GeodesicAnalyzer(window=20)
        for i in range(25):
            ga.update(5000.0 + i, atr=0.0)   # should not crash

    def test_window_deque_length(self):
        ga = GeodesicAnalyzer(window=20)
        for i in range(50):
            ga.update(5000.0 + i, atr=10.0)
        assert len(ga._closes) == 20


# ─────────────────────────────────────────────────────────────────────────────
# GravitationalLens
# ─────────────────────────────────────────────────────────────────────────────

class TestGravitationalLens:

    def test_mu_small_ctl(self):
        """ctl=0, M<2 → mu = max(0.3, M/3.0) = 0.3."""
        gl = GravitationalLens()
        mu = gl.update(5000.0, 1000.0, "TIMELIKE", ctl=0, atr=10.0)
        # M = 0+1=1, mu = max(0.3, 1/3) = 0.333
        assert mu == pytest.approx(1/3.0, abs=1e-4)

    def test_mu_low_m(self):
        """M=1 → mu = max(0.3, 1/3) ≈ 0.333."""
        gl = GravitationalLens()
        mu = gl.update(5000.0, 1000.0, "TIMELIKE", ctl=0, atr=10.0)
        assert 0.3 <= mu <= 0.5

    def test_mu_increases_near_vwap(self):
        """Price very close to VWAP → r small → R_E/(r+R_E) → 1 → mu → 2."""
        gl = GravitationalLens()
        # Build up TIMELIKE history at price ~5000
        for _ in range(5):
            gl.update(5000.0, 100.0, "TIMELIKE", ctl=5, atr=10.0)
        # Feed same price → r ≈ 0 → mu → 2
        mu = gl.update(5000.0, 100.0, "TIMELIKE", ctl=5, atr=10.0)
        assert mu > 1.5

    def test_mu_decreases_far_from_vwap(self):
        """Price far from VWAP → r large → mu → 1."""
        gl = GravitationalLens()
        for _ in range(5):
            gl.update(5000.0, 100.0, "TIMELIKE", ctl=5, atr=10.0)
        mu = gl.update(50000.0, 100.0, "TIMELIKE", ctl=5, atr=10.0)
        # mu = 1 + R_E/(r+R_E), r very large → approaches 1
        assert 1.0 <= mu < 1.5

    def test_mu_always_positive(self):
        gl = GravitationalLens()
        for i in range(20):
            mu = gl.update(5000.0 + i * 10, float(i * 100), "TIMELIKE", ctl=i, atr=15.0)
            assert mu > 0


# ─────────────────────────────────────────────────────────────────────────────
# HawkingMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestHawkingMonitor:

    def test_initial_state(self):
        hm = HawkingMonitor()
        assert hm.ht == 0.0

    def test_zero_std_no_crash(self):
        hm = HawkingMonitor()
        ht = hm.update(5000.0, 5000.0, 0.0)
        assert ht == 0.0

    def test_ht_increases_on_price_spike(self):
        hm = HawkingMonitor()
        hm.update(5000.0, 5000.0, 50.0)   # z=0, ht=0
        ht = hm.update(5200.0, 5000.0, 50.0)  # z=4, ht = 4*(4-0)=16
        assert ht > 0

    def test_is_hot(self):
        hm = HawkingMonitor()
        hm.update(5000.0, 5000.0, 10.0)
        hm.update(5200.0, 5000.0, 10.0)   # large z → hot
        assert hm.is_hot

    def test_is_inverted(self):
        hm = HawkingMonitor()
        hm.update(5100.0, 5000.0, 50.0)   # z = 2
        ht = hm.update(4900.0, 5000.0, 50.0)  # z = -2, ht = (-2)*(-2-2) = 8? no: ht = z*(z-prev_z)
        # z=-2, prev_z=2, ht = -2*(-2-2) = -2*(-4) = 8 → not inverted
        # Test genuine inverted: z goes from high to higher (second derivative negative)
        hm2 = HawkingMonitor()
        hm2.update(5000.0, 5050.0, 50.0)   # z = -1
        hm2.update(4900.0, 5050.0, 50.0)   # z = -3, ht = -3*(-3-(-1)) = -3*(-2) = 6
        # We need ht < -1.5; let's engineer it:
        hm3 = HawkingMonitor()
        hm3._pz = 3.0
        hm3.update(4850.0, 5000.0, 50.0)  # z = -3, ht = -3*(-3-3) = 18 (not inverted)
        hm4 = HawkingMonitor()
        hm4._pz = -1.0
        ht = hm4.update(5000.0 - 50*2, 5000.0, 50.0)  # z=-2, ht=-2*(-2-(-1))=-2*(-1)=2
        # Direct test: set pz such that ht < -1.5
        hm5 = HawkingMonitor()
        hm5._pz = 2.0
        ht = hm5.update(4900.0, 5000.0, 50.0)   # z=(4900-5000)/50=-2, ht=-2*(-2-2)=8 → not inverted
        # ht inverted when z moves from positive to less positive (or more negative)
        # z*(z-prev_z) < 0 when z and (z-prev_z) have opposite signs
        hm6 = HawkingMonitor()
        hm6._pz = -3.0
        ht = hm6.update(4950.0, 5000.0, 50.0)  # z=-1, ht=-1*(-1-(-3))=-1*(2)=-2 → IS inverted
        assert hm6.is_inverted

    def test_pz_updates(self):
        hm = HawkingMonitor()
        hm.update(5100.0, 5000.0, 50.0)  # z=2
        assert hm._pz == pytest.approx(2.0, abs=1e-6)
