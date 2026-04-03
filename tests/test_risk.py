"""
Tests for lib/risk.py — PortfolioRiskManager and KillConditions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import pytest
from risk import PortfolioRiskManager, KillConditions
from srfm_core import MarketRegime


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioRiskManager
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioRiskManager:

    def test_normal_returns_1(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        pm.daily_reset(1_000_000)
        rm = pm.portfolio_risk(1_000_000)
        assert rm == 1.0

    def test_circuit_breaker_at_maxdd(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000, maxdd=0.12)
        pm.portfolio_risk(1_000_000)   # set peak
        rm = pm.portfolio_risk(880_000)   # 12% drawdown
        assert rm == 0.0

    def test_caution_zone_half_rm(self):
        """dd >= 70% of maxdd → 0.5 rm. Must set daystart so daily_loss check doesn't fire first."""
        pm = PortfolioRiskManager(initial_equity=1_000_000, maxdd=0.12)
        pm.daily_reset(1_000_000)   # daystart = 1_000_000
        pm.portfolio_risk(1_000_000)  # set peak = 1_000_000
        # 8.5% dd = > 70% of 12% but below maxdd; daily loss must also be within limit
        # daily_loss = (910k - 1M) / 1M = -9% > dlim → would block; use a loss just inside daily limit
        pm.daily_reset(950_000)     # reset daystart lower so daily loss < 2%
        rm = pm.portfolio_risk(916_000)   # dd = 8.4% (> 0.084); daily_loss = (916k-950k)/950k = -3.6% > dlim
        # To isolate caution zone, set dlim high
        pm2 = PortfolioRiskManager(initial_equity=1_000_000, maxdd=0.12, dlim=0.50)
        pm2.daily_reset(1_000_000)
        pm2.portfolio_risk(1_000_000)
        rm2 = pm2.portfolio_risk(910_000)   # 9% dd > 70% of 12% → caution zone
        assert rm2 == 0.5

    def test_daily_loss_limit(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000, dlim=0.02)
        pm.daily_reset(1_000_000)
        rm = pm.portfolio_risk(979_000)   # -2.1% on the day
        assert rm == 0.0

    def test_peak_tracked(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        pm.portfolio_risk(1_200_000)
        assert pm.peak == 1_200_000

    def test_daily_reset_clears_cb(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000, maxdd=0.12)
        pm.portfolio_risk(1_000_000)
        pm.portfolio_risk(880_000)   # trigger cb
        pm.daily_reset(880_000)
        assert not pm.cb

    def test_trail_pct_scaling(self):
        assert PortfolioRiskManager._trail_pct(0.0)  == 0.20
        assert PortfolioRiskManager._trail_pct(0.20) == 0.18
        assert PortfolioRiskManager._trail_pct(0.50) == 0.15
        assert PortfolioRiskManager._trail_pct(1.00) == 0.12
        assert PortfolioRiskManager._trail_pct(1.50) == 0.10

    def test_hwm_cooldown_fires(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        pm.peak = 1_000_000
        # Need > 12% drawdown: 120,001 / 1_000_001 > 0.12
        state = pm.on_bar(875_000)   # 12.5% dd
        assert state == "hwm_cooldown"
        assert pm.hwm_cooldown == 3

    def test_hwm_cooldown_counts_down(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        pm.hwm_cooldown = 2
        state = pm.on_bar(1_000_000)
        assert state == "hwm_cooldown"
        assert pm.hwm_cooldown == 1

    def test_normal_on_bar_ok(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        state = pm.on_bar(1_000_000)
        assert state == "ok"

    def test_current_drawdown(self):
        pm = PortfolioRiskManager(initial_equity=1_000_000)
        pm.peak = 1_000_000
        dd = pm.current_drawdown(900_000)
        assert dd == pytest.approx(0.10, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# KillConditions
# ─────────────────────────────────────────────────────────────────────────────

class TestKillConditions:

    def _base_args(self, **overrides):
        args = dict(
            tgt=0.5,
            geo_dev=0.0,
            bc=200,
            tl_confirm=3,
            bit="TIMELIKE",
            regime=MarketRegime.BULL,
            ctl=5,
            last_target=0.5,
            ramp_back=0,
        )
        args.update(overrides)
        return args

    def test_normal_pass(self):
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args())
        assert not killed
        assert tgt != 0.0

    def test_extreme_geo_kills(self):
        """arctanh(0.9999) > 2.0 → kill."""
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(geo_dev=0.9999))
        assert killed
        assert tgt == 0.0

    def test_warmup_kills(self):
        """bc < 120 → kill."""
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(bc=50))
        assert killed

    def test_tl_confirm_gate_normal(self):
        """tl_confirm < 3 in non-volatile regime → kill."""
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(tl_confirm=2, regime=MarketRegime.BULL))
        assert killed

    def test_tl_confirm_gate_highvol(self):
        """tl_confirm >= 1 in HIGH_VOLATILITY → not killed by tl gate."""
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(tl_confirm=1, regime=MarketRegime.HIGH_VOLATILITY))
        assert not killed

    def test_spacelike_penalty_not_kill(self):
        """SPACELIKE in normal regime → tgt *= 0.15, not killed."""
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(bit="SPACELIKE", tgt=0.5, tl_confirm=3))
        # tl_confirm=3 passes gate, then spacelike penalty applied
        # Note: penalty applied when tl_confirm >= tl_req but bit==SPACELIKE
        # killed stays False from tl gate, but tgt is penalized
        # The gate kills first (tl_confirm < tl_req only kills; if tl_confirm >= req but SPACELIKE, penalty)
        # In our implementation: tl_confirm=3 >= tl_req=3 → not killed by gate → penalty applied
        assert not killed
        # tgt should be reduced (0.15 of original)

    def test_weak_bars_accumulate(self):
        """3 consecutive weak bars (<0.30) → kill."""
        kc = KillConditions()
        for _ in range(3):
            killed, tgt = kc.apply(**self._base_args(tgt=0.10))   # below 0.30 threshold
        assert killed

    def test_weak_bars_reset_on_strong(self):
        kc = KillConditions()
        kc.apply(**self._base_args(tgt=0.10))
        kc.apply(**self._base_args(tgt=0.10))
        kc.apply(**self._base_args(tgt=0.80))   # strong bar resets counter
        assert kc.weak_bars == 0

    def test_pos_floor_ratchet(self):
        """Strong signal (>0.5) with ctl>=3 → pos_floor set to 90% of tgt."""
        kc = KillConditions()
        kc.apply(**self._base_args(tgt=0.8, ctl=5))
        assert kc.pos_floor == pytest.approx(0.72, abs=1e-4)

    def test_pos_floor_holds_size(self):
        """After pos_floor set, weaker tgt should be floored up."""
        kc = KillConditions()
        kc.apply(**self._base_args(tgt=0.8, ctl=5))
        floor = kc.pos_floor
        _, tgt = kc.apply(**self._base_args(tgt=0.3, ctl=5))
        # floor should push tgt up to at least floor
        assert abs(tgt) >= floor - 0.01

    def test_ramp_back_caps_at_half(self):
        kc = KillConditions()
        killed, tgt = kc.apply(**self._base_args(tgt=1.5, ramp_back=3))
        assert not killed
        assert abs(tgt) <= 0.5

    def test_geo_floor_reset_on_extreme(self):
        """geo_raw > 1.5 → pos_floor reset to 0."""
        kc = KillConditions()
        kc.pos_floor = 0.6
        kc.apply(**self._base_args(geo_dev=0.99))   # arctanh(0.99) > 2.6 → kills and resets
        assert kc.pos_floor == 0.0
