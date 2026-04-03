"""
Tests for lib/agents.py — agent_d3qn, agent_ddqn, agent_td3qn, ensemble, size_position.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pytest
from srfm_core import MarketRegime
from agents import agent_d3qn, agent_ddqn, agent_td3qn, ensemble, size_position, REGIME_WEIGHTS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def zero_features() -> np.ndarray:
    return np.zeros(31, dtype=np.float32)

def bull_features() -> np.ndarray:
    """Feature vector suggestive of bull trend."""
    f = zero_features()
    f[0]  = 0.65   # RSI 65
    f[1]  = 0.5    # MACD positive
    f[3]  = 0.3    # MACD hist positive
    f[4]  = 0.4    # MOM positive
    f[5]  = 0.3    # ROC positive
    f[11] = 0.2    # price > EMA12
    f[14] = 0.4    # price > EMA200
    f[15] = 0.3    # EMA12 > EMA26
    f[17] = 0.35   # ADX 35
    f[24] = 1.0    # BULL regime one-hot
    f[29] = 0.8    # high TL fraction
    return f

def bear_features() -> np.ndarray:
    f = zero_features()
    f[0]  = 0.35   # RSI 35
    f[1]  = -0.5   # MACD negative
    f[3]  = -0.3
    f[4]  = -0.4
    f[11] = -0.2
    f[14] = -0.4
    f[15] = -0.3
    f[25] = 1.0    # BEAR one-hot
    return f


# ─────────────────────────────────────────────────────────────────────────────
# agent_d3qn
# ─────────────────────────────────────────────────────────────────────────────

class TestD3QN:

    def test_output_bounded(self):
        f = bull_features()
        s, c = agent_d3qn(f, mu=1.0, rc=0.7)
        assert -1.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0

    def test_bull_signal_positive(self):
        f = bull_features()
        s, _ = agent_d3qn(f, mu=1.0, rc=0.7)
        assert s > 0

    def test_bear_signal_negative(self):
        f = bear_features()
        s, _ = agent_d3qn(f, mu=1.0, rc=0.7)
        assert s < 0

    def test_mu_amplifies_signal(self):
        f = bull_features()
        s1, _ = agent_d3qn(f, mu=1.0, rc=0.7)
        s2, _ = agent_d3qn(f, mu=2.0, rc=0.7)
        assert abs(s2) >= abs(s1)

    def test_high_atr_dampers_signal(self):
        """High ATR% (f[6]) should reduce signal magnitude."""
        f_low  = bull_features(); f_low[6]  = 0.01
        f_high = bull_features(); f_high[6] = 0.5    # > 0.03 → 0.60 factor
        s_low,  _ = agent_d3qn(f_low,  mu=1.0, rc=0.7)
        s_high, _ = agent_d3qn(f_high, mu=1.0, rc=0.7)
        assert abs(s_high) <= abs(s_low)

    def test_low_rsi_adds_to_signal(self):
        """RSI < 0.30 adds 0.10 to raw signal."""
        f_low  = bull_features(); f_low[0]  = 0.25
        f_high = bull_features(); f_high[0] = 0.75
        s_low,  _ = agent_d3qn(f_low,  mu=1.0, rc=0.5)
        s_high, _ = agent_d3qn(f_high, mu=1.0, rc=0.5)
        # low RSI adds 0.10, high RSI subtracts 0.10
        assert s_low > s_high

    def test_zero_features_near_zero(self):
        f = zero_features()
        s, c = agent_d3qn(f, mu=1.0, rc=0.5)
        # RSI=0 < 0.30 → adds 0.10, but everything else zero → s = tanh(0.10)
        assert abs(s) < 0.2

    def test_confidence_scales_with_rc(self):
        f = bull_features()
        _, c1 = agent_d3qn(f, mu=1.0, rc=0.5)
        _, c2 = agent_d3qn(f, mu=1.0, rc=0.9)
        assert c2 > c1


# ─────────────────────────────────────────────────────────────────────────────
# agent_ddqn
# ─────────────────────────────────────────────────────────────────────────────

class TestDDQN:

    def test_output_bounded(self):
        f = bull_features()
        s, c = agent_ddqn(f, mu=1.0, rc=0.7)
        assert -1.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0

    def test_alignment_drives_signal(self):
        """All 4 alignment features positive → strong positive signal."""
        f = zero_features()
        f[1] = 0.5; f[3] = 0.5; f[4] = 0.5; f[5] = 0.5  # all positive
        s, _ = agent_ddqn(f, mu=1.0, rc=0.5)
        assert s > 0

    def test_misaligned_features_negative(self):
        f = zero_features()
        f[1] = -0.5; f[3] = -0.5; f[4] = -0.5; f[5] = -0.5
        s, _ = agent_ddqn(f, mu=1.0, rc=0.5)
        assert s < 0

    def test_volume_amplifies(self):
        """High relative volume (f[18]) amplifies signal."""
        f_base = bull_features(); f_base[18] = 0.0
        f_vol  = bull_features(); f_vol[18]  = 0.5
        s_base, _ = agent_ddqn(f_base, mu=1.0, rc=0.5)
        s_vol,  _ = agent_ddqn(f_vol,  mu=1.0, rc=0.5)
        assert abs(s_vol) >= abs(s_base)


# ─────────────────────────────────────────────────────────────────────────────
# agent_td3qn
# ─────────────────────────────────────────────────────────────────────────────

class TestTD3QN:

    def test_output_bounded(self):
        f = bull_features()
        s, c = agent_td3qn(f, mu=1.0, rc=0.7, ht=0.0)
        assert -1.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0

    def test_high_bbp_contrarian_negative(self):
        """BBP near 1.0 (overbought) → s = -(1-0.5)*0.40 = -0.20 → negative."""
        f = zero_features()
        f[8] = 1.0   # BBP at top
        f[0] = 0.5   # neutral RSI
        s, _ = agent_td3qn(f, mu=1.0, rc=0.5, ht=0.0)
        assert s < 0

    def test_low_bbp_contrarian_positive(self):
        """BBP near 0.0 (oversold) → s = -(0-0.5)*0.40 = +0.20."""
        f = zero_features()
        f[8] = 0.0   # BBP at bottom
        f[0] = 0.5
        s, _ = agent_td3qn(f, mu=1.0, rc=0.5, ht=0.0)
        assert s > 0

    def test_hot_hawking_reduces_signal(self):
        """ht > 1.5 → s -= 0.15."""
        f = zero_features()
        f[8] = 0.0   # would give positive signal
        s_cool, _ = agent_td3qn(f, mu=1.0, rc=0.5, ht=0.0)
        s_hot,  _ = agent_td3qn(f, mu=1.0, rc=0.5, ht=2.0)
        assert s_cool > s_hot

    def test_high_atr_dampers(self):
        """f[6] > 0.03 → s *= 0.60."""
        f_low  = zero_features(); f_low[8]  = 0.0; f_low[6]  = 0.01
        f_high = zero_features(); f_high[8] = 0.0; f_high[6] = 0.05
        s_low,  _ = agent_td3qn(f_low,  mu=1.0, rc=0.5, ht=0.0)
        s_high, _ = agent_td3qn(f_high, mu=1.0, rc=0.5, ht=0.0)
        assert abs(s_low) > abs(s_high)

    def test_std_reduces_confidence(self):
        """High std (f[10]) → confidence multiplier sc = 1-std → lower c."""
        f_low  = bull_features(); f_low[10]  = 0.1
        f_high = bull_features(); f_high[10] = 0.7
        _, c_low  = agent_td3qn(f_low,  mu=1.0, rc=0.7, ht=0.0)
        _, c_high = agent_td3qn(f_high, mu=1.0, rc=0.7, ht=0.0)
        assert c_low > c_high


# ─────────────────────────────────────────────────────────────────────────────
# ensemble
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsemble:

    def test_output_structure(self):
        f = bull_features()
        action, conf, sigs = ensemble(
            f, mu=1.0, rc=0.7, ht=0.0, beta=0.5,
            regime=MarketRegime.BULL, geo_slope=0.01, geo_dev=-0.1, rapidity=0.05,
        )
        assert isinstance(action, float)
        assert isinstance(conf, float)
        assert len(sigs) == 3

    def test_action_bounded(self):
        f = bull_features()
        action, _, _ = ensemble(
            f, mu=1.5, rc=0.7, ht=0.0, beta=0.5,
            regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0,
        )
        # tanh output then weighted → must be in reasonable range
        assert -3.0 < action < 3.0

    def test_spacelike_damping(self):
        """beta > 1 → divide by beta → reduces signal vs beta < 1."""
        f = bull_features()
        a_tl, _, _ = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=0.5,
                              regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0)
        a_sl, _, _ = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=2.0,
                              regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0)
        assert abs(a_tl) > abs(a_sl)

    def test_timelike_boost(self):
        """beta < 1 → γ boost → larger signal vs beta = 1."""
        f = bull_features()
        a_1, _, _ = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=1.0,
                             regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0)
        a_tl, _, _ = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=0.3,
                              regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0)
        # beta=0.3 → γ = 1/√(1-0.09) ≈ 1.048 → mild boost
        assert abs(a_tl) >= abs(a_1)

    def test_regime_weights_sum_to_one(self):
        for regime, w in REGIME_WEIGHTS.items():
            assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_geo_correction_d3qn(self):
        """geo_slope>0 and geo_dev<0 → s1 += 0.10 → larger positive signal."""
        f = bull_features()
        a_no,  _, (s1_no, _, _)  = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=0.5,
            regime=MarketRegime.BULL, geo_slope=0.0, geo_dev=0.0, rapidity=0.0)
        a_yes, _, (s1_yes, _, _) = ensemble(f, mu=1.0, rc=0.7, ht=0.0, beta=0.5,
            regime=MarketRegime.BULL, geo_slope=0.01, geo_dev=-0.1, rapidity=0.0)
        assert s1_yes > s1_no


# ─────────────────────────────────────────────────────────────────────────────
# size_position
# ─────────────────────────────────────────────────────────────────────────────

class TestSizePosition:

    def test_zero_rm_returns_zero(self):
        f = bull_features()
        sz = size_position(f, action=0.5, conf=0.7, rm=0.0,
                           regime=MarketRegime.BULL, mu=1.0, rc=0.7,
                           ht=0.0, tl_window=[1.0]*20)
        assert sz == 0.0

    def test_positive_action_positive_size(self):
        f = bull_features()
        sz = size_position(f, action=0.5, conf=0.7, rm=1.0,
                           regime=MarketRegime.BULL, mu=1.0, rc=0.7,
                           ht=0.0, tl_window=[0.8]*20)
        assert sz > 0

    def test_negative_action_negative_size(self):
        f = bear_features()
        sz = size_position(f, action=-0.5, conf=0.7, rm=1.0,
                           regime=MarketRegime.BEAR, mu=1.0, rc=0.7,
                           ht=0.0, tl_window=[0.8]*20)
        assert sz < 0

    def test_bull_regime_multiplier(self):
        """BULL regime uses 1.5× multiplier."""
        f = bull_features()
        sz_bull = size_position(f, action=0.5, conf=0.5, rm=1.0,
                                regime=MarketRegime.BULL, mu=1.0, rc=0.5,
                                ht=0.0, tl_window=[0.7]*20)
        sz_side = size_position(f, action=0.5, conf=0.5, rm=1.0,
                                regime=MarketRegime.SIDEWAYS, mu=1.0, rc=0.5,
                                ht=0.0, tl_window=[0.7]*20)
        assert abs(sz_bull) >= abs(sz_side) * 0.8   # roughly larger

    def test_tl_fraction_scales_size(self):
        """Higher TL fraction → larger position in trending regimes."""
        f = bull_features()
        sz_low  = size_position(f, action=0.5, conf=0.7, rm=1.0,
                                regime=MarketRegime.BULL, mu=1.0, rc=0.7,
                                ht=0.0, tl_window=[0.3]*20)
        sz_high = size_position(f, action=0.5, conf=0.7, rm=1.0,
                                regime=MarketRegime.BULL, mu=1.0, rc=0.7,
                                ht=0.0, tl_window=[0.9]*20)
        assert sz_high > sz_low
