"""
Tests for lib/regime.py — RegimeDetector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import pytest
from srfm_core import MarketRegime
from regime import RegimeDetector


class TestRegimeDetector:

    def _make_bull(self, atr_ratio=1.0, adx=30.0):
        """Return args that should give BULL regime."""
        base = 5000.0
        return dict(
            price=base, ema12=base+20, ema26=base+15,
            ema50=base+10, ema200=base-100,
            adx=adx, atr=10.0,
        )

    def _make_bear(self, adx=30.0):
        base = 5000.0
        return dict(
            price=base, ema12=base-20, ema26=base-15,
            ema50=base-10, ema200=base+100,
            adx=adx, atr=10.0,
        )

    def test_bull_full_stack(self):
        """Full EMA bull stack + ADX > 14 → BULL."""
        rd = RegimeDetector()
        # Warm ATR so atr_ratio < 1.5
        for _ in range(5):
            rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 30.0, 10.0)
        regime, conf = rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 30.0, 10.0)
        assert regime == MarketRegime.BULL

    def test_bear_full_stack(self):
        """Full EMA bear stack + ADX > 14 → BEAR."""
        rd = RegimeDetector()
        for _ in range(5):
            rd.update(5000.0, 4980.0, 4985.0, 4990.0, 5100.0, 30.0, 10.0)
        regime, _ = rd.update(5000.0, 4980.0, 4985.0, 4990.0, 5100.0, 30.0, 10.0)
        assert regime == MarketRegime.BEAR

    def test_high_volatility_from_atr_spike(self):
        """atr_ratio >= 1.5 → HIGH_VOLATILITY."""
        rd = RegimeDetector()
        # Seed with low ATR
        for _ in range(10):
            rd.update(5000.0, 5010.0, 5005.0, 5000.0, 4900.0, 20.0, 10.0)
        # Now spike ATR by 2×
        regime, conf = rd.update(5000.0, 5010.0, 5005.0, 5000.0, 4900.0, 20.0, 20.0)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_sideways_low_adx(self):
        """Bull-ish structure but ADX < 14 → SIDEWAYS."""
        rd = RegimeDetector()
        for _ in range(5):
            rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 10.0, 10.0)
        regime, _ = rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 10.0, 10.0)
        assert regime == MarketRegime.SIDEWAYS

    def test_confidence_bounded(self):
        rd = RegimeDetector()
        for _ in range(10):
            _, conf = rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 30.0, 10.0)
            assert 0.0 <= conf <= 1.0

    def test_bars_in_regime_counter(self):
        rd = RegimeDetector()
        for i in range(5):
            rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 30.0, 10.0)
        assert rd.bars_in_regime >= 0

    def test_is_trending(self):
        rd = RegimeDetector()
        for _ in range(5):
            rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 30.0, 10.0)
        assert rd.is_trending

    def test_is_crisis(self):
        rd = RegimeDetector()
        for _ in range(10):
            rd.update(5000.0, 5010.0, 5005.0, 5000.0, 4900.0, 20.0, 10.0)
        rd.update(5000.0, 5010.0, 5005.0, 5000.0, 4900.0, 20.0, 25.0)
        assert rd.is_crisis

    def test_regime_confidence_high_adx_bull(self):
        """ADX=60 → confidence near 0.95."""
        rd = RegimeDetector()
        for _ in range(5):
            rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 60.0, 10.0)
        regime, conf = rd.update(5000.0, 5020.0, 5015.0, 5010.0, 4900.0, 60.0, 10.0)
        assert regime == MarketRegime.BULL
        assert conf == pytest.approx(0.95, abs=0.05)

    def test_atr_window_default(self):
        rd = RegimeDetector(atr_window=50)
        assert rd._atr_hist.maxlen == 50
