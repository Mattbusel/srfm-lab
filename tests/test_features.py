"""
Tests for lib/features.py — compute_features() and index constants.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pytest
from features import compute_features, FEATURE_NAMES, F_RSI, F_MACD, F_D12, F_EX, F_BBP, F_BULL, F_TLF
from srfm_core import MarketRegime


def _base_inputs(**overrides):
    """Sensible default inputs for a healthy bull bar."""
    args = dict(
        close=5000.0,
        rsi=60.0,
        macd_val=5.0,
        macd_sig=4.0,
        macd_hist=1.0,
        mom=20.0,
        roc=0.5,
        atr=25.0,
        bb_upper=5050.0,
        bb_lower=4950.0,
        bb_middle=5000.0,
        std=25.0,
        ema12=4990.0,
        ema26=4980.0,
        ema50=4970.0,
        ema200=4800.0,
        adx=30.0,
        volume_window=[1000.0] * 20,
        open_window=[4998.0, 4996.0, 4994.0, 4992.0, 4990.0],
        close_window=[5000.0, 4990.0, 4980.0, 4970.0, 4960.0,
                      4950.0, 4940.0, 4930.0, 4920.0, 4910.0],
        tl_window=[1.0] * 20,
        beta_window=[0.5] * 5,
        bit="TIMELIKE",
        regime=MarketRegime.BULL,
        ht=0.2,
    )
    args.update(overrides)
    return args


class TestComputeFeatures:

    def test_returns_array(self):
        f = compute_features(**_base_inputs())
        assert isinstance(f, np.ndarray)

    def test_length_31(self):
        f = compute_features(**_base_inputs())
        assert len(f) == 31

    def test_dtype_float32(self):
        f = compute_features(**_base_inputs())
        assert f.dtype == np.float32

    def test_all_clipped_to_3(self):
        f = compute_features(**_base_inputs())
        assert np.all(f >= -3.0)
        assert np.all(f <= 3.0)

    def test_rsi_index(self):
        """f[0] = rsi/100."""
        f = compute_features(**_base_inputs(rsi=70.0))
        assert f[F_RSI] == pytest.approx(0.70, abs=1e-4)

    def test_rsi_index_boundary(self):
        f = compute_features(**_base_inputs(rsi=100.0))
        assert f[F_RSI] == pytest.approx(1.0, abs=1e-4)

    def test_bbp_clipped_0_to_1(self):
        """f[8] = bbp clipped to [0,1]."""
        f = compute_features(**_base_inputs())
        assert 0.0 <= f[F_BBP] <= 1.0

    def test_bbp_mid_price(self):
        """Price at BB middle → BBP = 0.5."""
        f = compute_features(**_base_inputs(close=5000.0, bb_upper=5050.0, bb_lower=4950.0, bb_middle=5000.0))
        assert f[F_BBP] == pytest.approx(0.5, abs=1e-4)

    def test_regime_onehot_bull(self):
        f = compute_features(**_base_inputs(regime=MarketRegime.BULL))
        assert f[F_BULL] == 1.0
        assert f[25] == 0.0   # BEAR
        assert f[26] == 0.0   # SIDEWAYS

    def test_regime_onehot_bear(self):
        f = compute_features(**_base_inputs(regime=MarketRegime.BEAR))
        assert f[25] == 1.0
        assert f[F_BULL] == 0.0

    def test_spacelike_reduces_macd_weight(self):
        """cw=0.3 for SPACELIKE → MACD features smaller than TIMELIKE."""
        f_tl = compute_features(**_base_inputs(bit="TIMELIKE"))
        f_sl = compute_features(**_base_inputs(bit="SPACELIKE"))
        assert abs(f_tl[F_MACD]) > abs(f_sl[F_MACD])

    def test_tlf_index(self):
        """f[29] = mean of tl_window."""
        f = compute_features(**_base_inputs(tl_window=[1.0]*10 + [0.0]*10))
        assert f[F_TLF] == pytest.approx(0.5, abs=1e-4)

    def test_d12_positive_when_price_above_ema(self):
        """price > ema12 → d12 > 0."""
        f = compute_features(**_base_inputs(close=5100.0, ema12=5000.0))
        assert f[F_D12] > 0

    def test_ex_positive_when_ema12_above_ema26(self):
        f = compute_features(**_base_inputs(ema12=5010.0, ema26=4990.0))
        assert f[F_EX] > 0

    def test_returns_none_on_zero_close(self):
        f = compute_features(**_base_inputs(close=0.0))
        assert f is None

    def test_returns_none_on_zero_atr(self):
        f = compute_features(**_base_inputs(atr=0.0))
        assert f is None

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 31

    def test_no_nan_in_output(self):
        f = compute_features(**_base_inputs())
        assert not np.any(np.isnan(f))

    def test_no_inf_in_output(self):
        f = compute_features(**_base_inputs())
        assert not np.any(np.isinf(f))

    def test_short_windows_graceful(self):
        """Short close_window → r3 and r10 default to 0 gracefully."""
        f = compute_features(**_base_inputs(close_window=[5000.0, 4990.0]))   # only 2 bars
        assert f is not None
        assert not np.any(np.isnan(f))

    def test_empty_tl_window(self):
        """Empty tl_window → tlf defaults to 0.5."""
        f = compute_features(**_base_inputs(tl_window=[]))
        assert f[F_TLF] == pytest.approx(0.5, abs=1e-4)
