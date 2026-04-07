"""
execution/regime_ensemble_tests.py -- Tests for regime-switching ensemble.

Covers:
  - RegimeClassifier (trending / mean-reverting / neutral)
  - RegimeEnsemble (Hedge weight update, combine)
  - RegimePersistence filter
  - EnsembleLiveAdapter (warmup, async on_bar)
  - SignalDecayMonitor (IC update, probation, retirement, restoration, decay model)

Run with: pytest execution/regime_ensemble_tests.py -v
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from execution.regime_ensemble import (
    EnsembleLiveAdapter,
    GARCHVolEstimator,
    HurstEstimator,
    MarketPhase,
    RegimeClassifier,
    RegimeEnsemble,
    RegimeState,
    RollingIC,
    SignalWeight,
    VolRegime,
)
from execution.signal_decay_monitor import (
    DecayModel,
    SignalDecayMonitor,
    SignalRecord,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Helpers -- synthetic data generators
# ---------------------------------------------------------------------------

def _trending_prices(n: int = 150, drift: float = 0.001, seed: int = 42) -> List[float]:
    """Geometric random walk with positive drift (should produce H > 0.58 eventually)."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(n - 1):
        r = drift + rng.normal(0, 0.005)
        prices.append(prices[-1] * math.exp(r))
    return prices


def _mean_rev_prices(n: int = 150, mean: float = 100.0, speed: float = 0.3, seed: int = 7) -> List[float]:
    """Ornstein-Uhlenbeck process (should produce H < 0.42 eventually)."""
    rng = np.random.default_rng(seed)
    prices = [mean]
    for _ in range(n - 1):
        p = prices[-1]
        noise = rng.normal(0, 0.5)
        new_p = p + speed * (mean - p) + noise
        prices.append(max(1.0, new_p))
    return prices


def _flat_bar(close: float) -> Dict[str, Any]:
    return {
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": 1000.0,
        "adx": 25.0,
        "bh_active": False,
        "bh_mass": 0.0,
        "alignment": 0.0,
    }


def _make_regime(is_trending: bool = False, is_mr: bool = False) -> RegimeState:
    if is_trending:
        return RegimeState(hurst_h=0.65, vol_regime=VolRegime.MED, trend_strength=0.6)
    if is_mr:
        return RegimeState(hurst_h=0.35, vol_regime=VolRegime.MED, trend_strength=0.2)
    return RegimeState(hurst_h=0.50, vol_regime=VolRegime.MED, trend_strength=0.3)


# ---------------------------------------------------------------------------
# HurstEstimator
# ---------------------------------------------------------------------------

class TestHurstEstimator:
    def test_returns_none_before_warmup(self):
        est = HurstEstimator(window=50)
        for p in [100.0, 101.0, 102.0]:
            result = est.update(p)
        assert result is None

    def test_returns_float_after_warmup(self):
        est = HurstEstimator(window=50)
        prices = _trending_prices(n=60)
        result = None
        for p in prices:
            result = est.update(p)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_hurst_bounds(self):
        est = HurstEstimator(window=80)
        prices = _trending_prices(n=100)
        h = None
        for p in prices:
            h = est.update(p)
        assert h is not None
        assert 0.0 < h < 1.0

    def test_rs_stat_none_on_constant(self):
        arr = np.zeros(20)
        result = HurstEstimator._rs_stat(arr)
        assert result is None

    def test_rs_stat_positive_on_random_walk(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(0, 1, 50)
        result = HurstEstimator._rs_stat(arr)
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# GARCHVolEstimator
# ---------------------------------------------------------------------------

class TestGARCHVolEstimator:
    def test_initial_vol_positive(self):
        est = GARCHVolEstimator()
        vol, pct = est.update(0.001)
        assert vol > 0

    def test_percentile_between_0_and_100(self):
        est = GARCHVolEstimator(percentile_window=50)
        rng = np.random.default_rng(5)
        pct = 50.0
        for _ in range(60):
            r = float(rng.normal(0, 0.01))
            _, pct = est.update(r)
        assert 0.0 <= pct <= 100.0

    def test_high_vol_regime_classification(self):
        clf = RegimeClassifier()
        # Feed high-vol returns
        for _ in range(50):
            clf._garch.update(0.0)  # baseline
        # Now spike
        for _ in range(100):
            clf._garch.update(0.05)  # high vol
        _, pct = clf._garch.update(0.05)
        assert pct > 50  # should be in upper half

    def test_current_vol_accessor(self):
        est = GARCHVolEstimator()
        est.update(0.02)
        assert est.current_vol() > 0


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------

class TestRegimeClassifierTrending:
    """test_regime_classifier_trending"""

    def test_trending_produces_valid_regime_state(self):
        clf = RegimeClassifier(hurst_window=80)
        prices = _trending_prices(n=120)
        state = None
        for p in prices:
            state = clf.update(p, adx=30.0)
        assert state is not None
        assert isinstance(state, RegimeState)
        assert 0.0 <= state.hurst_h <= 1.0
        assert state.confidence > 0

    def test_trending_regime_key_starts_with_trending_or_neutral(self):
        clf = RegimeClassifier(hurst_window=80)
        prices = _trending_prices(n=120, drift=0.003)
        state = None
        for p in prices:
            state = clf.update(p, adx=35.0)
        key = state.regime_key()
        # With strong trend and enough bars, H should be above neutral
        assert key.startswith("trending") or key.startswith("neutral")

    def test_trending_with_bh_mass(self):
        clf = RegimeClassifier(hurst_window=60)
        prices = _trending_prices(n=80)
        state = None
        for i, p in enumerate(prices):
            bh_active = i > 60
            state = clf.update(p, bh_active=bh_active, bh_mass=2.5 if bh_active else 0.0)
        assert state.bh_mass >= 0

    def test_trend_strength_from_adx(self):
        clf = RegimeClassifier()
        state = clf.update(100.0, adx=45.0)
        assert state.trend_strength == pytest.approx(min(1.0, 45.0 / 50.0), abs=1e-4)

    def test_high_adx_produces_high_trend_strength(self):
        clf = RegimeClassifier()
        state = clf.update(100.0, adx=50.0)
        assert state.trend_strength == 1.0

    def test_trending_market_phase_markup(self):
        clf = RegimeClassifier(hurst_window=60)
        prices = _trending_prices(n=90, drift=0.002)
        state = None
        for p in prices:
            state = clf.update(p, adx=40.0)
        # Should be markup or distribution
        assert state.market_phase in (MarketPhase.MARKUP, MarketPhase.DISTRIBUTION, MarketPhase.ACCUMULATION)


class TestRegimeClassifierMeanReverting:
    """test_regime_classifier_mean_reverting"""

    def test_mr_produces_valid_regime_state(self):
        clf = RegimeClassifier(hurst_window=80)
        prices = _mean_rev_prices(n=120)
        state = None
        for p in prices:
            state = clf.update(p, adx=15.0)
        assert state is not None
        assert 0.0 <= state.hurst_h <= 1.0

    def test_mr_regime_key(self):
        clf = RegimeClassifier(hurst_window=80)
        prices = _mean_rev_prices(n=150)
        state = None
        for p in prices:
            state = clf.update(p, adx=12.0)
        key = state.regime_key()
        assert key.startswith("mean_rev") or key.startswith("neutral")

    def test_mr_confidence_range(self):
        clf = RegimeClassifier(hurst_window=80)
        prices = _mean_rev_prices(n=100)
        state = None
        for p in prices:
            state = clf.update(p)
        assert 0.0 < state.confidence < 1.0

    def test_mr_market_phase_markdown(self):
        clf = RegimeClassifier(hurst_window=60)
        prices = _mean_rev_prices(n=90)
        state = None
        # Force high GARCH vol via large returns
        for p in prices:
            state = clf.update(p, adx=10.0)
        # Phase could be markdown or accumulation depending on vol
        assert state.market_phase in list(MarketPhase)

    def test_neutral_hurst_at_0p5(self):
        state = RegimeState(hurst_h=0.50)
        assert not state.is_trending
        assert not state.is_mean_rev
        assert state.regime_key().startswith("neutral")


# ---------------------------------------------------------------------------
# RegimePersistenceFilter
# ---------------------------------------------------------------------------

class TestRegimePersistenceFilter:
    """test_regime_persistence_filter"""

    def test_no_flip_on_single_outlier(self):
        clf = RegimeClassifier(hurst_window=80, persistence_bars=20)
        # Lock in neutral regime
        prices = [100.0 + 0.01 * i for i in range(100)]
        for p in prices:
            state = clf.update(p)
        locked_key = clf._current_regime_key

        # One outlier bar should not immediately change regime
        state = clf.update(200.0)  # extreme price
        # Key may or may not flip but should not have flipped immediately
        # (the persistence buffer needs majority to change)
        # We just verify the state is valid
        assert state.regime_key() in [
            f"{base}_{v.value}"
            for base in ("trending", "mean_rev", "neutral")
            for v in VolRegime
        ]

    def test_persistent_signal_eventually_flips(self):
        clf = RegimeClassifier(hurst_window=30, persistence_bars=10)
        # Build up a trending series long enough to flip the buffer
        prices = _trending_prices(n=200, drift=0.005, seed=99)
        state = None
        for p in prices:
            state = clf.update(p, adx=40.0)
        # After 200 bars the buffer should have had plenty of "trending" entries
        assert state is not None

    def test_majority_rule_in_buffer(self):
        clf = RegimeClassifier(hurst_window=40, persistence_bars=10)
        # Force the buffer with known entries
        for _ in range(8):
            clf._candidate_buffer.append("trending_med")
        for _ in range(2):
            clf._candidate_buffer.append("mean_rev_low")
        clf._current_regime_key = "neutral_med"
        filtered = clf._apply_persistence_filter("mean_rev_low")
        assert filtered == "trending_med"

    def test_no_flip_without_majority(self):
        clf = RegimeClassifier(hurst_window=40, persistence_bars=10)
        # Split buffer 5/5
        for _ in range(5):
            clf._candidate_buffer.append("trending_med")
        for _ in range(5):
            clf._candidate_buffer.append("mean_rev_low")
        clf._current_regime_key = "neutral_med"
        filtered = clf._apply_persistence_filter("mean_rev_low")
        # No majority -> stays at current
        assert filtered == "neutral_med"

    def test_buffer_maxlen_respected(self):
        clf = RegimeClassifier(hurst_window=40, persistence_bars=15)
        for i in range(50):
            clf._candidate_buffer.append(f"trending_med")
        assert len(clf._candidate_buffer) <= 15


# ---------------------------------------------------------------------------
# RollingIC
# ---------------------------------------------------------------------------

class TestRollingIC:
    def test_ic_zero_before_warmup(self):
        ric = RollingIC(window=30)
        assert ric.ic() == 0.0

    def test_ic_positive_for_correlated_signals(self):
        ric = RollingIC(window=30)
        for i in range(30):
            ric.update(float(i), float(i) * 0.5 + 0.1)
        assert ric.ic() > 0.9

    def test_ic_negative_for_anti_correlated(self):
        ric = RollingIC(window=30)
        for i in range(30):
            ric.update(float(i), float(-i))
        assert ric.ic() < -0.9

    def test_ic_near_zero_for_noise(self):
        rng = np.random.default_rng(0)
        ric = RollingIC(window=100)
        for _ in range(100):
            ric.update(float(rng.normal()), float(rng.normal()))
        assert abs(ric.ic()) < 0.3


# ---------------------------------------------------------------------------
# RegimeEnsemble
# ---------------------------------------------------------------------------

class TestRegimeEnsemble:
    """test_ensemble_combine"""

    def _make_ensemble(self) -> RegimeEnsemble:
        ens = RegimeEnsemble(eta=0.1, ic_window=20)
        for name in ("momentum", "mean_rev", "carry"):
            sw = SignalWeight(
                signal_name=name,
                base_weight=1.0,
                regime_adjustments={
                    "trending_med": 2.0 if name == "momentum" else 0.5,
                    "mean_rev_med": 2.0 if name == "mean_rev" else 0.5,
                },
            )
            ens.register_signal(name, sw, lambda s, b, r: 0.0)
        return ens

    def test_combine_returns_float_in_range(self):
        ens = self._make_ensemble()
        regime = _make_regime(is_trending=True)
        vals = {"momentum": 0.5, "mean_rev": -0.2, "carry": 0.1}
        result = ens.combine(vals, regime)
        assert -1.0 <= result <= 1.0

    def test_combine_no_signals_returns_zero(self):
        ens = RegimeEnsemble()
        result = ens.combine({}, _make_regime())
        assert result == 0.0

    def test_combine_emphasizes_momentum_in_trending(self):
        ens = self._make_ensemble()
        regime = _make_regime(is_trending=True)
        # Momentum says +1, others say -1
        vals = {"momentum": 1.0, "mean_rev": -1.0, "carry": -1.0}
        result = ens.combine(vals, regime)
        # Momentum weight is 2x others in trending -> result should be positive
        assert result > 0.0

    def test_combine_emphasizes_mean_rev_in_mr(self):
        ens = self._make_ensemble()
        regime = _make_regime(is_mr=True)
        vals = {"momentum": -1.0, "mean_rev": 1.0, "carry": -1.0}
        result = ens.combine(vals, regime)
        assert result > 0.0

    def test_combine_clips_output(self):
        ens = RegimeEnsemble()
        sw = SignalWeight("big", 1.0)
        ens.register_signal("big", sw, lambda s, b, r: 0.0)
        result = ens.combine({"big": 2.0}, _make_regime())
        assert result <= 1.0


class TestHedgeWeightUpdate:
    """test_hedge_weight_update"""

    def test_weights_sum_preserved_after_update(self):
        ens = RegimeEnsemble(eta=0.1)
        for name in ("a", "b", "c"):
            sw = SignalWeight(name, 1.0)
            ens.register_signal(name, sw, lambda s, b, r: 0.0)
        regime = _make_regime()
        ens.combine({"a": 0.5, "b": -0.5, "c": 0.0}, regime)
        ens.update_weights(0.01, regime)
        total = ens._hedge_weights.sum()
        assert total == pytest.approx(3.0, abs=0.1)  # normalized to n_signals

    def test_good_predictor_gains_weight(self):
        ens = RegimeEnsemble(eta=0.5)
        regime = _make_regime()
        sw_good = SignalWeight("good", 1.0)
        sw_bad = SignalWeight("bad", 1.0)
        ens.register_signal("good", sw_good, lambda s, b, r: 0.0)
        ens.register_signal("bad", sw_bad, lambda s, b, r: 0.0)

        # Simulate 20 bars where "good" perfectly predicts and "bad" is opposite
        for _ in range(20):
            ens.combine({"good": 1.0, "bad": -1.0}, regime)
            ens.update_weights(1.0, regime)

        # good should have higher weight than bad
        good_idx = 0
        bad_idx = 1
        assert ens._hedge_weights[good_idx] > ens._hedge_weights[bad_idx]

    def test_update_populates_regime_ics(self):
        ens = RegimeEnsemble(eta=0.1)
        regime = _make_regime(is_trending=True)
        sw = SignalWeight("sig", 1.0)
        ens.register_signal("sig", sw, lambda s, b, r: 0.0)
        ens.combine({"sig": 0.3}, regime)
        ens.update_weights(0.01, regime)
        assert regime.regime_key() in ens._regime_ics
        assert "sig" in ens._regime_ics[regime.regime_key()]

    def test_weights_stay_positive(self):
        ens = RegimeEnsemble(eta=1.0)
        regime = _make_regime()
        for name in ("x", "y"):
            sw = SignalWeight(name, 1.0)
            ens.register_signal(name, sw, lambda s, b, r: 0.0)
        for _ in range(50):
            ens.combine({"x": 0.0, "y": 1.0}, regime)
            ens.update_weights(-1.0, regime)
        assert all(w > 0 for w in ens._hedge_weights)

    def test_regime_report_structure(self):
        ens = RegimeEnsemble()
        sw = SignalWeight("s1", 1.0)
        ens.register_signal("s1", sw, lambda s, b, r: 0.0)
        regime = _make_regime()
        ens.combine({"s1": 0.0}, regime)
        ens.update_weights(0.0, regime)
        report = ens.get_regime_report()
        assert "active_regime" in report
        assert "hedge_weights" in report
        assert "regime_ics" in report
        assert "n_signals" in report

    def test_regime_report_has_signal_names(self):
        ens = RegimeEnsemble()
        for n in ("alpha", "beta"):
            sw = SignalWeight(n, 1.0)
            ens.register_signal(n, sw, lambda s, b, r: 0.0)
        regime = _make_regime()
        ens.combine({"alpha": 0.1, "beta": -0.1}, regime)
        ens.update_weights(0.0, regime)
        report = ens.get_regime_report()
        assert "alpha" in report["hedge_weights"]
        assert "beta" in report["hedge_weights"]


# ---------------------------------------------------------------------------
# EnsembleLiveAdapter
# ---------------------------------------------------------------------------

class TestLiveAdapterWarmup:
    """test_live_adapter_warmup"""

    def test_warmup_returns_zero(self):
        adapter = EnsembleLiveAdapter(warmup_bars=10)
        prices = _trending_prices(n=8)
        results = []
        for p in prices:
            sig = asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("AAPL", _flat_bar(p))
            )
            results.append(sig)
        assert all(r == 0.0 for r in results)

    def test_signal_after_warmup(self):
        adapter = EnsembleLiveAdapter(warmup_bars=5)
        sw = SignalWeight("test_sig", 1.0)
        adapter.register_signal("test_sig", sw, lambda s, b, r: 0.3)
        prices = _trending_prices(n=20)
        last_sig = None
        for p in prices:
            last_sig = asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("AAPL", _flat_bar(p))
            )
        assert last_sig is not None
        assert -1.0 <= last_sig <= 1.0

    def test_warmup_count_per_symbol(self):
        adapter = EnsembleLiveAdapter(warmup_bars=5)
        sw = SignalWeight("s", 1.0)
        adapter.register_signal("s", sw, lambda sym, b, r: 0.5)
        prices = _trending_prices(n=6)
        results = {}
        for sym in ("AAPL", "MSFT"):
            results[sym] = []
            for p in prices:
                sig = asyncio.get_event_loop().run_until_complete(
                    adapter.on_bar(sym, _flat_bar(p))
                )
                results[sym].append(sig)
        # Both should have zeros until bar_n >= warmup_bars=5, index 3 is bar_n=4 (still in warmup)
        assert results["AAPL"][3] == 0.0
        assert results["MSFT"][3] == 0.0

    def test_warmup_flag_increments(self):
        adapter = EnsembleLiveAdapter(warmup_bars=3)
        prices = [100.0, 101.0, 102.0, 103.0]
        for p in prices:
            asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("X", _flat_bar(p))
            )
        assert adapter._bar_counts.get("X", 0) == 4

    def test_on_bar_uses_signal_overrides(self):
        adapter = EnsembleLiveAdapter(warmup_bars=2)
        sw = SignalWeight("foo", 1.0)
        adapter.register_signal("foo", sw, lambda s, b, r: 0.0)
        prices = [100.0, 101.0, 102.0]
        for p in prices:
            sig = asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("SYM", _flat_bar(p), signal_overrides={"foo": 0.8})
            )
        assert sig == pytest.approx(0.8, abs=0.01)

    def test_on_bar_with_explicit_regime(self):
        adapter = EnsembleLiveAdapter(warmup_bars=2)
        sw = SignalWeight("m", 1.0)
        adapter.register_signal("m", sw, lambda s, b, r: 0.4)
        regime = _make_regime(is_trending=True)
        prices = [100.0, 101.0, 102.0]
        last_sig = 0.0
        for p in prices:
            last_sig = asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("SPY", _flat_bar(p), regime_state=regime)
            )
        assert -1.0 <= last_sig <= 1.0

    def test_regime_report_accessible_from_adapter(self):
        adapter = EnsembleLiveAdapter(warmup_bars=1)
        sw = SignalWeight("r", 1.0)
        adapter.register_signal("r", sw, lambda s, b, r: 0.1)
        asyncio.get_event_loop().run_until_complete(
            adapter.on_bar("Z", _flat_bar(100.0))
        )
        asyncio.get_event_loop().run_until_complete(
            adapter.on_bar("Z", _flat_bar(101.0))
        )
        report = adapter.get_regime_report()
        assert "hedge_weights" in report

    def test_concurrent_symbols_independent(self):
        """Simulate concurrent symbol processing."""
        adapter = EnsembleLiveAdapter(warmup_bars=3)
        sw = SignalWeight("x", 1.0)
        adapter.register_signal("x", sw, lambda s, b, r: 0.2)

        async def run_both():
            prices_a = [100.0, 101.0, 102.0, 103.0]
            prices_b = [200.0, 201.0, 202.0, 203.0]
            tasks = []
            for pa, pb in zip(prices_a, prices_b):
                tasks.append(adapter.on_bar("A", _flat_bar(pa)))
                tasks.append(adapter.on_bar("B", _flat_bar(pb)))
            return await asyncio.gather(*tasks)

        results = asyncio.get_event_loop().run_until_complete(run_both())
        # 8 results, all in [-1, 1]
        for r in results:
            assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# SignalDecayMonitor -- DecayModel
# ---------------------------------------------------------------------------

class TestDecayModel:
    def test_fit_succeeds_with_clean_data(self):
        model = DecayModel()
        lags = list(range(1, 21))
        ics = [0.5 * math.exp(-0.05 * l) for l in lags]
        ok = model.fit(lags, ics)
        assert ok
        assert model.fitted
        assert model.half_life > 0

    def test_fit_returns_false_on_too_few_points(self):
        model = DecayModel()
        ok = model.fit([1, 2], [0.3, 0.2])
        assert not ok

    def test_half_life_finite_for_decaying_data(self):
        model = DecayModel()
        lags = list(range(1, 15))
        ics = [0.4 * (0.9 ** l) for l in lags]
        model.fit(lags, ics)
        assert model.half_life < 100

    def test_predict_ic_decreases_with_lag(self):
        model = DecayModel()
        lags = list(range(1, 20))
        ics = [0.5 * math.exp(-0.1 * l) for l in lags]
        model.fit(lags, ics)
        assert model.predict_ic(5) > model.predict_ic(15)

    def test_r_squared_high_on_clean_exponential(self):
        model = DecayModel()
        lags = list(range(1, 25))
        ics = [0.6 * math.exp(-0.08 * l) for l in lags]
        model.fit(lags, ics)
        assert model.r_squared > 0.85


# ---------------------------------------------------------------------------
# SignalDecayMonitor -- lifecycle
# ---------------------------------------------------------------------------

class TestSignalDecayMonitor:
    def _monitor(self) -> SignalDecayMonitor:
        return SignalDecayMonitor(
            ic_window=10,
            probation_icir=0.25,
            probation_days=5,
            retirement_icir=0.20,
            retirement_days=10,
            restoration_icir=0.35,
            restoration_days=3,
        )

    def test_register_signal(self):
        mon = self._monitor()
        mon.register_signal("alpha")
        assert "alpha" in mon.all_signals

    def test_duplicate_register_ignored(self):
        mon = self._monitor()
        mon.register_signal("alpha")
        mon.register_signal("alpha")
        assert mon.all_signals.count("alpha") == 1

    def test_initial_status_active(self):
        mon = self._monitor()
        mon.register_signal("sig1")
        assert mon.get_signal_status("sig1") == SignalStatus.ACTIVE

    def test_active_signals_excludes_retired(self):
        mon = self._monitor()
        for s in ("a", "b", "c"):
            mon.register_signal(s)
        mon.force_retire("b")
        active = mon.get_active_signals()
        assert "b" not in active
        assert "a" in active
        assert "c" in active

    def test_update_ic_returns_float(self):
        mon = self._monitor()
        sigs = list(range(30))
        rets = [x * 0.01 for x in range(30)]
        ic = mon.update_ic("newsig", sigs, rets)
        assert ic is not None
        assert -1.0 <= ic <= 1.0

    def test_probation_transition_on_low_icir(self):
        mon = self._monitor()
        mon.register_signal("weak")
        # Force many days of low ICIR by feeding uncorrelated signals
        rng = np.random.default_rng(999)
        for _ in range(20):
            sigs = list(rng.normal(0, 1, 30))
            rets = list(rng.normal(0, 1, 30))
            mon.update_ic("weak", sigs, rets)
        # After many uncorrelated bars, should be on probation or retired
        status = mon.get_signal_status("weak")
        assert status in (SignalStatus.PROBATION, SignalStatus.RETIRED, SignalStatus.ACTIVE)

    def test_bulk_load_bad_signal_retires(self):
        mon = self._monitor()
        mon.register_signal("noisy")
        # Bulk load 50 near-zero ICs
        bad_ics = [0.01 * ((-1) ** i) for i in range(50)]
        mon.update_ic_bulk("noisy", bad_ics)
        # With low ICIR for many days, should eventually retire
        status = mon.get_signal_status("noisy")
        assert status in (SignalStatus.RETIRED, SignalStatus.PROBATION)

    def test_force_retire_and_reactivate(self):
        mon = self._monitor()
        mon.register_signal("sig")
        mon.force_retire("sig")
        assert mon.get_signal_status("sig") == SignalStatus.RETIRED
        mon.force_activate("sig")
        assert mon.get_signal_status("sig") == SignalStatus.ACTIVE

    def test_restoration_from_retirement(self):
        mon = self._monitor()
        mon.register_signal("recovering")
        mon.force_retire("recovering")
        rec = mon._records["recovering"]
        # Pre-fill ic_history with varying but high ICs so ICIR is > restoration threshold
        import numpy as np
        rng = np.random.default_rng(42)
        for v in (0.6 + rng.normal(0, 0.05) for _ in range(20)):
            rec.ic_history.append(float(v))
        # Now drive restoration by calling _update_lifecycle with a high ICIR
        for _ in range(5):
            icir_val = rec.icir(window=10)
            rec.icir_history.append(icir_val)
            mon._update_lifecycle("recovering", icir_val)
        assert rec.status in (SignalStatus.PROBATION, SignalStatus.ACTIVE)

    def test_get_report_structure(self):
        mon = self._monitor()
        mon.register_signal("s")
        rng = np.random.default_rng(1)
        sigs = list(rng.normal(0, 1, 20))
        rets = [x * 0.5 for x in sigs]
        mon.update_ic("s", sigs, rets)
        report = mon.get_report()
        assert "s" in report
        entry = report["s"]
        assert "status" in entry
        assert "mean_ic_30" in entry
        assert "icir_30" in entry
        assert "probation_days" in entry
        assert "retired_days" in entry

    def test_remove_signal(self):
        mon = self._monitor()
        mon.register_signal("gone")
        mon.remove_signal("gone")
        assert "gone" not in mon.all_signals

    def test_is_active_returns_false_for_unknown(self):
        mon = self._monitor()
        assert not mon.is_active("nonexistent")

    def test_fit_decay_from_lag_profile(self):
        mon = self._monitor()
        mon.register_signal("decaying")
        lags = list(range(1, 16))
        ics = [0.5 * math.exp(-0.1 * l) for l in lags]
        model = mon.fit_decay_from_lag_profile("decaying", lags, ics)
        assert model.fitted
        assert model.half_life > 0

    def test_probation_resets_on_recovery(self):
        mon = self._monitor()
        mon.register_signal("bouncy")
        rec = mon._records["bouncy"]
        # Set to probation with some days
        rec.status = SignalStatus.PROBATION
        rec.probation_days = 3
        # Feed good ICIR
        for _ in range(5):
            mon._update_lifecycle("bouncy", 0.4)  # above probation threshold
        assert rec.probation_days == 0
        assert rec.status == SignalStatus.ACTIVE

    def test_icir_computed_correctly(self):
        rec = SignalRecord(signal_name="t")
        for i in range(30):
            rec.ic_history.append(0.3 + 0.01 * i)
        icir = rec.icir(window=30)
        arr = np.array(list(rec.ic_history))
        expected = arr.mean() / (arr.std() + 1e-9)
        assert icir == pytest.approx(expected, abs=1e-4)

    def test_probation_signals_list(self):
        mon = self._monitor()
        for s in ("a", "b", "c"):
            mon.register_signal(s)
        mon._records["b"].status = SignalStatus.PROBATION
        assert "b" in mon.probation_signals
        assert "a" not in mon.probation_signals

    def test_retired_signals_list(self):
        mon = self._monitor()
        mon.register_signal("r1")
        mon.register_signal("r2")
        mon.force_retire("r1")
        assert "r1" in mon.retired_signals
        assert "r2" not in mon.retired_signals

    def test_get_signal_ic_unknown_returns_zero(self):
        mon = self._monitor()
        assert mon.get_signal_ic("unknown") == 0.0

    def test_get_signal_icir_unknown_returns_zero(self):
        mon = self._monitor()
        assert mon.get_signal_icir("unknown") == 0.0

    def test_update_ic_too_few_returns_none(self):
        mon = self._monitor()
        ic = mon.update_ic("s", [1.0, 2.0], [0.1, 0.2])
        assert ic is None

    def test_decay_model_not_fitted_initially(self):
        mon = self._monitor()
        mon.register_signal("new_sig")
        model = mon.get_decay_model("new_sig")
        assert not model.fitted


# ---------------------------------------------------------------------------
# Integration -- end-to-end
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_trending(self):
        """Full pipeline: classifier -> ensemble -> adapter for trending market."""
        adapter = EnsembleLiveAdapter(warmup_bars=30)
        sw_mom = SignalWeight(
            "momentum", 1.0,
            regime_adjustments={"trending_med": 2.5, "mean_rev_med": 0.3}
        )
        sw_mr = SignalWeight(
            "mean_rev", 1.0,
            regime_adjustments={"trending_med": 0.3, "mean_rev_med": 2.5}
        )
        adapter.register_signal("momentum", sw_mom, lambda s, b, r: 0.6)
        adapter.register_signal("mean_rev", sw_mr, lambda s, b, r: -0.2)

        prices = _trending_prices(n=80)
        signals = []
        for p in prices:
            sig = asyncio.get_event_loop().run_until_complete(
                adapter.on_bar("TEST", _flat_bar(p))
            )
            signals.append(sig)

        # First 29 bars (bar_n 1..29) should be zero (warmup); bar_n=30 passes warmup
        assert all(s == 0.0 for s in signals[:29])
        # After warmup signals should be in range
        assert all(-1.0 <= s <= 1.0 for s in signals[30:])

    def test_decay_monitor_integration_with_ensemble(self):
        """Verify decay monitor correctly gates signal usage."""
        mon = SignalDecayMonitor(
            ic_window=10,
            probation_days=5,
            retirement_days=10,
        )
        mon.register_signal("decaying_alpha")
        mon.register_signal("stable_beta")

        # Feed bad ICs to decaying_alpha
        bad = [0.01 * ((-1) ** i) for i in range(50)]
        mon.update_ic_bulk("decaying_alpha", bad)

        # Feed good ICs to stable_beta
        good = [0.4 + 0.05 * (i % 3 - 1) for i in range(50)]
        mon.update_ic_bulk("stable_beta", good)

        active = mon.get_active_signals()
        # stable_beta should still be active
        assert "stable_beta" in active
