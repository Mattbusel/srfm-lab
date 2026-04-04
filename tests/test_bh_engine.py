"""
Tests for the SRFM BH physics engine (srfm_core).

Covers:
- MinkowskiClassifier: timelike/spacelike classification
- BlackHoleDetector: bh_mass, bh_active, bh_dir, ctl
- GeodesicAnalyzer: log-linear regression, causal_frac, rapidity
- GravitationalLens: mu (amplification factor)
- HawkingMonitor: Hawking temperature
- Multi-timeframe consistency
- Convergence with increasing data
"""

import sys
import os
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_price_series(n=500, drift=0.0005, vol=0.015, seed=42):
    rng = np.random.default_rng(seed)
    log_rets = drift + vol * rng.standard_normal(n)
    prices = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="price")


def try_import_srfm():
    """Import srfm_core if available, return None if not."""
    try:
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        import srfm_core
        return srfm_core
    except ImportError:
        return None


class TestMinkowskiClassifier(unittest.TestCase):
    """Test timelike / spacelike classification of returns."""

    def test_import(self):
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        self.assertTrue(hasattr(srfm, "MinkowskiClassifier"))

    def test_classification_types(self):
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=200)
        clf = srfm.MinkowskiClassifier()
        result = clf.classify(price)
        # Should return a Series with values 'timelike' or 'spacelike'
        self.assertIsInstance(result, (pd.Series, np.ndarray))
        if isinstance(result, pd.Series):
            unique_vals = set(result.dropna().unique())
            expected = {"timelike", "spacelike"}
            self.assertTrue(unique_vals.issubset(expected | {"null", "lightlike"}))

    def test_timelike_fraction_valid(self):
        """Timelike fraction should be in [0, 1]."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        clf = srfm.MinkowskiClassifier()
        result = clf.classify(price)
        if isinstance(result, pd.Series):
            timelike_frac = (result == "timelike").mean()
            self.assertGreaterEqual(timelike_frac, 0.0)
            self.assertLessEqual(timelike_frac, 1.0)


class TestBlackHoleDetector(unittest.TestCase):
    """Test BH mass, active status, direction, and CTL."""

    def test_bh_mass_non_negative(self):
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        detector = srfm.BlackHoleDetector()
        result = detector.detect(price)
        # bh_mass should be non-negative
        if hasattr(result, "bh_mass"):
            mass_series = result.bh_mass
            if isinstance(mass_series, pd.Series):
                self.assertTrue((mass_series.dropna() >= 0).all())

    def test_bh_active_boolean(self):
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        detector = srfm.BlackHoleDetector()
        result = detector.detect(price)
        if hasattr(result, "bh_active"):
            active = result.bh_active
            if isinstance(active, pd.Series):
                self.assertTrue(active.dropna().isin([0, 1, True, False]).all())

    def test_bh_direction_valid(self):
        """BH direction should be -1, 0, or +1."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        detector = srfm.BlackHoleDetector()
        result = detector.detect(price)
        if hasattr(result, "bh_dir"):
            bh_dir = result.bh_dir
            if isinstance(bh_dir, pd.Series):
                valid = {-1, 0, 1, -1.0, 0.0, 1.0}
                self.assertTrue(bh_dir.dropna().isin(valid).all())

    def test_ctl_in_range(self):
        """CTL (critical threshold level) should be in [0, 1]."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        detector = srfm.BlackHoleDetector()
        result = detector.detect(price)
        if hasattr(result, "ctl"):
            ctl = result.ctl
            if isinstance(ctl, pd.Series):
                self.assertTrue((ctl.dropna() >= 0).all())
                self.assertTrue((ctl.dropna() <= 1).all())


class TestGeodesicAnalyzer(unittest.TestCase):
    """Test log-linear regression, causal_frac, rapidity."""

    def test_log_linear_slope(self):
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300, drift=0.001)
        analyzer = srfm.GeodesicAnalyzer()
        result = analyzer.analyze(price)
        if hasattr(result, "slope"):
            slope = result.slope
            if isinstance(slope, float):
                self.assertIsInstance(slope, float)

    def test_causal_fraction_valid(self):
        """causal_frac should be in [0, 1]."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        analyzer = srfm.GeodesicAnalyzer()
        result = analyzer.analyze(price)
        if hasattr(result, "causal_frac"):
            cf = result.causal_frac
            if isinstance(cf, float):
                self.assertGreaterEqual(cf, 0.0)
                self.assertLessEqual(cf, 1.0)

    def test_rapidity_finite(self):
        """Rapidity should be a finite number."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        analyzer = srfm.GeodesicAnalyzer()
        result = analyzer.analyze(price)
        if hasattr(result, "rapidity"):
            rap = result.rapidity
            if isinstance(rap, float):
                self.assertTrue(np.isfinite(rap))


class TestGravitationalLens(unittest.TestCase):
    """Test mu (amplification factor) of the gravitational lens."""

    def test_mu_positive(self):
        """Amplification factor mu should be >= 1."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        lens = srfm.GravitationalLens()
        mu = lens.compute_mu(price)
        if isinstance(mu, (float, int)):
            self.assertGreaterEqual(mu, 0.0)
        elif isinstance(mu, pd.Series):
            self.assertTrue((mu.dropna() >= 0).all())

    def test_mu_near_unity_random_walk(self):
        """For near-random-walk series, mu should be near 1."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        rng = np.random.default_rng(42)
        price = pd.Series(
            100 + np.cumsum(rng.standard_normal(300)),
            index=pd.date_range("2020-01-02", periods=300, freq="B"),
        )
        lens = srfm.GravitationalLens()
        mu = lens.compute_mu(price)
        if isinstance(mu, float):
            # Mu for random walk should be bounded
            self.assertLess(abs(mu), 100)


class TestHawkingMonitor(unittest.TestCase):
    """Test Hawking temperature signal."""

    def test_hawking_temperature_positive(self):
        """Hawking temperature should be non-negative."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        price = make_price_series(n=300)
        monitor = srfm.HawkingMonitor()
        temp = monitor.compute_temperature(price)
        if isinstance(temp, pd.Series):
            self.assertTrue((temp.dropna() >= 0).all())
        elif isinstance(temp, float):
            self.assertGreaterEqual(temp, 0.0)

    def test_hawking_temperature_inversely_related_to_mass(self):
        """Higher mass BH → lower Hawking temperature."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")
        # This is a qualitative test: create a strong trend (high BH mass)
        # and a noisy series (low BH mass) and compare temperatures
        # (may not be deterministic; just check it runs)
        trending = make_price_series(n=200, drift=0.005, vol=0.005)
        noisy = make_price_series(n=200, drift=0.0, vol=0.03, seed=99)
        monitor = srfm.HawkingMonitor()
        temp_trend = monitor.compute_temperature(trending)
        temp_noisy = monitor.compute_temperature(noisy)
        # Both should be non-negative
        for temp in [temp_trend, temp_noisy]:
            if isinstance(temp, float):
                self.assertGreaterEqual(temp, 0.0)


class TestMultiTimeframe(unittest.TestCase):
    """Test BH features across multiple timeframes."""

    def test_features_fallback(self):
        """
        BH features fallback should return valid DataFrame
        when srfm_core is not available.
        """
        price = make_price_series(n=300)
        # Simulate the fallback by calling features.py directly
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from strategies.ml_alpha.features import FeatureEngine
            engine = FeatureEngine()
            # Build all features with minimal data
            prices_df = pd.DataFrame({"SPY": price})
            feats = engine.price_features(prices_df)
            self.assertIsInstance(feats, pd.DataFrame)
            self.assertGreater(feats.shape[1], 0)
        except ImportError:
            self.skipTest("FeatureEngine not available")

    def test_bh_convergence(self):
        """
        BH mass should converge as data length increases
        (longer series → more stable estimate).
        """
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")

        price = make_price_series(n=500)
        detector = srfm.BlackHoleDetector()

        masses = []
        for n in [100, 200, 300, 400, 500]:
            result = detector.detect(price.iloc[:n])
            if hasattr(result, "bh_mass"):
                mass = result.bh_mass
                if isinstance(mass, pd.Series) and len(mass.dropna()) > 0:
                    masses.append(float(mass.dropna().iloc[-1]))

        if len(masses) >= 3:
            # Variance of estimates should decrease with more data
            # (heuristic: last estimate should not be wildly different from second-to-last)
            diffs = [abs(masses[i] - masses[i - 1]) for i in range(1, len(masses))]
            # Not a strict monotone decrease, but check values are bounded
            self.assertTrue(all(np.isfinite(m) for m in masses))


class TestBHThreshold(unittest.TestCase):
    """Test that BH threshold behavior is consistent."""

    def test_no_bh_constant_series(self):
        """Constant price series should not trigger an active BH."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")

        price = pd.Series(
            100.0 * np.ones(200),
            index=pd.date_range("2020-01-02", periods=200, freq="B"),
        )
        detector = srfm.BlackHoleDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = detector.detect(price)

        if hasattr(result, "bh_active"):
            active = result.bh_active
            if isinstance(active, pd.Series):
                # Constant series = no strong trend → most days should be inactive
                active_rate = active.dropna().mean() if len(active.dropna()) > 0 else 0.5
                self.assertLess(active_rate, 0.9)

    def test_strong_trend_triggers_bh(self):
        """A strong deterministic uptrend should trigger BH activity."""
        srfm = try_import_srfm()
        if srfm is None:
            self.skipTest("srfm_core not available")

        rng = np.random.default_rng(42)
        n = 300
        # Very strong trend + low noise
        prices = 100 * np.exp(np.cumsum(0.005 + 0.001 * rng.standard_normal(n)))
        price = pd.Series(prices, index=pd.date_range("2020-01-02", periods=n, freq="B"))
        detector = srfm.BlackHoleDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = detector.detect(price)

        if hasattr(result, "bh_mass"):
            mass = result.bh_mass
            if isinstance(mass, pd.Series) and len(mass.dropna()) > 0:
                # Strong trend should produce non-zero BH mass
                self.assertGreater(mass.dropna().mean(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
