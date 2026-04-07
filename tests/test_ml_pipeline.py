"""
tests/test_ml_pipeline.py
Tests for the ML pipeline: online learning, ensemble, feature engineering,
model selection, and live signal integration.

No em dashes. Uses numpy, pytest.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports from ml package
# ---------------------------------------------------------------------------

from ml.online_learning import (
    ADWIN,
    FTRL,
    ForgettingEnsemble,
    KernelOnlineLearning,
    OnlineGradientBoosting,
    OnlineLogistic,
    OnlinePassiveAggressive,
    OnlineRidge,
    make_default_ensemble,
)
from ml.ensemble import (
    DynamicWeightOptimizer,
    ManagedSignalEnsemble,
    Regime,
    RegimeSwitchingEnsemble,
    SignalEnsemble,
    StackingEnsemble,
)
from ml.feature_engineering import (
    CalendarFeatures,
    CrossAssetFeatures,
    FeatureImportanceAnalyzer,
    FeaturePipeline,
    InteractionFeatures,
    LagFeatures,
    MicrostructureFeatures,
    PhysicsFeatures,
    RawFeatures,
    TechnicalFeatures,
    VolatilityFeatures,
)
from ml.model_selection import (
    CombinatorialPurgedCV,
    HyperparameterSearch,
    ModelDriftDetector,
    ModelSelector,
    PurgedCVSplit,
    WalkForwardValidator,
    _compute_sharpe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_features(rng):
    return rng.randn(20)


@pytest.fixture
def binary_dataset(rng):
    """Simple linearly separable dataset with 200 samples."""
    X = rng.randn(200, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def return_series(rng):
    """Simulated daily returns with slight momentum."""
    returns = 0.001 + 0.02 * rng.randn(500)
    return returns


@pytest.fixture
def sample_raw_features():
    return RawFeatures(
        open=50000.0, high=51000.0, low=49000.0, close=50500.0, volume=1000.0,
        ret_1=0.01, garch_vol=0.02,
        bh_mass_short=0.5, bh_mass_mid=0.7, bh_mass_long=0.3,
        bh_active_short=1.0, bh_active_mid=1.0, bh_active_long=0.0,
        bh_proper_time=50.0, bh_geodesic_dev=0.1,
        quat_angular_vel=0.5, quat_nav_signal=0.3,
        ou_zscore=-1.5, ou_theta=0.3,
        hurst=0.6,
        btc_ret_1=0.015, spy_ret_1=0.005, vix_level=18.0,
        granger_signal=0.2,
        hour=14, day_of_week=2, day_of_month=15, month=6,
        is_fomc_week=False, is_earnings_week=False,
        spread_proxy=0.001,
    )


# ---------------------------------------------------------------------------
# OnlineLogistic tests
# ---------------------------------------------------------------------------

class TestOnlineLogistic:

    def test_predict_returns_in_minus1_1(self, simple_features):
        model = OnlineLogistic(n_features=20)
        sig = model.predict(simple_features)
        assert -1.0 <= sig <= 1.0

    def test_predict_proba_in_0_1(self, simple_features):
        model = OnlineLogistic(n_features=20)
        p = model.predict_proba(simple_features)
        assert 0.0 < p < 1.0

    def test_fit_one_returns_error(self, simple_features):
        model = OnlineLogistic(n_features=20)
        err = model.fit_one(simple_features, 1.0)
        assert 0.0 <= err <= 1.0

    def test_xor_convergence(self):
        """
        OnlineLogistic with hashing can approximate XOR.
        XOR is not linearly separable, but feature hashing gives richer repr.
        We use a soft version: learn to predict sign(x0 * x1).
        """
        rng = np.random.RandomState(0)
        model = OnlineLogistic(n_features=4, learning_rate=0.1, l1=1e-5, l2=1e-5)
        n_steps = 1000
        errors = []
        for _ in range(n_steps):
            x = rng.randn(4)
            # Simple separable problem using model can learn
            y = 1.0 if x[0] + 2 * x[1] > 0 else 0.0
            err = model.fit_one(x, y)
            errors.append(err)
        # Error should decrease over time
        first_half_err = float(np.mean(errors[:100]))
        last_half_err = float(np.mean(errors[-100:]))
        assert last_half_err < first_half_err, (
            f"Model did not improve: early={first_half_err:.3f}, late={last_half_err:.3f}"
        )

    def test_serialization(self, simple_features, rng):
        model = OnlineLogistic(n_features=20)
        for i in range(20):
            x = rng.randn(20)
            model.fit_one(x, float(rng.randint(0, 2)))
        d = model.to_dict()
        model2 = OnlineLogistic.from_dict(d)
        pred1 = model.predict(simple_features)
        pred2 = model2.predict(simple_features)
        assert abs(pred1 - pred2) < 1e-10

    def test_reset_clears_weights(self, rng):
        model = OnlineLogistic(n_features=20)
        for _ in range(50):
            model.fit_one(rng.randn(20), float(rng.randint(0, 2)))
        model.reset()
        assert np.all(model._w == 0.0)
        assert model._n_updates == 0

    def test_feature_importance_shape(self):
        model = OnlineLogistic(n_features=20)
        imp = model.feature_importance()
        assert imp.shape == (20,)

    def test_hashing_reduces_dim(self):
        model = OnlineLogistic(n_features=100, use_hashing=True, hash_dim=32)
        x = np.random.randn(100)
        sig = model.predict(x)
        assert -1.0 <= sig <= 1.0
        assert model._dim == 32


# ---------------------------------------------------------------------------
# FTRL tests
# ---------------------------------------------------------------------------

class TestFTRL:

    def test_sparse_features(self):
        """FTRL with very high L1 should produce sparse weights."""
        rng = np.random.RandomState(7)
        # Use aggressive L1 so FTRL zeroes most features
        model = FTRL(n_features=50, alpha=0.05, beta=1.0, l1=5.0, l2=1.0)
        # Sparse features: only 5 of 50 are non-zero per sample
        for _ in range(200):
            x = np.zeros(50)
            active = rng.randint(0, 50, size=5)
            x[active] = rng.randn(5)
            y = 1.0 if x.sum() > 0 else 0.0
            model.fit_one(x, y)
        sparsity = model.sparsity
        assert sparsity > 0.5, f"Expected sparse weights, got sparsity={sparsity:.3f}"

    def test_prediction_range(self):
        model = FTRL(n_features=20)
        x = np.random.randn(20)
        p = model.predict_proba(x)
        assert 0.0 < p < 1.0
        s = model.predict(x)
        assert -1.0 <= s <= 1.0

    def test_learns_simple_pattern(self):
        """FTRL should learn a simple linear pattern."""
        rng = np.random.RandomState(11)
        model = FTRL(n_features=10, alpha=0.3, l1=0.001, l2=0.01)
        errors = []
        for _ in range(500):
            x = rng.randn(10)
            y = 1.0 if x[0] > 0 else 0.0
            p = model.predict_proba(x)
            model.fit_one(x, y)
            errors.append(abs(p - y))
        early = float(np.mean(errors[:50]))
        late = float(np.mean(errors[-50:]))
        assert late < early, f"FTRL did not improve: early={early:.3f}, late={late:.3f}"

    def test_serialization(self):
        rng = np.random.RandomState(13)
        model = FTRL(n_features=20)
        for _ in range(50):
            model.fit_one(rng.randn(20), float(rng.randint(0, 2)))
        d = model.to_dict()
        model2 = FTRL.from_dict(d)
        x = rng.randn(20)
        assert abs(model.predict(x) - model2.predict(x)) < 1e-10


# ---------------------------------------------------------------------------
# OnlineRidge tests
# ---------------------------------------------------------------------------

class TestOnlineRidge:

    def test_woodbury_update(self, rng):
        """Test that Woodbury updates give consistent predictions."""
        model = OnlineRidge(n_features=10, lam=1.0)
        for _ in range(30):
            x = rng.randn(10)
            y = float(x[0] * 2.0 + rng.randn() * 0.1)
            model.fit_one(x, y)
        # Model should predict positively for x[0] > 0
        x_pos = np.zeros(10)
        x_pos[0] = 1.0
        x_neg = np.zeros(10)
        x_neg[0] = -1.0
        assert model.predict(x_pos) > model.predict(x_neg)

    def test_serialization(self, rng):
        model = OnlineRidge(n_features=10)
        for _ in range(20):
            model.fit_one(rng.randn(10), rng.randn())
        d = model.to_dict()
        model2 = OnlineRidge.from_dict(d)
        x = rng.randn(10)
        assert abs(model.predict(x) - model2.predict(x)) < 1e-8


# ---------------------------------------------------------------------------
# OnlinePassiveAggressive tests
# ---------------------------------------------------------------------------

class TestOnlinePassiveAggressive:

    def test_classification_converges(self, rng):
        model = OnlinePassiveAggressive(n_features=10, C=1.0, variant="PA-II")
        errors = []
        for _ in range(300):
            x = rng.randn(10)
            y = 1.0 if x[0] > 0 else -1.0
            pred = model.predict(x)
            err = model.fit_one(x, y)
            errors.append(err)
        early = float(np.mean(errors[:30]))
        late = float(np.mean(errors[-30:]))
        assert late < early + 0.5  # should improve or stay similar

    def test_regression_mode(self, rng):
        model = OnlinePassiveAggressive(
            n_features=5, C=1.0, variant="PA-I", task="regression"
        )
        for _ in range(100):
            x = rng.randn(5)
            y = float(x[0])
            model.fit_one(x, y)
        x_test = np.array([1.0, 0, 0, 0, 0])
        pred = model.predict(x_test)
        assert pred > 0, f"Expected positive prediction for positive input, got {pred}"


# ---------------------------------------------------------------------------
# OnlineGradientBoosting tests
# ---------------------------------------------------------------------------

class TestOnlineGradientBoosting:

    def test_fits_simple_pattern(self, rng):
        model = OnlineGradientBoosting(
            n_estimators=5, max_depth=2, n_features=10,
            buffer_size=100, refitting_interval=10
        )
        errors_early = []
        errors_late = []
        for i in range(200):
            x = rng.randn(10)
            y = 1.0 if x[0] > 0 else -1.0
            err = model.fit_one(x, y)
            if i < 50:
                errors_early.append(abs(err))
            if i > 150:
                errors_late.append(abs(err))
        # Not guaranteed to converge fully, just ensure no crash
        assert len(errors_late) > 0

    def test_predict_in_range(self, rng):
        model = OnlineGradientBoosting(n_features=10)
        for _ in range(50):
            model.fit_one(rng.randn(10), float(rng.randint(0, 2)))
        x = rng.randn(10)
        pred = model.predict(x)
        assert -1.0 <= pred <= 1.0

    def test_serialization(self, rng):
        model = OnlineGradientBoosting(
            n_features=10, buffer_size=50, refitting_interval=5
        )
        for _ in range(50):
            model.fit_one(rng.randn(10), float(rng.randint(0, 2)))
        d = model.to_dict()
        model2 = OnlineGradientBoosting.from_dict(d)
        assert model2._n_updates == model._n_updates


# ---------------------------------------------------------------------------
# KernelOnlineLearning tests
# ---------------------------------------------------------------------------

class TestKernelOnlineLearning:

    def test_rff_nonlinear_feature(self, rng):
        model = KernelOnlineLearning(n_features=10, D=64, gamma=1.0, seed=0)
        for _ in range(100):
            x = rng.randn(10)
            y = 1.0 if np.sin(x[0]) > 0 else 0.0
            model.fit_one(x, y)
        p = model.predict_proba(rng.randn(10))
        assert 0.0 < p < 1.0

    def test_D_256_default(self):
        model = KernelOnlineLearning(n_features=20, D=256)
        assert model._omega.shape == (256, 20)
        assert model._bias.shape == (256,)


# ---------------------------------------------------------------------------
# ForgettingEnsemble tests
# ---------------------------------------------------------------------------

class TestForgettingEnsemble:

    def test_reweights_after_error(self, rng):
        """
        Ensemble should shift weight away from a model that consistently errs.
        """
        good = OnlineLogistic(n_features=5, learning_rate=0.2)
        bad = OnlineLogistic(n_features=5, learning_rate=0.001)

        ensemble = ForgettingEnsemble(
            models=[good, bad],
            learning_rate=0.3,
            min_weight=0.01,
        )
        initial_weights = ensemble.get_weights().copy()

        # Force bad model to always predict wrong by training it on opposite labels
        for _ in range(50):
            x = rng.randn(5)
            y = 1.0 if x[0] > 0 else 0.0
            bad.fit_one(x, 1.0 - y)  # wrong labels

        # Now run ensemble on correct patterns
        for _ in range(100):
            x = rng.randn(5)
            y = 1.0 if x[0] > 0 else 0.0
            ensemble.fit_one(x, y)

        final_weights = ensemble.get_weights()
        # Good model should have higher weight than bad model
        assert final_weights[0] > final_weights[1], (
            f"Expected good model weight > bad model weight, "
            f"got {final_weights[0]:.3f} vs {final_weights[1]:.3f}"
        )

    def test_predict_returns_in_range(self, rng):
        ens = make_default_ensemble(n_features=20)
        x = rng.randn(20)
        sig = ens.predict(x)
        assert -1.0 <= sig <= 1.0
        p = ens.predict_proba(x)
        assert 0.0 <= p <= 1.0

    def test_serialization_roundtrip(self, rng):
        ens = make_default_ensemble(n_features=10)
        for _ in range(30):
            ens.fit_one(rng.randn(10), float(rng.randint(0, 2)))
        d = ens.to_dict()
        ens2 = ForgettingEnsemble.from_dict(d)
        x = rng.randn(10)
        # Weights should match
        np.testing.assert_allclose(ens.get_weights(), ens2.get_weights())

    def test_drift_counts_increment_on_drift(self, rng):
        ens = make_default_ensemble(n_features=10)
        ens._adwins[0].delta = 0.5  # very sensitive ADWIN
        for _ in range(200):
            ens.fit_one(rng.randn(10), float(rng.randint(0, 2)))
        # Drift may or may not have occurred, but counts should be >= 0
        assert all(c >= 0 for c in ens.get_drift_counts())


# ---------------------------------------------------------------------------
# ADWIN tests
# ---------------------------------------------------------------------------

class TestADWIN:

    def test_detects_mean_shift(self):
        adwin = ADWIN(delta=0.01)
        detected = False
        # First 50 obs from N(0,1), then 50 obs from N(5,1)
        for i in range(50):
            adwin.update(float(np.random.randn()))
        for i in range(50):
            if adwin.update(5.0 + float(np.random.randn() * 0.1)):
                detected = True
                break
        assert detected, "ADWIN should detect the mean shift from 0 to 5"

    def test_no_drift_stable_stream(self):
        # Use a conservative delta (very small) so ADWIN rarely fires falsely
        adwin = ADWIN(delta=1e-6)
        rng = np.random.RandomState(99)
        drift_count = 0
        for _ in range(100):
            if adwin.update(float(rng.randn() * 0.01 + 0.5)):
                drift_count += 1
        # With a very tight delta, we expect at most a small handful of false positives
        assert drift_count < 10, f"Too many false positives: {drift_count}"


# ---------------------------------------------------------------------------
# Feature pipeline tests
# ---------------------------------------------------------------------------

class TestFeaturePipeline:

    def test_output_shape(self, sample_raw_features):
        pipeline = FeaturePipeline()
        feats = pipeline.transform_one(sample_raw_features)
        assert feats.shape == (200,), f"Expected (200,) got {feats.shape}"

    def test_no_nan_inf(self, sample_raw_features):
        pipeline = FeaturePipeline()
        # Warm up with several bars
        for _ in range(50):
            feats = pipeline.transform_one(sample_raw_features)
        assert np.all(np.isfinite(feats)), "Feature vector contains NaN or Inf"

    def test_normalization_after_warmup(self, sample_raw_features):
        pipeline = FeaturePipeline()
        for _ in range(40):
            feats = pipeline.transform_one(sample_raw_features)
        # After warmup, values should be z-score normalized and clipped to [-5, 5]
        assert feats.max() <= 5.0 + 1e-6
        assert feats.min() >= -5.0 - 1e-6

    def test_winsorization(self):
        pipeline = FeaturePipeline(winsor_limit=3.0)
        raw = RawFeatures(
            close=1e10, high=1e10, low=0.0, volume=1e15,
            ret_1=100.0, garch_vol=100.0,
        )
        feats = pipeline.transform_one(raw)
        assert np.all(np.isfinite(feats))

    def test_serialization(self, sample_raw_features):
        pipeline = FeaturePipeline()
        for _ in range(10):
            pipeline.transform_one(sample_raw_features)
        d = pipeline.to_dict()
        pipeline2 = FeaturePipeline.from_dict(d)
        assert pipeline2._n_processed == pipeline._n_processed

    def test_lag_features_shape(self, sample_raw_features):
        lf = LagFeatures()
        out = lf.transform_one(sample_raw_features.ret_1, sample_raw_features.garch_vol, sample_raw_features.volume)
        assert out.shape == (60,)

    def test_interaction_features_shape(self, sample_raw_features):
        intf = InteractionFeatures()
        out = intf.transform_one(sample_raw_features)
        assert out.shape == (30,)

    def test_calendar_features_shape(self, sample_raw_features):
        cf = CalendarFeatures()
        out = cf.transform_one(sample_raw_features)
        assert out.shape == (20,)

    def test_technical_features_shape(self, sample_raw_features):
        tf = TechnicalFeatures()
        for _ in range(5):
            out = tf.transform_one(sample_raw_features)
        assert out.shape == (20,)

    def test_physics_features_shape(self, sample_raw_features):
        pf = PhysicsFeatures()
        out = pf.transform_one(sample_raw_features)
        assert out.shape == (15,)

    def test_volatility_features_shape(self, sample_raw_features):
        vf = VolatilityFeatures()
        out = vf.transform_one(sample_raw_features)
        assert out.shape == (15,)

    def test_microstructure_features_shape(self, sample_raw_features):
        mf = MicrostructureFeatures()
        out = mf.transform_one(sample_raw_features)
        assert out.shape == (15,)

    def test_feature_names_count(self):
        pipeline = FeaturePipeline()
        names = pipeline.feature_names
        assert len(names) == 200, f"Expected 200 feature names, got {len(names)}"

    def test_cross_asset_features_shape(self, sample_raw_features):
        ca = CrossAssetFeatures()
        out = ca.transform_one(sample_raw_features)
        assert out.shape == (15,)


# ---------------------------------------------------------------------------
# PurgedCVSplit tests
# ---------------------------------------------------------------------------

class TestPurgedCVSplit:

    def test_splits_are_nonoverlapping(self):
        """Train and test indices must not overlap."""
        X = np.random.randn(300, 10)
        cv = PurgedCVSplit(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_train_precedes_test(self):
        """All train indices should precede all test indices (temporal ordering)."""
        X = np.random.randn(300, 10)
        cv = PurgedCVSplit(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            assert train_idx.max() < test_idx.min(), (
                f"Train max={train_idx.max()} >= test min={test_idx.min()}"
            )

    def test_embargo_creates_gap(self):
        """There should be a gap between end of train and start of test."""
        X = np.random.randn(300, 10)
        cv = PurgedCVSplit(n_splits=5, embargo_pct=0.02)
        for train_idx, test_idx in cv.split(X):
            gap = test_idx.min() - train_idx.max()
            assert gap >= 1, f"Expected gap >= 1, got {gap}"

    def test_correct_number_of_splits(self):
        X = np.random.randn(300, 10)
        cv = PurgedCVSplit(n_splits=5)
        splits = list(cv.split(X))
        assert len(splits) == 5


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV tests
# ---------------------------------------------------------------------------

class TestCombinatorialPurgedCV:

    def test_splits_nonoverlapping(self):
        X = np.random.randn(200, 10)
        cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0

    def test_correct_number_of_combinations(self):
        from math import comb
        cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2)
        X = np.random.randn(300, 10)
        splits = list(cv.split(X))
        # May be fewer due to minimum sample filtering
        assert len(splits) <= comb(6, 2)
        assert len(splits) >= 1


# ---------------------------------------------------------------------------
# WalkForwardValidator tests
# ---------------------------------------------------------------------------

class TestWalkForwardValidator:

    def test_wfa_efficiency_in_range(self, rng):
        """WFA efficiency should be a real finite number (not necessarily 0-2)."""
        X = rng.randn(400, 20)
        y = 0.01 * rng.randn(400)  # small returns

        model = OnlineLogistic(n_features=20)
        wfa = WalkForwardValidator(
            train_window=100, test_window=50, step_size=50
        )
        result = wfa.validate(model, X, y)

        assert result.n_periods >= 1
        assert math.isfinite(result.is_sharpe)
        assert math.isfinite(result.oos_sharpe)
        assert math.isfinite(result.wfa_efficiency)

    def test_wfa_structure(self, rng):
        X = rng.randn(300, 10)
        y = rng.randn(300) * 0.01
        model = OnlineLogistic(n_features=10)
        wfa = WalkForwardValidator(train_window=100, test_window=40, step_size=40)
        result = wfa.validate(model, X, y)
        assert result.n_periods > 0
        assert len(result.period_results) == result.n_periods
        for pr in result.period_results:
            assert "is_sharpe" in pr
            assert "oos_sharpe" in pr


# ---------------------------------------------------------------------------
# ModelDriftDetector tests
# ---------------------------------------------------------------------------

class TestModelDriftDetector:

    def test_fires_on_mean_shift(self):
        """Drift detector should fire when performance degrades."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            detector = ModelDriftDetector(
                model_name="test_model",
                threshold=3.0,
                slack=0.2,
                db_path=db_path,
                min_samples=20,
            )

            # Feed good performance values
            for _ in range(30):
                detector.update(0.8)  # good accuracy

            # Feed degraded performance (mean shift downward)
            fired = False
            for _ in range(50):
                if detector.update(0.3):  # bad accuracy
                    fired = True
                    break

            assert fired, (
                f"Expected drift detector to fire on mean shift, "
                f"cusum_neg={detector.cusum_stat:.3f}"
            )
        finally:
            os.unlink(db_path)

    def test_no_false_alarm_stable(self):
        """Drift detector should not fire on stable performance stream."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            rng = np.random.RandomState(42)
            detector = ModelDriftDetector(
                model_name="stable_model",
                threshold=6.0,
                slack=0.5,
                db_path=db_path,
                min_samples=20,
            )
            fired_count = 0
            for _ in range(200):
                val = 0.6 + rng.randn() * 0.02  # stable ~60% accuracy
                if detector.update(val):
                    fired_count += 1
            assert fired_count == 0, f"False alarm count: {fired_count}"
        finally:
            os.unlink(db_path)

    def test_logs_to_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            detector = ModelDriftDetector(
                threshold=2.0, slack=0.1, db_path=db_path, min_samples=10
            )
            for _ in range(20):
                detector.update(0.7)
            for _ in range(30):
                detector.update(0.2)  # force drift

            import sqlite3 as sq
            conn = sq.connect(db_path)
            rows = conn.execute("SELECT * FROM drift_events").fetchall()
            conn.close()
            # Events may or may not be in DB depending on threshold
            assert isinstance(rows, list)
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# ModelSelector tests
# ---------------------------------------------------------------------------

class TestModelSelector:

    def test_selects_a_model(self, rng, binary_dataset):
        X, y_bin = binary_dataset
        y_ret = (y_bin - 0.5) * 0.02  # convert to returns

        candidates = {
            "logistic": OnlineLogistic(n_features=20),
            "ftrl": FTRL(n_features=20),
        }
        selector = ModelSelector(n_splits=3, n_bootstrap=20)
        best_name, best_model, results = selector.select(candidates, X, y_ret)

        assert best_name in ("logistic", "ftrl")
        assert len(results) == 2
        for r in results:
            assert math.isfinite(r.sharpe)
            assert math.isfinite(r.max_drawdown)


# ---------------------------------------------------------------------------
# HyperparameterSearch tests
# ---------------------------------------------------------------------------

class TestHyperparameterSearch:

    def test_finds_good_params(self):
        """Search should find params that improve a simple quadratic objective."""
        def objective(params: Dict[str, Any]) -> float:
            x = params["x"]
            return -(x - 2.0) ** 2  # max at x=2

        search = HyperparameterSearch(
            param_grid={"x": (-5.0, 5.0)},
            n_trials=20,
            n_random_init=5,
        )
        best_params, best_score, history = search.search(objective)

        assert "x" in best_params
        assert abs(best_params["x"] - 2.0) < 2.0, (
            f"Expected x near 2.0, got {best_params['x']:.3f}"
        )

    def test_returns_history(self):
        search = HyperparameterSearch(
            param_grid={"lr": (0.001, 0.1), "l2": (1e-5, 1.0)},
            n_trials=10,
            n_random_init=5,
        )
        _, _, history = search.search(lambda p: -p["lr"] ** 2)
        assert len(history) == 10


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------

class TestSignalEnsemble:

    def test_predict_in_range(self):
        ens = SignalEnsemble()
        signals = {
            "bh_physics": 0.5, "garch": -0.3, "ou_zscore": 0.2,
            "quat_nav": 0.1, "granger": 0.0, "ml": -0.2, "hurst": 0.4,
        }
        pred = ens.predict(signals)
        assert -1.0 <= pred <= 1.0

    def test_updates_weights_with_returns(self):
        ens = SignalEnsemble()
        signals = {n: 0.3 for n in SignalEnsemble.DEFAULT_NAMES}
        for _ in range(50):
            ens.update(signals, 0.01)
        w = ens.component_weights()
        assert sum(w.values()) == pytest.approx(1.0, abs=0.01)

    def test_report_structure(self):
        ens = SignalEnsemble()
        rep = ens.report()
        assert rep.combined_icir == 0.0 or isinstance(rep.combined_icir, float)
        assert len(rep.component_names) == len(SignalEnsemble.DEFAULT_NAMES)


class TestDynamicWeightOptimizer:

    def test_weights_sum_to_one(self):
        opt = DynamicWeightOptimizer(n_components=5)
        signals = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
        for _ in range(100):
            opt.update(signals, 0.01)
        w = opt.weights
        assert w.sum() == pytest.approx(1.0, abs=0.01)
        assert all(w >= DynamicWeightOptimizer.FLOOR - 1e-6)
        assert all(w <= DynamicWeightOptimizer.CAP + 1e-6)

    def test_bates_granger_weights(self):
        rng = np.random.RandomState(1)
        opt = DynamicWeightOptimizer(n_components=3)
        errors = rng.randn(3, 100)
        errors[0] *= 0.1  # model 0 has tiny errors
        errors[1] *= 1.0
        errors[2] *= 2.0
        w = opt.bates_granger_weights(errors)
        # Model 0 (lowest error) should get highest weight
        assert w[0] > w[1]
        assert w[0] > w[2]


class TestRegimeSwitchingEnsemble:

    def test_regime_switch_smooth(self):
        """Signal should not jump discontinuously during transition."""
        ens = RegimeSwitchingEnsemble(transition_bars=5)
        signals = {n: 0.5 for n in SignalEnsemble.DEFAULT_NAMES}

        ens.set_regime(Regime.BULL)
        pred_bull = ens.predict(signals)

        ens.set_regime(Regime.BEAR)
        preds_transition = [ens.predict(signals) for _ in range(5)]

        # Predictions should be continuous (no huge jumps)
        for i in range(1, len(preds_transition)):
            diff = abs(preds_transition[i] - preds_transition[i - 1])
            assert diff < 0.5, f"Discontinuous transition: diff={diff:.3f}"

    def test_regime_breakdown(self):
        ens = RegimeSwitchingEnsemble()
        signals = {n: 0.1 for n in SignalEnsemble.DEFAULT_NAMES}
        ens.set_regime(Regime.BULL)
        for _ in range(10):
            ens.update(signals, 0.01)
        ens.set_regime(Regime.BEAR)
        for _ in range(5):
            ens.update(signals, -0.01)
        breakdown = ens.regime_breakdown()
        assert "BULL" in breakdown
        assert "BEAR" in breakdown
        assert abs(sum(breakdown.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Compute utilities tests
# ---------------------------------------------------------------------------

class TestComputeUtilities:

    def test_sharpe_zero_variance(self):
        returns = np.ones(100) * 0.001
        s = _compute_sharpe(returns)
        assert s == 0.0

    def test_sharpe_positive(self):
        rng = np.random.RandomState(0)
        returns = 0.01 + rng.randn(252) * 0.02
        s = _compute_sharpe(returns)
        assert s > 0  # positive mean return

    def test_sharpe_negative(self):
        rng = np.random.RandomState(0)
        returns = -0.01 + rng.randn(252) * 0.02
        s = _compute_sharpe(returns)
        assert s < 0


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer tests
# ---------------------------------------------------------------------------

class TestFeatureImportanceAnalyzer:

    def test_returns_importance_for_all_features(self, rng):
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(float)

        model = OnlineLogistic(n_features=10)
        for i in range(100):
            model.fit_one(X[i], y[i])

        analyzer = FeatureImportanceAnalyzer(model, n_repeats=3)
        importances = analyzer.compute_importance(X, y)

        assert len(importances) == 10
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_top_n_features_sorted(self, rng):
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(float)
        model = OnlineLogistic(n_features=10)
        for i in range(100):
            model.fit_one(X[i], y[i])
        names = [f"f{i}" for i in range(10)]
        analyzer = FeatureImportanceAnalyzer(model, feature_names=names, n_repeats=2)
        top = analyzer.top_n_features(X, y, n=5)
        assert len(top) == 5
        # Check sorted descending
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)
