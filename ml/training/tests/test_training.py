# ml/training/tests/test_training.py -- tests for model registry, experiment tracker,
# feature importance, online trainer, and hyperparameter tuner
from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_dir(tmp_path):
    return str(tmp_path / "registry")


@pytest.fixture
def tracking_dir(tmp_path):
    return str(tmp_path / "mlruns")


@pytest.fixture
def checkpoint_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def registry(registry_dir):
    from ml.training.model_registry import ModelRegistry
    return ModelRegistry(registry_dir)


@pytest.fixture
def tracker(tracking_dir):
    from ml.training.experiment_tracker import ExperimentTracker
    return ExperimentTracker(tracking_dir)


def _make_metadata(name: str = "test_model", **kwargs):
    from ml.training.model_registry import ModelMetadata
    return ModelMetadata(
        model_id="",
        name=name,
        version="",
        created_at=datetime.utcnow(),
        trained_on_data="2024-01-01/2024-12-31",
        feature_names=["f1", "f2", "f3"],
        metrics={"sharpe": 1.5, "ic": 0.08, "icir": 0.6},
        hyperparams={"lr": 0.01, "depth": 5},
        status="candidate",
        parent_model_id=None,
        **kwargs,
    )


class _LinearModel:
    """Minimal linear model for testing."""

    def __init__(self, n_features: int = 3):
        self.coef_: np.ndarray = np.ones(n_features)
        self.intercept_ = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearModel":
        # OLS closed-form
        X = np.asarray(X)
        y = np.asarray(y)
        Xb = np.column_stack([X, np.ones(len(X))])
        try:
            params = np.linalg.lstsq(Xb, y, rcond=None)[0]
            self.coef_ = params[:-1]
            self.intercept_ = params[-1]
        except np.linalg.LinAlgError:
            pass
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:

    def test_register_returns_model_id(self, registry):
        meta = _make_metadata()
        model = _LinearModel()
        model_id = registry.register("test_model", model, meta)
        assert isinstance(model_id, str)
        assert len(model_id) == 36  # UUID

    def test_register_increments_version(self, registry):
        for i in range(3):
            model_id = registry.register("versioned", _LinearModel(), _make_metadata())
        records = registry.list_models("versioned")
        versions = [r.version for r in records]
        assert "1.0.0" in versions
        assert "1.0.1" in versions
        assert "1.0.2" in versions

    def test_load_by_id(self, registry):
        meta = _make_metadata()
        model = _LinearModel(3)
        model_id = registry.register("load_test", model, meta)

        loaded_model, loaded_meta = registry.load(model_id)
        assert loaded_meta.model_id == model_id
        assert loaded_meta.name == "load_test"
        assert isinstance(loaded_model, _LinearModel)

    def test_load_latest(self, registry):
        registry.register("latest_test", _LinearModel(), _make_metadata(metrics={"ic": 0.05}))
        time.sleep(0.01)
        registry.register("latest_test", _LinearModel(), _make_metadata(metrics={"ic": 0.10}))

        _, meta = registry.load_latest("latest_test")
        # Should load the most recently registered
        assert meta.metrics["ic"] == pytest.approx(0.10)

    def test_load_latest_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load_latest("nonexistent_model")

    def test_list_models_all(self, registry):
        registry.register("a", _LinearModel(), _make_metadata())
        registry.register("b", _LinearModel(), _make_metadata())
        records = registry.list_models()
        names = {r.name for r in records}
        assert "a" in names and "b" in names

    def test_list_models_filtered(self, registry):
        registry.register("alpha", _LinearModel(), _make_metadata())
        registry.register("beta", _LinearModel(), _make_metadata())
        records = registry.list_models("alpha")
        assert all(r.name == "alpha" for r in records)
        assert len(records) >= 1

    def test_archive(self, registry):
        model_id = registry.register("archive_test", _LinearModel(), _make_metadata())
        registry.archive(model_id)
        records = registry.list_models("archive_test")
        assert records[0].status == "archived"

    def test_archive_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.archive("00000000-0000-0000-0000-000000000000")

    def test_promote_to_production(self, registry):
        model_id = registry.register("prod_test", _LinearModel(), _make_metadata())
        registry.promote_to_production(model_id)

        _, meta = registry.get_production_model("prod_test")
        assert meta.status == "production"
        assert meta.model_id == model_id

    def test_promote_demotes_old_production(self, registry):
        id1 = registry.register("multi_prod", _LinearModel(), _make_metadata())
        registry.promote_to_production(id1)

        id2 = registry.register("multi_prod", _LinearModel(), _make_metadata())
        registry.promote_to_production(id2)

        _, meta1 = registry.load(id1)
        _, meta2 = registry.load(id2)
        assert meta1.status == "candidate"
        assert meta2.status == "production"

    def test_get_production_model_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_production_model("no_prod_here")

    def test_compare_models(self, registry):
        meta1 = _make_metadata(metrics={"ic": 0.05, "sharpe": 1.0})
        meta2 = _make_metadata(metrics={"ic": 0.10, "sharpe": 1.5})
        id1 = registry.register("cmp", _LinearModel(), meta1)
        id2 = registry.register("cmp", _LinearModel(), meta2)

        result = registry.compare_models(id1, id2, primary_metric="ic")
        assert result.winner == id2
        assert result.metric_deltas["ic"] == pytest.approx(0.05, abs=1e-6)

    def test_get_lineage(self, registry):
        id1 = registry.register("lineage", _LinearModel(), _make_metadata())
        meta2 = _make_metadata()
        meta2.parent_model_id = id1
        meta2.model_id = ""
        id2 = registry.register("lineage", _LinearModel(), meta2)

        chain = registry.get_lineage(id2)
        # Chain should include both ancestors
        chain_ids = [m.model_id for m in chain]
        assert id1 in chain_ids
        assert id2 in chain_ids

    def test_update_metrics(self, registry):
        model_id = registry.register("metrics_upd", _LinearModel(), _make_metadata())
        registry.update_metrics(model_id, {"live_ic": 0.07})
        _, meta = registry.load(model_id)
        assert "live_ic" in meta.metrics

    def test_count_by_status(self, registry):
        id1 = registry.register("cnt", _LinearModel(), _make_metadata())
        id2 = registry.register("cnt", _LinearModel(), _make_metadata())
        registry.archive(id1)
        registry.promote_to_production(id2)
        counts = registry.count_by_status()
        assert counts.get("archived", 0) >= 1
        assert counts.get("production", 0) >= 1

    def test_search_by_min_metric(self, registry):
        registry.register("search", _LinearModel(), _make_metadata(metrics={"ic": 0.03}))
        registry.register("search", _LinearModel(), _make_metadata(metrics={"ic": 0.09}))
        results = registry.search(name="search", min_metric={"ic": 0.08})
        assert all(r.metrics.get("ic", 0) >= 0.08 for r in results)


# ---------------------------------------------------------------------------
# ExperimentTracker tests
# ---------------------------------------------------------------------------


class TestExperimentTracker:

    def test_start_run_returns_context(self, tracker):
        ctx = tracker.start_run("exp_a", {"lr": 0.01})
        assert ctx.run_id
        tracker.end_run(ctx.run_id)

    def test_context_manager_completes(self, tracker):
        with tracker.start_run("exp_b", {}) as run:
            run.log_metric("ic", 0.07)

        record = tracker.get_run(run.run_id)
        assert record.status == "completed"
        assert "ic" in record.metrics

    def test_context_manager_fails_on_exception(self, tracker):
        run_id = None
        try:
            with tracker.start_run("exp_fail", {}) as run:
                run_id = run.run_id
                raise ValueError("deliberate failure")
        except ValueError:
            pass

        record = tracker.get_run(run_id)
        assert record.status == "failed"

    def test_log_metric_with_steps(self, tracker):
        with tracker.start_run("steps_exp", {}) as run:
            for step in range(5):
                run.log_metric("loss", 1.0 - step * 0.1, step=step)

        record = tracker.get_run(run.run_id)
        assert len(record.metrics["loss"]) == 5

    def test_log_artifact_and_load(self, tracker):
        data = {"key": [1, 2, 3]}
        with tracker.start_run("artifact_exp", {}) as run:
            run.log_artifact("my_dict", data)

        loaded = tracker.load_artifact(run.run_id, "my_dict")
        assert loaded == data

    def test_compare_runs(self, tracker):
        ids = []
        for ic_val in [0.05, 0.08, 0.10]:
            with tracker.start_run("compare_exp", {"ic_target": ic_val}) as run:
                run.log_metric("ic", ic_val)
            ids.append(run.run_id)

        df = tracker.compare_runs(ids, "ic")
        assert len(df) == 3
        assert df.iloc[0]["ic"] == pytest.approx(0.10)

    def test_best_run(self, tracker):
        for ic_val in [0.03, 0.07, 0.12]:
            with tracker.start_run("best_exp", {"ic": ic_val}) as run:
                run.log_metric("ic", ic_val)

        best = tracker.best_run("best_exp", "ic", n=1)
        assert len(best) == 1
        assert best[0].metrics["ic"][-1] == pytest.approx(0.12)

    def test_best_run_top_n(self, tracker):
        for ic_val in [0.01, 0.05, 0.09, 0.11]:
            with tracker.start_run("topn_exp", {}) as run:
                run.log_metric("ic", ic_val)

        best = tracker.best_run("topn_exp", "ic", n=2)
        assert len(best) == 2
        scores = [r.metrics["ic"][-1] for r in best]
        assert all(s >= 0.09 for s in scores)

    def test_list_experiments(self, tracker):
        tracker.start_run("exp_list_1", {}).end_run = lambda s: None  # noqa: E731
        with tracker.start_run("exp_list_1", {}):
            pass
        with tracker.start_run("exp_list_2", {}):
            pass
        exps = tracker.list_experiments()
        assert "exp_list_1" in exps
        assert "exp_list_2" in exps

    def test_metric_history(self, tracker):
        with tracker.start_run("hist_exp", {}) as run:
            for i in range(4):
                run.log_metric("val_ic", float(i) * 0.02, step=i)

        hist = tracker.metric_history(run.run_id, "val_ic")
        assert len(hist) == 4
        steps = [h[0] for h in hist]
        vals = [h[1] for h in hist]
        assert steps == [0, 1, 2, 3]
        assert vals[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HyperparameterTuner tests
# ---------------------------------------------------------------------------


class TestHyperparameterTuner:

    def test_grid_search_finds_optimum(self, tracker):
        from ml.training.experiment_tracker import HyperparameterTuner

        def objective(params: Dict) -> float:
            # Optimum at lr=0.01, depth=3
            lr_score = -abs(params["lr"] - 0.01) * 100
            depth_score = -abs(params["depth"] - 3)
            return lr_score + depth_score

        tuner = HyperparameterTuner(tracker, experiment_name="grid_test", metric="ic")
        result = tuner.grid_search(
            objective,
            {"lr": [0.001, 0.01, 0.1], "depth": [2, 3, 5]},
            n_jobs=1,
        )

        assert result.best_params["lr"] == pytest.approx(0.01)
        assert result.best_params["depth"] == 3
        assert len(result.all_results) == 9  # 3 x 3

    def test_random_search_returns_n_trials(self, tracker):
        from ml.training.experiment_tracker import HyperparameterTuner

        def objective(params: Dict) -> float:
            return float(params.get("x", 0))

        tuner = HyperparameterTuner(tracker, experiment_name="rand_test", metric="ic")
        result = tuner.random_search(
            objective,
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
            n_trials=10,
            n_jobs=1,
            seed=42,
        )
        assert len(result.all_results) == 10
        assert result.best_score == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer tests
# ---------------------------------------------------------------------------


class TestFeatureImportanceAnalyzer:

    def _make_data(self, n: int = 200, n_feats: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.standard_normal((n, n_feats)),
            columns=[f"f{i}" for i in range(n_feats)],
        )
        # y depends strongly on f0, weakly on f1
        y = pd.Series(2 * X["f0"] + 0.5 * X["f1"] + rng.standard_normal(n) * 0.3)
        return X, y

    def test_permutation_importance_shape(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data()
        model = _LinearModel(5)
        model.fit(X.values, y.values)

        analyzer = FeatureImportanceAnalyzer(random_state=0)
        result = analyzer.permutation_importance(model, X, y, n_repeats=5)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance_mean" in result.columns
        assert len(result) == 5

    def test_permutation_importance_top_feature(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data(n=300)
        model = _LinearModel(5)
        model.fit(X.values, y.values)

        analyzer = FeatureImportanceAnalyzer(random_state=0)
        result = analyzer.permutation_importance(model, X, y, n_repeats=10)

        # f0 should be the most important
        assert result.iloc[0]["feature"] == "f0"

    def test_shap_linear_shape(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data()
        model = _LinearModel(5)
        model.fit(X.values, y.values)

        analyzer = FeatureImportanceAnalyzer()
        shap = analyzer.shap_values(model, X, method="linear")
        assert shap.shape == (len(X), 5)

    def test_shap_summary(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data()
        model = _LinearModel(5)
        model.fit(X.values, y.values)

        analyzer = FeatureImportanceAnalyzer()
        shap = analyzer.shap_values(model, X, method="linear")
        summary = analyzer.shap_summary(shap, list(X.columns))
        # f0 should have highest mean abs SHAP
        assert summary.index[0] == "f0"

    def test_mutual_information(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data(n=500)
        analyzer = FeatureImportanceAnalyzer()
        mi = analyzer.mutual_information(X, y)

        assert isinstance(mi, pd.Series)
        assert len(mi) == 5
        assert mi.index[0] == "f0"  # f0 has highest MI with y

    def test_correlation_importance(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data(n=300)
        analyzer = FeatureImportanceAnalyzer()
        corr = analyzer.correlation_importance(X, y)

        assert isinstance(corr, pd.Series)
        assert corr.index[0] == "f0"

    def test_forward_selection(self):
        from ml.training.feature_importance import FeatureImportanceAnalyzer
        X, y = self._make_data(n=300)
        analyzer = FeatureImportanceAnalyzer()

        def fit_fn(Xs, ys):
            m = _LinearModel(Xs.shape[1])
            m.fit(Xs, ys)
            return m

        selected = analyzer.forward_selection(X, y, None, max_features=3, fit_fn=fit_fn)
        assert isinstance(selected, list)
        assert "f0" in selected  # f0 must be selected (most informative)
        assert len(selected) <= 3


# ---------------------------------------------------------------------------
# FeatureImportanceReport tests
# ---------------------------------------------------------------------------


class TestFeatureImportanceReport:

    def test_markdown_report(self):
        from ml.training.feature_importance import FeatureImportanceReport
        scores = pd.Series({"mom_roc": 0.5, "ofi_imbalance": 0.3, "hurst_d": 0.1})
        report = FeatureImportanceReport(importance_scores=scores)
        md = report.to_markdown()
        assert "Feature Importance Report" in md
        assert "mom_roc" in md
        assert "technical" in md
        assert "microstructure" in md

    def test_redundant_pairs(self):
        from ml.training.feature_importance import FeatureImportanceReport
        scores = pd.Series({"a": 0.5, "b": 0.3, "c": 0.1})
        corr = pd.DataFrame(
            [[1.0, 0.95, 0.2], [0.95, 1.0, 0.1], [0.2, 0.1, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        report = FeatureImportanceReport(
            importance_scores=scores, correlation_matrix=corr, redundancy_threshold=0.9
        )
        pairs = report.redundant_pairs()
        assert len(pairs) == 1
        assert set(pairs[0][:2]) == {"a", "b"}

    def test_suggested_drops(self):
        from ml.training.feature_importance import FeatureImportanceReport
        scores = pd.Series({"a": 0.5, "b": 0.3})
        corr = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]], index=["a", "b"], columns=["a", "b"]
        )
        report = FeatureImportanceReport(
            importance_scores=scores, correlation_matrix=corr, redundancy_threshold=0.9
        )
        drops = report.suggested_drops()
        # b has lower importance, so it should be suggested for dropping
        assert "b" in drops


# ---------------------------------------------------------------------------
# ConceptDriftDetector tests
# ---------------------------------------------------------------------------


class TestConceptDriftDetector:

    def test_no_drift_on_stable_errors(self):
        from ml.training.online_trainer import ConceptDriftDetector
        detector = ConceptDriftDetector(target_error=0.1, h=5.0, slack=0.5)
        rng = np.random.default_rng(0)
        for _ in range(100):
            pred = rng.normal(0.5, 0.05)
            actual = rng.normal(0.5, 0.05)
            detector.update(pred, actual)
        assert not detector.is_drifting()
        assert detector.drift_score() < 0.5

    def test_drift_detected_on_large_errors(self):
        from ml.training.online_trainer import ConceptDriftDetector
        detector = ConceptDriftDetector(target_error=0.01, h=2.0, slack=0.01)
        # Inject massive errors
        for _ in range(50):
            detector.update(0.0, 1.0)  # error = 1.0 >> target
        assert detector.is_drifting()
        assert detector.drift_score() == pytest.approx(1.0, abs=0.01)

    def test_reset_clears_cusum(self):
        from ml.training.online_trainer import ConceptDriftDetector
        detector = ConceptDriftDetector(target_error=0.01, h=2.0, slack=0.01)
        for _ in range(50):
            detector.update(0.0, 1.0)
        assert detector.is_drifting()
        detector.reset()
        assert not detector.is_drifting()
        assert detector.drift_score() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# IncrementalSGDSignal tests
# ---------------------------------------------------------------------------


class TestIncrementalSGDSignal:

    def test_initial_prediction_is_half(self, checkpoint_dir):
        from ml.training.online_trainer import IncrementalSGDSignal
        model = IncrementalSGDSignal(n_features=6, checkpoint_dir=checkpoint_dir)
        pred = model.predict(np.zeros(6))
        assert pred == pytest.approx(0.5)

    def test_update_changes_weights(self, checkpoint_dir):
        from ml.training.online_trainer import IncrementalSGDSignal
        model = IncrementalSGDSignal(n_features=3, checkpoint_dir=checkpoint_dir)
        weights_before = model.weights.copy()
        model.update(np.array([1.0, 0.0, 0.0]), 1.0)
        assert not np.allclose(model.weights, weights_before)

    def test_learning_rate_decays(self, checkpoint_dir):
        from ml.training.online_trainer import IncrementalSGDSignal
        model = IncrementalSGDSignal(
            n_features=3, learning_rate=0.1, decay_every=10, lr_decay=0.9,
            checkpoint_dir=checkpoint_dir
        )
        for _ in range(20):
            model.update(np.ones(3), 1.0)
        assert model.lr < 0.1

    def test_checkpoint_saved(self, checkpoint_dir):
        from ml.training.online_trainer import IncrementalSGDSignal
        model = IncrementalSGDSignal(
            n_features=3, checkpoint_dir=checkpoint_dir, checkpoint_every=5
        )
        for _ in range(5):
            model.update(np.ones(3), 1.0)
        ckpt = model.latest_checkpoint()
        assert ckpt is not None
        assert Path(ckpt).exists()

    def test_model_learns_separable_pattern(self, checkpoint_dir):
        """SGD should learn to predict label=1 when feature[0] > 0."""
        from ml.training.online_trainer import IncrementalSGDSignal
        model = IncrementalSGDSignal(n_features=3, learning_rate=0.1, checkpoint_dir=checkpoint_dir)
        rng = np.random.default_rng(42)

        for _ in range(500):
            x = rng.normal(size=3)
            label = 1.0 if x[0] > 0 else 0.0
            model.update(x, label)

        # After training, positive x[0] should give higher probability
        pos_pred = model.predict(np.array([2.0, 0.0, 0.0]))
        neg_pred = model.predict(np.array([-2.0, 0.0, 0.0]))
        assert pos_pred > neg_pred


# ---------------------------------------------------------------------------
# OnlineTrainer tests
# ---------------------------------------------------------------------------


class TestOnlineTrainer:

    def test_update_and_predict(self, checkpoint_dir):
        from ml.training.online_trainer import OnlineTrainer
        trainer = OnlineTrainer("test_signal", checkpoint_dir=checkpoint_dir)
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        trainer.update(features, 1.0)
        pred = trainer.predict(features)
        assert 0.0 <= pred <= 1.0

    def test_performance_since_retrain(self, checkpoint_dir):
        from ml.training.online_trainer import OnlineTrainer
        trainer = OnlineTrainer("perf_test", checkpoint_dir=checkpoint_dir)
        rng = np.random.default_rng(1)

        for _ in range(60):
            features = rng.normal(size=6)
            label = float(features[0] > 0)
            trainer.update(features, label)

        perf = trainer.performance_since_last_retrain()
        assert "ic" in perf
        assert "mean_loss" in perf
        assert perf["n_samples"] == 60

    def test_should_retrain_triggered_by_drift(self, checkpoint_dir):
        from ml.training.online_trainer import OnlineTrainer
        # Use very tight drift parameters to force detection quickly
        trainer = OnlineTrainer(
            "drift_test",
            drift_h=1.0,
            checkpoint_dir=checkpoint_dir,
        )
        trainer.drift_detector.target_error = 0.001
        trainer.drift_detector.slack = 0.001
        trainer.drift_detector.h = 0.5

        for _ in range(30):
            trainer.update(np.zeros(6), 1.0)  # large constant error

        assert trainer.should_retrain()

    def test_acknowledge_retrain_resets(self, checkpoint_dir):
        from ml.training.online_trainer import OnlineTrainer
        trainer = OnlineTrainer("ack_test", checkpoint_dir=checkpoint_dir)
        rng = np.random.default_rng(2)
        for _ in range(20):
            trainer.update(rng.normal(size=6), 1.0)

        trainer.acknowledge_retrain()
        assert trainer.drift_detector._cusum_pos == 0.0
        assert len(trainer._pred_buffer) == 0

    def test_summary_keys(self, checkpoint_dir):
        from ml.training.online_trainer import OnlineTrainer
        trainer = OnlineTrainer("summary_test", checkpoint_dir=checkpoint_dir)
        summary = trainer.summary()
        for key in ("model_name", "total_updates", "ic", "drift_score"):
            assert key in summary
