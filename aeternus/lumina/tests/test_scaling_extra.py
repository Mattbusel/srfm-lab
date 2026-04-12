"""Tests for scaling.py extended components."""
import pytest
import torch
import torch.nn as nn

class TestModelSizeProfile:
    def test_from_model(self):
        from scaling import ModelSizeProfile
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        profile = ModelSizeProfile.from_model(model)
        assert profile.n_params_total > 0
        assert profile.memory_fp32_mb > 0

    def test_trainable_le_total(self):
        from scaling import ModelSizeProfile
        model = nn.Linear(64, 64)
        model.weight.requires_grad_(False)
        profile = ModelSizeProfile.from_model(model)
        assert profile.n_params_trainable <= profile.n_params_total

    def test_str_representation(self):
        from scaling import ModelSizeProfile
        model = nn.Linear(32, 64)
        profile = ModelSizeProfile.from_model(model)
        s = str(profile)
        assert 'params' in s.lower() or 'Memory' in s


class TestChinchillaPredictor:
    def test_optimal_allocation_returns_dict(self):
        from scaling import ChinchillaPredictor
        result = ChinchillaPredictor.optimal_allocation(1e21)
        assert 'n_params' in result
        assert 'n_tokens' in result
        assert 'predicted_loss' in result

    def test_predict_loss_positive(self):
        from scaling import ChinchillaPredictor
        loss = ChinchillaPredictor.predict_loss(1e9, 2e10)
        assert loss > 0

    def test_more_params_less_loss(self):
        from scaling import ChinchillaPredictor
        L_small = ChinchillaPredictor.predict_loss(1e8, 1e10)
        L_large = ChinchillaPredictor.predict_loss(1e10, 1e10)
        assert L_large < L_small

    def test_iso_flop_frontier(self):
        from scaling import ChinchillaPredictor
        result = ChinchillaPredictor.iso_flop_frontier(1e21, n_points=10)
        assert len(result['n_params']) > 0
        assert len(result['loss']) == len(result['n_params'])

    def test_required_tokens(self):
        from scaling import ChinchillaPredictor
        tokens = ChinchillaPredictor.required_tokens(1e9, target_loss=3.0)
        assert tokens > 0


class TestComputeOptimalTrainer:
    def test_budget_fraction(self):
        from scaling import ComputeOptimalTrainer
        trainer = ComputeOptimalTrainer(
            n_params=1e9,
            compute_budget_flops=1e21,
            flops_per_step_estimate=1e18,
        )
        trainer.record_step(loss=3.5)
        assert 0 < trainer.budget_fraction_used() < 1

    def test_should_stop_on_budget(self):
        from scaling import ComputeOptimalTrainer
        trainer = ComputeOptimalTrainer(
            n_params=1e6,
            compute_budget_flops=5.0,
            flops_per_step_estimate=6.0,
        )
        trainer.record_step(loss=3.0)
        assert trainer.should_stop(current_loss=3.0)

    def test_summary(self):
        from scaling import ComputeOptimalTrainer
        trainer = ComputeOptimalTrainer(1e8, 1e18, 1e15)
        for i in range(10):
            trainer.record_step(loss=3.0 - i * 0.1)
        s = trainer.training_summary()
        assert 'steps' in s
        assert s['steps'] == 10


class TestGradientNoiseScaleMonitor:
    def test_update_and_noise_scale(self):
        from scaling import GradientNoiseScaleMonitor
        model = nn.Linear(16, 4)
        monitor = GradientNoiseScaleMonitor(model)
        # Simulate a backward pass
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        monitor.update()
        ns = monitor.noise_scale()
        assert not (ns != ns)  # not NaN

    def test_optimal_batch_size(self):
        from scaling import GradientNoiseScaleMonitor
        model = nn.Linear(16, 4)
        monitor = GradientNoiseScaleMonitor(model)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        monitor.update()
        bs = monitor.optimal_batch_size(current_batch_size=32)
        assert isinstance(bs, int)
        assert bs >= 1


class TestModelEfficiencyAnalyzer:
    def test_efficiency_metrics(self):
        from scaling import ModelEfficencyAnalyzer
        model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
        analyzer = ModelEfficencyAnalyzer(model)
        metrics = analyzer.compute_efficiency(target_loss=2.5)
        assert 'memory_fp32_mb' in metrics

    def test_layer_breakdown(self):
        from scaling import ModelEfficencyAnalyzer
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
        analyzer = ModelEfficencyAnalyzer(model)
        breakdown = analyzer.layer_param_breakdown()
        assert len(breakdown) > 0
        assert sum(breakdown.values()) > 0

    def test_suggest_compression(self):
        from scaling import ModelEfficencyAnalyzer
        model = nn.Embedding(1000, 64)
        analyzer = ModelEfficencyAnalyzer(model)
        suggestions = analyzer.suggest_compression(target_compression=0.5)
        assert 'suggestions' in suggestions
        assert len(suggestions['suggestions']) > 0


class TestNeuralScalingLawFitter:
    def test_fit_and_predict(self):
        from scaling import NeuralScalingLawFitter
        fitter = NeuralScalingLawFitter()
        # Synthetic data from Chinchilla
        from scaling import ChinchillaPredictor as CP
        ns = [1e6, 1e7, 1e8, 1e9]
        ds = [2e7, 2e8, 2e9, 2e10]
        ls = [CP.predict_loss(n, d) for n, d in zip(ns, ds)]
        params = fitter.fit(ns, ds, ls)
        assert 'E' in params
        assert 'alpha' in params
        # Test predict
        pred = fitter.predict(1e8, 2e9)
        assert pred > 0


@pytest.mark.parametrize('n_params,n_tokens', [
    (1e6, 2e7),
    (1e7, 2e8),
    (1e8, 2e9),
    (1e9, 2e10),
    (7e9, 1.4e11),  # Chinchilla 70B equivalent
])
def test_chinchilla_loss_positive(n_params, n_tokens):
    from scaling import ChinchillaPredictor
    loss = ChinchillaPredictor.predict_loss(n_params, n_tokens)
    assert loss > 0
    assert loss < 10.0
