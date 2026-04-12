"""Tests for interpretability.py extended components."""
import pytest
import torch
import torch.nn as nn
import numpy as np

def _make_simple_model(in_dim=16, out_dim=4, seq=None):
    if seq is None:
        return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))
    # Sequence model
    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, 32)
            self.lin2 = nn.Linear(32, out_dim)
        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))
    return SeqModel()

class TestLIMEExplainer:
    def test_explain_returns_dict(self):
        from interpretability import LIMEExplainer
        model = _make_simple_model(16, 4)
        explainer = LIMEExplainer(model, n_samples=20)
        x = torch.randn(1, 8, 16)
        result = explainer.explain(x)
        assert 'feature_importance' in result
        assert 'time_importance' in result

    def test_feature_importance_shape(self):
        from interpretability import LIMEExplainer
        model = _make_simple_model(8, 2)
        explainer = LIMEExplainer(model, n_samples=10)
        x = torch.randn(1, 4, 8)
        result = explainer.explain(x)
        assert result['feature_importance'].shape == (4, 8)
        assert result['time_importance'].shape == (4,)

    def test_no_nan_in_importance(self):
        from interpretability import LIMEExplainer
        model = _make_simple_model(8, 2)
        explainer = LIMEExplainer(model, n_samples=15)
        x = torch.randn(1, 6, 8)
        result = explainer.explain(x)
        assert not torch.isnan(result['feature_importance']).any()

class TestShapleyValueEstimator:
    def test_shap_values_shape(self):
        from interpretability import ShapleyValueEstimator
        model = _make_simple_model(8, 1)
        bg = torch.zeros(5, 4, 8)
        estimator = ShapleyValueEstimator(model, bg, n_samples=10)
        x = torch.randn(2, 4, 8)
        shap = estimator.shap_values(x)
        assert shap.shape == (2, 4, 8)

    def test_shap_no_nan(self):
        from interpretability import ShapleyValueEstimator
        model = _make_simple_model(4, 1)
        bg = torch.zeros(3, 2, 4)
        estimator = ShapleyValueEstimator(model, bg, n_samples=5)
        x = torch.randn(1, 2, 4)
        shap = estimator.shap_values(x)
        assert not torch.isnan(shap).any()

class TestCausalAttentionAnalyzer:
    def test_analyze_returns_dict(self):
        from interpretability import CausalAttentionAnalyzer
        model = nn.Sequential(
            nn.Linear(16, 16),
        )
        analyzer = CausalAttentionAnalyzer(model)
        x = torch.randn(2, 4, 16)
        analysis = analyzer.analyze_attention(x)
        # No attention layers -> empty dict
        assert isinstance(analysis, dict)

    def test_with_mha_model(self):
        from interpretability import CausalAttentionAnalyzer
        class MHAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(16, 4, batch_first=True)
            def forward(self, x):
                out, _ = self.attn(x, x, x)
                return out
        model = MHAModel()
        analyzer = CausalAttentionAnalyzer(model)
        x = torch.randn(2, 8, 16)
        analysis = analyzer.analyze_attention(x)
        assert len(analysis) > 0

class TestWhatIfAnalyzer:
    def test_feature_ablation(self):
        from interpretability import WhatIfAnalyzer
        model = _make_simple_model(8, 2)
        analyzer = WhatIfAnalyzer(model)
        x = torch.randn(2, 4, 8)
        result = analyzer.feature_ablation(x)
        assert 'feature_ablation_scores' in result
        assert len(result['feature_ablation_scores']) == 8

    def test_timestep_ablation(self):
        from interpretability import WhatIfAnalyzer
        model = _make_simple_model(8, 2)
        analyzer = WhatIfAnalyzer(model)
        x = torch.randn(2, 6, 8)
        result = analyzer.timestep_ablation(x)
        assert len(result['timestep_ablation_scores']) == 6

    def test_sensitivity_analysis(self):
        from interpretability import WhatIfAnalyzer
        model = _make_simple_model(4, 2)
        analyzer = WhatIfAnalyzer(model)
        x = torch.randn(1, 4, 4)
        result = analyzer.sensitivity_analysis(x, feature_idx=0, n_points=10)
        assert len(result['predictions']) == 10
        assert 'monotonicity' in result

class TestModelCardGenerator:
    def test_generate_card(self):
        from interpretability import ModelCardGenerator
        model = nn.Linear(32, 8)
        gen = ModelCardGenerator(model, 'TestModel', '1.0')
        card = gen.generate_card(
            task_description='Test task',
            training_data='Synthetic data',
            evaluation_results={'sharpe': 1.5},
        )
        assert 'model_name' in card
        assert card['model_architecture']['total_parameters'] > 0

    def test_format_markdown(self):
        from interpretability import ModelCardGenerator
        model = nn.Linear(16, 4)
        gen = ModelCardGenerator(model, 'LuminaTest', '0.1')
        card = gen.generate_card('Test', 'Data')
        md = gen.format_markdown(card)
        assert '# LuminaTest' in md
        assert '## Limitations' in md

class TestFinancialInterpretabilityReport:
    def test_feature_importance_report(self):
        from interpretability import FinancialInterpretabilityReport
        model = _make_simple_model(8, 2)
        rep = FinancialInterpretabilityReport(model)
        x = torch.randn(2, 4, 8)
        result = rep.feature_importance_report(x)
        assert 'top_features' in result
        assert 'most_important_timestep' in result

    def test_full_report(self):
        from interpretability import FinancialInterpretabilityReport
        model = _make_simple_model(4, 2)
        rep = FinancialInterpretabilityReport(model)
        x = torch.randn(2, 4, 4)
        result = rep.generate_full_report(x)
        assert 'feature_importance' in result
        assert 'model_complexity' in result
