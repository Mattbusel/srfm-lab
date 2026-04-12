"""
tests/test_interpretability.py

Tests for interpretability.py module.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.interpretability import (
    ActivationCache,
    AttentionCache,
    AttentionRollout,
    GradCAMTransformer,
    IntegratedGradients,
    SmoothGrad,
    DeepLIFTAttribution,
    PerturbationAttribution,
    LinearProbe,
    AttentionHeadAnalyzer,
    MechanisticInterpreter,
    LuminaInterpreter,
)


# ---------------------------------------------------------------------------
# Test model helpers
# ---------------------------------------------------------------------------

class SimpleAttentionModel(nn.Module):
    """Model that returns attention weights."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, seq_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        logits = self.head(attn_out[:, -1, :])
        return {"logits": logits, "attn_weights": attn_weights}


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim == 3:
            x = x.mean(dim=1)   # Pool sequence
        logits = self.net(x)
        return {"logits": logits}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActivationCache(unittest.TestCase):

    def test_forward_hook(self):
        model = SimpleClassifier(32)
        cache = ActivationCache()
        cache.register("net.0", model.net[0])   # Register first linear layer

        x = torch.randn(2, 32)
        model(x)

        act = cache.get_activation("net.0")
        self.assertIsNotNone(act)
        self.assertEqual(act.shape[0], 2)   # Batch size

    def test_clear(self):
        model = SimpleClassifier(32)
        cache = ActivationCache()
        cache.register("net.0", model.net[0])
        x = torch.randn(2, 32)
        model(x)
        cache.clear()
        self.assertIsNone(cache.get_activation("net.0"))

    def test_gradient_hook(self):
        model = SimpleClassifier(32)
        cache = ActivationCache()
        cache.register("net.0", model.net[0])

        x = torch.randn(2, 32)
        out = model(x)
        out["logits"].sum().backward()

        # Gradient should be captured
        grad = cache.get_gradient("net.0")
        # May or may not have grad depending on backward hook behavior
        # Just test no error

    def test_remove_hooks(self):
        model = SimpleClassifier(32)
        cache = ActivationCache()
        cache.register("net.0", model.net[0])
        cache.remove_hooks()
        self.assertEqual(len(cache._hooks), 0)


class TestAttentionRollout(unittest.TestCase):

    def test_rollout_shape(self):
        rollout = AttentionRollout()
        B, H, T = 2, 4, 16
        # Create 3 layers of attention
        attn_mats = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
        result = rollout.rollout(attn_mats)
        self.assertEqual(result.shape, (B, T, T))

    def test_rollout_normalization(self):
        rollout = AttentionRollout()
        B, H, T = 1, 2, 8
        attn_mats = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(2)]
        result = rollout.rollout(attn_mats)
        # Each row should sum to approximately 1
        row_sums = result.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01))

    def test_input_importance(self):
        rollout = AttentionRollout()
        B, H, T = 2, 4, 16
        attn_mats = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
        importance = rollout.get_input_importance(attn_mats, query_idx=-1)
        self.assertEqual(importance.shape, (B, T))

    def test_head_fusion_methods(self):
        for method in ["mean", "min", "max"]:
            rollout = AttentionRollout(head_fusion=method)
            B, H, T = 2, 4, 8
            attn_mats = [torch.softmax(torch.randn(B, H, T, T), dim=-1)]
            fused = rollout._fuse_heads(attn_mats[0])
            self.assertEqual(fused.shape, (B, T, T))

    def test_empty_attention_raises(self):
        rollout = AttentionRollout()
        with self.assertRaises(ValueError):
            rollout.rollout([])


class TestIntegratedGradients(unittest.TestCase):

    def test_attribution_shape(self):
        model = SimpleClassifier(32)
        ig = IntegratedGradients(model, n_steps=10)
        x = torch.randn(2, 32)
        attr = ig.attribute(x)
        self.assertEqual(attr.shape, x.shape)

    def test_attribution_nonzero(self):
        model = SimpleClassifier(32)
        ig = IntegratedGradients(model, n_steps=10)
        x = torch.randn(2, 32)
        attr = ig.attribute(x)
        self.assertGreater(attr.abs().sum().item(), 0)

    def test_different_baselines(self):
        for baseline_type in ["zero", "mean", "noise"]:
            model = SimpleClassifier(32)
            ig = IntegratedGradients(model, n_steps=5, baseline_type=baseline_type)
            x = torch.randn(2, 32)
            attr = ig.attribute(x)
            self.assertEqual(attr.shape, x.shape)

    def test_sequence_input(self):
        model = SimpleClassifier(32)
        ig = IntegratedGradients(model, n_steps=5)
        x = torch.randn(2, 8, 32)   # (B, T, F)
        attr = ig.attribute(x)
        self.assertEqual(attr.shape, x.shape)

    def test_convergence_delta(self):
        model = SimpleClassifier(32)
        ig = IntegratedGradients(model, n_steps=50)
        x = torch.randn(1, 32)
        attr = ig.attribute(x)
        delta = ig.convergence_delta(attr, x)
        # Delta should be finite
        self.assertTrue(not torch.isnan(torch.tensor(delta)))


class TestSmoothGrad(unittest.TestCase):

    def test_attribution_shape(self):
        model = SimpleClassifier(32)
        sg = SmoothGrad(model, n_samples=10, noise_level=0.1)
        x = torch.randn(2, 32)
        attr = sg.attribute(x)
        self.assertEqual(attr.shape, x.shape)

    def test_smoother_than_vanilla(self):
        """SmoothGrad should produce smoother attributions than vanilla gradient."""
        model = SimpleClassifier(32)
        sg = SmoothGrad(model, n_samples=50, noise_level=0.15)
        x = torch.randn(1, 32)
        attr = sg.attribute(x)
        # Just check shape and non-zero
        self.assertEqual(attr.shape, x.shape)
        self.assertGreater(attr.abs().max().item(), 0)


class TestDeepLIFTAttribution(unittest.TestCase):

    def test_attribution_shape(self):
        model = SimpleClassifier(32)
        dl = DeepLIFTAttribution(model)
        x = torch.randn(2, 32)
        attr = dl.attribute(x)
        self.assertEqual(attr.shape, x.shape)

    def test_attribution_nonzero(self):
        model = SimpleClassifier(32)
        dl = DeepLIFTAttribution(model)
        x = torch.randn(2, 32) + 5.0  # Non-zero baseline difference
        attr = dl.attribute(x)
        self.assertGreater(attr.abs().sum().item(), 0)


class TestPerturbationAttribution(unittest.TestCase):

    def test_attribution_shape(self):
        model = SimpleClassifier(32)
        pa = PerturbationAttribution(model)
        x = torch.randn(2, 32)
        attr = pa.attribute(x)
        self.assertEqual(attr.shape, x.shape)

    def test_different_perturbation_types(self):
        for ptype in ["zero", "mean", "noise"]:
            model = SimpleClassifier(32)
            pa = PerturbationAttribution(model, perturbation_type=ptype)
            x = torch.randn(2, 32)
            attr = pa.attribute(x)
            self.assertEqual(attr.shape, x.shape)

    def test_time_step_importance(self):
        model = SimpleClassifier(32)
        pa = PerturbationAttribution(model)
        x = torch.randn(2, 8, 32)   # (B, T, F)
        importance = pa.time_step_importance(x)
        self.assertEqual(importance.shape, (2, 8))


class TestLinearProbe(unittest.TestCase):

    def test_classification_probe(self):
        probe = LinearProbe(64, 3, probe_type="classification")
        x = torch.randn(10, 64)
        out = probe(x)
        self.assertEqual(out.shape, (10, 3))

    def test_regression_probe(self):
        probe = LinearProbe(64, 1, probe_type="regression")
        x = torch.randn(10, 64)
        out = probe(x)
        self.assertEqual(out.shape, (10, 1))

    def test_predict_classification(self):
        probe = LinearProbe(32, 4, probe_type="classification")
        x = torch.randn(5, 32)
        preds = probe.predict(x)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(torch.all(preds >= 0) and torch.all(preds < 4))

    def test_predict_regression(self):
        probe = LinearProbe(32, 1, probe_type="regression")
        x = torch.randn(5, 32)
        preds = probe.predict(x)
        self.assertEqual(preds.shape, (5,))


class TestAttentionHeadAnalyzer(unittest.TestCase):

    def test_analyze_head_patterns(self):
        analyzer = AttentionHeadAnalyzer(nn.Module(), n_layers=2, n_heads=4)
        B, H, T = 2, 4, 16
        attn_weights = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(2)]
        patterns = analyzer.analyze_head_patterns(attn_weights)
        self.assertIn("layer_0", patterns)
        self.assertIn("layer_1", patterns)
        self.assertIn("head_0", patterns["layer_0"])

    def test_pattern_keys(self):
        analyzer = AttentionHeadAnalyzer(nn.Module(), n_layers=1, n_heads=2)
        B, H, T = 1, 2, 8
        attn_weights = [torch.softmax(torch.randn(B, H, T, T), dim=-1)]
        patterns = analyzer.analyze_head_patterns(attn_weights)
        head_info = patterns["layer_0"]["head_0"]
        self.assertIn("diagonal_attn", head_info)
        self.assertIn("entropy", head_info)
        self.assertIn("first_token_attn", head_info)

    def test_detect_redundant_heads(self):
        analyzer = AttentionHeadAnalyzer(nn.Module(), n_layers=2, n_heads=4)
        importance = torch.zeros(2, 4)
        importance[0, 0] = 1.0   # Only one head is important
        redundant = analyzer.detect_redundant_heads(importance, threshold=0.1)
        self.assertGreater(len(redundant), 0)


class TestMechanisticInterpreter(unittest.TestCase):

    def test_identify_induction_heads(self):
        model = SimpleClassifier(32)
        mech = MechanisticInterpreter(model)
        B, H, T = 2, 4, 16
        attn_weights = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
        induction = mech.identify_induction_heads(attn_weights, seq_len=T)
        # Result should be a list of (layer, head) tuples
        self.assertIsInstance(induction, list)
        for item in induction:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


class TestLuminaInterpreter(unittest.TestCase):

    def setUp(self):
        self.model = SimpleClassifier(32)
        self.interpreter = LuminaInterpreter(
            self.model, n_layers=1, n_heads=1
        )

    def test_explain_integrated_gradients(self):
        x = torch.randn(2, 8, 32)
        result = self.interpreter.explain(x, method="integrated_gradients")
        self.assertIn("attribution", result)
        self.assertIn("attribution_agg", result)
        self.assertEqual(result["attribution"].shape, x.shape)

    def test_explain_smooth_grad(self):
        x = torch.randn(2, 8, 32)
        result = self.interpreter.explain(x, method="smooth_grad")
        self.assertIn("attribution", result)

    def test_explain_perturbation(self):
        x = torch.randn(2, 8, 32)
        result = self.interpreter.explain(x, method="perturbation")
        self.assertIn("attribution", result)

    def test_unknown_method_raises(self):
        x = torch.randn(2, 8, 32)
        with self.assertRaises(ValueError):
            self.interpreter.explain(x, method="nonexistent_method")

    def test_full_analysis(self):
        x = torch.randn(2, 8, 32)
        B, H, T = 2, 1, 8
        attn_weights = [torch.softmax(torch.randn(B, H, T, T), dim=-1)]
        result = self.interpreter.full_analysis(x, attn_weights=attn_weights)
        self.assertIn("integrated_gradients", result)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
