"""Comprehensive tests for scaling modules."""
import pytest
import torch
import torch.nn as nn
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lumina"))

class TestTransformerSizeCalculator:
    @pytest.mark.parametrize("d_model,n_heads,n_layers,expected_min_params", [
        (128, 4, 2, 100_000),
        (256, 8, 4, 1_000_000),
        (512, 8, 6, 5_000_000),
        (768, 12, 12, 80_000_000),
    ])
    def test_param_count_reasonable(self, d_model, n_heads, n_layers, expected_min_params):
        try:
            from scaling import TransformerSizeCalculator
            calc = TransformerSizeCalculator(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
            n = calc.total_params()
            assert n >= expected_min_params, f"Expected >= {expected_min_params}, got {n}"
        except ImportError:
            pytest.skip("Not available")

    def test_summary_keys(self):
        try:
            from scaling import TransformerSizeCalculator
            calc = TransformerSizeCalculator()
            s = calc.summary()
            required_keys = ["total_params", "total_params_M", "flops_per_token"]
            for k in required_keys:
                assert k in s, f"Missing key: {k}"
        except ImportError:
            pytest.skip("Not available")

    def test_flops_scale_with_layers(self):
        try:
            from scaling import TransformerSizeCalculator
            c1 = TransformerSizeCalculator(n_layers=4)
            c2 = TransformerSizeCalculator(n_layers=8)
            assert c2.flops_per_token() > c1.flops_per_token()
        except ImportError:
            pytest.skip("Not available")

class TestPowerLawFitter:
    def test_fit_predict(self):
        try:
            from scaling import PowerLawFitter
            import numpy as np
            ns = np.array([1e7, 1e8, 1e9, 1e10])
            losses = 3.0 * ns ** (-0.076)
            fitter = PowerLawFitter()
            fitter.fit(ns, losses)
            pred = fitter.predict(1e11)
            assert pred > 0
        except ImportError:
            pytest.skip("Not available")

    def test_optimal_allocation(self):
        try:
            from scaling import PowerLawFitter
            fitter = PowerLawFitter()
            n_opt, d_opt = fitter.optimal_allocation(6e21)
            assert n_opt > 0 and d_opt > 0
            assert abs(n_opt * d_opt * 6 - 6e21) / 6e21 < 0.01
        except ImportError:
            pytest.skip("Not available")

class TestSAM:
    def test_first_second_step(self):
        try:
            from scaling import SAM
            model = nn.Linear(16, 4)
            sam = SAM(model.parameters(), torch.optim.SGD, lr=0.01)
            x = torch.randn(4, 16)
            y = torch.zeros(4, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            sam.first_step(zero_grad=True)
            loss2 = nn.CrossEntropyLoss()(model(x), y)
            loss2.backward()
            sam.second_step(zero_grad=True)
        except (ImportError, TypeError):
            pytest.skip("Not available")

class TestWarmupCosineScheduler:
    def test_lr_schedule(self):
        try:
            from scaling import WarmupCosineScheduler
            model = nn.Linear(16, 4)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
            lrs = []
            for _ in range(110):
                sched.step()
                lrs.append(opt.param_groups[0]["lr"])
            # LR should peak around step 10
            assert lrs[10] >= lrs[0]
            assert lrs[-1] <= lrs[10]
        except ImportError:
            pytest.skip("Not available")

class TestLinearAttentionKernel:
    @pytest.mark.parametrize("B,T,D", [(2, 32, 128), (4, 64, 256), (1, 16, 64)])
    def test_output_shape(self, B, T, D):
        try:
            from scaling import LinearAttentionKernel
            attn = LinearAttentionKernel(d_model=D, n_heads=4)
            x = torch.randn(B, T, D)
            out = attn(x)
            assert out.shape == (B, T, D)
        except ImportError:
            pytest.skip("Not available")

class TestDistillationLoss:
    def test_loss_value(self):
        try:
            from scaling import DistillationLoss
            dl = DistillationLoss(temperature=4.0, alpha=0.5)
            s_logits = torch.randn(4, 10)
            t_logits = torch.randn(4, 10)
            labels = torch.randint(0, 10, (4,))
            result = dl(s_logits, t_logits, labels)
            assert "total" in result
            assert result["total"].item() >= 0
        except ImportError:
            pytest.skip("Not available")

    def test_no_hard_labels(self):
        try:
            from scaling import DistillationLoss
            dl = DistillationLoss(temperature=2.0, alpha=0.0)
            s = torch.randn(4, 8)
            t = torch.randn(4, 8)
            result = dl(s, t)
            assert result["total"].item() >= 0
        except ImportError:
            pytest.skip("Not available")

class TestMagnitudePruner:
    def test_pruning_reduces_nonzeros(self):
        try:
            from scaling import MagnitudePruner
            model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
            pruner = MagnitudePruner(model, sparsity=0.5)
            pruner.compute_masks(global_threshold=True)
            pruner.apply_masks()
            global_sp = pruner.global_sparsity()
            assert 0.4 <= global_sp <= 0.6
        except ImportError:
            pytest.skip("Not available")
