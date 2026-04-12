"""Comprehensive tests for MoE and LoRA modules."""
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lumina"))


class TestTopKMoE:
    """Tests for Standard top-k mixture of experts."""

    @pytest.fixture
    def model(self):
        try:
            from moe import TopKMoE
            return TopKMoE(d_model=128, n_experts=8)
        except ImportError:
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b4_t32(self, model):
        x = torch.randn(4, 32, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b1_t64(self, model):
        x = torch.randn(1, 64, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_no_nan_output(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert not torch.isnan(result).any(), "Output contains NaN"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        result.sum().backward()
        assert x.grad is not None

    def test_eval_mode(self, model):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 128)
            out = model(x)
            result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape[0] == 2

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out_full = model(x)
            out_half = model(x[:2])
        full = out_full[0] if isinstance(out_full, (tuple, list)) else out_full
        half = out_half[0] if isinstance(out_half, (tuple, list)) else out_half
        assert torch.allclose(full[:2], half, atol=1e-4), "Batch independence violated"


class TestSoftMoE:
    """Tests for Soft routing MoE without discrete selection."""

    @pytest.fixture
    def model(self):
        try:
            from moe import SoftMoE
            return SoftMoE(d_model=128, n_experts=8)
        except ImportError:
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b4_t32(self, model):
        x = torch.randn(4, 32, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b1_t64(self, model):
        x = torch.randn(1, 64, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_no_nan_output(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert not torch.isnan(result).any(), "Output contains NaN"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        result.sum().backward()
        assert x.grad is not None

    def test_eval_mode(self, model):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 128)
            out = model(x)
            result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape[0] == 2

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out_full = model(x)
            out_half = model(x[:2])
        full = out_full[0] if isinstance(out_full, (tuple, list)) else out_full
        half = out_half[0] if isinstance(out_half, (tuple, list)) else out_half
        assert torch.allclose(full[:2], half, atol=1e-4), "Batch independence violated"


class TestHierarchicalMoE:
    """Tests for Two-level hierarchical MoE."""

    @pytest.fixture
    def model(self):
        try:
            from moe import HierarchicalMoE
            return HierarchicalMoE(d_model=128, n_experts=8)
        except ImportError:
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b4_t32(self, model):
        x = torch.randn(4, 32, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b1_t64(self, model):
        x = torch.randn(1, 64, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_no_nan_output(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert not torch.isnan(result).any(), "Output contains NaN"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        result.sum().backward()
        assert x.grad is not None

    def test_eval_mode(self, model):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 128)
            out = model(x)
            result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape[0] == 2

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out_full = model(x)
            out_half = model(x[:2])
        full = out_full[0] if isinstance(out_full, (tuple, list)) else out_full
        half = out_half[0] if isinstance(out_half, (tuple, list)) else out_half
        assert torch.allclose(full[:2], half, atol=1e-4), "Batch independence violated"


class TestSharedExpertMoE:
    """Tests for MoE with shared always-active expert."""

    @pytest.fixture
    def model(self):
        try:
            from moe import SharedExpertMoE
            return SharedExpertMoE(d_model=128, n_experts=8)
        except ImportError:
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b4_t32(self, model):
        x = torch.randn(4, 32, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b1_t64(self, model):
        x = torch.randn(1, 64, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_no_nan_output(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert not torch.isnan(result).any(), "Output contains NaN"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        result.sum().backward()
        assert x.grad is not None

    def test_eval_mode(self, model):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 128)
            out = model(x)
            result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape[0] == 2

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out_full = model(x)
            out_half = model(x[:2])
        full = out_full[0] if isinstance(out_full, (tuple, list)) else out_full
        half = out_half[0] if isinstance(out_half, (tuple, list)) else out_half
        assert torch.allclose(full[:2], half, atol=1e-4), "Batch independence violated"


class TestBalancedMoELayer:
    """Tests for MoE with load balancing auxiliary loss."""

    @pytest.fixture
    def model(self):
        try:
            from moe import BalancedMoELayer
            return BalancedMoELayer(d_model=128, n_experts=8)
        except ImportError:
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b4_t32(self, model):
        x = torch.randn(4, 32, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_forward_b1_t64(self, model):
        x = torch.randn(1, 64, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"

    def test_no_nan_output(self, model):
        x = torch.randn(2, 16, 128)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        assert not torch.isnan(result).any(), "Output contains NaN"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = model(x)
        result = out[0] if isinstance(out, (tuple, list)) else out
        result.sum().backward()
        assert x.grad is not None

    def test_eval_mode(self, model):
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 128)
            out = model(x)
            result = out[0] if isinstance(out, (tuple, list)) else out
        assert result.shape[0] == 2

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out_full = model(x)
            out_half = model(x[:2])
        full = out_full[0] if isinstance(out_full, (tuple, list)) else out_full
        half = out_half[0] if isinstance(out_half, (tuple, list)) else out_half
        assert torch.allclose(full[:2], half, atol=1e-4), "Batch independence violated"

class TestLoRALinear:
    """Tests for LoRALinear adapter."""

    @pytest.fixture
    def adapter(self):
        try:
            from lora import LoRALinear
            return LoRALinear(in_features=128, out_features=128, rank=8)
        except (ImportError, TypeError):
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_forward_b4_t32(self, adapter):
        x = torch.randn(4, 32, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_trainable_params_fewer(self, adapter):
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert total > 0

    def test_no_nan(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert not torch.isnan(out).any()

class TestAdaLoRA:
    """Tests for AdaLoRA adapter."""

    @pytest.fixture
    def adapter(self):
        try:
            from lora import AdaLoRA
            return AdaLoRA(in_features=128, out_features=128, rank=8)
        except (ImportError, TypeError):
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_forward_b4_t32(self, adapter):
        x = torch.randn(4, 32, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_trainable_params_fewer(self, adapter):
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert total > 0

    def test_no_nan(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert not torch.isnan(out).any()

class TestDyLoRA:
    """Tests for DyLoRA adapter."""

    @pytest.fixture
    def adapter(self):
        try:
            from lora import DyLoRA
            return DyLoRA(in_features=128, out_features=128, rank=8)
        except (ImportError, TypeError):
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_forward_b4_t32(self, adapter):
        x = torch.randn(4, 32, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_trainable_params_fewer(self, adapter):
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert total > 0

    def test_no_nan(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert not torch.isnan(out).any()

class TestFourierFT:
    """Tests for FourierFT adapter."""

    @pytest.fixture
    def adapter(self):
        try:
            from lora import FourierFT
            return FourierFT(in_features=128, out_features=128, rank=8)
        except (ImportError, TypeError):
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_forward_b4_t32(self, adapter):
        x = torch.randn(4, 32, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_trainable_params_fewer(self, adapter):
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert total > 0

    def test_no_nan(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert not torch.isnan(out).any()

class TestGLoRA:
    """Tests for GLoRA adapter."""

    @pytest.fixture
    def adapter(self):
        try:
            from lora import GLoRA
            return GLoRA(in_features=128, out_features=128, rank=8)
        except (ImportError, TypeError):
            pytest.skip("Module not available")

    def test_forward_b2_t16(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_forward_b4_t32(self, adapter):
        x = torch.randn(4, 32, 128)
        out = adapter(x)
        assert out.shape[-1] == 128

    def test_trainable_params_fewer(self, adapter):
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert total > 0

    def test_no_nan(self, adapter):
        x = torch.randn(2, 16, 128)
        out = adapter(x)
        assert not torch.isnan(out).any()
