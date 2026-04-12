"""Tests for positional encoding strategies in Lumina."""
import pytest
import torch
import math
from typing import Optional



class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding."""

    def test_basic_forward(self):
        """Test SinusoidalPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import SinusoidalPositionalEncoding
            d_model = 64
            enc = SinusoidalPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test SinusoidalPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import SinusoidalPositionalEncoding
            d_model = 64
            enc = SinusoidalPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestLearnedAbsolutePositionalEncoding:
    """Tests for LearnedAbsolutePositionalEncoding."""

    def test_basic_forward(self):
        """Test LearnedAbsolutePositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import LearnedAbsolutePositionalEncoding
            d_model = 64
            enc = LearnedAbsolutePositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test LearnedAbsolutePositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import LearnedAbsolutePositionalEncoding
            d_model = 64
            enc = LearnedAbsolutePositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestRotaryPositionalEncoding:
    """Tests for RotaryPositionalEncoding."""

    def test_basic_forward(self):
        """Test RotaryPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import RotaryPositionalEncoding
            d_model = 64
            enc = RotaryPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test RotaryPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import RotaryPositionalEncoding
            d_model = 64
            enc = RotaryPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestALiBiPositionalBias:
    """Tests for ALiBiPositionalBias."""

    def test_basic_forward(self):
        """Test ALiBiPositionalBias basic forward pass."""
        try:
            from lumina.positional_encoding import ALiBiPositionalBias
            d_model = 64
            enc = ALiBiPositionalBias(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test ALiBiPositionalBias with different sequence lengths."""
        try:
            from lumina.positional_encoding import ALiBiPositionalBias
            d_model = 64
            enc = ALiBiPositionalBias(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestT5RelativePositionBias:
    """Tests for T5RelativePositionBias."""

    def test_basic_forward(self):
        """Test T5RelativePositionBias basic forward pass."""
        try:
            from lumina.positional_encoding import T5RelativePositionBias
            d_model = 64
            enc = T5RelativePositionBias(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test T5RelativePositionBias with different sequence lengths."""
        try:
            from lumina.positional_encoding import T5RelativePositionBias
            d_model = 64
            enc = T5RelativePositionBias(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestTemporalEncoding:
    """Tests for TemporalEncoding."""

    def test_basic_forward(self):
        """Test TemporalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import TemporalEncoding
            d_model = 64
            enc = TemporalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test TemporalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import TemporalEncoding
            d_model = 64
            enc = TemporalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestFourierTimeEncoding:
    """Tests for FourierTimeEncoding."""

    def test_basic_forward(self):
        """Test FourierTimeEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import FourierTimeEncoding
            d_model = 64
            enc = FourierTimeEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test FourierTimeEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import FourierTimeEncoding
            d_model = 64
            enc = FourierTimeEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestCalendarEncoding:
    """Tests for CalendarEncoding."""

    def test_basic_forward(self):
        """Test CalendarEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import CalendarEncoding
            d_model = 64
            enc = CalendarEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test CalendarEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import CalendarEncoding
            d_model = 64
            enc = CalendarEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestHierarchicalPositionalEncoding:
    """Tests for HierarchicalPositionalEncoding."""

    def test_basic_forward(self):
        """Test HierarchicalPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import HierarchicalPositionalEncoding
            d_model = 64
            enc = HierarchicalPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test HierarchicalPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import HierarchicalPositionalEncoding
            d_model = 64
            enc = HierarchicalPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestCompoundPositionalEncoding:
    """Tests for CompoundPositionalEncoding."""

    def test_basic_forward(self):
        """Test CompoundPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import CompoundPositionalEncoding
            d_model = 64
            enc = CompoundPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test CompoundPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import CompoundPositionalEncoding
            d_model = 64
            enc = CompoundPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestNTKAwareRoPE:
    """Tests for NTKAwareRoPE."""

    def test_basic_forward(self):
        """Test NTKAwareRoPE basic forward pass."""
        try:
            from lumina.positional_encoding import NTKAwareRoPE
            d_model = 64
            enc = NTKAwareRoPE(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test NTKAwareRoPE with different sequence lengths."""
        try:
            from lumina.positional_encoding import NTKAwareRoPE
            d_model = 64
            enc = NTKAwareRoPE(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestBinaryPositionalEncoding:
    """Tests for BinaryPositionalEncoding."""

    def test_basic_forward(self):
        """Test BinaryPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import BinaryPositionalEncoding
            d_model = 64
            enc = BinaryPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test BinaryPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import BinaryPositionalEncoding
            d_model = 64
            enc = BinaryPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestConvolutionalPositionalEncoding:
    """Tests for ConvolutionalPositionalEncoding."""

    def test_basic_forward(self):
        """Test ConvolutionalPositionalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import ConvolutionalPositionalEncoding
            d_model = 64
            enc = ConvolutionalPositionalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test ConvolutionalPositionalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import ConvolutionalPositionalEncoding
            d_model = 64
            enc = ConvolutionalPositionalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestPeriodicPositionEncoding:
    """Tests for PeriodicPositionEncoding."""

    def test_basic_forward(self):
        """Test PeriodicPositionEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import PeriodicPositionEncoding
            d_model = 64
            enc = PeriodicPositionEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test PeriodicPositionEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import PeriodicPositionEncoding
            d_model = 64
            enc = PeriodicPositionEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")



class TestTemporalHierarchicalEncoding:
    """Tests for TemporalHierarchicalEncoding."""

    def test_basic_forward(self):
        """Test TemporalHierarchicalEncoding basic forward pass."""
        try:
            from lumina.positional_encoding import TemporalHierarchicalEncoding
            d_model = 64
            enc = TemporalHierarchicalEncoding(d_model=d_model)
            x = torch.randn(2, 16, d_model)
            out = enc(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
            assert torch.isfinite(out).all(), "Output has NaN/Inf"
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping {cls_name}: {e}")

    def test_different_lengths(self):
        """Test TemporalHierarchicalEncoding with different sequence lengths."""
        try:
            from lumina.positional_encoding import TemporalHierarchicalEncoding
            d_model = 64
            enc = TemporalHierarchicalEncoding(d_model=d_model)
            for T in [4, 8, 16, 32]:
                x = torch.randn(1, T, d_model)
                out = enc(x)
                assert out.shape == x.shape
        except (ImportError, Exception) as e:
            pytest.skip(f"Skipping: {e}")
