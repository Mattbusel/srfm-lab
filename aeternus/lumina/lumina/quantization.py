"""
quantization.py
===============
Model quantization for faster MoE inference in Lumina.

Schemes:
  1. INT8 static quantization for expert linear layers (calibrated on financial data)
  2. FP16 mixed precision inference with automatic loss scaling
  3. GPTQ-style weight quantization for large expert FFN layers
  4. Quantization-Aware Training (QAT) for minimal accuracy loss
  5. Benchmark: accuracy degradation vs latency improvement per scheme
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.ao.quantization import (
    QConfig,
    get_default_qconfig,
    prepare,
    convert,
    QuantStub,
    DeQuantStub,
    fuse_modules,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BITS = 8                # INT8 quantization
GPTQ_BITS = 4                  # GPTQ uses 4-bit
GPTQ_GROUP_SIZE = 128           # GPTQ weight group size
AMP_INIT_SCALE = 2.0 ** 16     # AMP loss scaler initial value
CALIBRATION_BATCHES = 100

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QuantizationConfig:
    scheme: str = "int8_static"         # int8_static | fp16_amp | gptq | qat | none
    bits: int = DEFAULT_BITS
    symmetric: bool = True
    per_channel: bool = True
    calibration_batches: int = CALIBRATION_BATCHES
    gptq_group_size: int = GPTQ_GROUP_SIZE
    gptq_percdamp: float = 0.01
    qat_epochs: int = 3
    qat_lr: float = 1e-5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclasses.dataclass
class QuantizationBenchmarkResult:
    scheme: str
    batch_size: int
    seq_len: int
    fp32_latency_ms: float
    quantized_latency_ms: float
    speedup: float
    fp32_output_norm: float
    quant_output_norm: float
    relative_error: float
    memory_fp32_mb: float
    memory_quant_mb: float
    memory_reduction: float


# ---------------------------------------------------------------------------
# Calibration data generator (financial data proxy)
# ---------------------------------------------------------------------------


class FinancialCalibrationDataset:
    """
    Generates calibration data representative of financial LOB features.
    In production this would be loaded from historical data.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        seq_len: int = 64,
        batch_size: int = 16,
        n_batches: int = CALIBRATION_BATCHES,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
        self.dtype = dtype
        self._rng = np.random.default_rng(42)

    def __iter__(self):
        for _ in range(self.n_batches):
            # Simulate realistic LOB feature distributions:
            # - Returns: fat-tailed (t-distribution with low df)
            # - Order imbalances: bounded [-1, 1]
            # - Volume features: log-normal
            returns = self._rng.standard_t(df=3, size=(self.batch_size, self.seq_len, self.hidden_dim // 4))
            imbalances = self._rng.uniform(-1, 1, (self.batch_size, self.seq_len, self.hidden_dim // 4))
            volumes = np.exp(self._rng.normal(0, 1, (self.batch_size, self.seq_len, self.hidden_dim // 4)))
            spreads = np.abs(self._rng.normal(0, 0.5, (self.batch_size, self.seq_len, self.hidden_dim // 4)))

            features = np.concatenate([returns, imbalances, volumes, spreads], axis=-1)
            features = features.astype(np.float32)

            # Clip extremes (representative of real preprocessing)
            features = np.clip(features, -10.0, 10.0)

            yield torch.from_numpy(features).to(self.device).to(self.dtype)

    def __len__(self):
        return self.n_batches


# ---------------------------------------------------------------------------
# 1. INT8 Static Quantization
# ---------------------------------------------------------------------------


class INT8StaticQuantizer:
    """
    Static INT8 quantization for MoE expert linear layers.
    Uses PyTorch's native quantization framework with per-channel weight quantization
    and per-tensor activation quantization (calibrated on financial data).
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        hidden_dim: int = 512,
    ) -> nn.Module:
        """
        Quantize the model to INT8 using static quantization.
        Returns the quantized model (CPU only for static quant).
        """
        model = copy.deepcopy(model).cpu().eval()

        if calibration_data is None:
            calibration_data = FinancialCalibrationDataset(
                hidden_dim=hidden_dim,
                n_batches=self.config.calibration_batches,
                device="cpu",
            )

        # Wrap with quant/dequant stubs
        model_wrapped = _QuantizableWrapper(model)

        # Select qconfig
        if self.config.per_channel:
            qconfig = get_default_qconfig("fbgemm")  # per-channel for weights
        else:
            qconfig = get_default_qconfig("qnnpack")  # per-tensor

        model_wrapped.qconfig = qconfig

        # Prepare for calibration
        try:
            model_prepared = prepare(model_wrapped, inplace=False)
        except Exception as e:
            logger.warning(f"INT8 prepare failed ({e}), trying fx-based quantization")
            return self._quantize_fx(model, calibration_data, hidden_dim)

        # Run calibration passes
        logger.info(f"Running INT8 calibration ({self.config.calibration_batches} batches)...")
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                try:
                    model_prepared(batch)
                except Exception:
                    break

        # Convert to quantized model
        try:
            model_quant = convert(model_prepared, inplace=False)
            logger.info("INT8 static quantization complete")
            return model_quant
        except Exception as e:
            logger.warning(f"INT8 convert failed ({e}), returning original model")
            return model

    def _quantize_fx(
        self,
        model: nn.Module,
        calibration_data: Any,
        hidden_dim: int,
    ) -> nn.Module:
        """FX-graph-mode quantization (more robust than eager mode)."""
        try:
            qconfig_mapping = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
            example_input = torch.randn(1, 32, hidden_dim)
            model_prepared = prepare_fx(model, qconfig_mapping, example_input)

            with torch.no_grad():
                for i, batch in enumerate(calibration_data):
                    if i >= self.config.calibration_batches:
                        break
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    try:
                        model_prepared(batch)
                    except Exception:
                        break

            model_quant = convert_fx(model_prepared)
            logger.info("INT8 FX quantization complete")
            return model_quant
        except Exception as e:
            logger.warning(f"INT8 FX quantization failed ({e})")
            return model


class _QuantizableWrapper(nn.Module):
    """Thin wrapper to add QuantStub/DeQuantStub to arbitrary models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# 2. FP16 Mixed Precision Inference
# ---------------------------------------------------------------------------


class FP16MixedPrecisionManager:
    """
    Manages FP16 mixed-precision inference using torch.cuda.amp.
    Includes automatic loss scaling for training stability.
    """

    def __init__(
        self,
        init_scale: float = AMP_INIT_SCALE,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self._scaler = (
            torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=self.enabled,
            )
            if torch.cuda.is_available()
            else None
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def autocast_ctx(self):
        """Return an autocast context for inference."""
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return _null_context()

    def scale_loss(self, loss: Tensor) -> Tensor:
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    @torch.no_grad()
    def infer(self, model: nn.Module, inputs: Tensor) -> Tensor:
        """Run FP16 inference."""
        with self.autocast_ctx:
            return model(inputs)

    @property
    def loss_scale(self) -> float:
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        if self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state)


class _null_context:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


def convert_model_to_fp16(
    model: nn.Module,
    exclude_layers: Optional[List[str]] = None,
    keep_fp32_modules: Optional[List[type]] = None,
) -> nn.Module:
    """
    Convert model to FP16, keeping certain layers in FP32 for stability.

    Layers kept in FP32 by default:
    - LayerNorm (numerical stability)
    - Softmax (precision needed for top-k routing)
    - Embedding layers
    """
    exclude_types = keep_fp32_modules or [
        nn.LayerNorm,
        nn.Embedding,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    ]

    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if exclude_layers and any(exc in name for exc in exclude_layers):
            continue
        if any(isinstance(module, t) for t in exclude_types):
            # Keep in FP32
            module.float()
        else:
            # Convert to FP16
            try:
                module.half()
            except Exception:
                pass

    return model


# ---------------------------------------------------------------------------
# 3. GPTQ-Style Weight Quantization
# ---------------------------------------------------------------------------


class GPTQQuantizer:
    """
    GPTQ (Gradient-Free Post-Training Quantization) for large expert FFN layers.

    Algorithm (Frantar et al., 2022):
    1. Compute the Hessian of each weight matrix using activations
    2. Quantize weights column-by-column, compensating for quantization error
       using the inverse Hessian

    This achieves near-lossless 4-bit weight quantization.
    """

    def __init__(
        self,
        bits: int = GPTQ_BITS,
        group_size: int = GPTQ_GROUP_SIZE,
        percdamp: float = 0.01,
        sym: bool = False,
        act_order: bool = True,   # order columns by activation magnitude
    ):
        self.bits = bits
        self.group_size = group_size
        self.percdamp = percdamp
        self.sym = sym
        self.act_order = act_order

    def quantize_layer(
        self,
        weight: Tensor,
        hessian: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize a single weight matrix W using the GPTQ algorithm.

        Args:
            weight:  (out_features, in_features) float32
            hessian: (in_features, in_features) float32 — Hessian of loss w.r.t. input

        Returns:
            quantized_weight: (out_features, in_features) float32 (dequantized)
            scales:           (out_features, n_groups) float32
            zeros:            (out_features, n_groups) float32
        """
        W = weight.clone().float()
        n_rows, n_cols = W.shape
        n_groups = math.ceil(n_cols / self.group_size)

        scales = torch.zeros(n_rows, n_groups, device=W.device)
        zeros = torch.zeros(n_rows, n_groups, device=W.device)
        Q = torch.zeros_like(W)

        # Regularize Hessian
        H = hessian.clone().float()
        dead = H.diagonal() == 0
        H[dead, dead] = 1.0
        damp = self.percdamp * H.diagonal().mean()
        H.diagonal().add_(damp)

        # Cholesky decomposition of H (more stable than direct inverse)
        try:
            H_inv = torch.linalg.inv(H)
        except Exception:
            H_inv = torch.eye(n_cols, device=W.device)
            logger.warning("GPTQ: Hessian inversion failed, using identity")

        # Optional: reorder columns by activation importance
        perm = None
        if self.act_order:
            perm = H.diagonal().argsort(descending=True)
            W = W[:, perm]
            H_inv = H_inv[perm][:, perm]

        # Quantize column-by-column (or group-by-group)
        for col_start in range(0, n_cols, 1):
            col_end = min(col_start + 1, n_cols)
            g = col_start // self.group_size

            w = W[:, col_start:col_end]

            # Compute scale and zero-point for this group (first col of group)
            if col_start % self.group_size == 0:
                w_group = W[:, col_start: min(col_start + self.group_size, n_cols)]
                if self.sym:
                    scale = w_group.abs().max(dim=1).values / ((2 ** (self.bits - 1)) - 1)
                    scale = scale.clamp(min=1e-8)
                    zero = torch.zeros_like(scale)
                else:
                    w_min = w_group.min(dim=1).values
                    w_max = w_group.max(dim=1).values
                    scale = (w_max - w_min) / (2 ** self.bits - 1)
                    scale = scale.clamp(min=1e-8)
                    zero = -w_min / scale
                scales[:, g] = scale
                zeros[:, g] = zero

            # Quantize
            g = col_start // self.group_size
            sc = scales[:, g].unsqueeze(1)
            zp = zeros[:, g].unsqueeze(1)

            q = (w / sc + zp).round().clamp(0, 2 ** self.bits - 1)
            q_dequant = (q - zp) * sc

            Q[:, col_start:col_end] = q_dequant

            # Propagate quantization error to remaining columns
            err = (w - q_dequant) / H_inv[col_start:col_end, col_start:col_end].diag().unsqueeze(0)
            if col_start + 1 < n_cols:
                W[:, col_start + 1:] -= err @ H_inv[col_start:col_end, col_start + 1:]

        # Restore original column order
        if perm is not None:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm]

        return Q, scales, zeros

    def quantize_model_experts(
        self,
        model: nn.Module,
        calibration_data: Any,
        hidden_dim: int = 512,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Quantize all expert FFN layers in the model using GPTQ.
        Hooks into the model to collect Hessians from calibration data.
        """
        model = copy.deepcopy(model).to(device).eval()

        # Find all linear layers in expert modules
        linear_layers = {}
        for name, module in model.named_modules():
            if "expert" in name.lower() and isinstance(module, nn.Linear):
                linear_layers[name] = module

        if not linear_layers:
            logger.warning("GPTQ: no expert linear layers found")
            return model

        # Collect Hessians via forward hooks
        hessians: Dict[str, Tensor] = {}
        hooks: List[Any] = []
        n_samples = 0

        def make_hook(layer_name: str, in_dim: int):
            hessians[layer_name] = torch.zeros(in_dim, in_dim, device=device)

            def hook(module, input, output):
                nonlocal n_samples
                x = input[0].detach().float()
                if x.dim() == 3:
                    x = x.view(-1, x.shape[-1])
                hessians[layer_name] += x.T @ x
                n_samples += 1
            return hook

        for name, layer in linear_layers.items():
            h = make_hook(name, layer.in_features)
            hooks.append(layer.register_forward_hook(h))

        # Run calibration
        logger.info(f"GPTQ: collecting Hessians from {len(linear_layers)} layers...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.group_size:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                try:
                    batch = batch.to(device)
                    model(batch)
                except Exception:
                    break

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Quantize each layer
        for name, layer in linear_layers.items():
            if name not in hessians:
                continue
            logger.debug(f"GPTQ quantizing layer: {name}")
            H = hessians[name] / max(n_samples, 1)

            try:
                q_weight, scales, zeros = self.quantize_layer(layer.weight.data, H)
                layer.weight.data = q_weight
            except Exception as e:
                logger.warning(f"GPTQ failed for layer {name}: {e}")

        logger.info(f"GPTQ quantization complete: {len(linear_layers)} layers processed")
        return model


# ---------------------------------------------------------------------------
# Packed INT4 weight storage
# ---------------------------------------------------------------------------


class PackedInt4Weight:
    """
    Packs two 4-bit weights into one INT8 byte for memory efficiency.
    Used to store GPTQ quantized weights compactly.
    """

    def __init__(self, weight: Tensor, bits: int = 4):
        assert bits == 4, "Only 4-bit packing supported"
        self.bits = bits
        self.shape = weight.shape
        self.packed = self._pack(weight.int())

    def _pack(self, weight: Tensor) -> Tensor:
        """Pack int4 weights: 2 values per byte."""
        flat = weight.view(-1)
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
        lo = flat[0::2] & 0xF
        hi = flat[1::2] & 0xF
        packed = (hi << 4) | lo
        return packed.to(torch.uint8)

    def unpack(self) -> Tensor:
        """Unpack int4 weights back to int8."""
        lo = (self.packed & 0xF).int()
        hi = ((self.packed >> 4) & 0xF).int()
        interleaved = torch.stack([lo, hi], dim=1).view(-1)
        n = math.prod(self.shape)
        return interleaved[:n].view(self.shape)

    def memory_bytes(self) -> int:
        return self.packed.numel()

    def original_memory_bytes(self) -> int:
        return math.prod(self.shape) * 4  # float32


# ---------------------------------------------------------------------------
# 4. Quantization-Aware Training (QAT)
# ---------------------------------------------------------------------------


class FakeQuantize(nn.Module):
    """
    Fake-quantize operation for QAT.
    During forward: quantize then dequantize (simulates quantization error).
    Gradients flow through via STE (Straight-Through Estimator).
    """

    def __init__(
        self,
        bits: int = 8,
        per_channel: bool = False,
        symmetric: bool = True,
        ch_dim: int = 0,
    ):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.ch_dim = ch_dim
        self.enabled = True

        # Running scale/zero-point (updated via EMA)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.register_buffer("n_calibrated", torch.tensor(0))

        self.q_min = -(2 ** (bits - 1)) if symmetric else 0
        self.q_max = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1

    def update_stats(self, x: Tensor) -> None:
        """Update scale/zero-point estimates from a new batch."""
        with torch.no_grad():
            if self.per_channel:
                dims = list(range(x.dim()))
                dims.remove(self.ch_dim)
                x_min = x.amin(dim=dims)
                x_max = x.amax(dim=dims)
            else:
                x_min = x.min()
                x_max = x.max()

            if self.symmetric:
                scale = x_max.abs().clamp(min=1e-8) / self.q_max
                zp = torch.zeros_like(scale)
            else:
                scale = (x_max - x_min).clamp(min=1e-8) / (self.q_max - self.q_min)
                zp = (-x_min / scale).round().clamp(self.q_min, self.q_max)

            # EMA update
            alpha = 0.1
            self.scale.copy_((1 - alpha) * self.scale + alpha * scale.mean())
            self.zero_point.copy_((1 - alpha) * self.zero_point + alpha * zp.mean())
            self.n_calibrated += 1

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled or not self.training:
            return x

        self.update_stats(x)

        # Fake quantize: quantize then dequantize
        x_q = (x / self.scale + self.zero_point).round().clamp(self.q_min, self.q_max)
        x_dq = (x_q - self.zero_point) * self.scale

        # STE: gradient passes through as if identity
        return x + (x_dq - x).detach()


class QATExpertFFN(nn.Module):
    """
    Expert FFN with fake-quantize nodes for QAT.
    Drop-in replacement for SwiGLUExpertFFN during QAT.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        bits: int = 8,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

        # Fake-quantize for weights (per-channel)
        self.fq_gate_w = FakeQuantize(bits=bits, per_channel=True, symmetric=True)
        self.fq_up_w = FakeQuantize(bits=bits, per_channel=True, symmetric=True)
        self.fq_down_w = FakeQuantize(bits=bits, per_channel=True, symmetric=True)

        # Fake-quantize for activations (per-tensor)
        self.fq_input = FakeQuantize(bits=bits, per_channel=False, symmetric=False)
        self.fq_ffn = FakeQuantize(bits=bits, per_channel=False, symmetric=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fq_input(x)

        gate_w = self.fq_gate_w(self.w_gate.weight)
        up_w = self.fq_up_w(self.w_up.weight)
        down_w = self.fq_down_w(self.w_down.weight)

        gate = F.silu(F.linear(x, gate_w))
        up = F.linear(x, up_w)
        ffn = self.fq_ffn(gate * up)
        return F.linear(ffn, down_w)


class QATTrainer:
    """
    Trains a model with fake-quantize nodes to minimize accuracy loss
    from quantization.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        lr: float = 1e-5,
        epochs: int = 3,
        device: str = "cpu",
    ):
        self.model = model
        self.bits = bits
        self.lr = lr
        self.epochs = epochs
        self.device = device

    def insert_fake_quantize(self, model: nn.Module) -> nn.Module:
        """Replace all expert linear layers with QAT-enabled versions."""
        model = copy.deepcopy(model)
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                fq = FakeQuantize(bits=self.bits)
                setattr(model, name, nn.Sequential(fq, module))
            else:
                self.insert_fake_quantize(module)
        return model

    def train(
        self,
        dataloader: Any,
        loss_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """Run QAT fine-tuning."""
        model = copy.deepcopy(self.model).to(self.device)
        model.train()

        # Disable fake quantize for first epoch (warm up)
        self._set_fake_quant(model, enabled=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * 100
        )

        for epoch in range(self.epochs):
            # Enable fake quantize after first epoch
            if epoch == 1:
                self._set_fake_quant(model, enabled=True)

            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None

                optimizer.zero_grad(set_to_none=True)

                output = model(inputs)

                if targets is not None and loss_fn is not None:
                    loss = loss_fn(output, targets)
                elif targets is not None:
                    loss = F.mse_loss(output.float(), targets.float())
                else:
                    # Self-supervised: minimize output entropy variance as proxy
                    loss = output.float().var() * 0.0 + output.float().pow(2).mean() * 0.01

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            logger.info(f"QAT Epoch {epoch + 1}/{self.epochs}: loss={avg:.6f}")

        # Freeze fake-quantize parameters
        self._set_fake_quant(model, enabled=False)
        logger.info("QAT training complete")
        return model

    @staticmethod
    def _set_fake_quant(model: nn.Module, enabled: bool) -> None:
        for m in model.modules():
            if isinstance(m, FakeQuantize):
                m.enabled = enabled


# ---------------------------------------------------------------------------
# 5. Quantization Benchmark
# ---------------------------------------------------------------------------


class QuantizationBenchmark:
    """
    Benchmarks different quantization schemes:
    - FP32 (baseline)
    - FP16 AMP
    - INT8 static
    - GPTQ-4bit

    Reports:
    - Latency speedup
    - Memory reduction
    - Relative output error (accuracy proxy)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = 512,
        device: str = "cuda",
    ):
        self.model = model
        self.hidden_dim = hidden_dim
        self.device = device if torch.cuda.is_available() else "cpu"
        self._results: List[QuantizationBenchmarkResult] = []

    def run(
        self,
        batch_sizes: Optional[List[int]] = None,
        seq_lens: Optional[List[int]] = None,
        warmup: int = 5,
        repeat: int = 30,
    ) -> List[QuantizationBenchmarkResult]:
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        if seq_lens is None:
            seq_lens = [32, 64]

        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for scheme in ["fp32", "fp16", "int8", "gptq"]:
                    try:
                        r = self._benchmark_scheme(
                            scheme, batch_size, seq_len, warmup, repeat
                        )
                        results.append(r)
                        logger.info(
                            f"[{scheme}] B={batch_size} S={seq_len}: "
                            f"speedup={r.speedup:.2f}x, "
                            f"error={r.relative_error:.4f}, "
                            f"mem_reduction={r.memory_reduction:.2f}x"
                        )
                    except Exception as e:
                        logger.warning(f"Benchmark [{scheme}] failed: {e}")

        self._results = results
        return results

    def _benchmark_scheme(
        self,
        scheme: str,
        batch_size: int,
        seq_len: int,
        warmup: int,
        repeat: int,
    ) -> QuantizationBenchmarkResult:
        """Benchmark a single quantization scheme."""
        dummy = torch.randn(
            batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float32
        )

        # Get FP32 reference
        fp32_model = copy.deepcopy(self.model).to(self.device).eval()
        fp32_lat = self._measure_latency(fp32_model, dummy, warmup, repeat)
        with torch.no_grad():
            fp32_out = fp32_model(dummy)
        fp32_mem = self._estimate_model_memory(fp32_model)

        # Build quantized model
        quant_model, quant_dummy = self._build_quantized(scheme, dummy)
        quant_lat = self._measure_latency(quant_model, quant_dummy, warmup, repeat)

        with torch.no_grad():
            quant_out = quant_model(quant_dummy)
            if quant_out.dtype != torch.float32:
                quant_out = quant_out.float()
        quant_mem = self._estimate_model_memory(quant_model)

        # Relative error
        try:
            rel_err = (
                (fp32_out.float() - quant_out.float()).norm()
                / fp32_out.float().norm().clamp(min=1e-8)
            ).item()
        except Exception:
            rel_err = float("nan")

        return QuantizationBenchmarkResult(
            scheme=scheme,
            batch_size=batch_size,
            seq_len=seq_len,
            fp32_latency_ms=fp32_lat,
            quantized_latency_ms=quant_lat,
            speedup=fp32_lat / max(quant_lat, 1e-9),
            fp32_output_norm=fp32_out.float().norm().item(),
            quant_output_norm=quant_out.float().norm().item(),
            relative_error=rel_err,
            memory_fp32_mb=fp32_mem,
            memory_quant_mb=quant_mem,
            memory_reduction=fp32_mem / max(quant_mem, 1e-9),
        )

    def _build_quantized(self, scheme: str, dummy: Tensor) -> Tuple[nn.Module, Tensor]:
        """Build quantized model for the given scheme."""
        if scheme == "fp32":
            return copy.deepcopy(self.model).to(self.device).eval(), dummy

        elif scheme == "fp16":
            model_fp16 = convert_model_to_fp16(self.model)
            model_fp16 = model_fp16.to(self.device).eval()
            dummy_fp16 = dummy.half()
            return model_fp16, dummy_fp16

        elif scheme == "int8":
            config = QuantizationConfig(scheme="int8_static", calibration_batches=5)
            quantizer = INT8StaticQuantizer(config)
            calib_data = [dummy.cpu()[:2]] * 5
            model_int8 = quantizer.quantize(self.model, calib_data, self.hidden_dim)
            return model_int8.cpu().eval(), dummy.cpu()

        elif scheme == "gptq":
            gptq = GPTQQuantizer(bits=4, group_size=64)
            calib_data = [dummy.cpu()[:2]] * 5
            model_gptq = gptq.quantize_model_experts(
                self.model, calib_data, self.hidden_dim, device="cpu"
            )
            return model_gptq.eval(), dummy.cpu()

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def _measure_latency(
        self,
        model: nn.Module,
        dummy: Tensor,
        warmup: int,
        repeat: int,
    ) -> float:
        """Measure mean inference latency in ms."""
        device = next(model.parameters()).device

        with torch.no_grad():
            for _ in range(warmup):
                try:
                    model(dummy)
                except Exception:
                    break

        if str(device) != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(repeat):
                t0 = time.perf_counter()
                try:
                    model(dummy)
                except Exception:
                    break
                if str(device) != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        return float(np.mean(latencies)) if latencies else 0.0

    @staticmethod
    def _estimate_model_memory(model: nn.Module) -> float:
        """Estimate model memory in MB."""
        total = 0
        for p in model.parameters():
            total += p.nelement() * p.element_size()
        for b in model.buffers():
            total += b.nelement() * b.element_size()
        return total / (1024 ** 2)

    def print_report(self) -> None:
        """Print a formatted benchmark report."""
        print("\n" + "=" * 90)
        print("Quantization Benchmark Report")
        print("=" * 90)
        print(
            f"{'Scheme':<10} {'B':>4} {'S':>5} "
            f"{'FP32 ms':>9} {'Quant ms':>10} {'Speedup':>9} "
            f"{'Rel Err':>9} {'Mem Reduc':>10}"
        )
        print("-" * 90)
        for r in self._results:
            print(
                f"{r.scheme:<10} {r.batch_size:>4} {r.seq_len:>5} "
                f"{r.fp32_latency_ms:>9.2f} {r.quantized_latency_ms:>10.2f} "
                f"{r.speedup:>8.2f}x {r.relative_error:>9.4f} "
                f"{r.memory_reduction:>9.2f}x"
            )
        print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Quantization-aware MoE wrapper
# ---------------------------------------------------------------------------


class QuantizedMoEInferenceEngine:
    """
    Wraps a quantized model to provide the same inference API as
    MoEInferenceEngine but with quantized weights.
    """

    def __init__(
        self,
        quantized_model: nn.Module,
        scheme: str = "int8",
        device: str = "cpu",
    ):
        self.model = quantized_model
        self.scheme = scheme
        self.device = torch.device(device)
        self.model.eval()

        self._fp16_manager: Optional[FP16MixedPrecisionManager] = None
        if scheme == "fp16":
            self._fp16_manager = FP16MixedPrecisionManager(enabled=True)

    @torch.no_grad()
    def infer(self, features: Tensor) -> Tensor:
        features = features.to(self.device)
        if self.scheme == "fp16":
            features = features.half()
            with self._fp16_manager.autocast_ctx:
                return self.model(features).float()
        return self.model(features)

    def benchmark(
        self,
        batch_sizes: Optional[List[int]] = None,
        seq_lens: Optional[List[int]] = None,
        hidden_dim: int = 512,
    ) -> List[Dict[str, Any]]:
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        if seq_lens is None:
            seq_lens = [32, 64]

        results = []
        for bs in batch_sizes:
            for sl in seq_lens:
                dummy = torch.randn(bs, sl, hidden_dim, device=self.device)
                lats = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    self.infer(dummy)
                    t1 = time.perf_counter()
                    lats.append((t1 - t0) * 1000.0)
                lats_np = np.array(lats[3:])
                results.append({
                    "scheme": self.scheme,
                    "batch_size": bs,
                    "seq_len": sl,
                    "mean_ms": float(lats_np.mean()),
                    "p99_ms": float(np.percentile(lats_np, 99)),
                    "tokens_per_sec": float(bs * sl / (lats_np.mean() / 1000.0)),
                })
        return results


# ---------------------------------------------------------------------------
# Utility: count quantized parameters
# ---------------------------------------------------------------------------


def count_quantized_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by dtype."""
    counts: Dict[str, int] = {}
    for name, p in model.named_parameters():
        dtype_str = str(p.dtype)
        counts[dtype_str] = counts.get(dtype_str, 0) + p.numel()
    return counts


def model_size_mb(model: nn.Module) -> float:
    """Return model parameter size in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)


# ---------------------------------------------------------------------------
# Auto-select best quantization scheme
# ---------------------------------------------------------------------------


def auto_quantize(
    model: nn.Module,
    calibration_data: Optional[Any] = None,
    hidden_dim: int = 512,
    latency_budget_ms: float = 50.0,
    accuracy_tolerance: float = 0.05,
    device: str = "cuda",
) -> Tuple[nn.Module, str]:
    """
    Automatically select and apply the best quantization scheme based on
    latency budget and accuracy tolerance.

    Strategy:
    1. Try FP16 (fastest, minimal accuracy loss on modern hardware)
    2. If not fast enough, try INT8
    3. If still not fast enough, try GPTQ-4bit
    4. Return the first scheme that meets both constraints
    """
    schemes_to_try = ["fp16", "int8", "gptq"] if torch.cuda.is_available() else ["int8", "gptq"]

    dummy = torch.randn(8, 32, hidden_dim)
    fp32_model = copy.deepcopy(model).eval()
    with torch.no_grad():
        fp32_ref = fp32_model(dummy.to("cpu" if not torch.cuda.is_available() else device))
    fp32_ref = fp32_ref.float()

    bench = QuantizationBenchmark(model, hidden_dim, device=device)

    for scheme in schemes_to_try:
        try:
            quant_model, quant_dummy = bench._build_quantized(scheme, dummy)
            latency = bench._measure_latency(quant_model, quant_dummy, warmup=3, repeat=10)

            with torch.no_grad():
                quant_out = quant_model(quant_dummy).float()

            rel_err = (fp32_ref - quant_out).norm() / fp32_ref.norm().clamp(min=1e-8)
            rel_err = float(rel_err.item())

            logger.info(
                f"auto_quantize [{scheme}]: latency={latency:.2f}ms, rel_err={rel_err:.4f}"
            )

            if latency <= latency_budget_ms and rel_err <= accuracy_tolerance:
                logger.info(f"auto_quantize: selected scheme '{scheme}'")
                return quant_model, scheme
        except Exception as e:
            logger.warning(f"auto_quantize [{scheme}] failed: {e}")

    logger.warning("auto_quantize: no scheme met constraints, returning FP32 model")
    return model, "fp32"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from lumina.moe_inference_engine import MoEConfig, LuminaMoEModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MoEConfig(
        num_experts=4,
        hidden_dim=256,
        ffn_dim=512,
        device=device,
        dtype=torch.float32,
        use_ped=False,
        use_prefetch=False,
        use_kv_cache=False,
        use_speculative_routing=False,
    )
    model = LuminaMoEModel(config, num_layers=2)
    model.eval()

    print(f"Model size: {model_size_mb(model):.2f} MB")

    bench = QuantizationBenchmark(model, hidden_dim=256, device="cpu")
    results = bench.run(batch_sizes=[1, 4], seq_lens=[32])
    bench.print_report()


# ---------------------------------------------------------------------------
# Extended: Quantization sensitivity analysis
# ---------------------------------------------------------------------------


class QuantizationSensitivityAnalyzer:
    """
    Analyzes the sensitivity of each layer to quantization error.
    Layers with high sensitivity should be kept in FP32 or use higher bits.
    Uses a Hessian-trace proxy (input gradient variance) as sensitivity metric.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self._sensitivities: Dict[str, float] = {}
        self._hooks: List[Any] = []

    def compute_sensitivity(
        self,
        calibration_data: Any,
        n_batches: int = 20,
    ) -> Dict[str, float]:
        """
        Compute per-layer quantization sensitivity.
        Higher sensitivity = more accuracy loss from quantization.
        """
        grad_vars: Dict[str, List[float]] = {}

        def make_hook(name: str):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    var = float(grad_output[0].float().var().item())
                    if name not in grad_vars:
                        grad_vars[name] = []
                    grad_vars[name].append(var)
            return hook

        # Register backward hooks on all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_full_backward_hook(make_hook(name))
                self._hooks.append(h)

        self.model.eval()
        for i, batch in enumerate(calibration_data):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            if not isinstance(x, Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            x.requires_grad_(False)

            try:
                self.model.zero_grad()
                out = self.model(x)
                # Use squared sum as a proxy loss
                loss = out.float().pow(2).sum()
                loss.backward()
            except Exception:
                break

        # Remove hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Aggregate sensitivities
        for name, vars_ in grad_vars.items():
            self._sensitivities[name] = float(np.mean(vars_))

        return dict(sorted(
            self._sensitivities.items(),
            key=lambda x: x[1],
            reverse=True,
        ))

    def get_quantization_order(self) -> List[str]:
        """
        Return layers sorted by sensitivity (least sensitive first).
        This is the order in which they should be quantized.
        """
        return list(sorted(self._sensitivities, key=lambda n: self._sensitivities[n]))

    def suggest_bit_widths(
        self,
        base_bits: int = 8,
        high_sensitivity_bits: int = 16,
        sensitivity_threshold: float = 1.0,
    ) -> Dict[str, int]:
        """
        Suggest per-layer bit widths based on sensitivity.
        High-sensitivity layers get more bits to preserve accuracy.
        """
        suggestions = {}
        for name, sens in self._sensitivities.items():
            if sens > sensitivity_threshold:
                suggestions[name] = high_sensitivity_bits
            else:
                suggestions[name] = base_bits
        return suggestions


# ---------------------------------------------------------------------------
# Extended: Mixed-precision aware layer swapper
# ---------------------------------------------------------------------------


class MixedPrecisionLayerSwapper:
    """
    Intelligently swaps individual layers to different precisions:
    - Expert FFN layers -> INT8 (highest savings)
    - Router layers -> FP16 (needs precision for top-k stability)
    - LayerNorm -> FP32 (numerical stability)
    - Attention layers -> BF16 (good balance)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_dtypes: Dict[str, torch.dtype] = {}

    def swap(
        self,
        layer_dtype_map: Optional[Dict[str, torch.dtype]] = None,
        sensitivity: Optional[Dict[str, float]] = None,
    ) -> nn.Module:
        """
        Swap layers to target dtypes.
        layer_dtype_map: name -> dtype overrides
        sensitivity: from QuantizationSensitivityAnalyzer
        """
        if layer_dtype_map is None:
            layer_dtype_map = {}

        # Record original dtypes
        for name, p in self.model.named_parameters():
            self._original_dtypes[name] = p.dtype

        for name, module in self.model.named_modules():
            # Determine target dtype
            target_dtype = layer_dtype_map.get(name, None)

            if target_dtype is None:
                # Auto-assign based on layer type
                if isinstance(module, nn.LayerNorm):
                    target_dtype = torch.float32
                elif "router" in name.lower() or "gate" in name.lower():
                    target_dtype = torch.float16
                elif "expert" in name.lower() and isinstance(module, nn.Linear):
                    target_dtype = torch.float16  # INT8 requires quantizer
                elif isinstance(module, nn.Linear):
                    target_dtype = torch.bfloat16

            if target_dtype is not None:
                try:
                    module.to(target_dtype)
                except Exception:
                    pass

        return self.model

    def restore(self) -> nn.Module:
        """Restore all layers to their original dtypes."""
        for name, p in self.model.named_parameters():
            orig_dtype = self._original_dtypes.get(name)
            if orig_dtype is not None:
                p.data = p.data.to(orig_dtype)
        return self.model


# ---------------------------------------------------------------------------
# Extended: Smooth quantization (SmoothQuant style)
# ---------------------------------------------------------------------------


class SmoothQuantTransformer:
    """
    Applies SmoothQuant (Xiao et al., 2022) to migrate quantization difficulty
    from activations to weights.

    For a linear layer Y = X @ W:
    - Activation X can have outlier channels that are hard to quantize
    - SmoothQuant: Y = (X / s) @ (s * W)  where s = per-channel scale
    - s is chosen to balance the quantization difficulty between X and W

    After smoothing, both X/s and s*W have similar per-channel magnitudes,
    making INT8 quantization much more accurate.
    """

    def __init__(
        self,
        alpha: float = 0.5,   # migration strength (0=all on activations, 1=all on weights)
    ):
        self.alpha = alpha

    def compute_scale(
        self,
        activation_abs_max: Tensor,   # (hidden_dim,) max abs of activations per channel
        weight_abs_max: Tensor,        # (hidden_dim,) max abs of weight rows
    ) -> Tensor:
        """
        Compute per-channel scale factors for SmoothQuant migration.
        s_i = (max|X_i|)^alpha / (max|W_i|)^(1-alpha)
        """
        scale = (
            activation_abs_max.pow(self.alpha)
            / weight_abs_max.pow(1 - self.alpha).clamp(min=1e-8)
        )
        return scale.clamp(min=1e-8)

    def smooth_layer(
        self,
        linear: nn.Linear,
        act_max: Tensor,
    ) -> Tuple[nn.Linear, Tensor]:
        """
        Apply SmoothQuant to a single linear layer.
        Returns (modified_linear, scale_factors).
        The preceding layer's output must be divided by scale_factors.
        """
        weight = linear.weight.data  # (out_features, in_features)
        weight_max = weight.abs().max(dim=0).values  # per input channel

        scale = self.compute_scale(act_max, weight_max)

        # Smooth weights: W_new = diag(s) @ W (scale each input channel)
        linear.weight.data = weight * scale.unsqueeze(0)

        return linear, scale

    def calibrate_activation_stats(
        self,
        model: nn.Module,
        calibration_data: Any,
        target_module_names: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> Dict[str, Tensor]:
        """
        Collect per-channel activation statistics by running calibration data.
        Returns dict of layer_name -> per-channel max abs activation.
        """
        act_stats: Dict[str, Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                x = input[0].detach().float()
                if x.dim() == 3:
                    x = x.view(-1, x.shape[-1])
                per_ch_max = x.abs().max(dim=0).values
                if name in act_stats:
                    act_stats[name] = torch.maximum(act_stats[name], per_ch_max)
                else:
                    act_stats[name] = per_ch_max
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_module_names is None or name in target_module_names:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                if not isinstance(batch, Tensor):
                    batch = torch.tensor(batch, dtype=torch.float32)
                try:
                    model(batch.to(device))
                except Exception:
                    break
                if i >= 50:
                    break

        for h in hooks:
            h.remove()

        return act_stats

    def apply_to_model(
        self,
        model: nn.Module,
        calibration_data: Any,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Tensor]]:
        """Apply SmoothQuant to all linear layers in the model."""
        act_stats = self.calibrate_activation_stats(model, calibration_data, device=device)

        scales_applied = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in act_stats:
                _, scale = self.smooth_layer(module, act_stats[name].to(module.weight.device))
                scales_applied[name] = scale
                logger.debug(f"SmoothQuant: applied to {name}, scale range [{scale.min():.3f}, {scale.max():.3f}]")

        logger.info(f"SmoothQuant: applied to {len(scales_applied)} layers")
        return model, scales_applied


# ---------------------------------------------------------------------------
# Extended: Quantized expert cache
# ---------------------------------------------------------------------------


class QuantizedExpertCache:
    """
    Caches quantized expert weights in a compact format.
    Hot experts are kept in INT8 / INT4 on GPU for fast access.
    Cold experts are evicted to CPU in INT4 format.
    """

    def __init__(
        self,
        num_experts: int,
        max_gpu_experts: int = 4,
        bits: int = 8,
        device: str = "cuda",
    ):
        self.num_experts = num_experts
        self.max_gpu_experts = max_gpu_experts
        self.bits = bits
        self.device = device

        self._gpu_experts: OrderedDict = OrderedDict()   # LRU
        self._cpu_experts: Dict[int, Dict[str, Tensor]] = {}
        self._lock = threading.Lock()

    def store_expert(
        self,
        expert_id: int,
        state_dict: Dict[str, Tensor],
        quantized: bool = False,
    ) -> None:
        """Store expert weights (optionally already quantized)."""
        with self._lock:
            if quantized:
                self._cpu_experts[expert_id] = {k: v.cpu() for k, v in state_dict.items()}
            else:
                # Quantize to INT8
                q_state = {}
                for k, v in state_dict.items():
                    if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        scale = v.float().abs().max() / 127.0 + 1e-8
                        q = (v.float() / scale).round().clamp(-128, 127).to(torch.int8)
                        q_state[k] = q.cpu()
                        q_state[f"{k}_scale"] = scale.cpu()
                    else:
                        q_state[k] = v.cpu()
                self._cpu_experts[expert_id] = q_state

    def load_expert(self, expert_id: int) -> Optional[Dict[str, Tensor]]:
        """Load expert weights to GPU (dequantized to FP16)."""
        with self._lock:
            # Check GPU cache
            if expert_id in self._gpu_experts:
                self._gpu_experts.move_to_end(expert_id)
                return self._gpu_experts[expert_id]

            # Load from CPU and dequantize
            if expert_id not in self._cpu_experts:
                return None

            cpu_state = self._cpu_experts[expert_id]
            gpu_state = {}
            for k, v in cpu_state.items():
                if k.endswith("_scale"):
                    continue
                scale_key = f"{k}_scale"
                if scale_key in cpu_state:
                    scale = cpu_state[scale_key].to(self.device)
                    gpu_state[k] = v.to(self.device).to(torch.float16) * scale
                else:
                    gpu_state[k] = v.to(self.device)

            # Add to GPU cache with LRU eviction
            self._gpu_experts[expert_id] = gpu_state
            if len(self._gpu_experts) > self.max_gpu_experts:
                self._gpu_experts.popitem(last=False)

            return gpu_state

    def prefetch(self, expert_ids: List[int]) -> None:
        """Prefetch experts to GPU in background."""
        for eid in expert_ids:
            self.load_expert(eid)

    def memory_usage(self) -> Dict[str, Any]:
        """Report memory usage of the cache."""
        gpu_bytes = sum(
            sum(v.nbytes for v in sd.values())
            for sd in self._gpu_experts.values()
        )
        cpu_bytes = sum(
            sum(v.nbytes for v in sd.values())
            for sd in self._cpu_experts.values()
        )
        return {
            "gpu_experts_cached": len(self._gpu_experts),
            "gpu_bytes": gpu_bytes,
            "gpu_mb": gpu_bytes / (1024 ** 2),
            "cpu_experts_cached": len(self._cpu_experts),
            "cpu_bytes": cpu_bytes,
            "cpu_mb": cpu_bytes / (1024 ** 2),
        }
