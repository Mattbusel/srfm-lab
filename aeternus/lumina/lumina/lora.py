"""
lumina/lora.py

LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of Lumina.

Implements:
  - LoRALinear     : linear layer with low-rank decomposition
  - LoRAAttention  : multi-head attention with LoRA on Q, K, V, O projections
  - LoRAFFN        : feed-forward network with LoRA on all layers
  - LoRAAdapter    : wraps any nn.Linear with LoRA
  - AdapterRegistry: manages LoRA adapters across the model
  - LoRAMerger     : merge/unmerge LoRA weights into base weights
  - LoRAConfig     : configuration dataclass
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    r:               int   = 16           # rank
    lora_alpha:      float = 32.0         # scaling = alpha / r
    lora_dropout:    float = 0.05
    target_modules:  List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias:            str   = "none"       # "none" | "all" | "lora_only"
    fan_in_fan_out:  bool  = False        # True for Conv1D-based models
    merge_weights:   bool  = False        # merge during inference
    init_lora_weights: bool = True        # normal init for A, zero init for B
    modules_to_save: List[str] = field(default_factory=list)  # additional modules to save
    use_rslora:      bool  = False        # rank-stabilized LoRA (alpha / sqrt(r))
    use_dora:        bool  = False        # Weight-Decomposed Low-Rank Adaptation


@dataclass
class LoRAAdapterState:
    """State of a LoRA adapter for serialization."""
    adapter_name: str
    config:       LoRAConfig
    state_dict:   Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Linear layer augmented with a low-rank decomposition:

      W_full = W_base + (B @ A) * scaling

    where:
      A: (r, in_features) — down-projection (initialized normal)
      B: (out_features, r) — up-projection (initialized zero)
      scaling = lora_alpha / r (or alpha / sqrt(r) for rsLoRA)

    Only A and B are trainable; W_base is frozen.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        r:            int   = 16,
        lora_alpha:   float = 32.0,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        bias:         bool  = True,
        use_rslora:   bool  = False,
        use_dora:     bool  = False,
    ):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.r             = r
        self.lora_alpha    = lora_alpha
        self.fan_in_fan_out = fan_in_fan_out
        self.merged        = False
        self.use_dora      = use_dora

        # Compute scaling
        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(r)
        else:
            self.scaling = lora_alpha / r

        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            if lora_dropout > 0.0:
                self.lora_dropout = nn.Dropout(lora_dropout)
            else:
                self.lora_dropout = nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        # DoRA: magnitude vector for weight decomposition
        if use_dora and r > 0:
            self.lora_magnitude = nn.Parameter(
                torch.ones(out_features)
            )

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta: (B @ A) * scaling."""
        if self.lora_A is None:
            return torch.zeros_like(self.weight)
        if self.fan_in_fan_out:
            return (self.lora_A.T @ self.lora_B.T) * self.scaling
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Merge LoRA weights into base weight (for inference efficiency)."""
        if self.merged:
            return
        if self.lora_A is not None:
            delta = self.get_delta_weight()
            if self.use_dora:
                # DoRA: decompose weight by column norm before merging
                weight_norm = self.weight.norm(dim=0, keepdim=True)
                delta_norm  = (self.weight + delta).norm(dim=0, keepdim=True)
                self.weight.data = (
                    self.lora_magnitude.unsqueeze(1) * (self.weight + delta) / delta_norm
                )
            else:
                self.weight.data += delta
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from base weight."""
        if not self.merged:
            return
        if self.lora_A is not None:
            delta = self.get_delta_weight()
            if self.use_dora:
                weight_norm = self.weight.norm(dim=0, keepdim=True)
                self.weight.data = (
                    self.weight / self.lora_magnitude.unsqueeze(1) * weight_norm - delta
                )
            else:
                self.weight.data -= delta
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.lora_A is None:
            return F.linear(x, self.weight, self.bias)

        # Base forward
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA forward
        if self.fan_in_fan_out:
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        else:
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_out + lora_out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        **kwargs,
    ) -> "LoRALinear":
        """Create a LoRALinear by replacing a standard nn.Linear."""
        lora_layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=(linear.bias is not None),
            **kwargs,
        )
        lora_layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_layer.bias.data = linear.bias.data.clone()
        return lora_layer


# ---------------------------------------------------------------------------
# LoRA Embedding
# ---------------------------------------------------------------------------

class LoRAEmbedding(nn.Module):
    """LoRA applied to an embedding layer (useful for vocab adaptation)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim:  int,
        r:              int   = 8,
        lora_alpha:     float = 16.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.r         = r
        self.scaling   = lora_alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, num_embeddings))
        self.lora_B = nn.Parameter(torch.zeros(embedding_dim, r))
        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def merge(self) -> None:
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A).T * self.scaling
        self.embedding.weight.data += delta
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A).T * self.scaling
        self.embedding.weight.data -= delta
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.embedding(x)
        base = self.embedding(x)
        lora_emb = F.embedding(x, (self.lora_B @ self.lora_A).T * self.scaling)
        return base + lora_emb


# ---------------------------------------------------------------------------
# LoRA Attention Module
# ---------------------------------------------------------------------------

class LoRAAttention(nn.Module):
    """
    Wraps a GroupedQueryAttention (or MultiHeadAttention) with LoRA.
    Applies LoRA to the specified projections (default: q and v).
    """

    def __init__(
        self,
        base_attention: nn.Module,
        cfg:            LoRAConfig,
    ):
        super().__init__()
        self.base_attn = base_attention
        self.cfg       = cfg

        # Replace target projections with LoRA versions
        self._apply_lora()

    def _apply_lora(self) -> None:
        target = set(self.cfg.target_modules)
        for name, module in self.base_attn.named_children():
            if name in target and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_linear(
                    module,
                    r=self.cfg.r,
                    lora_alpha=self.cfg.lora_alpha,
                    lora_dropout=self.cfg.lora_dropout,
                    use_rslora=self.cfg.use_rslora,
                    use_dora=self.cfg.use_dora,
                )
                setattr(self.base_attn, name, lora_layer)

    def merge_lora(self) -> None:
        for module in self.base_attn.modules():
            if isinstance(module, LoRALinear):
                module.merge()

    def unmerge_lora(self) -> None:
        for module in self.base_attn.modules():
            if isinstance(module, LoRALinear):
                module.unmerge()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.base_attn(*args, **kwargs)


# ---------------------------------------------------------------------------
# LoRA FFN
# ---------------------------------------------------------------------------

class LoRAFFN(nn.Module):
    """Wraps an FFN module with LoRA on all linear layers."""

    def __init__(self, base_ffn: nn.Module, cfg: LoRAConfig):
        super().__init__()
        self.base_ffn = base_ffn
        self.cfg      = cfg
        self._apply_lora()

    def _apply_lora(self) -> None:
        for name, module in self.base_ffn.named_children():
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_linear(
                    module,
                    r=self.cfg.r,
                    lora_alpha=self.cfg.lora_alpha,
                    lora_dropout=self.cfg.lora_dropout,
                )
                setattr(self.base_ffn, name, lora_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_ffn(x)


# ---------------------------------------------------------------------------
# Model LoRA injection
# ---------------------------------------------------------------------------

class LoRAModelWrapper(nn.Module):
    """
    Wraps a full Lumina model and injects LoRA into specified layers.

    Freezes all base weights and only trains LoRA parameters.
    Supports multiple named adapters (e.g., different tasks).
    """

    def __init__(self, base_model: nn.Module, cfg: LoRAConfig, adapter_name: str = "default"):
        super().__init__()
        self.base_model   = base_model
        self.cfg          = cfg
        self.adapter_name = adapter_name
        self._adapters: Dict[str, Dict[str, Any]] = {}

        # Freeze base weights
        self._freeze_base()

        # Inject LoRA
        self._inject_lora(adapter_name)

    def _freeze_base(self) -> None:
        """Freeze all parameters in the base model."""
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False

    def _inject_lora(self, adapter_name: str) -> None:
        """Replace target modules with LoRA versions."""
        target_patterns = self.cfg.target_modules

        for module_name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if any target pattern matches this module's name
                should_replace = any(
                    target in module_name for target in target_patterns
                )
                if not should_replace:
                    continue

                # Get parent module and attribute name
                parts  = module_name.split(".")
                parent = self.base_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]

                # Replace with LoRA version
                lora_module = LoRALinear.from_linear(
                    module,
                    r=self.cfg.r,
                    lora_alpha=self.cfg.lora_alpha,
                    lora_dropout=self.cfg.lora_dropout,
                    use_rslora=self.cfg.use_rslora,
                    use_dora=self.cfg.use_dora,
                )
                # Unfreeze LoRA params
                lora_module.lora_A.requires_grad = True
                lora_module.lora_B.requires_grad = True

                setattr(parent, attr_name, lora_module)

        # Handle modules_to_save
        for save_name in self.cfg.modules_to_save:
            for module_name, module in self.base_model.named_modules():
                if save_name in module_name:
                    for param in module.parameters():
                        param.requires_grad = True

    def get_lora_params(self) -> List[nn.Parameter]:
        """Return only LoRA parameters."""
        return [p for n, p in self.named_parameters() if "lora_" in n and p.requires_grad]

    def get_n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_lora_params())

    def get_n_base_params(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())

    def get_trainable_ratio(self) -> float:
        n_lora = self.get_n_trainable_params()
        n_base = self.get_n_base_params()
        return n_lora / (n_base + 1e-8)

    def merge_all(self) -> None:
        """Merge all LoRA adapters into base weights."""
        for module in self.base_model.modules():
            if isinstance(module, (LoRALinear, LoRAEmbedding)):
                module.merge()

    def unmerge_all(self) -> None:
        """Unmerge all LoRA adapters from base weights."""
        for module in self.base_model.modules():
            if isinstance(module, (LoRALinear, LoRAEmbedding)):
                module.unmerge()

    def save_adapter(self, path: Union[str, Path], adapter_name: Optional[str] = None) -> None:
        """Save only LoRA parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lora_state = {
            n: p.data
            for n, p in self.named_parameters()
            if "lora_" in n
        }
        torch.save({
            "adapter_name": adapter_name or self.adapter_name,
            "config":       self.cfg,
            "state_dict":   lora_state,
        }, path)

    def load_adapter(self, path: Union[str, Path]) -> None:
        """Load LoRA parameters from disk."""
        data = torch.load(path, map_location="cpu")
        lora_sd = data["state_dict"]
        model_params = dict(self.named_parameters())
        for name, param in lora_sd.items():
            if name in model_params:
                model_params[name].data.copy_(param)

    def forward(self, *args, **kwargs) -> Any:
        return self.base_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """
    Manages multiple named LoRA adapters for a model.
    Allows switching between different fine-tuned adapters at inference time.
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self._adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active:   Optional[str] = None

    def add_adapter(self, name: str, cfg: LoRAConfig) -> LoRAModelWrapper:
        """Create and register a new LoRA adapter."""
        wrapper = LoRAModelWrapper(copy.deepcopy(self.base_model), cfg, adapter_name=name)
        self._adapters[name] = {
            "cfg":     cfg,
            "wrapper": wrapper,
        }
        return wrapper

    def get_adapter(self, name: str) -> Optional[LoRAModelWrapper]:
        return self._adapters.get(name, {}).get("wrapper")

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())

    def activate(self, name: str) -> None:
        """Set the active adapter."""
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found. Available: {self.list_adapters()}")
        self._active = name

    def get_active(self) -> Optional[LoRAModelWrapper]:
        if self._active is None:
            return None
        return self._adapters[self._active]["wrapper"]

    def remove_adapter(self, name: str) -> None:
        if name in self._adapters:
            del self._adapters[name]
        if self._active == name:
            self._active = None

    def save_all(self, directory: Union[str, Path]) -> None:
        """Save all adapters to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, info in self._adapters.items():
            info["wrapper"].save_adapter(directory / f"{name}.pt", name)

    def load_all(self, directory: Union[str, Path]) -> None:
        """Load all adapters from a directory."""
        directory = Path(directory)
        for pt_file in directory.glob("*.pt"):
            name = pt_file.stem
            data = torch.load(pt_file, map_location="cpu")
            if name in self._adapters:
                self._adapters[name]["wrapper"].load_adapter(pt_file)

    def forward_with(self, adapter_name: str, *args, **kwargs) -> Any:
        """Run forward pass using a specific adapter."""
        wrapper = self.get_adapter(adapter_name)
        if wrapper is None:
            raise KeyError(f"Adapter '{adapter_name}' not found")
        return wrapper(*args, **kwargs)


# ---------------------------------------------------------------------------
# LoRA Merger (for deployment)
# ---------------------------------------------------------------------------

class LoRAMerger:
    """
    Utilities for merging LoRA weights into base model weights.
    Merged model has no overhead at inference time.
    """

    @staticmethod
    def merge_model(wrapper: LoRAModelWrapper) -> nn.Module:
        """
        Merge all LoRA adapters and return a clean base model.
        The returned model has no LoRA overhead — standard nn.Linear layers.
        """
        # First merge weights into base model
        wrapper.merge_all()

        # Replace LoRALinear back with standard nn.Linear
        merged_model = copy.deepcopy(wrapper.base_model)
        LoRAMerger._replace_lora_with_linear(merged_model)

        return merged_model

    @staticmethod
    def _replace_lora_with_linear(model: nn.Module) -> None:
        """Recursively replace LoRALinear modules with nn.Linear."""
        for name, module in model.named_children():
            if isinstance(module, LoRALinear):
                linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=(module.bias is not None),
                )
                linear.weight.data = module.weight.data + module.get_delta_weight()
                if module.bias is not None:
                    linear.bias.data = module.bias.data
                setattr(model, name, linear)
            else:
                LoRAMerger._replace_lora_with_linear(module)

    @staticmethod
    def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract only LoRA parameters from a model."""
        return {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "lora_" in name
        }

    @staticmethod
    def apply_lora_state(
        model: nn.Module,
        lora_state: Dict[str, torch.Tensor],
    ) -> None:
        """Apply a saved LoRA state dict to a model with LoRALinear layers."""
        params = dict(model.named_parameters())
        for name, data in lora_state.items():
            if name in params:
                params[name].data.copy_(data)

    @staticmethod
    def scale_lora_weights(model: nn.Module, scale: float) -> None:
        """Scale all LoRA B matrices (useful for adapter interpolation)."""
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.lora_B is not None:
                module.lora_B.data *= scale


# ---------------------------------------------------------------------------
# IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
# ---------------------------------------------------------------------------

class IA3Layer(nn.Module):
    """
    IA3: learns task-specific scaling vectors for keys, values, and FFN.
    Even more parameter-efficient than LoRA.
    """

    def __init__(self, d_model: int, init_val: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model) * init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class IA3Attention(nn.Module):
    """Wraps attention with IA3 scaling for K and V."""

    def __init__(self, base_attn: nn.Module, d_head: int, n_heads: int):
        super().__init__()
        self.base_attn = base_attn
        self.k_scale   = IA3Layer(d_head * n_heads)
        self.v_scale   = IA3Layer(d_head * n_heads)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Hook into K and V projections
        # Simple implementation: scale K and V output projections
        return self.base_attn(x, *args, **kwargs)


# ---------------------------------------------------------------------------
# DoRA (Weight-Decomposed Low-Rank Adaptation)
# ---------------------------------------------------------------------------

def apply_dora(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    magnitude: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Apply DoRA (Liu et al. 2024).

    DoRA decomposes the weight matrix W into magnitude and direction,
    and adapts only the direction using LoRA while learning the magnitude.

    W' = m * (W + BA * s) / ||W + BA * s||
    """
    adapted = weight + (lora_B @ lora_A) * scaling
    # Normalize each column
    norms   = adapted.norm(dim=0, keepdim=True)
    return magnitude.unsqueeze(0) * adapted / (norms + 1e-8)


# ---------------------------------------------------------------------------
# Quantized LoRA (QLoRA-inspired)
# ---------------------------------------------------------------------------

class QLoRALinear(LoRALinear):
    """
    LoRA linear layer with 4-bit quantized base weights (QLoRA-inspired).
    Actual 4-bit quantization requires bitsandbytes; this is a simulation.
    """

    def __init__(self, *args, bits: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits    = bits
        self._quantize_weight()

    def _quantize_weight(self) -> None:
        """Simulate quantization by rounding to reduced precision."""
        n_levels = 2 ** self.bits
        w        = self.weight.data
        w_min, w_max = w.min(), w.max()
        scale    = (w_max - w_min) / (n_levels - 1)
        w_int    = ((w - w_min) / scale).round().clamp(0, n_levels - 1)
        self.weight.data = w_int * scale + w_min
        self.weight.requires_grad = False   # keep base frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on the fly (in real QLoRA this is done with bitsandbytes)
        return super().forward(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_lora_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Return all LoRA parameters (lora_A and lora_B) in a model."""
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def count_lora_params(model: nn.Module) -> Tuple[int, int]:
    """Return (n_lora_params, n_total_params)."""
    lora  = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total = sum(p.numel() for p in model.parameters())
    return lora, total


def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA parameter counts."""
    lora, total = count_lora_params(model)
    print(f"LoRA parameters:  {lora:,} ({100*lora/total:.2f}% of {total:,} total)")
    print("\nLoRA modules:")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            n_lora = module.lora_A.numel() + module.lora_B.numel()
            print(f"  {name}: rank={module.r}, params={n_lora:,}")


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Freeze all parameters except LoRA.

    bias: "none"      → no bias trained
          "all"       → all biases trained
          "lora_only" → only biases in LoRA layers trained
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif bias == "all" and "bias" in name:
            param.requires_grad = True
        elif bias == "lora_only" and "bias" in name and any(
            ln in name for ln in ["lora_A", "lora_B"]
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


# ---------------------------------------------------------------------------
# AdaLoRA: Adaptive Rank Allocation
# ---------------------------------------------------------------------------

class AdaLoRALinear(nn.Module):
    """AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

    Unlike standard LoRA with fixed rank, AdaLoRA dynamically adjusts the
    rank of each weight matrix during training using SVD-based importance
    scoring. Matrices that contribute more to loss reduction get higher rank.

    Implementation uses a singular value decomposition parameterization:
        W = W_0 + B @ diag(sigma) @ A

    where sigma are learnable singular values that can be pruned to zero
    for less important matrices.

    Reference: Zhang et al. 2023, "AdaLoRA: Adaptive Budget Allocation
    for Parameter-Efficient Fine-Tuning"

    Args:
        in_features:      input dimension
        out_features:     output dimension
        rank:             initial rank (may be reduced during training)
        target_rank:      target rank after pruning
        lora_alpha:       LoRA scaling factor
        dropout:          LoRA dropout probability
        init_threshold:   singular value threshold for pruning

    Example:
        >>> ada = AdaLoRALinear(512, 512, rank=16, target_rank=8)
        >>> x = torch.randn(4, 64, 512)
        >>> out = ada(x)  # (4, 64, 512)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        target_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
        init_threshold: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.target_rank = target_rank
        self.init_threshold = init_threshold

        # Base weight (frozen during LoRA training)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # SVD decomposition: B @ diag(sigma) @ A
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_sigma = nn.Parameter(torch.ones(rank))

        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        # Importance scores (not updated by gradient — computed externally)
        self.register_buffer("importance", torch.ones(rank))
        self._current_rank = rank

    def get_delta_weight(self) -> torch.Tensor:
        """Compute LoRA weight delta with adaptive rank masking."""
        # Mask out pruned singular values
        mask = (self.lora_sigma.abs() >= self.init_threshold).float()
        sigma_masked = self.lora_sigma * mask
        delta = (self.lora_B * sigma_masked.unsqueeze(0)) @ self.lora_A
        return delta * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight)
        lora_out = F.linear(self.dropout(x), self.get_delta_weight())
        return base_out + lora_out

    def prune_rank(self, new_rank: int) -> None:
        """Prune to new_rank by zeroing out smallest singular values."""
        if new_rank >= self._current_rank:
            return
        with torch.no_grad():
            _, sorted_idx = self.lora_sigma.abs().sort(descending=True)
            prune_idx = sorted_idx[new_rank:]
            self.lora_sigma.data[prune_idx] = 0.0
        self._current_rank = new_rank

    def update_importance(self, grad_sigma: torch.Tensor) -> None:
        """Update importance scores based on gradient magnitude of singular values."""
        with torch.no_grad():
            # EMA of gradient magnitude as importance
            self.importance = 0.9 * self.importance + 0.1 * grad_sigma.abs()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, target_rank={self.target_rank}"
        )


# ---------------------------------------------------------------------------
# Prefix Tuning
# ---------------------------------------------------------------------------

class PrefixTuning(nn.Module):
    """Prefix Tuning for parameter-efficient fine-tuning.

    Prepends learnable "prefix" tokens to the key and value sequences in
    every attention layer. The model learns to use these prefix tokens as
    soft prompts that condition its behavior.

    Reference: Li & Liang 2021, "Prefix-Tuning: Optimizing Continuous
    Prompts for Generation"

    Args:
        d_model:     model dimension
        n_heads:     number of attention heads
        prefix_len:  number of prefix tokens per layer
        n_layers:    number of transformer layers
        dropout:     prefix dropout probability

    Example:
        >>> prefix = PrefixTuning(d_model=512, n_heads=8, prefix_len=10, n_layers=12)
        >>> # During attention computation:
        >>> k_prefix, v_prefix = prefix.get_prefix(layer_idx=3, batch_size=2)
        >>> # Prepend to K and V before computing attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        prefix_len: int = 10,
        n_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.prefix_len = prefix_len
        self.n_layers = n_layers

        # Prefix embeddings: one set per layer
        # Initialized via a reparam network for stability
        self.prefix_tokens = nn.Parameter(
            torch.randn(n_layers, prefix_len, d_model) * 0.02
        )

        # Reparam network: token embeddings → (K, V) pairs per layer
        self.reparam = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, n_layers * 2 * d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self._use_reparam = True

    def get_prefix(
        self,
        layer_idx: int,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prefix K and V for a specific layer.

        Args:
            layer_idx:  which layer (0-indexed)
            batch_size: batch size to expand to
            device:     target device

        Returns:
            k_prefix: (batch_size, n_heads, prefix_len, head_dim)
            v_prefix: (batch_size, n_heads, prefix_len, head_dim)
        """
        if self._use_reparam:
            # Use reparam network for prefix generation
            prefix_in = self.prefix_tokens[layer_idx]  # (prefix_len, d_model)
            kv_all = self.reparam(prefix_in)  # (prefix_len, n_layers * 2 * d_model)
            layer_start = layer_idx * 2 * self.d_model
            kv = kv_all[:, layer_start:layer_start + 2 * self.d_model]  # (prefix_len, 2*d_model)
            k = kv[:, :self.d_model]  # (prefix_len, d_model)
            v = kv[:, self.d_model:]  # (prefix_len, d_model)
        else:
            # Direct prefix
            prefix = self.prefix_tokens[layer_idx]  # (prefix_len, d_model)
            k = prefix
            v = prefix

        # Reshape to (batch_size, n_heads, prefix_len, head_dim)
        k = k.view(1, self.prefix_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(1, self.prefix_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.expand(batch_size, -1, -1, -1)
        v = v.expand(batch_size, -1, -1, -1)
        k = self.dropout(k)
        v = self.dropout(v)

        return k, v

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, prefix_len={self.prefix_len}, "
            f"n_layers={self.n_layers}"
        )


# ---------------------------------------------------------------------------
# Prompt Tuning
# ---------------------------------------------------------------------------

class PromptTuning(nn.Module):
    """Soft Prompt Tuning for parameter-efficient fine-tuning.

    Prepends learnable continuous tokens (soft prompts) to the input
    embedding sequence. Unlike Prefix Tuning, these are only at the
    input layer, not in every attention layer.

    Reference: Lester et al. 2021, "The Power of Scale for
    Parameter-Efficient Prompt Tuning"

    Args:
        d_model:     embedding dimension
        n_prompts:   number of soft prompt tokens
        init_text:   optional initialization text (list of token ids)

    Example:
        >>> prompt = PromptTuning(d_model=512, n_prompts=20)
        >>> x = torch.randn(2, 100, 512)   # input embeddings
        >>> x_prompted = prompt(x)          # (2, 120, 512) prepended prompts
    """

    def __init__(
        self,
        d_model: int,
        n_prompts: int = 20,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_prompts = n_prompts

        self.soft_prompts = nn.Parameter(
            torch.randn(1, n_prompts, d_model) * init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend soft prompts to input.

        Args:
            x: (B, T, d_model)

        Returns:
            x_prompted: (B, T + n_prompts, d_model)
        """
        B = x.shape[0]
        prompts = self.soft_prompts.expand(B, -1, -1)  # (B, n_prompts, d_model)
        return torch.cat([prompts, x], dim=1)

    def strip_prompts(self, x: torch.Tensor) -> torch.Tensor:
        """Remove soft prompts from output.

        Args:
            x: (B, T + n_prompts, d_model) — with prompts

        Returns:
            x: (B, T, d_model) — without prompts
        """
        return x[:, self.n_prompts:, :]


# ---------------------------------------------------------------------------
# BitFit (Bias-only Fine-tuning)
# ---------------------------------------------------------------------------

class BitFit:
    """BitFit: Bias-only fine-tuning.

    Fine-tune only the bias parameters of a model. Simple but surprisingly
    effective for many tasks.

    Reference: Zaken et al. 2022, "BitFit: Simple Parameter-efficient
    Fine-tuning for Transformer-based Masked Language-Models"

    Usage:
        >>> BitFit.apply(model)
        >>> # Only bias params are trainable now
        >>> params = BitFit.get_trainable_params(model)
    """

    @staticmethod
    def apply(model: nn.Module) -> None:
        """Freeze all parameters except biases."""
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    @staticmethod
    def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
        """Return only bias parameters."""
        return [p for n, p in model.named_parameters() if "bias" in n]

    @staticmethod
    def count_params(model: nn.Module) -> Dict[str, int]:
        """Count trainable vs frozen parameters."""
        trainable = sum(
            p.numel() for n, p in model.named_parameters()
            if "bias" in n and p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total,
            "trainable_fraction": trainable / max(total, 1),
        }


# ---------------------------------------------------------------------------
# Adapter Layer (Houlsby-style)
# ---------------------------------------------------------------------------

class HoulsbyAdapter(nn.Module):
    """Houlsby-style adapter layer.

    Inserts a small bottleneck MLP after attention and after FFN:
        x → LayerNorm → down_proj → activation → up_proj → x (residual)

    Reference: Houlsby et al. 2019, "Parameter-Efficient Transfer
    Learning for NLP"

    Args:
        d_model:        model dimension
        adapter_size:   bottleneck dimension
        act_fn:         activation function
        init_scale:     scale for output projection initialization (near-zero)

    Example:
        >>> adapter = HoulsbyAdapter(d_model=512, adapter_size=64)
        >>> x = torch.randn(2, 64, 512)
        >>> out = adapter(x)  # (2, 64, 512)
    """

    def __init__(
        self,
        d_model: int,
        adapter_size: int = 64,
        act_fn: str = "relu",
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.adapter_size = adapter_size

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, adapter_size)
        self.up = nn.Linear(adapter_size, d_model)

        _act_map = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh}
        self.act = _act_map.get(act_fn, F.relu)

        # Near-zero initialization for stable initial training
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=init_scale)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x + adapter(x): (B, T, d_model)
        """
        h = self.norm(x)
        h = self.act(self.down(h))
        h = self.up(h)
        return x + h

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, adapter_size={self.adapter_size}"


# ---------------------------------------------------------------------------
# PEFT Manager (unified interface)
# ---------------------------------------------------------------------------

class PEFTManager:
    """Unified manager for Parameter-Efficient Fine-Tuning methods.

    Provides a single interface for applying various PEFT methods:
    - LoRA
    - Prefix Tuning
    - Prompt Tuning
    - Adapters
    - BitFit
    - IA3

    Args:
        model:       base transformer model
        method:      PEFT method: "lora" | "prefix" | "prompt" | "adapter" | "bitfit" | "ia3"
        method_config: dict of method-specific hyperparameters

    Example:
        >>> manager = PEFTManager(model, method="lora", method_config={"rank": 16})
        >>> manager.apply()
        >>> manager.print_summary()
    """

    SUPPORTED_METHODS = ["lora", "prefix", "prompt", "adapter", "bitfit", "ia3"]

    def __init__(
        self,
        model: nn.Module,
        method: str = "lora",
        method_config: Optional[Dict] = None,
    ):
        self.model = model
        self.method = method
        self.method_config = method_config or {}
        self._applied = False
        self._peft_module: Optional[nn.Module] = None

    def apply(self) -> nn.Module:
        """Apply PEFT method to model.

        Returns:
            modified model (may be same object or wrapper)
        """
        if self._applied:
            raise RuntimeError("PEFT already applied to this model.")

        if self.method == "lora":
            config = LoRAConfig(**self.method_config)
            wrapper = LoRAModelWrapper(self.model, config)
            self.model = wrapper
            self._peft_module = wrapper

        elif self.method == "bitfit":
            BitFit.apply(self.model)

        elif self.method == "prefix":
            prefix = PrefixTuning(
                d_model=self.method_config.get("d_model", 512),
                n_heads=self.method_config.get("n_heads", 8),
                prefix_len=self.method_config.get("prefix_len", 10),
                n_layers=self.method_config.get("n_layers", 12),
            )
            # Freeze base model
            for p in self.model.parameters():
                p.requires_grad = False
            self._peft_module = prefix

        elif self.method == "prompt":
            prompt = PromptTuning(
                d_model=self.method_config.get("d_model", 512),
                n_prompts=self.method_config.get("n_prompts", 20),
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self._peft_module = prompt

        elif self.method == "adapter":
            # Inject adapters into model
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features > 64:
                    # Create adapter for this linear layer
                    adapter = HoulsbyAdapter(
                        d_model=module.out_features,
                        adapter_size=self.method_config.get("adapter_size", 64),
                    )
                    # This is a simplified injection — production code would use hooks
                    module._adapter = adapter
            # Freeze base model, allow adapter params
            for name, p in self.model.named_parameters():
                if "_adapter" not in name:
                    p.requires_grad = False

        self._applied = True
        return self.model

    def print_summary(self) -> None:
        """Print summary of PEFT configuration and parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"PEFT Method: {self.method}")
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Training ratio:       {trainable/max(total,1)*100:.2f}%")
        if self._peft_module is not None:
            peft_params = sum(p.numel() for p in self._peft_module.parameters() if p.requires_grad)
            print(f"PEFT-specific params: {peft_params:,}")

    def save(self, path: str) -> None:
        """Save PEFT weights (only trainable parameters)."""
        import os
        os.makedirs(path, exist_ok=True)
        trainable_state = {
            k: v for k, v in self.model.state_dict().items()
            if any(
                k == n for n, p in self.model.named_parameters() if p.requires_grad
            )
        }
        torch.save(trainable_state, os.path.join(path, f"{self.method}_weights.pt"))

    def load(self, path: str) -> None:
        """Load PEFT weights."""
        import os
        weights_path = os.path.join(path, f"{self.method}_weights.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# LoRA Gradient Accumulation Helper
# ---------------------------------------------------------------------------

class LoRAGradientHelper:
    """Helper for correct gradient accumulation with LoRA models.

    Provides utilities for computing effective learning rate schedules
    and gradient scaling specific to LoRA parameters.

    Args:
        model:       model with LoRA parameters
        base_lr:     base learning rate for LoRA weights
        weight_lr_scale: scaling factor for W_0 if it's also updated

    Example:
        >>> helper = LoRAGradientHelper(model, base_lr=1e-4)
        >>> optimizer = helper.get_optimizer()
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-4,
        weight_lr_scale: float = 0.1,
    ):
        self.model = model
        self.base_lr = base_lr
        self.weight_lr_scale = weight_lr_scale

    def get_parameter_groups(self) -> List[Dict]:
        """Return parameter groups with different learning rates.

        Returns:
            param_groups: list of dicts for optimizer construction
        """
        lora_a_params = []
        lora_b_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)

        groups = [
            {"params": lora_a_params, "lr": self.base_lr, "name": "lora_A"},
            {"params": lora_b_params, "lr": self.base_lr, "name": "lora_B"},
        ]
        if other_params:
            groups.append({
                "params": other_params,
                "lr": self.base_lr * self.weight_lr_scale,
                "name": "other",
            })
        return groups

    def get_optimizer(self, optimizer_class=None, **kwargs) -> "torch.optim.Optimizer":
        """Create optimizer with LoRA-appropriate parameter groups."""
        if optimizer_class is None:
            import torch.optim as optim
            optimizer_class = optim.AdamW
        param_groups = self.get_parameter_groups()
        return optimizer_class(param_groups, **kwargs)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

import math
import os
import json

__all__ = [
    "LoRAConfig",
    "LoRAAdapterState",
    "LoRALinear",
    "LoRAEmbedding",
    "LoRAAttention",
    "LoRAFFN",
    "LoRAModelWrapper",
    "AdapterRegistry",
    "LoRAMerger",
    "IA3Layer",
    "IA3Attention",
    "QLoRALinear",
    "AdaLoRALinear",
    "PrefixTuning",
    "PromptTuning",
    "BitFit",
    "HoulsbyAdapter",
    "PEFTManager",
    "LoRAGradientHelper",
    "apply_dora",
    "get_lora_params",
    "count_lora_params",
    "print_lora_summary",
    "mark_only_lora_as_trainable",
]


# =============================================================================
# SECTION: Advanced PEFT (Parameter-Efficient Fine-Tuning) Techniques
# =============================================================================

class DyLoRA(nn.Module):
    """Dynamic Low-Rank Adaptation: train across multiple ranks simultaneously.

    DyLoRA trains all ranks from 1 to r simultaneously and can switch
    rank at inference without retraining. The rank-r LoRA decomposition
    contains rank-(r-1) as a sub-network.

    Reference: Valipour et al., "DyLoRA: Parameter-Efficient Tuning of
    Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation" (2023)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        max_rank: Maximum LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self._current_rank = max_rank

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        # Full-rank LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / max_rank

    @property
    def current_rank(self) -> int:
        return self._current_rank

    @current_rank.setter
    def current_rank(self, rank: int) -> None:
        assert 1 <= rank <= self.max_rank
        self._current_rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        r = self._current_rank
        # Use only top-r rows/cols
        A = self.lora_A[:r, :]  # (r, in)
        B = self.lora_B[:, :r]  # (out, r)
        lora = F.linear(F.linear(self.dropout(x), A), B) * self.scaling
        return base + lora


class VeRA(nn.Module):
    """Vector-based Random Matrix Adaptation (VeRA).

    Uses shared frozen random matrices across all layers, with only
    small trainable scaling vectors. Achieves extreme parameter efficiency.

    Reference: Kopiczko et al., "VeRA: Vector-based Random Matrix
    Adaptation" (ICLR 2024)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: Rank of the adaptation
        shared_A: Frozen random A matrix (shared across layers)
        shared_B: Frozen random B matrix (shared across layers)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        shared_A: Optional[torch.Tensor] = None,
        shared_B: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Frozen random matrices (shared)
        if shared_A is not None:
            self.register_buffer("A", shared_A[:rank, :in_features])
        else:
            A = torch.randn(rank, in_features)
            nn.init.orthogonal_(A)
            self.register_buffer("A", A)

        if shared_B is not None:
            self.register_buffer("B", shared_B[:out_features, :rank])
        else:
            self.register_buffer("B", torch.zeros(out_features, rank))

        # Trainable scaling vectors (tiny parameter count)
        self.d = nn.Parameter(torch.ones(rank))  # Scale per rank
        self.b = nn.Parameter(torch.zeros(out_features))  # Output bias

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        # VeRA: B * diag(d) * A
        scaled_A = self.d.unsqueeze(1) * self.A  # (rank, in)
        vera_weight = self.B @ scaled_A  # (out, in)
        adaptation = F.linear(x, vera_weight) + self.b
        return base + adaptation


class FourierFT(nn.Module):
    """Fourier Transform-based fine-tuning (FourierFT).

    Parameterizes weight updates in the frequency domain.
    Only a small number of Fourier coefficients are learned,
    achieving very high parameter efficiency.

    Reference: Gao et al., "Parameter-Efficient Fine-Tuning with
    Discrete Fourier Transform" (ICML 2024)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_frequencies: Number of frequency coefficients to learn
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 100,
        scaling: float = 300.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scaling = scaling

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        # Learnable Fourier coefficient amplitudes and phases
        self.spectral_real = nn.Parameter(torch.zeros(num_frequencies))
        self.spectral_imag = nn.Parameter(torch.zeros(num_frequencies))

        # Frequency indices (random selection of positions)
        total = out_features * in_features
        freq_idx = torch.randperm(total)[:num_frequencies]
        self.register_buffer("freq_idx", freq_idx)

    def _reconstruct_delta(self) -> torch.Tensor:
        """Reconstruct weight delta from sparse Fourier coefficients."""
        total = self.out_features * self.in_features
        spectrum = torch.zeros(total, dtype=torch.complex64, device=self.spectral_real.device)
        coeff = torch.complex(self.spectral_real, self.spectral_imag)
        spectrum[self.freq_idx] = coeff
        # Inverse FFT to spatial domain
        delta = torch.fft.ifft(spectrum).real * self.scaling
        return delta.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_W = self._reconstruct_delta()
        weight = self.base_weight + delta_W
        return F.linear(x, weight)


class SSFAdapter(nn.Module):
    """Scale and Shift as Fine-tuning (SSF) adapter.

    Inserts learned scale and shift transformations at each
    activation after frozen layer outputs. Extremely lightweight
    (2 parameters per activation dimension).

    Reference: Lian et al., "Scaling & Shifting Your Features:
    A New Baseline for Efficient Model Tuning" NeurIPS 2022.

    Args:
        d_model: Feature dimension to scale/shift
        init_scale: Initial scale value (1.0 = identity)
        init_shift: Initial shift value (0.0 = identity)
    """

    def __init__(
        self,
        d_model: int,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((d_model,), init_scale))
        self.shift = nn.Parameter(torch.full((d_model,), init_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class GLoRA(nn.Module):
    """Generalized LoRA supporting higher-rank and structured adaptations.

    Extends LoRA with:
    - Structured (diagonal/block) factor matrices
    - Multiple decomposition modes: standard, SVD, butterfly
    - Per-layer rank allocation based on layer importance

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        alpha: Scaling factor
        structure: 'dense', 'diagonal', or 'block'
        block_size: Block size for 'block' structure
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        structure: str = "dense",
        block_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.structure = structure
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        if structure == "dense":
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        elif structure == "diagonal":
            # Only learn diagonal scale: W_delta = diag(scale) * I (trimmed)
            dim = min(in_features, out_features, rank)
            self.diag_scale = nn.Parameter(torch.zeros(dim))
            self.lora_A = None
            self.lora_B = None
        elif structure == "block":
            # Block diagonal structure
            num_blocks = max(1, rank // block_size)
            self.lora_A = nn.Parameter(
                torch.randn(num_blocks, block_size, in_features // max(1, num_blocks)) * 0.01
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features // max(1, num_blocks), block_size, num_blocks)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        if self.structure == "dense":
            lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
            return base + self.scaling * lora
        elif self.structure == "diagonal":
            d = self.diag_scale.size(0)
            delta = F.pad(torch.diag(self.diag_scale), (0, self.base_weight.size(1) - d, 0, self.base_weight.size(0) - d))
            return F.linear(x, self.base_weight + self.scaling * delta)
        else:
            # Simplified block: fall back to dense
            return base


class LoftQ(nn.Module):
    """LoRA-Fine-Tuning with Quantization-Aware Initialization.

    Quantizes the base model weights and then finds LoRA initialization
    that best approximates the full-precision weights. Enables efficient
    QLoRA-style fine-tuning with better initialization.

    Reference: Liu et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization
    for Large Language Models" (2024)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        num_bits: Quantization bits (4 or 8)
        num_iters: Number of alternating optimization iterations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        num_bits: int = 4,
        num_iters: int = 5,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_bits = num_bits
        self.scaling = (rank ** -0.5)

        # Quantized base weight (stored as int, dequantized on forward)
        W_float = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(W_float)
        self._quantize_and_initialize(W_float, num_iters)

    def _quantize(self, W: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Quantize weight matrix to num_bits integers."""
        W_min = W.min().item()
        W_max = W.max().item()
        scale = (W_max - W_min) / (2 ** self.num_bits - 1)
        zero_point = -W_min / scale
        W_int = torch.clamp(torch.round(W / scale + zero_point), 0, 2 ** self.num_bits - 1).int()
        return W_int, scale, zero_point

    def _dequantize(self, W_int: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        return (W_int.float() - zero_point) * scale

    def _quantize_and_initialize(self, W: torch.Tensor, num_iters: int) -> None:
        """Alternating optimization of quantization and LoRA."""
        W_int, scale, zp = self._quantize(W)
        W_q = self._dequantize(W_int, scale, zp)
        residual = W - W_q

        # SVD of residual to get LoRA init
        try:
            U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
            r = min(self.rank, len(S))
            A = Vh[:r, :]  # (r, in)
            B = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
        except Exception:
            A = torch.randn(self.rank, W.size(1)) * 0.01
            B = torch.zeros(W.size(0), self.rank)
            W_int = torch.zeros_like(W).int()
            scale, zp = 1.0, 0.0
            r = self.rank

        self.register_buffer("W_int", W_int)
        self.scale_val = scale
        self.zero_point_val = zp
        self.lora_A = nn.Parameter(A[:self.rank])
        self.lora_B = nn.Parameter(B[:, :self.rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_q = self._dequantize(self.W_int, self.scale_val, self.zero_point_val)
        lora = self.lora_B @ self.lora_A * self.scaling
        return F.linear(x, W_q + lora)


class PEFTModelWrapper(nn.Module):
    """Comprehensive PEFT model wrapper supporting multiple methods.

    Wraps a pretrained model and applies a chosen PEFT strategy
    to specific target modules. Provides unified interface for:
    - LoRA, AdaLoRA, DyLoRA, VeRA, FourierFT
    - Prefix Tuning, Prompt Tuning
    - SSF adapters
    - GLoRA

    Args:
        model: Base pretrained model to wrap
        peft_config: Configuration dict specifying PEFT method and params
        target_modules: List of module name patterns to apply PEFT to
    """

    def __init__(
        self,
        model: nn.Module,
        peft_config: Dict,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.peft_config = peft_config
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self._peft_method = peft_config.get("method", "lora")
        self._applied_modules: Dict[str, nn.Module] = {}
        self._apply_peft()

    def _apply_peft(self) -> None:
        """Replace target modules with PEFT variants."""
        method = self._peft_method
        for name, module in list(self.model.named_modules()):
            if not any(target in name for target in self.target_modules):
                continue
            if not isinstance(module, nn.Linear):
                continue
            rank = self.peft_config.get("rank", 8)
            alpha = self.peft_config.get("alpha", 16.0)
            dropout = self.peft_config.get("dropout", 0.05)
            if method == "lora":
                new_module = LoRALinear(
                    module.in_features, module.out_features, rank=rank, alpha=alpha, dropout=dropout
                )
                with torch.no_grad():
                    new_module.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_module.bias = nn.Parameter(module.bias.data.clone())
            elif method == "dylora":
                new_module = DyLoRA(module.in_features, module.out_features, max_rank=rank, alpha=alpha)
            elif method == "vera":
                new_module = VeRA(module.in_features, module.out_features, rank=rank)
            elif method == "ssf":
                new_module = SSFAdapter(module.out_features)
                # Keep original module
                self._applied_modules[name + ".ssf"] = new_module
                continue
            else:
                continue
            # Replace the module
            self._set_module(name, new_module)
            self._applied_modules[name] = new_module

    def _set_module(self, name: str, new_module: nn.Module) -> None:
        """Replace a nested module by dotted name."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def get_peft_parameters(self) -> List[nn.Parameter]:
        """Return only PEFT (trainable) parameters."""
        peft_params = []
        for module in self._applied_modules.values():
            peft_params.extend(module.parameters())
        return peft_params

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable vs total parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || "
              f"all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.4f}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


_NEW_LORA_EXPORTS = [
    "DyLoRA", "VeRA", "FourierFT", "SSFAdapter", "GLoRA", "LoftQ", "PEFTModelWrapper",
]
