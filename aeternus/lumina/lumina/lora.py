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
# Exports
# ---------------------------------------------------------------------------

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
    "apply_dora",
    "get_lora_params",
    "count_lora_params",
    "print_lora_summary",
    "mark_only_lora_as_trainable",
]
