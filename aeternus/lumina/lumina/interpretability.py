"""
lumina/interpretability.py

Model interpretability tools for Lumina financial foundation model.

Covers:
  - Attention rollout (Abnar & Zuidema, 2020)
  - Gradient-weighted attention (GradCAM for transformers)
  - SHAP values via DeepLIFT approximation
  - Integrated Gradients (Sundararajan et al., 2017)
  - Smooth Gradients
  - Feature attribution via input perturbation
  - Probing classifiers for learned representations
  - Attention head analysis (head importance, head specialization)
  - Circuit-level mechanistic interpretability scaffolding
  - Logit lens for intermediate layer predictions
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

class ActivationCache:
    """
    Registers forward hooks to cache intermediate activations and
    backward hooks to cache gradients.
    """

    def __init__(self):
        self._activations: Dict[str, Tensor] = {}
        self._gradients: Dict[str, Tensor] = {}
        self._hooks: List[Any] = []

    def register(self, name: str, module: nn.Module) -> None:
        def fwd_hook(mod, inp, out):
            if isinstance(out, tuple):
                self._activations[name] = out[0].detach()
            else:
                self._activations[name] = out.detach()

        def bwd_hook(mod, grad_in, grad_out):
            if isinstance(grad_out, tuple) and grad_out[0] is not None:
                self._gradients[name] = grad_out[0].detach()
            elif not isinstance(grad_out, tuple) and grad_out is not None:
                self._gradients[name] = grad_out.detach()

        self._hooks.append(module.register_forward_hook(fwd_hook))
        self._hooks.append(module.register_full_backward_hook(bwd_hook))

    def get_activation(self, name: str) -> Optional[Tensor]:
        return self._activations.get(name)

    def get_gradient(self, name: str) -> Optional[Tensor]:
        return self._gradients.get(name)

    def clear(self) -> None:
        self._activations.clear()
        self._gradients.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


class AttentionCache:
    """
    Specifically caches attention weights from all transformer layers.
    """

    def __init__(self):
        self._attn_weights: Dict[str, Tensor] = {}
        self._hooks: List[Any] = []

    def register_attention_module(self, name: str, module: nn.Module) -> None:
        """Register a forward hook on an attention module."""
        def hook(mod, inp, out):
            # Most attention modules return (output, attn_weights) or just output
            if isinstance(out, tuple) and len(out) >= 2:
                if out[1] is not None and isinstance(out[1], Tensor):
                    self._attn_weights[name] = out[1].detach()
            elif hasattr(mod, '_last_attn_weights'):
                self._attn_weights[name] = mod._last_attn_weights.detach()

        self._hooks.append(module.register_forward_hook(hook))

    def get_attn_weights(self) -> Dict[str, Tensor]:
        return dict(self._attn_weights)

    def clear(self) -> None:
        self._attn_weights.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

class AttentionRollout:
    """
    Attention rollout (Abnar & Zuidema, 2020).

    Propagates attention through all layers to get the effective
    attention from the output token to the input tokens.

    A_rollout = A_1 * A_2 * ... * A_L (with identity residual)
    """

    def __init__(self, discard_ratio: float = 0.9, head_fusion: str = "mean"):
        """
        Args:
            discard_ratio: Fraction of lowest attentions to zero out.
            head_fusion: How to aggregate across heads ("mean" | "min" | "max").
        """
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def _fuse_heads(self, attn: Tensor) -> Tensor:
        """
        Aggregate attention across heads.
        attn: (B, H, T, T)
        Returns: (B, T, T)
        """
        if self.head_fusion == "mean":
            return attn.mean(dim=1)
        elif self.head_fusion == "min":
            return attn.min(dim=1).values
        elif self.head_fusion == "max":
            return attn.max(dim=1).values
        else:
            raise ValueError(f"Unknown head fusion: {self.head_fusion}")

    def _discard_low_attention(self, attn: Tensor) -> Tensor:
        """Zero out the lowest discard_ratio attention values."""
        flat = attn.view(attn.size(0), -1)
        threshold_idx = int(flat.shape[-1] * self.discard_ratio)
        threshold, _ = flat.sort(dim=-1)
        threshold = threshold[:, threshold_idx: threshold_idx + 1]
        mask = attn >= threshold.unsqueeze(-1)
        attn = attn * mask.float()
        return attn

    def rollout(self, attention_matrices: List[Tensor]) -> Tensor:
        """
        Compute rollout attention.

        Args:
            attention_matrices: List of (B, H, T, T) attention tensors, one per layer.

        Returns:
            Rollout attention (B, T, T).
        """
        if not attention_matrices:
            raise ValueError("No attention matrices provided.")

        B, H, T, _ = attention_matrices[0].shape
        device = attention_matrices[0].device

        result = torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        for attn in attention_matrices:
            fused = self._fuse_heads(attn)         # (B, T, T)
            fused = self._discard_low_attention(fused)
            # Normalize each row
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-9)
            # Add identity (residual connection) and renormalize
            fused = 0.5 * fused + 0.5 * torch.eye(T, device=device).unsqueeze(0)
            fused = fused / fused.sum(dim=-1, keepdim=True)
            result = torch.bmm(fused, result)

        return result

    def get_input_importance(
        self,
        attention_matrices: List[Tensor],
        query_idx: int = -1,
    ) -> Tensor:
        """
        Get importance scores for input tokens with respect to a specific query position.
        Defaults to last token (for causal/decoder models).

        Returns: (B, T) importance scores.
        """
        rollout = self.rollout(attention_matrices)
        if query_idx == -1:
            query_idx = rollout.shape[-1] - 1
        return rollout[:, query_idx, :]  # (B, T)


# ---------------------------------------------------------------------------
# Gradient-weighted attention (GradCAM for transformers)
# ---------------------------------------------------------------------------

class GradCAMTransformer:
    """
    GradCAM adapted for transformer models (Chefer et al., 2021).

    Combines attention weights with gradient signal to produce
    relevancy scores for each input token.

    relevancy = relu(gradient * attention)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._cache = ActivationCache()
        self._attn_cache = AttentionCache()

    def register_layer(self, name: str, module: nn.Module) -> None:
        self._cache.register(name, module)

    def compute(
        self,
        input_ids: Tensor,
        target_class: Optional[int] = None,
        attention_layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute GradCAM relevancy scores.

        Returns:
            Dict with 'relevancy' (B, T) and per-layer relevancies.
        """
        input_ids.requires_grad_(True) if input_ids.dtype == torch.float else None
        self.model.eval()

        outputs = self.model(input_ids)

        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("output", None))
            loss = outputs.get("loss", None)
        else:
            logits = outputs
            loss = None

        if logits is not None:
            if target_class is not None:
                score = logits[:, target_class].sum()
            else:
                score = logits.max(dim=-1).values.sum()
        elif loss is not None:
            score = -loss  # Higher = better
        else:
            raise ValueError("Could not find logits or loss in model output.")

        self.model.zero_grad()
        score.backward(retain_graph=True)

        relevancies = {}
        for name in (attention_layer_names or self._cache._activations.keys()):
            act = self._cache.get_activation(name)
            grad = self._cache.get_gradient(name)
            if act is None or grad is None:
                continue
            # GradCAM: weight activations by mean gradient
            weights = grad.mean(dim=-1, keepdim=True)  # (B, T, 1)
            relevancy = F.relu(act * weights).mean(dim=-1)  # (B, T)
            relevancies[name] = relevancy

        # Aggregate across layers
        if relevancies:
            stacked = torch.stack(list(relevancies.values()), dim=0)
            agg_relevancy = stacked.mean(dim=0)
        else:
            agg_relevancy = torch.zeros(input_ids.shape[:2])

        return {"relevancy": agg_relevancy, "per_layer": relevancies}


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017).

    Attributes importance by integrating gradients along a path
    from a baseline input to the actual input.

    IG_i(x) = (x_i - x'_i) * integral_0^1 [d F(x' + alpha*(x-x')) / d x_i] d alpha
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        baseline_type: str = "zero",
    ):
        self.model = model
        self.n_steps = n_steps
        self.baseline_type = baseline_type

    def _get_baseline(self, x: Tensor) -> Tensor:
        if self.baseline_type == "zero":
            return torch.zeros_like(x)
        elif self.baseline_type == "mean":
            return x.mean(dim=0, keepdim=True).expand_as(x)
        elif self.baseline_type == "noise":
            return torch.randn_like(x) * 0.001
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

    def attribute(
        self,
        x: Tensor,
        target_fn: Optional[Callable[[Any], Tensor]] = None,
        baseline: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute integrated gradients.

        Args:
            x: Input tensor (B, T, F) or (B, F).
            target_fn: Maps model output to scalar (or uses output.sum()).
            baseline: Optional custom baseline.

        Returns:
            Attribution tensor of same shape as x.
        """
        if baseline is None:
            baseline = self._get_baseline(x)

        x = x.float()
        baseline = baseline.float()

        # Interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=x.device)
        grads = []

        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            with torch.enable_grad():
                outputs = self.model(interpolated)
                if isinstance(outputs, dict):
                    target = outputs.get("logits", outputs.get("output"))
                    if target is None:
                        target = -outputs["loss"]
                    else:
                        target = target.sum()
                else:
                    target = outputs.sum() if target_fn is None else target_fn(outputs)

            grad = torch.autograd.grad(target, interpolated, create_graph=False)[0]
            grads.append(grad.detach())

        # Approximate integral via trapezoidal rule
        grads_tensor = torch.stack(grads, dim=0)  # (n_steps+1, B, ...)
        integral = (grads_tensor[:-1] + grads_tensor[1:]).mean(dim=0) / 2.0

        # Multiply by (x - baseline)
        attribution = (x - baseline) * integral
        return attribution

    def convergence_delta(
        self,
        attribution: Tensor,
        x: Tensor,
        baseline: Optional[Tensor] = None,
        target_fn: Optional[Callable] = None,
    ) -> float:
        """
        Check completeness axiom: sum of attributions ≈ F(x) - F(baseline).
        A small delta indicates good approximation.
        """
        if baseline is None:
            baseline = self._get_baseline(x)

        with torch.no_grad():
            out_x = self.model(x)
            out_b = self.model(baseline)
            if isinstance(out_x, dict):
                score_x = out_x.get("logits", torch.zeros(1)).sum().item()
                score_b = out_b.get("logits", torch.zeros(1)).sum().item()
            else:
                score_x = out_x.sum().item()
                score_b = out_b.sum().item()

        attribution_sum = attribution.sum().item()
        expected_delta = score_x - score_b
        return abs(attribution_sum - expected_delta)


# ---------------------------------------------------------------------------
# SmoothGrad
# ---------------------------------------------------------------------------

class SmoothGrad:
    """
    SmoothGrad (Smilkov et al., 2017).

    Reduces noise in gradient-based saliency maps by averaging
    gradients over many noisy copies of the input.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        noise_level: float = 0.1,
    ):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def attribute(
        self,
        x: Tensor,
        target_fn: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute SmoothGrad attribution.

        Returns: Attribution of same shape as x.
        """
        x = x.float()
        total_grads = torch.zeros_like(x)
        noise_std = (x.max() - x.min()).item() * self.noise_level

        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * noise_std
            x_noisy = (x + noise).detach().requires_grad_(True)

            with torch.enable_grad():
                outputs = self.model(x_noisy)
                if isinstance(outputs, dict):
                    target = outputs.get("logits", outputs.get("output"))
                    if target is not None:
                        target = target.sum()
                    else:
                        target = -outputs["loss"]
                else:
                    target = outputs.sum() if target_fn is None else target_fn(outputs)

            grad = torch.autograd.grad(target, x_noisy, create_graph=False)[0]
            total_grads += grad.detach()

        return total_grads / self.n_samples


# ---------------------------------------------------------------------------
# SHAP-style DeepLIFT approximation
# ---------------------------------------------------------------------------

class DeepLIFTAttribution:
    """
    DeepLIFT (Shrikumar et al., 2017) approximation.
    Assigns contributions based on differences from a reference (baseline).

    C_i = delta_out * (x_i - x'_i) / delta_in
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_type: str = "zero",
    ):
        self.model = model
        self.baseline_type = baseline_type
        self._cache = ActivationCache()

    def _get_baseline(self, x: Tensor) -> Tensor:
        if self.baseline_type == "zero":
            return torch.zeros_like(x)
        elif self.baseline_type == "mean":
            return x.mean(dim=(0, 1), keepdim=True).expand_as(x)
        return torch.zeros_like(x)

    def attribute(self, x: Tensor, target_idx: Optional[int] = None) -> Tensor:
        """
        Approximate DeepLIFT contributions.

        Returns attribution of same shape as x.
        """
        baseline = self._get_baseline(x)

        with torch.no_grad():
            out_x = self.model(x)
            out_b = self.model(baseline)

            if isinstance(out_x, dict):
                logits_x = out_x.get("logits", out_x.get("output", torch.zeros(1)))
                logits_b = out_b.get("logits", out_b.get("output", torch.zeros(1)))
            else:
                logits_x, logits_b = out_x, out_b

            if target_idx is not None:
                delta_out = logits_x[:, target_idx] - logits_b[:, target_idx]
            else:
                delta_out = (logits_x - logits_b).sum(dim=-1)  # (B,)

        # Use gradient as proxy for multiplier
        x_req = x.float().detach().requires_grad_(True)
        outputs = self.model(x_req)
        if isinstance(outputs, dict):
            score = outputs.get("logits", outputs.get("output", torch.zeros(1))).sum()
        else:
            score = outputs.sum()
        grad = torch.autograd.grad(score, x_req)[0].detach()

        delta_in = x - baseline
        attribution = grad * delta_in
        return attribution


# ---------------------------------------------------------------------------
# Probing classifiers
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    A lightweight linear probe for testing what information is
    encoded in intermediate representations.

    Trains a logistic regression / linear regression on frozen features.
    """

    def __init__(self, input_dim: int, output_dim: int, probe_type: str = "classification"):
        super().__init__()
        assert probe_type in ("classification", "regression")
        self.probe_type = probe_type
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            if self.probe_type == "classification":
                return logits.argmax(dim=-1)
            return logits.squeeze(-1)


class ProbingClassifierSuite:
    """
    Trains and evaluates multiple linear probes on intermediate representations.

    Useful for understanding what each layer "knows" about:
      - Market regime
      - Volatility level
      - Return direction
      - Sector/asset identity
      - Time-of-day, day-of-week
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self._probes: Dict[str, Dict[str, LinearProbe]] = {}
        self._cache = ActivationCache()

    def register_layer_probe(
        self,
        layer_name: str,
        module: nn.Module,
        target_name: str,
        output_dim: int,
        probe_type: str = "classification",
    ) -> None:
        """Register a probe for a specific layer and target."""
        self._cache.register(layer_name, module)
        if layer_name not in self._probes:
            self._probes[layer_name] = {}
        # Will be initialized when first features are extracted
        self._probes[layer_name][target_name] = {
            "probe": None,
            "output_dim": output_dim,
            "probe_type": probe_type,
        }

    def extract_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        max_samples: int = 5000,
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract intermediate activations for a layer.
        Returns (features, labels).
        """
        self.model.eval()
        features_list, labels_list = [], []
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if n >= max_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1]
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("features")).to(self.device)
                    y = batch.get("labels", None)
                else:
                    continue

                self._cache.clear()
                _ = self.model(x)
                act = self._cache.get_activation(layer_name)

                if act is not None:
                    if act.ndim == 3:
                        act = act.mean(dim=1)  # Pool over sequence
                    features_list.append(act.cpu())
                    if y is not None:
                        labels_list.append(y.cpu() if isinstance(y, Tensor) else torch.tensor(y))
                    n += len(x)

        features = torch.cat(features_list, dim=0) if features_list else torch.empty(0)
        labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty(0)
        return features, labels

    def train_probe(
        self,
        layer_name: str,
        target_name: str,
        features: Tensor,
        labels: Tensor,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Train a linear probe for (layer, target) pair."""
        info = self._probes[layer_name][target_name]
        input_dim = features.shape[-1]
        output_dim = info["output_dim"]
        probe_type = info["probe_type"]

        probe = LinearProbe(input_dim, output_dim, probe_type).to(self.device)
        info["probe"] = probe
        self._probes[layer_name][target_name]["probe"] = probe

        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Split train/val 80/20
        n = len(features)
        n_train = int(0.8 * n)
        x_train, x_val = features[:n_train], features[n_train:]
        y_train, y_val = labels[:n_train], labels[n_train:]

        best_val_loss = float("inf")
        for epoch in range(n_epochs):
            probe.train()
            optimizer.zero_grad()
            pred = probe(x_train)
            if probe_type == "classification":
                loss = F.cross_entropy(pred, y_train.long())
            else:
                loss = F.mse_loss(pred.squeeze(), y_train.float())
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(x_val)
            if probe_type == "classification":
                val_loss = F.cross_entropy(val_pred, y_val.long()).item()
                val_acc = (val_pred.argmax(dim=-1) == y_val.long()).float().mean().item()
            else:
                val_loss = F.mse_loss(val_pred.squeeze(), y_val.float()).item()
                val_acc = 0.0

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "layer": layer_name,
            "target": target_name,
        }

    def probing_accuracy_by_layer(self) -> Dict[str, Dict[str, float]]:
        """Return probing accuracy organized by layer and target."""
        results = {}
        for layer_name, targets in self._probes.items():
            results[layer_name] = {}
            for target_name, info in targets.items():
                if info["probe"] is not None:
                    results[layer_name][target_name] = info.get("val_acc", float("nan"))
        return results


# ---------------------------------------------------------------------------
# Attention head analysis
# ---------------------------------------------------------------------------

class AttentionHeadAnalyzer:
    """
    Analyzes individual attention heads:
      - Head importance scores via gradient-based pruning metric
      - Head specialization (what does each head attend to?)
      - Redundancy detection
    """

    def __init__(self, model: nn.Module, n_layers: int, n_heads: int):
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads

    def compute_head_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_batches: int = 10,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Compute importance of each attention head.
        Uses gradient magnitude w.r.t. head output as proxy.

        Returns: (n_layers, n_heads) importance matrix.
        """
        device = device or torch.device("cpu")
        self.model = self.model.to(device)

        importance = torch.zeros(self.n_layers, self.n_heads)
        n_seen = 0

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            elif isinstance(batch, dict):
                x = batch.get("input_ids", batch.get("features")).to(device)
            else:
                continue

            self.model.zero_grad()
            outputs = self.model(x)
            if isinstance(outputs, dict):
                loss = outputs.get("loss", outputs.get("logits", torch.zeros(1)).sum())
            else:
                loss = outputs.sum()
            loss.backward()

            # Look for attention-related parameters
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    # Try to find q/k/v parameters for this layer
                    for name, param in self.model.named_parameters():
                        if f"layers.{layer_idx}" in name or f"layer.{layer_idx}" in name:
                            if "q_proj" in name or "query" in name:
                                if param.grad is not None:
                                    head_dim = param.shape[0] // self.n_heads
                                    start = head_idx * head_dim
                                    end = (head_idx + 1) * head_dim
                                    importance[layer_idx, head_idx] += param.grad[start:end].abs().mean().item()
            n_seen += 1

        if n_seen > 0:
            importance /= n_seen
        return importance

    def analyze_head_patterns(
        self,
        attn_weights: List[Tensor],
    ) -> Dict[str, Any]:
        """
        Analyze patterns in attention matrices.

        Args:
            attn_weights: List of (B, H, T, T) tensors per layer.

        Returns:
            Dict with pattern statistics per head.
        """
        results = {}
        for layer_idx, attn in enumerate(attn_weights):
            B, H, T, _ = attn.shape
            layer_results = {}
            for h in range(H):
                head_attn = attn[:, h, :, :].mean(dim=0)  # (T, T)
                # Diagonal attention (local)
                diag = torch.diag(head_attn).mean().item()
                # Previous token attention (shift-1)
                if T > 1:
                    prev_token = torch.diag(head_attn, -1).mean().item() if T > 1 else 0.0
                else:
                    prev_token = 0.0
                # First token attention (global/CLS)
                first_token = head_attn[:, 0].mean().item()
                # Entropy (how spread out the attention is)
                attn_clamped = head_attn.clamp(min=1e-9)
                entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1).mean().item()

                layer_results[f"head_{h}"] = {
                    "diagonal_attn": diag,
                    "prev_token_attn": prev_token,
                    "first_token_attn": first_token,
                    "entropy": entropy,
                    "is_local": diag > 0.3,
                    "is_global": first_token > 0.3,
                    "is_previous": prev_token > 0.3,
                }
            results[f"layer_{layer_idx}"] = layer_results
        return results

    def detect_redundant_heads(
        self,
        importance: Tensor,
        threshold: float = 0.01,
    ) -> List[Tuple[int, int]]:
        """Return list of (layer, head) pairs that are below importance threshold."""
        redundant = []
        max_importance = importance.max().item()
        for layer in range(importance.shape[0]):
            for head in range(importance.shape[1]):
                if importance[layer, head].item() < threshold * max_importance:
                    redundant.append((layer, head))
        return redundant

    def head_ablation_study(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        metric_fn: Callable,
        device: Optional[torch.device] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        Ablation study: zero out each head and measure performance drop.
        Returns dict mapping (layer, head) -> performance drop.
        """
        device = device or torch.device("cpu")
        baseline = metric_fn(model, dataloader, device)
        drops = {}

        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                # Find and zero out this head's parameters
                ablated_model = self._zero_head(model, layer_idx, head_idx)
                performance = metric_fn(ablated_model, dataloader, device)
                drops[(layer_idx, head_idx)] = baseline - performance
                del ablated_model

        return drops

    def _zero_head(self, model: nn.Module, layer_idx: int, head_idx: int) -> nn.Module:
        """Return a copy of model with specific head zeroed out."""
        ablated = copy.deepcopy(model)
        for name, param in ablated.named_parameters():
            if (f"layers.{layer_idx}" in name or f"layer.{layer_idx}" in name):
                if any(x in name for x in ["q_proj", "query"]):
                    head_dim = param.shape[0] // self.n_heads
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    param.data[start:end].zero_()
        return ablated


import copy


# ---------------------------------------------------------------------------
# Circuit-level mechanistic interpretability
# ---------------------------------------------------------------------------

@dataclass
class Circuit:
    """Represents a computational circuit (subgraph) in the model."""
    name: str
    components: List[str]       # Module names in this circuit
    task: str                   # What task this circuit performs
    importance: float = 0.0


class MechanisticInterpreter:
    """
    Scaffolding for circuit-level mechanistic interpretability.

    Based on Elhage et al. (2021) and Olah et al. (2020).
    Identifies circuits responsible for specific computations.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._circuits: List[Circuit] = []

    def identify_induction_heads(
        self,
        attn_weights: List[Tensor],
        seq_len: int,
    ) -> List[Tuple[int, int]]:
        """
        Identify induction heads: heads that attend to position (i - n + 1)
        when at position i, after seeing a pattern [A][B]...[A].

        Returns list of (layer, head) pairs that exhibit induction behavior.
        """
        induction_heads = []
        for layer_idx, attn in enumerate(attn_weights):
            B, H, T, _ = attn.shape
            for h in range(H):
                head_attn = attn[:, h, :, :].mean(dim=0)  # (T, T)
                # Induction heads: strong off-diagonal at position -seq shift
                if T > 2:
                    # Check if attention peaks at T positions back
                    diag_val = 0.0
                    count = 0
                    for t in range(1, min(T, seq_len)):
                        if t < T:
                            diag_val += head_attn[t, t - 1].item()
                            count += 1
                    avg_induction = diag_val / max(1, count)
                    if avg_induction > 0.2:
                        induction_heads.append((layer_idx, h))
        return induction_heads

    def activation_patching(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        module: nn.Module,
        metric_fn: Callable,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Activation patching: replace activations from corrupted run with
        clean activations and measure recovery in performance.

        High recovery = module is important for the task.
        Returns the recovery score (higher = more important).
        """
        device = device or torch.device("cpu")
        cache = ActivationCache()

        # Run clean forward and cache
        cache.register("target", module)
        self.model(clean_input.to(device))
        clean_act = cache.get_activation("target")
        cache.remove_hooks()

        # Run corrupted and measure baseline
        with torch.no_grad():
            corrupt_out = self.model(corrupted_input.to(device))
        baseline_metric = metric_fn(corrupt_out)

        # Patch: run corrupted with clean activations injected
        def patch_hook(mod, inp, out):
            if isinstance(out, tuple):
                return (clean_act.to(device),) + out[1:]
            return clean_act.to(device)

        hook = module.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_out = self.model(corrupted_input.to(device))
        hook.remove()

        patched_metric = metric_fn(patched_out)
        return patched_metric - baseline_metric

    def logit_lens(
        self,
        model: nn.Module,
        input_ids: Tensor,
        unembedding: nn.Linear,
        layer_names: List[str],
        modules: List[nn.Module],
    ) -> Dict[str, Tensor]:
        """
        Logit lens (nostalgebraist, 2020).
        Projects intermediate representations through the unembedding matrix
        to see what token the model "thinks" it will output at each layer.

        Returns: Dict[layer_name -> (B, T, V) logits]
        """
        cache = ActivationCache()
        for name, module in zip(layer_names, modules):
            cache.register(name, module)

        self.model.eval()
        with torch.no_grad():
            self.model(input_ids)

        logit_lens_results = {}
        for name in layer_names:
            act = cache.get_activation(name)
            if act is not None:
                # Normalize if the model uses layer norm before unembedding
                logits = unembedding(act)
                logit_lens_results[name] = logits

        cache.remove_hooks()
        return logit_lens_results


# ---------------------------------------------------------------------------
# Feature attribution via perturbation
# ---------------------------------------------------------------------------

class PerturbationAttribution:
    """
    Model-agnostic feature attribution via input perturbation.

    Estimates feature importance by masking/perturbing each feature
    and measuring the change in model output.
    """

    def __init__(
        self,
        model: nn.Module,
        perturbation_type: str = "zero",   # "zero" | "mean" | "noise"
        n_samples: int = 1,                # For stochastic perturbations
    ):
        self.model = model
        self.perturbation_type = perturbation_type
        self.n_samples = n_samples

    def attribute(
        self,
        x: Tensor,
        baseline_score: Optional[float] = None,
        target_fn: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute feature importance by perturbation.
        Returns attribution of same shape as x.
        """
        self.model.eval()
        x_np = x.float()

        with torch.no_grad():
            orig_out = self.model(x_np)
            if isinstance(orig_out, dict):
                orig_score = orig_out.get("logits", orig_out.get("output", torch.zeros(1))).sum().item()
            else:
                orig_score = (target_fn(orig_out) if target_fn else orig_out).sum().item()

        attribution = torch.zeros_like(x_np)
        *batch_dims, n_features = x_np.shape

        for feat_idx in range(n_features):
            x_perturbed = x_np.clone()
            if self.perturbation_type == "zero":
                x_perturbed[..., feat_idx] = 0.0
            elif self.perturbation_type == "mean":
                x_perturbed[..., feat_idx] = x_np[..., feat_idx].mean()
            elif self.perturbation_type == "noise":
                x_perturbed[..., feat_idx] = torch.randn_like(x_np[..., feat_idx]) * x_np[..., feat_idx].std()

            with torch.no_grad():
                pert_out = self.model(x_perturbed)
                if isinstance(pert_out, dict):
                    pert_score = pert_out.get("logits", pert_out.get("output", torch.zeros(1))).sum().item()
                else:
                    pert_score = (target_fn(pert_out) if target_fn else pert_out).sum().item()

            attribution[..., feat_idx] = orig_score - pert_score

        return attribution

    def time_step_importance(
        self,
        x: Tensor,
        target_fn: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute importance of each time step by masking entire time steps.
        x: (B, T, F)
        Returns: (B, T) importance scores.
        """
        self.model.eval()
        B, T, F = x.shape

        with torch.no_grad():
            orig_out = self.model(x)
            if isinstance(orig_out, dict):
                orig_score = orig_out.get("logits", orig_out.get("output", torch.zeros(B, 1))).sum(dim=-1)
            else:
                orig_score = (target_fn(orig_out) if target_fn else orig_out).sum(dim=-1)

        importance = torch.zeros(B, T, device=x.device)
        for t in range(T):
            x_masked = x.clone()
            x_masked[:, t, :] = 0.0
            with torch.no_grad():
                masked_out = self.model(x_masked)
                if isinstance(masked_out, dict):
                    masked_score = masked_out.get(
                        "logits", masked_out.get("output", torch.zeros(B, 1))
                    ).sum(dim=-1)
                else:
                    masked_score = (target_fn(masked_out) if target_fn else masked_out).sum(dim=-1)
            importance[:, t] = (orig_score - masked_score)

        return importance


# ---------------------------------------------------------------------------
# Comprehensive interpretability analysis
# ---------------------------------------------------------------------------

class LuminaInterpreter:
    """
    Unified interpretability interface for Lumina models.

    Provides a convenient API for all attribution methods.
    """

    def __init__(
        self,
        model: nn.Module,
        n_layers: int = 12,
        n_heads: int = 12,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)

        self.integrated_gradients = IntegratedGradients(model)
        self.smooth_grad = SmoothGrad(model)
        self.deep_lift = DeepLIFTAttribution(model)
        self.perturbation = PerturbationAttribution(model)
        self.rollout = AttentionRollout()
        self.gradcam = GradCAMTransformer(model)
        self.head_analyzer = AttentionHeadAnalyzer(model, n_layers, n_heads)
        self.mechanistic = MechanisticInterpreter(model)

    def explain(
        self,
        x: Tensor,
        method: str = "integrated_gradients",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified explanation method.

        Args:
            x: Input tensor.
            method: "integrated_gradients" | "smooth_grad" | "deep_lift" |
                    "perturbation" | "gradcam"
            **kwargs: Method-specific arguments.

        Returns:
            Dict with attribution and metadata.
        """
        x = x.to(self.device)
        self.model.eval()

        if method == "integrated_gradients":
            attr = self.integrated_gradients.attribute(x, **kwargs)
        elif method == "smooth_grad":
            attr = self.smooth_grad.attribute(x, **kwargs)
        elif method == "deep_lift":
            attr = self.deep_lift.attribute(x, **kwargs)
        elif method == "perturbation":
            attr = self.perturbation.attribute(x, **kwargs)
        elif method == "gradcam":
            result = self.gradcam.compute(x, **kwargs)
            return result
        else:
            raise ValueError(f"Unknown method: {method}")

        # Aggregate attribution across features
        if attr.ndim == 3:
            attr_agg = attr.abs().mean(dim=-1)  # (B, T)
        else:
            attr_agg = attr.abs()

        return {
            "method": method,
            "attribution": attr,
            "attribution_agg": attr_agg,
            "top_features": attr_agg.argmax(dim=-1).tolist(),
        }

    def full_analysis(
        self,
        x: Tensor,
        attn_weights: Optional[List[Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Run all attribution methods and compile a full analysis report.
        """
        results = {}

        for method in ["integrated_gradients", "smooth_grad", "perturbation"]:
            try:
                results[method] = self.explain(x, method=method)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")

        if attn_weights:
            rollout_attn = self.rollout.get_input_importance(attn_weights)
            results["attention_rollout"] = {
                "importance": rollout_attn,
                "method": "attention_rollout",
            }

            head_patterns = self.head_analyzer.analyze_head_patterns(attn_weights)
            results["head_patterns"] = head_patterns

            induction = self.mechanistic.identify_induction_heads(
                attn_weights, seq_len=x.shape[1]
            )
            results["induction_heads"] = induction

        return results


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Hooks
    "ActivationCache",
    "AttentionCache",
    # Attention rollout
    "AttentionRollout",
    # GradCAM
    "GradCAMTransformer",
    # Attribution methods
    "IntegratedGradients",
    "SmoothGrad",
    "DeepLIFTAttribution",
    "PerturbationAttribution",
    # Probing
    "LinearProbe",
    "ProbingClassifierSuite",
    # Head analysis
    "AttentionHeadAnalyzer",
    # Mechanistic
    "Circuit",
    "MechanisticInterpreter",
    # Unified
    "LuminaInterpreter",
]
