#!/usr/bin/env python3
"""Mega expansion 5 - large additions to lora.py, moe.py, continual_learning.py, deployment.py, rlhf.py"""
import os, textwrap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def append(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    lines = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {lines} lines")
    return lines

# ─── lora.py additions ────────────────────────────────────────────────────────
LORA_ADD = '''

# ============================================================
# Extended LoRA Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fan_in_fan_out: bool = False
    bias: str = "none"  # none | all | lora_only
    modules_to_save: List[str] = field(default_factory=list)
    init_lora_weights: bool = True
    lora_on_all_linear: bool = False

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer (Hu et al. 2022)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._merged = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B already zeros

    def merge_weights(self):
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if self.fan_in_fan_out:
            delta = delta.T
        self.weight.data += delta
        self._merged = True

    def unmerge_weights(self):
        if not self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if self.fan_in_fan_out:
            delta = delta.T
        self.weight.data -= delta
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}, merged={self._merged}")


class LoRAEmbedding(nn.Module):
    """LoRA-adapted embedding layer."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.lora_A = nn.Parameter(torch.empty(num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim))

        nn.init.normal_(self.weight)
        nn.init.normal_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.embedding(x, self.weight)
        lora_A_x = F.embedding(x, self.lora_A)
        lora_out = (lora_A_x @ self.lora_B) * self.scaling
        return base + lora_out


class LoRAConv1d(nn.Module):
    """LoRA-adapted 1-D convolution for sequence models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 4,
        alpha: float = 8.0,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.scaling = alpha / rank

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        # LoRA on unfolded kernel: (out, in*k) -> low rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_channels * kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        # Build delta kernel from lora
        delta_w = (self.lora_B @ self.lora_A).view(self.out_channels, self.in_channels, self.kernel_size)
        delta_w = delta_w * self.scaling
        lora_out = F.conv1d(x, delta_w, padding=self.conv.padding[0], stride=self.conv.stride[0])
        return base + lora_out


class AdaLoRA(nn.Module):
    """Adaptive LoRA (Zhang et al. 2023) with singular value decomposition budget."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_rank: int = 12,
        target_rank: int = 4,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_rank = initial_rank
        self.target_rank = target_rank
        self.scaling = alpha / initial_rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # SVD-form: W_delta = P * diag(sigma) * Q^T
        self.lora_P = nn.Parameter(torch.empty(out_features, initial_rank))
        self.lora_sigma = nn.Parameter(torch.ones(initial_rank))
        self.lora_Q = nn.Parameter(torch.empty(in_features, initial_rank))

        nn.init.orthogonal_(self.lora_P)
        nn.init.orthogonal_(self.lora_Q)

        self.importance_scores = torch.zeros(initial_rank)
        self.register_buffer("_importance", self.importance_scores)

    def prune_to_rank(self, new_rank: int):
        """Prune singular vectors to new_rank based on importance."""
        idx = self._importance.topk(new_rank).indices
        with torch.no_grad():
            self.lora_P.data = self.lora_P.data[:, idx]
            self.lora_sigma.data = self.lora_sigma.data[idx]
            self.lora_Q.data = self.lora_Q.data[:, idx]
        self.initial_rank = new_rank
        self._importance = torch.zeros(new_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        # delta W = P * diag(sigma) * Q^T
        delta_w = (self.lora_P * self.lora_sigma.unsqueeze(0)) @ self.lora_Q.T
        lora_out = F.linear(x, delta_w * self.scaling)
        # update importance scores (EMA of sigma gradient norms)
        if self.training and self.lora_sigma.grad is not None:
            with torch.no_grad():
                self._importance = 0.9 * self._importance + 0.1 * self.lora_sigma.grad.abs()
        return base + lora_out


class VeRALayer(nn.Module):
    """VeRA (Kopiczko et al. 2024) - Vector-based Random Matrix Adaptation.

    Shares frozen random matrices across all layers; only adapts tiny scale vectors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Shared frozen random matrices (same seed -> same matrices across layers)
        gen = torch.Generator()
        gen.manual_seed(seed)
        A = torch.randn(rank, in_features, generator=gen)
        B = torch.randn(out_features, rank, generator=gen)
        self.register_buffer("shared_A", A / math.sqrt(rank))
        self.register_buffer("shared_B", B / math.sqrt(out_features))

        # Trainable scale vectors (very few params)
        self.vera_b = nn.Parameter(torch.ones(rank))
        self.vera_d = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        # delta = diag(d) * B * diag(b) * A
        Ax = F.linear(x, self.shared_A)          # (..., rank)
        bAx = Ax * self.vera_b                    # element-wise scale
        BAx = F.linear(bAx, self.shared_B)        # (..., out)
        dBAx = BAx * self.vera_d                  # element-wise scale
        return base + dBAx


class DoRALayer(nn.Module):
    """DoRA (Liu et al. 2024) - Weight-Decomposed Low-Rank Adaptation.

    Decomposes pretrained weight into magnitude + direction, adapts direction with LoRA.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen pretrained weight direction (normalized)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Magnitude vector (trainable)
        self.magnitude = nn.Parameter(torch.ones(out_features, 1))

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapted weight direction = W + delta (low-rank)
        delta = (self.lora_B @ self.lora_A) * self.scaling
        adapted = self.weight + delta
        # Normalize to unit column norm, then scale by magnitude
        norm = adapted.norm(dim=1, keepdim=True).clamp(min=1e-8)
        w_eff = self.magnitude * (adapted / norm)
        return F.linear(x, w_eff)


class LoRAMoELayer(nn.Module):
    """Mixture-of-LoRA-Experts: multiple LoRA adapters with soft routing.

    Each token gets a mixture of LoRA adaptations, enabling task-conditional adaptation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Expert LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Router
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        B, T, D = x.shape

        # Route
        logits = self.router(x)  # (B, T, E)
        weights = F.softmax(logits, dim=-1)

        # Top-k selection
        topk_vals, topk_idx = weights.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Compute each expert's delta
        # lora_A: (E, r, D), lora_B: (E, out, r)
        x_flat = x.reshape(-1, D)
        expert_out = torch.zeros(B * T, self.out_features, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            idx = topk_idx[:, :, k].reshape(-1)  # (B*T,)
            w = topk_vals[:, :, k].reshape(-1, 1)  # (B*T, 1)

            # Gather per-token expert matrices (expensive but correct)
            A_k = self.lora_A[idx]  # (B*T, r, D)
            B_k = self.lora_B[idx]  # (B*T, out, r)

            # delta_x = B_k @ A_k @ x -> (B*T, out)
            Ax = torch.bmm(A_k, x_flat.unsqueeze(-1)).squeeze(-1)  # (B*T, r)
            BAx = torch.bmm(B_k, Ax.unsqueeze(-1)).squeeze(-1)     # (B*T, out)
            expert_out = expert_out + w * BAx * self.scaling

        return base + expert_out.reshape(B, T, self.out_features)


class LoRAGradientProjection(nn.Module):
    """Gradient projection for LoRA: keeps gradients in low-rank subspace (GaLore style)."""

    def __init__(self, weight: nn.Parameter, rank: int = 8, update_freq: int = 200):
        super().__init__()
        self.weight = weight
        self.rank = rank
        self.update_freq = update_freq
        self._step = 0
        self._proj_matrix: Optional[torch.Tensor] = None

    def update_projection(self):
        """Recompute SVD projection matrix from current weight gradient."""
        if self.weight.grad is None:
            return
        G = self.weight.grad.data
        # Thin SVD
        try:
            U, S, Vt = torch.linalg.svd(G, full_matrices=False)
            self._proj_matrix = U[:, :self.rank]  # (out, rank)
        except Exception:
            pass

    def project_gradient(self):
        """Project gradient into low-rank subspace."""
        if self._proj_matrix is None or self.weight.grad is None:
            return
        P = self._proj_matrix
        G = self.weight.grad.data
        # G_proj = P @ P^T @ G
        self.weight.grad.data = P @ (P.T @ G)

    def step(self):
        self._step += 1
        if self._step % self.update_freq == 0:
            self.update_projection()
        self.project_gradient()


class LoRAModelWrapper(nn.Module):
    """Wraps a base model and injects LoRA layers into target modules."""

    def __init__(self, model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.model = model
        self.config = config
        self._lora_layers: Dict[str, LoRALinear] = {}
        self._inject_lora()
        self._freeze_base_model()

    def _inject_lora(self):
        """Replace target linear layers with LoRALinear."""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            should_inject = (
                self.config.lora_on_all_linear or
                any(t in name for t in self.config.target_modules)
            )
            if not should_inject:
                continue
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.model if parent_name == "" else dict(self.model.named_modules())[parent_name]
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                bias=module.bias is not None,
            )
            lora_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data)
            setattr(parent, child_name, lora_layer)
            self._lora_layers[name] = lora_layer

    def _freeze_base_model(self):
        """Freeze all parameters except LoRA adapters."""
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into base model and return unmodified model."""
        for layer in self._lora_layers.values():
            layer.merge_weights()
        return self.model

    def get_lora_parameters(self) -> List[nn.Parameter]:
        params = []
        for layer in self._lora_layers.values():
            params.extend([layer.lora_A, layer.lora_B])
        return params

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {}
        for name, layer in self._lora_layers.items():
            sd[f"{name}.lora_A"] = layer.lora_A.data
            sd[f"{name}.lora_B"] = layer.lora_B.data
        return sd

    def load_lora_state_dict(self, sd: Dict[str, torch.Tensor]):
        for name, layer in self._lora_layers.items():
            if f"{name}.lora_A" in sd:
                layer.lora_A.data.copy_(sd[f"{name}.lora_A"])
            if f"{name}.lora_B" in sd:
                layer.lora_B.data.copy_(sd[f"{name}.lora_B"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_ratio(self) -> float:
        return self.num_trainable_parameters() / max(1, self.num_total_parameters())


class LoRATrainer:
    """High-level trainer for LoRA fine-tuning with gradient clipping and checkpointing."""

    def __init__(
        self,
        model: LoRAModelWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./lora_checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.step = 0
        self.losses: List[float] = []

    def train_step(self, batch: Dict[str, torch.Tensor], loss_fn) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(**batch)
        loss = loss_fn(output, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"lora_{tag}.pt")
        torch.save({
            "step": self.step,
            "lora_state_dict": self.model.lora_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        return path

    def load_checkpoint(self, tag: str = "latest"):
        path = os.path.join(self.checkpoint_dir, f"lora_{tag}.pt")
        ckpt = torch.load(path, map_location="cpu")
        self.step = ckpt["step"]
        self.model.load_lora_state_dict(ckpt["lora_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def compute_average_loss(self, last_n: int = 100) -> float:
        if not self.losses:
            return float("nan")
        return sum(self.losses[-last_n:]) / len(self.losses[-last_n:])


class QuantizedLoRALinear(nn.Module):
    """QLoRA-style: frozen INT8 base weight + FP32 LoRA adapters (Dettmers et al. 2023)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.bits = bits

        # Simulated quantized weight (stored as INT8)
        weight_fp32 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight_fp32, a=math.sqrt(5))
        scale = weight_fp32.abs().max() / (2 ** (bits - 1) - 1)
        weight_int = (weight_fp32 / scale).round().clamp(-(2**(bits-1)), 2**(bits-1)-1).to(torch.int8)
        self.register_buffer("weight_int", weight_int)
        self.register_buffer("weight_scale", scale.unsqueeze(0))

        # LoRA adapters in FP32 (not quantized)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def dequantize_weight(self) -> torch.Tensor:
        return self.weight_int.float() * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight()
        base = F.linear(x, w)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora_out


class MultiTaskLoRA(nn.Module):
    """Multi-task LoRA with task-specific adapters selected by task id."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tasks: int,
        rank: int = 8,
        alpha: float = 16.0,
        shared_rank: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tasks = num_tasks
        self.scaling = alpha / rank
        self.shared_scaling = alpha / shared_rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Shared adapter (applied to all tasks)
        self.shared_A = nn.Parameter(torch.empty(shared_rank, in_features))
        self.shared_B = nn.Parameter(torch.zeros(out_features, shared_rank))
        nn.init.kaiming_uniform_(self.shared_A, a=math.sqrt(5))

        # Task-specific adapters
        self.task_A = nn.Parameter(torch.empty(num_tasks, rank, in_features))
        self.task_B = nn.Parameter(torch.zeros(num_tasks, out_features, rank))
        nn.init.kaiming_uniform_(self.task_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        base = F.linear(x, self.weight)
        shared_out = F.linear(F.linear(x, self.shared_A), self.shared_B) * self.shared_scaling
        if task_id is not None:
            A = self.task_A[task_id]  # (rank, D)
            B = self.task_B[task_id]  # (out, rank)
            task_out = F.linear(F.linear(x, A), B) * self.scaling
        else:
            task_out = 0.0
        return base + shared_out + task_out


# ─── LoRA rank schedule ───────────────────────────────────────────────────────

class RankScheduler:
    """Gradually reduces LoRA rank during training (rank annealing)."""

    def __init__(self, initial_rank: int, final_rank: int, total_steps: int):
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.total_steps = total_steps

    def get_rank(self, step: int) -> int:
        frac = min(step / self.total_steps, 1.0)
        rank = int(self.initial_rank + frac * (self.final_rank - self.initial_rank))
        return max(rank, self.final_rank)


class LoRARegularizer:
    """Regularization for LoRA weights: nuclear norm, Frobenius norm, orthogonality."""

    def __init__(
        self,
        nuclear_weight: float = 0.0,
        frobenius_weight: float = 1e-4,
        ortho_weight: float = 0.0,
    ):
        self.nuclear_weight = nuclear_weight
        self.frobenius_weight = frobenius_weight
        self.ortho_weight = ortho_weight

    def __call__(self, lora_layers: List[LoRALinear]) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for layer in lora_layers:
            delta = layer.lora_B @ layer.lora_A  # (out, in)
            if self.nuclear_weight > 0:
                loss = loss + self.nuclear_weight * torch.linalg.matrix_norm(delta, ord="nuc")
            if self.frobenius_weight > 0:
                loss = loss + self.frobenius_weight * delta.norm(p="fro")
            if self.ortho_weight > 0:
                # Orthogonality: A A^T should be identity
                AtA = layer.lora_A @ layer.lora_A.T
                I = torch.eye(AtA.shape[0], device=AtA.device, dtype=AtA.dtype)
                loss = loss + self.ortho_weight * (AtA - I).norm(p="fro")
        return loss
'''

# ─── moe.py additions ─────────────────────────────────────────────────────────
MOE_ADD = '''

# ============================================================
# Extended MoE Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
import torch.distributed as dist


class ExpertCapacity:
    """Computes per-expert capacity for load-balanced dispatch."""

    @staticmethod
    def compute(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity: int = 1) -> int:
        capacity = math.ceil(capacity_factor * num_tokens / num_experts)
        return max(capacity, min_capacity)


class RouterOutput(NamedTuple):
    dispatch_mask: torch.Tensor      # (B*T, E, C) one-hot
    combine_weights: torch.Tensor    # (B*T, E, C) float
    router_probs: torch.Tensor       # (B*T, E)
    load_loss: torch.Tensor          # scalar


class TopKRouter(nn.Module):
    """Top-K router with auxiliary load balancing loss (Switch/GShard style)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "rsample",  # rsample | none
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gate_policy = noisy_gate_policy
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

        if noisy_gate_policy == "rsample":
            self.noise_gate = nn.Linear(d_model, num_experts, bias=False)
            nn.init.normal_(self.noise_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """x: (N, D) where N = B*T"""
        N, D = x.shape
        E = self.num_experts
        C = ExpertCapacity.compute(N, E, self.capacity_factor)

        logits = self.gate(x)  # (N, E)

        if self.training and self.noisy_gate_policy == "rsample":
            noise_std = F.softplus(self.noise_gate(x))
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        router_probs = F.softmax(logits, dim=-1)  # (N, E)

        # Top-k selection
        topk_vals, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Build dispatch and combine tensors
        dispatch_mask = torch.zeros(N, E, C, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, E, C, device=x.device, dtype=x.dtype)

        # Assign tokens to experts with capacity constraint
        expert_counts = torch.zeros(E, dtype=torch.long, device=x.device)
        for token_idx in range(N):
            for k_idx in range(self.top_k):
                e = topk_idx[token_idx, k_idx].item()
                c = expert_counts[e].item()
                if c < C:
                    dispatch_mask[token_idx, e, c] = True
                    combine_weights[token_idx, e, c] = topk_vals[token_idx, k_idx]
                    expert_counts[e] += 1

        # Auxiliary load balancing loss
        # L_aux = alpha * E * sum_e(f_e * p_e)
        f = (topk_idx == torch.arange(E, device=x.device).unsqueeze(0).unsqueeze(0)).float().mean(0).mean(0)
        p = router_probs.mean(0)
        load_loss = self.aux_loss_weight * E * (f * p).sum()

        return RouterOutput(dispatch_mask, combine_weights, router_probs, load_loss)


class ExpertFFN(nn.Module):
    """Single expert FFN (feed-forward network)."""

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoELayer(nn.Module):
    """Sparse Mixture-of-Experts layer with top-K routing and capacity constraints."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
        dropout: float = 0.0,
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor, aux_loss_weight=aux_loss_weight)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, activation, dropout) for _ in range(num_experts)
        ])
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        router_out = self.router(x_flat)
        self.aux_loss = router_out.load_loss

        dispatch = router_out.dispatch_mask   # (N, E, C)
        combine = router_out.combine_weights  # (N, E, C)
        C = dispatch.shape[-1]

        output = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            # Gather tokens dispatched to expert e
            # dispatch[:, e, :] -> (N, C) bool
            token_indices = dispatch[:, e, :].any(dim=-1).nonzero(as_tuple=True)[0]  # (K,)
            if token_indices.numel() == 0:
                continue
            expert_in = x_flat[token_indices]  # (K, D)
            expert_out = expert(expert_in)       # (K, D)
            # Accumulate with combine weights
            for slot in range(C):
                dispatched = dispatch[token_indices, e, slot]  # (K,)
                weight = combine[token_indices, e, slot]        # (K,)
                output[token_indices] += dispatched.float().unsqueeze(-1) * weight.unsqueeze(-1) * expert_out

        return output.reshape(B, T, D)


class FusedMoELayer(nn.Module):
    """Fused MoE using batched matrix multiply for efficiency (no Python loop over experts)."""

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff

        # All expert weights stacked
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.router = nn.Linear(d_model, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            idx = indices[:, k]     # (N,)
            w = weights[:, k]       # (N,)
            # Gather expert weights per token
            w1_k = self.w1[idx]     # (N, d_ff, D)
            w2_k = self.w2[idx]     # (N, D, d_ff)
            h = torch.bmm(w1_k, x_flat.unsqueeze(-1)).squeeze(-1)  # (N, d_ff)
            h = F.gelu(h)
            out_k = torch.bmm(w2_k, h.unsqueeze(-1)).squeeze(-1)   # (N, D)
            output = output + w.unsqueeze(-1) * out_k

        return output.reshape(B, T, D)


class ExpertChoiceLayer(nn.Module):
    """Expert-Choice MoE (Zhou et al. 2022): each expert selects top-k tokens.

    Guarantees perfect load balance; each expert processes exactly C tokens.
    """

    def __init__(self, d_model: int, num_experts: int = 8, expert_capacity: int = 32, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = min(self.expert_capacity, N)

        # Router: (N, E) -> each expert picks top-C tokens
        logits = self.router(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=0)  # softmax over tokens dimension for expert-choice

        # Expert i selects top-C tokens by its column
        output = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            expert_probs = probs[:, e]            # (N,)
            topC_vals, topC_idx = expert_probs.topk(C)
            selected = x_flat[topC_idx]           # (C, D)
            expert_out = expert(selected)          # (C, D)
            output.scatter_add_(0, topC_idx.unsqueeze(-1).expand(-1, D),
                                topC_vals.unsqueeze(-1) * expert_out)

        return output.reshape(B, T, D)


class SwitchTransformerLayer(nn.Module):
    """Switch Transformer FFN layer (Fedus et al. 2021) with top-1 routing."""

    def __init__(self, d_model: int, num_experts: int = 8, d_ff: Optional[int] = None,
                 capacity_factor: float = 1.5, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=dropout) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = ExpertCapacity.compute(N, self.num_experts, self.capacity_factor)

        logits = self.router(x_flat)              # (N, E)
        probs = F.softmax(logits, dim=-1)          # (N, E)
        chosen_e = probs.argmax(dim=-1)            # (N,)
        chosen_p = probs.max(dim=-1).values        # (N,)

        # Aux loss
        f = F.one_hot(chosen_e, self.num_experts).float().mean(0)
        p = probs.mean(0)
        self.aux_loss = self.num_experts * (f * p).sum()

        output = torch.zeros_like(x_flat)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for i in range(N):
            e = chosen_e[i].item()
            if counts[e] < C:
                expert_out = self.experts[e](x_flat[i:i+1])
                output[i] = chosen_p[i] * expert_out.squeeze(0)
                counts[e] += 1

        return output.reshape(B, T, D)


class MoETransformerBlock(nn.Module):
    """Transformer block with every-N-th FFN replaced by MoE layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        moe_layer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if moe_layer:
            self.ffn = SparseMoELayer(d_model, num_experts, top_k, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
        self.moe_layer = moe_layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        # FFN / MoE
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    @property
    def aux_loss(self) -> torch.Tensor:
        if self.moe_layer and hasattr(self.ffn, "aux_loss"):
            return self.ffn.aux_loss
        return torch.tensor(0.0)


class MoELanguageModel(nn.Module):
    """Simple MoE language model with alternating dense/MoE FFN layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        moe_every_n: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model, num_heads, num_experts, top_k,
                moe_layer=(i % moe_every_n == (moe_every_n - 1)),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))
        aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, "aux_loss"):
                aux_loss = aux_loss + layer.aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss


class DistributedMoE(nn.Module):
    """Distributed MoE where each rank holds a subset of experts (Expert Parallelism)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.world_size = world_size
        self.rank = rank

        assert num_experts % world_size == 0
        self.local_num_experts = num_experts // world_size

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.local_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(self.local_num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: run local experts only (in real usage, would all_to_all)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        local_start = self.rank * self.local_num_experts
        local_end = local_start + self.local_num_experts

        for k in range(self.top_k):
            idx = topk_idx[:, k]
            w = topk_vals[:, k]
            # Filter for local experts
            local_mask = (idx >= local_start) & (idx < local_end)
            if local_mask.any():
                local_idx = idx[local_mask] - local_start
                x_sel = x_flat[local_mask]
                for le in range(self.local_num_experts):
                    le_mask = local_idx == le
                    if le_mask.any():
                        out = self.local_experts[le](x_sel[le_mask])
                        tok_idx = local_mask.nonzero(as_tuple=True)[0][le_mask]
                        output[tok_idx] += w[tok_idx].unsqueeze(-1) * out

        return output.reshape(B, T, D)


class MoELoadBalancer:
    """Monitors and adjusts expert load balance during training."""

    def __init__(self, num_experts: int, ema_decay: float = 0.99):
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.expert_counts = torch.zeros(num_experts)
        self.total_tokens = 0

    def update(self, dispatch_mask: torch.Tensor):
        """dispatch_mask: (N, E) binary"""
        counts = dispatch_mask.float().sum(0).detach().cpu()
        self.expert_counts = self.ema_decay * self.expert_counts + (1 - self.ema_decay) * counts
        self.total_tokens += dispatch_mask.shape[0]

    def load_imbalance(self) -> float:
        """Returns coefficient of variation of expert loads."""
        mean = self.expert_counts.mean()
        std = self.expert_counts.std()
        return (std / (mean + 1e-8)).item()

    def underutilized_experts(self, threshold: float = 0.5) -> List[int]:
        mean = self.expert_counts.mean()
        return [i for i, c in enumerate(self.expert_counts) if c < threshold * mean]

    def report(self) -> Dict[str, Any]:
        return {
            "expert_counts": self.expert_counts.tolist(),
            "load_imbalance_cv": self.load_imbalance(),
            "underutilized": self.underutilized_experts(),
        }
'''

lora_lines = append("lora.py", LORA_ADD)
moe_lines = append("moe.py", MOE_ADD)

# ─── continual_learning.py additions ─────────────────────────────────────────
CL_ADD = '''

# ============================================================
# Extended Continual Learning Components
# ============================================================

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class ElasticWeightConsolidation(nn.Module):
    """EWC (Kirkpatrick et al. 2017): penalizes changes to important weights.

    Importance estimated by diagonal Fisher information matrix.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, n_fisher_samples: int = 200):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.n_fisher_samples = n_fisher_samples

        # Stores: {param_name: (optimal_param, fisher_diag)}
        self._anchors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def estimate_fisher(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Estimate Fisher information diagonal on current task data."""
        self.model.eval()
        fisher_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        count = 0
        for batch in dataloader:
            if count >= self.n_fisher_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)
            count += 1

        # Average
        for name in fisher_dict:
            fisher_dict[name] /= count

        # Store anchors
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._anchors[name] = (
                    param.data.clone(),
                    fisher_dict.get(name, torch.zeros_like(param.data)),
                )

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._anchors:
                optimal, fisher = self._anchors[name]
                optimal = optimal.to(param.device)
                fisher = fisher.to(param.device)
                loss = loss + (fisher * (param - optimal).pow(2)).sum()
        return 0.5 * self.ewc_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class OnlineEWC(ElasticWeightConsolidation):
    """Online EWC (Schwarz et al. 2018): running average of Fisher matrices across tasks."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, gamma: float = 0.95):
        super().__init__(model, ewc_lambda)
        self.gamma = gamma
        self._task_count = 0

    def consolidate(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Consolidate current task into running EWC anchor."""
        self.estimate_fisher(dataloader, loss_fn, device)
        self._task_count += 1

        if self._task_count > 1:
            # Decay old Fisher estimates
            for name in self._anchors:
                optimal, fisher = self._anchors[name]
                self._anchors[name] = (optimal, self.gamma * fisher)


class ProgressiveNeuralNetworks(nn.Module):
    """Progressive Neural Networks (Rusu et al. 2016).

    Adds new columns for each task; lateral connections from all previous columns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.columns: nn.ModuleList = nn.ModuleList()
        self.lateral_connections: nn.ModuleList = nn.ModuleList()
        self._add_column()

    def _add_column(self):
        """Add a new network column."""
        k = len(self.columns)
        col = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        ])
        self.columns.append(col)

        if k > 0:
            # Lateral adapters from all prev columns at each hidden layer
            laterals = nn.ModuleList()
            for prev_k in range(k):
                lat = nn.ModuleList([
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                ])
                laterals.append(lat)
            self.lateral_connections.append(laterals)

        # Freeze all previous columns
        for prev_k in range(k):
            for param in self.columns[prev_k].parameters():
                param.requires_grad_(False)

    def add_task(self):
        """Add a new column for a new task."""
        self._add_column()

    def forward(self, x: torch.Tensor, column_idx: Optional[int] = None) -> torch.Tensor:
        k = column_idx if column_idx is not None else len(self.columns) - 1
        col = self.columns[k]

        # Layer 1
        h_prev = [None] * k  # hidden states from prev columns at each layer
        h = F.relu(col[0](x))

        # Collect from prev columns layer 0
        prev_h = []
        for pk in range(k):
            prev_h.append(F.relu(self.columns[pk][0](x)))

        # Layer 2 with laterals
        h2_input = h
        if k > 0:
            laterals_for_k = self.lateral_connections[k - 1]
            for pk, lat in enumerate(laterals_for_k):
                h2_input = h2_input + lat[0](prev_h[pk])
        h = F.relu(col[1](h2_input))

        # Output layer
        out = col[2](h)
        return out


class PacketNetworks(nn.Module):
    """PackNet (Mallya & Lazebnik 2018): hard parameter isolation via binary masks."""

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self._masks: Dict[str, torch.Tensor] = {}
        self._task_masks: List[Dict[str, torch.Tensor]] = []
        self._current_task = 0

    def _get_prunable_params(self):
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dim() >= 2
        }

    def pack_task(self):
        """After training task k, prune low-magnitude weights and assign to task k+1."""
        params = self._get_prunable_params()

        # Find free parameters (not assigned to any previous task)
        task_mask = {}
        for name, param in params.items():
            existing_mask = self._masks.get(name, torch.zeros_like(param.data, dtype=torch.bool))
            free = ~existing_mask
            # Rank free weights by magnitude
            magnitudes = param.data.abs() * free.float()
            flat_mag = magnitudes.flatten()
            # Keep top (1-prune_ratio) fraction of free weights
            k = max(1, int(free.sum().item() * (1 - self.prune_ratio)))
            threshold = flat_mag.topk(k).values.min()
            new_mask = (magnitudes >= threshold) & free
            task_mask[name] = new_mask
            # Update global free mask
            self._masks[name] = existing_mask | new_mask

        self._task_masks.append(task_mask)
        self._current_task += 1

    def apply_task_mask(self, task_id: int):
        """Zero out parameters not belonging to task_id."""
        if task_id >= len(self._task_masks):
            return
        mask = self._task_masks[task_id]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    param.data *= mask[name].float()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ContinualNormalization(nn.Module):
    """Task-specific batch normalization statistics for continual learning."""

    def __init__(self, num_features: int, num_tasks: int = 10, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.eps = eps
        self.momentum = momentum
        self.current_task = 0

        # Per-task affine params
        self.weight = nn.Parameter(torch.ones(num_tasks, num_features))
        self.bias = nn.Parameter(torch.zeros(num_tasks, num_features))

        # Per-task running stats (not parameters)
        self.register_buffer("running_mean", torch.zeros(num_tasks, num_features))
        self.register_buffer("running_var", torch.ones(num_tasks, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.current_task
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean[t] = (1 - self.momentum) * self.running_mean[t] + self.momentum * mean.detach()
            self.running_var[t] = (1 - self.momentum) * self.running_var[t] + self.momentum * var.detach()
        else:
            mean = self.running_mean[t]
            var = self.running_var[t]

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.weight[t] * x_norm + self.bias[t]

    def set_task(self, task_id: int):
        assert 0 <= task_id < self.num_tasks
        self.current_task = task_id


class GradientEpisodicMemory(nn.Module):
    """GEM (Lopez-Paz & Ranzato 2017): project gradients to avoid forgetting.

    Maintains episodic memory per task and projects current gradient
    to satisfy dot-product constraints with memory gradients.
    """

    def __init__(self, model: nn.Module, memory_size_per_task: int = 100, n_tasks: int = 10):
        super().__init__()
        self.model = model
        self.memory_size_per_task = memory_size_per_task
        self.n_tasks = n_tasks

        self.memory_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prev_gradients: List[torch.Tensor] = []
        self._current_task = 0

    def store_memory(self, x: torch.Tensor, y: torch.Tensor):
        """Store a sample into episodic memory."""
        idx = torch.randperm(x.shape[0])[:self.memory_size_per_task]
        self.memory_data.append((x[idx].detach().clone(), y[idx].detach().clone()))

    def compute_memory_gradients(self, loss_fn: Callable, device: str = "cpu") -> List[torch.Tensor]:
        """Compute gradients on all past task memories."""
        past_grads = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        for mem_x, mem_y in self.memory_data[:-1]:  # exclude current task
            self.model.zero_grad()
            mem_x, mem_y = mem_x.to(device), mem_y.to(device)
            loss = loss_fn(self.model(mem_x), mem_y)
            loss.backward()
            grad = torch.cat([p.grad.data.flatten() for p in params if p.grad is not None])
            past_grads.append(grad.clone())
        return past_grads

    def project_gradient(self, current_grad: torch.Tensor, past_grads: List[torch.Tensor]) -> torch.Tensor:
        """Project current_grad so dot product with all past_grads is >= 0."""
        g = current_grad.clone()
        for pg in past_grads:
            dot = (g * pg).sum()
            if dot < 0:
                # Project: g = g - (g·pg / pg·pg) * pg
                g = g - (dot / (pg * pg).sum()) * pg
        return g

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MemoryReplayBuffer:
    """Experience replay buffer for continual learning."""

    def __init__(self, capacity: int = 5000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy  # reservoir | fifo | class_balanced
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self._n_seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add samples to replay buffer."""
        n = x.shape[0]
        for i in range(n):
            self._n_seen += 1
            if self.strategy == "reservoir":
                if len(self.buffer_x) < self.capacity:
                    self.buffer_x.append(x[i].clone())
                    self.buffer_y.append(y[i].clone())
                else:
                    j = torch.randint(0, self._n_seen, (1,)).item()
                    if j < self.capacity:
                        self.buffer_x[j] = x[i].clone()
                        self.buffer_y[j] = y[i].clone()
            else:  # fifo
                self.buffer_x.append(x[i].clone())
                self.buffer_y.append(y[i].clone())
                if len(self.buffer_x) > self.capacity:
                    self.buffer_x.pop(0)
                    self.buffer_y.pop(0)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items from buffer."""
        idx = torch.randperm(len(self.buffer_x))[:n]
        x = torch.stack([self.buffer_x[i] for i in idx])
        y = torch.stack([self.buffer_y[i] for i in idx])
        return x, y

    def __len__(self) -> int:
        return len(self.buffer_x)


class DualMemorySystem(nn.Module):
    """Complementary Learning System (CLS): hippocampus + neocortex dual memory.

    Fast-learning episodic memory (hippocampus) + slow-learning semantic memory (neocortex).
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        semantic_dim: int,
        hippo_memory_size: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.semantic_dim = semantic_dim

        # Hippocampus: fast-learning autoassociative memory
        self.hippo_keys = nn.Parameter(torch.randn(hippo_memory_size, input_dim))
        self.hippo_values = nn.Parameter(torch.randn(hippo_memory_size, memory_dim))

        # Neocortex: slow-learning semantic model
        self.neocortex = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, memory_dim),
        )

        # Integration gate
        self.gate = nn.Linear(2 * memory_dim, 1)

    def hippocampal_recall(self, x: torch.Tensor) -> torch.Tensor:
        """Soft attention-based recall from episodic memory."""
        # x: (B, D)
        sim = F.cosine_similarity(x.unsqueeze(1), self.hippo_keys.unsqueeze(0), dim=-1)  # (B, M)
        weights = F.softmax(sim / 0.1, dim=-1)  # (B, M)
        recalled = weights @ self.hippo_values  # (B, memory_dim)
        return recalled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hippo_out = self.hippocampal_recall(x)         # (B, memory_dim)
        neo_out = self.neocortex(x)                    # (B, memory_dim)
        # Gate between systems
        gate_input = torch.cat([hippo_out, neo_out], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))   # (B, 1)
        return alpha * hippo_out + (1 - alpha) * neo_out


class SynapticIntelligence(nn.Module):
    """Synaptic Intelligence (Zenke et al. 2017): online importance estimation via path integral."""

    def __init__(self, model: nn.Module, si_lambda: float = 0.1, damping: float = 0.1):
        super().__init__()
        self.model = model
        self.si_lambda = si_lambda
        self.damping = damping

        self._prev_params: Dict[str, torch.Tensor] = {}
        self._running_importance: Dict[str, torch.Tensor] = {}
        self._old_params: Dict[str, torch.Tensor] = {}
        self._W: Dict[str, torch.Tensor] = {}

        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.data.clone()
                self._running_importance[name] = torch.zeros_like(param.data)
                self._W[name] = torch.zeros_like(param.data)

    def update_w(self):
        """Update path integral W after each step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.data - self._prev_params[name]
                self._W[name] += -param.grad.data * delta
                self._prev_params[name] = param.data.clone()

    def consolidate(self):
        """Consolidate importance after task ends."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                omega = self._W[name] / ((param.data - self._old_params.get(name, param.data)).pow(2) + self.damping)
                self._running_importance[name] = self._running_importance.get(name, torch.zeros_like(omega)) + F.relu(omega)
                self._old_params[name] = param.data.clone()
                self._W[name] = torch.zeros_like(param.data)

    def si_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._old_params and name in self._running_importance:
                loss = loss + (self._running_importance[name] * (param - self._old_params[name]).pow(2)).sum()
        return self.si_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
'''

cl_lines = append("continual_learning.py", CL_ADD)
print(f"Total so far: lora={lora_lines}, moe={moe_lines}, cl={cl_lines}")

# ─── deployment.py additions ──────────────────────────────────────────────────
DEPLOY_ADD = '''

# ============================================================
# Extended Deployment Components
# ============================================================

import os
import time
import json
import math
import hashlib
import threading
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    max_batch_size: int = 32
    max_seq_len: int = 512
    timeout_ms: float = 100.0
    num_workers: int = 4
    device: str = "cpu"
    use_fp16: bool = False
    use_dynamic_batching: bool = True
    cache_size: int = 1000
    warmup_steps: int = 10
    log_requests: bool = True
    rate_limit_qps: Optional[float] = None


@dataclass
class Request:
    """Single inference request."""
    request_id: str
    inputs: Dict[str, torch.Tensor]
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Single inference response."""
    request_id: str
    outputs: Dict[str, torch.Tensor]
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class RequestQueue:
    """Priority queue for inference requests with timeout support."""

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        self._counter = 0

    def put(self, request: Request, block: bool = True, timeout: Optional[float] = None):
        # Negate priority for max-heap behavior
        item = (-request.priority, self._counter, request)
        self._counter += 1
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Request:
        _, _, request = self._queue.get(block=block, timeout=timeout)
        return request

    def get_batch(self, max_size: int, timeout_ms: float = 10.0) -> List[Request]:
        """Collect up to max_size requests within timeout."""
        batch = []
        deadline = time.time() + timeout_ms / 1000.0
        while len(batch) < max_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                req = self.get(block=True, timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break
        return batch

    def qsize(self) -> int:
        return self._queue.qsize()


class DynamicBatcher:
    """Dynamic batching: accumulates requests until max_batch or timeout."""

    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self._pending: List[Request] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def add(self, request: Request) -> Optional[List[Request]]:
        """Add request; returns batch to process if ready."""
        with self._lock:
            self._pending.append(request)
            should_flush = (
                len(self._pending) >= self.max_batch_size or
                (time.time() - self._last_flush) * 1000 >= self.timeout_ms
            )
            if should_flush:
                batch = self._pending.copy()
                self._pending.clear()
                self._last_flush = time.time()
                return batch
        return None

    def flush(self) -> List[Request]:
        with self._lock:
            batch = self._pending.copy()
            self._pending.clear()
            self._last_flush = time.time()
        return batch


class InferenceCache:
    """LRU cache for inference results keyed by input hash."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: Dict[str, Response] = {}
        self._access_order: deque = deque()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash_inputs(self, inputs: Dict[str, torch.Tensor]) -> str:
        parts = []
        for k in sorted(inputs.keys()):
            v = inputs[k]
            parts.append(f"{k}:{v.shape}:{v.sum().item():.6f}")
        return hashlib.md5(":".join(parts).encode()).hexdigest()

    def get(self, inputs: Dict[str, torch.Tensor]) -> Optional[Response]:
        key = self._hash_inputs(inputs)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            self.misses += 1
        return None

    def put(self, inputs: Dict[str, torch.Tensor], response: Response):
        key = self._hash_inputs(inputs)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.capacity:
                oldest = self._access_order.popleft()
                del self._cache[oldest]
            self._cache[key] = response
            self._access_order.append(key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ModelServer:
    """High-performance model server with batching, caching, and rate limiting."""

    def __init__(self, model: nn.Module, config: ServingConfig):
        self.model = model
        self.config = config
        self.model.eval()
        if config.use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(config.device)

        self.request_queue = RequestQueue(maxsize=10000)
        self.batcher = DynamicBatcher(config.max_batch_size, config.timeout_ms)
        self.cache = InferenceCache(config.cache_size)

        self._response_futures: Dict[str, Any] = {}
        self._workers: List[threading.Thread] = []
        self._running = False
        self._metrics = ServerMetrics()

        # Rate limiter
        if config.rate_limit_qps:
            self._rate_limiter = TokenBucketRateLimiter(config.rate_limit_qps)
        else:
            self._rate_limiter = None

    def start(self):
        self._running = True
        for i in range(self.config.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"server-worker-{i}")
            t.start()
            self._workers.append(t)

    def stop(self):
        self._running = False
        for t in self._workers:
            t.join(timeout=5.0)

    def _worker_loop(self):
        while self._running:
            try:
                requests = self.request_queue.get_batch(
                    self.config.max_batch_size, self.config.timeout_ms
                )
                if requests:
                    self._process_batch(requests)
                else:
                    time.sleep(0.001)
            except Exception as e:
                pass  # Log in production

    def _process_batch(self, requests: List[Request]):
        start = time.time()
        try:
            # Check cache for each request
            uncached = []
            results = {}
            for req in requests:
                cached = self.cache.get(req.inputs)
                if cached is not None:
                    results[req.request_id] = cached
                else:
                    uncached.append(req)

            if uncached:
                # Collate inputs
                keys = list(uncached[0].inputs.keys())
                batch_inputs = {}
                for k in keys:
                    batch_inputs[k] = torch.stack([r.inputs[k] for r in uncached]).to(self.config.device)

                # Inference
                with torch.no_grad():
                    if self.config.use_fp16:
                        with torch.autocast(device_type=self.config.device.split(":")[0]):
                            batch_out = self.model(**batch_inputs)
                    else:
                        batch_out = self.model(**batch_inputs)

                # Unbatch and cache
                latency_ms = (time.time() - start) * 1000 / len(uncached)
                for i, req in enumerate(uncached):
                    if isinstance(batch_out, torch.Tensor):
                        out_i = {"output": batch_out[i]}
                    elif isinstance(batch_out, dict):
                        out_i = {k: v[i] for k, v in batch_out.items()}
                    else:
                        out_i = {"output": batch_out[0][i]}
                    resp = Response(req.request_id, out_i, latency_ms)
                    results[req.request_id] = resp
                    self.cache.put(req.inputs, resp)

            total_latency = (time.time() - start) * 1000
            self._metrics.record_batch(len(requests), total_latency)

        except Exception as e:
            for req in requests:
                results[req.request_id] = Response(req.request_id, {}, error=str(e))

    def infer(self, inputs: Dict[str, torch.Tensor], priority: int = 0) -> str:
        """Submit inference request, returns request_id."""
        import uuid
        req_id = str(uuid.uuid4())[:8]
        req = Request(req_id, inputs, priority)
        self.request_queue.put(req)
        return req_id

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics.summary(),
            "cache_hit_rate": self.cache.hit_rate,
            "queue_depth": self.request_queue.qsize(),
        }


class TokenBucketRateLimiter:
    """Token bucket rate limiter for QPS control."""

    def __init__(self, rate_qps: float, burst: Optional[float] = None):
        self.rate = rate_qps
        self.burst = burst or rate_qps * 2
        self._tokens = self.burst
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, n_tokens: float = 1.0, block: bool = True) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens >= n_tokens:
                self._tokens -= n_tokens
                return True
            elif block:
                wait_time = (n_tokens - self._tokens) / self.rate
                time.sleep(wait_time)
                self._tokens = 0
                return True
            return False


class ServerMetrics:
    """Tracks server performance metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._total_requests = 0
        self._total_batches = 0
        self._start_time = time.time()

    def record_batch(self, batch_size: int, latency_ms: float):
        self._latencies.append(latency_ms)
        self._batch_sizes.append(batch_size)
        self._total_requests += batch_size
        self._total_batches += 1

    def summary(self) -> Dict[str, float]:
        uptime = time.time() - self._start_time
        lats = list(self._latencies)
        if not lats:
            return {"uptime_s": uptime, "total_requests": self._total_requests}
        return {
            "uptime_s": uptime,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "throughput_qps": self._total_requests / max(uptime, 1e-8),
            "avg_batch_size": statistics.mean(self._batch_sizes) if self._batch_sizes else 0.0,
            "p50_latency_ms": statistics.median(lats),
            "p95_latency_ms": sorted(lats)[int(0.95 * len(lats))],
            "p99_latency_ms": sorted(lats)[int(0.99 * len(lats))],
            "max_latency_ms": max(lats),
        }


class ModelVersionManager:
    """Manages multiple model versions with A/B testing and canary deployments."""

    def __init__(self):
        self._versions: Dict[str, nn.Module] = {}
        self._traffic_weights: Dict[str, float] = {}
        self._metrics: Dict[str, ServerMetrics] = {}
        self._active_version: str = ""

    def register(self, version_id: str, model: nn.Module, traffic_weight: float = 1.0):
        self._versions[version_id] = model
        self._traffic_weights[version_id] = traffic_weight
        self._metrics[version_id] = ServerMetrics()
        if not self._active_version:
            self._active_version = version_id
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self._traffic_weights.values())
        for k in self._traffic_weights:
            self._traffic_weights[k] /= total

    def route_request(self) -> str:
        """Select version based on traffic weights."""
        r = torch.rand(1).item()
        cumulative = 0.0
        for version_id, weight in self._traffic_weights.items():
            cumulative += weight
            if r <= cumulative:
                return version_id
        return self._active_version

    def get_model(self, version_id: Optional[str] = None) -> nn.Module:
        if version_id is None:
            version_id = self.route_request()
        return self._versions[version_id]

    def set_canary(self, canary_id: str, canary_fraction: float = 0.05):
        """Route canary_fraction of traffic to canary model."""
        for version_id in self._traffic_weights:
            if version_id == canary_id:
                self._traffic_weights[version_id] = canary_fraction
            else:
                self._traffic_weights[version_id] = (1 - canary_fraction) / max(
                    1, len(self._versions) - 1
                )
        self._normalize_weights()

    def promote_canary(self, canary_id: str):
        """Make canary the primary version."""
        self._active_version = canary_id
        for version_id in self._traffic_weights:
            self._traffic_weights[version_id] = 1.0 if version_id == canary_id else 0.0

    def version_comparison(self) -> Dict[str, Dict[str, float]]:
        return {vid: self._metrics[vid].summary() for vid in self._versions}


class GradientFreeShadowMode(nn.Module):
    """Shadow mode deployment: runs new model alongside production, logs discrepancies."""

    def __init__(self, production_model: nn.Module, shadow_model: nn.Module, log_dir: str = "./shadow_logs"):
        super().__init__()
        self.production = production_model
        self.shadow = shadow_model
        self.log_dir = log_dir
        self._discrepancies: List[float] = []
        os.makedirs(log_dir, exist_ok=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            prod_out = self.production(x, **kwargs)
            try:
                shadow_out = self.shadow(x, **kwargs)
                if isinstance(prod_out, torch.Tensor) and isinstance(shadow_out, torch.Tensor):
                    disc = (prod_out - shadow_out).abs().mean().item()
                    self._discrepancies.append(disc)
            except Exception:
                pass
        return prod_out

    def discrepancy_stats(self) -> Dict[str, float]:
        if not self._discrepancies:
            return {}
        return {
            "mean_discrepancy": statistics.mean(self._discrepancies),
            "max_discrepancy": max(self._discrepancies),
            "n_comparisons": len(self._discrepancies),
        }


class TorchServeAdapter:
    """Adapter to export Lumina models for TorchServe deployment."""

    def __init__(self, model: nn.Module, model_name: str, version: str = "1.0"):
        self.model = model
        self.model_name = model_name
        self.version = version

    def create_handler_script(self, output_path: str) -> str:
        lines = [
            chr(34)*3,
            "TorchServe handler for " + self.model_name + " v" + self.version,
            "Auto-generated by Lumina deployment module.",
            chr(34)*3,
            "import torch",
            "from ts.torch_handler.base_handler import BaseHandler",
            "",
            "",
            "class LuminaHandler(BaseHandler):",
            "    def initialize(self, context):",
            "        self.model = torch.jit.load(properties.get(\"model_dir\") + \"/model.pt\")",
            "    def preprocess(self, data): return torch.stack([torch.tensor(r.get(\"body\")) for r in data])",
            "    def inference(self, data): return self.model(data)",
            "    def postprocess(self, data): return data.tolist()",
        ]
        handler_content = chr(10).join(lines)
        with open(output_path, "w") as f:
            f.write(handler_content)
        return output_path

    def export_torchscript(self, example_input: torch.Tensor, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        traced = torch.jit.trace(self.model, example_input)
        path = os.path.join(output_dir, f"{self.model_name}.pt")
        traced.save(path)
        return path
'''

deploy_lines = append("deployment.py", DEPLOY_ADD)
print(f"deployment.py: {deploy_lines} lines")

# ─── rlhf.py additions ────────────────────────────────────────────────────────
RLHF_ADD = '''

# ============================================================
# Extended RLHF Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 16
    n_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95  # GAE lambda
    normalize_advantages: bool = True
    kl_target: float = 0.02
    kl_coef: float = 0.1
    adaptive_kl: bool = True


class RewardModel(nn.Module):
    """Reward model for RLHF: maps (prompt, response) -> scalar reward."""

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        pooling: str = "last_token",  # last_token | mean | cls
    ):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.base_model(input_ids)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.pooling == "last_token":
            if attention_mask is not None:
                last_pos = attention_mask.sum(-1) - 1
                pooled = hidden[torch.arange(hidden.shape[0]), last_pos]
            else:
                pooled = hidden[:, -1]
        elif self.pooling == "mean":
            if attention_mask is not None:
                pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            else:
                pooled = hidden.mean(1)
        else:  # cls
            pooled = hidden[:, 0]

        return self.value_head(pooled).squeeze(-1)


class PreferenceDataset:
    """Dataset of (prompt, chosen, rejected) preference pairs."""

    def __init__(self):
        self.data: List[Dict[str, torch.Tensor]] = []

    def add(self, prompt: torch.Tensor, chosen: torch.Tensor, rejected: torch.Tensor):
        self.data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class BradleyTerryLoss(nn.Module):
    """Bradley-Terry preference loss for reward model training."""

    def forward(self, reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
        """reward_chosen, reward_rejected: (B,) scalars"""
        # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
        return loss


class ListwiseLoss(nn.Module):
    """Listwise ranking loss using softmax over rewards."""

    def forward(self, rewards: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """rewards: (B, K), labels: (B, K) with 1 for best."""
        log_probs = F.log_softmax(rewards, dim=-1)
        loss = -(labels * log_probs).sum(-1).mean()
        return loss


class GeneralizedAdvantagEstimation:
    """Computes GAE (Schulman et al. 2016) returns and advantages."""

    @staticmethod
    def compute(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards: (T,), values: (T+1,), dones: (T,)
        Returns: returns (T,), advantages (T,)
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=rewards.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:T]
        return returns, advantages


class PPOTrainer:
    """PPO trainer for RLHF with KL penalty against reference policy."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        value_model: nn.Module,
        config: PPOConfig,
    ):
        self.policy = policy_model
        self.ref_policy = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.config = config

        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)
        for param in self.reward_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_model.parameters()),
            lr=config.learning_rate,
        )
        self.kl_coef = config.kl_coef
        self._step = 0

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward = reward_model(x, y) - kl_coef * KL(policy || ref)."""
        full_ids = torch.cat([input_ids, response_ids], dim=1)

        with torch.no_grad():
            rm_reward = self.reward_model(full_ids, attention_mask)

            # KL divergence: sum over response tokens
            policy_logits = self.policy(full_ids)
            ref_logits = self.ref_policy(full_ids)

            if isinstance(policy_logits, tuple):
                policy_logits = policy_logits[0]
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]

            # Only KL over response tokens
            T_prompt = input_ids.shape[1]
            policy_log_probs = F.log_softmax(policy_logits[:, T_prompt:-1], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, T_prompt:-1], dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - policy_log_probs)).sum(-1).sum(-1)

        return rm_reward - self.kl_coef * kl

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Single PPO update step."""
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        T_prompt = input_ids.shape[1]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.config.n_epochs):
            # Policy forward
            logits = self.policy(full_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            response_logits = logits[:, T_prompt - 1:-1]  # shift for next-token prediction
            log_probs = F.log_softmax(response_logits, dim=-1)
            # Gather token log probs
            new_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

            # PPO clipped surrogate
            ratio = (new_log_probs - old_log_probs).exp()
            adv = advantages
            if self.config.normalize_advantages:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value_model(full_ids[:, :T_prompt])
            if isinstance(values, tuple):
                values = values[0]
            if values.dim() > 1:
                values = values.squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(-1).mean()

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Adaptive KL
        with torch.no_grad():
            ref_logits = self.ref_policy(full_ids)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            p = F.softmax(logits[:, T_prompt - 1:-1], dim=-1)
            q = F.softmax(ref_logits[:, T_prompt - 1:-1], dim=-1)
            kl = (p * (p.log() - q.log())).sum(-1).mean().item()

        if self.config.adaptive_kl:
            if kl > 2 * self.config.kl_target:
                self.kl_coef *= 1.5
            elif kl < 0.5 * self.config.kl_target:
                self.kl_coef *= 0.75

        self._step += 1
        n = self.config.n_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "kl": kl,
            "kl_coef": self.kl_coef,
        }


class DirectPreferenceOptimization(nn.Module):
    """DPO (Rafailov et al. 2023): directly optimize preference without reward model."""

    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, beta: float = 0.1):
        super().__init__()
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta

        for param in self.ref.parameters():
            param.requires_grad_(False)

    def _log_prob(self, model: nn.Module, input_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probability of response given input."""
        full = torch.cat([input_ids, response_ids], dim=1)
        logits = model(full)
        if isinstance(logits, tuple):
            logits = logits[0]
        T = input_ids.shape[1]
        response_logits = logits[:, T - 1:-1]
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(-1)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss."""
        # Policy log probs
        pi_chosen = self._log_prob(self.policy, prompt_ids, chosen_ids)
        pi_rejected = self._log_prob(self.policy, prompt_ids, rejected_ids)

        # Reference log probs
        with torch.no_grad():
            ref_chosen = self._log_prob(self.ref, prompt_ids, chosen_ids)
            ref_rejected = self._log_prob(self.ref, prompt_ids, rejected_ids)

        # DPO objective
        logits = self.beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        loss = -F.logsigmoid(logits).mean()

        reward_chosen = (pi_chosen - ref_chosen).mean().item()
        reward_rejected = (pi_rejected - ref_rejected).mean().item()
        accuracy = (logits > 0).float().mean().item()

        return loss, {
            "loss": loss.item(),
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
            "reward_margin": reward_chosen - reward_rejected,
            "accuracy": accuracy,
        }


class IdentityPreferenceOptimization(nn.Module):
    """IPO (Azar et al. 2024): fixes DPO overfitting by using squared loss."""

    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, tau: float = 0.1):
        super().__init__()
        self.policy = policy_model
        self.ref = ref_model
        self.tau = tau

        for param in self.ref.parameters():
            param.requires_grad_(False)

    def _log_prob(self, model, input_ids, response_ids):
        full = torch.cat([input_ids, response_ids], dim=1)
        logits = model(full)
        if isinstance(logits, tuple):
            logits = logits[0]
        T = input_ids.shape[1]
        lp = F.log_softmax(logits[:, T - 1:-1], dim=-1)
        return lp.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

    def forward(self, prompt_ids, chosen_ids, rejected_ids):
        pi_w = self._log_prob(self.policy, prompt_ids, chosen_ids)
        pi_l = self._log_prob(self.policy, prompt_ids, rejected_ids)
        with torch.no_grad():
            ref_w = self._log_prob(self.ref, prompt_ids, chosen_ids)
            ref_l = self._log_prob(self.ref, prompt_ids, rejected_ids)

        h = (pi_w - ref_w) - (pi_l - ref_l)
        loss = ((h - 1 / (2 * self.tau)) ** 2).mean()
        return loss, {"loss": loss.item(), "h": h.mean().item()}


class RewardModelTrainer:
    """Trains reward model from human preference data."""

    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        loss_type: str = "bradley_terry",  # bradley_terry | listwise
    ):
        self.reward_model = reward_model
        self.loss_type = loss_type

        if loss_type == "bradley_terry":
            self.loss_fn = BradleyTerryLoss()
        else:
            self.loss_fn = ListwiseLoss()

        self.optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.step = 0
        self.losses: List[float] = []

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()

        full_chosen = torch.cat([prompt_ids, chosen_ids], dim=1)
        full_rejected = torch.cat([prompt_ids, rejected_ids], dim=1)

        r_chosen = self.reward_model(full_chosen)
        r_rejected = self.reward_model(full_rejected)

        loss = self.loss_fn(r_chosen, r_rejected)
        loss.backward()
        nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        self.losses.append(loss.item())
        return loss.item()

    def evaluate(self, dataset: PreferenceDataset, device: str = "cpu") -> Dict[str, float]:
        self.reward_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for item in dataset.data:
                full_c = torch.cat([item["prompt"], item["chosen"]], dim=0).unsqueeze(0).to(device)
                full_r = torch.cat([item["prompt"], item["rejected"]], dim=0).unsqueeze(0).to(device)
                r_c = self.reward_model(full_c)
                r_r = self.reward_model(full_r)
                correct += (r_c > r_r).item()
                total += 1
        self.reward_model.train()
        return {"preference_accuracy": correct / max(total, 1), "n_evaluated": total}


class ConstitutionalAIFilter(nn.Module):
    """Constitutional AI (Bai et al. 2022) critique-revision filter."""

    def __init__(self, critique_model: nn.Module, revision_model: nn.Module):
        super().__init__()
        self.critique = critique_model
        self.revision = revision_model
        self.principles: List[str] = [
            "Be helpful, harmless, and honest.",
            "Avoid harmful financial advice.",
            "Do not make false predictions about markets.",
            "Acknowledge uncertainty in financial forecasts.",
        ]

    def critique_response(self, prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Generate critique of response w.r.t. principles."""
        combined = torch.cat([prompt, response], dim=1)
        return self.critique(combined)

    def revise_response(
        self,
        prompt: torch.Tensor,
        response: torch.Tensor,
        critique: torch.Tensor,
    ) -> torch.Tensor:
        """Revise response based on critique."""
        combined = torch.cat([prompt, response, critique], dim=1)
        return self.revision(combined)

    def forward(self, prompt: torch.Tensor, response: torch.Tensor, n_revisions: int = 1) -> torch.Tensor:
        current = response
        for _ in range(n_revisions):
            critique = self.critique_response(prompt, current)
            current = self.revise_response(prompt, current, critique)
        return current
'''

rlhf_lines = append("rlhf.py", RLHF_ADD)

# Final count
import subprocess
result = subprocess.run(
    ["bash", "-c", "find /c/Users/Matthew/srfm-lab/aeternus/lumina -name '*.py' -o -name '*.yaml' | xargs wc -l 2>/dev/null | tail -1"],
    capture_output=True, text=True
)
print("GRAND TOTAL:", result.stdout.strip())
