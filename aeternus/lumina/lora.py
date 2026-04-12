

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
