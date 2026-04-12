"""Expand scaling.py with large additions."""
import os, sys

SCALING_PATH = os.path.join(os.path.dirname(__file__), "..", "lumina", "scaling.py")

CONTENT = '''

# =============================================================================
# SECTION: Neural Scaling Law Estimators
# =============================================================================

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List, Tuple, Dict


class PowerLawFitter:
    """Fit power-law scaling curves: L = a * N^b + c.

    Used to estimate how loss scales with model size or data size
    following Kaplan et al. (2020) and Hoffmann et al. (2022).
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.a = None
        self.b = None
        self.c = None
        self._fitted = False

    def fit(self, ns: np.ndarray, losses: np.ndarray) -> "PowerLawFitter":
        """Fit L = a * N^b + c to (N, L) data points using log-linear regression."""
        log_n = np.log(ns)
        log_l = np.log(losses)

        if self.fit_intercept:
            A = np.vstack([log_n, np.ones_like(log_n)]).T
            result = np.linalg.lstsq(A, log_l, rcond=None)
            self.b, log_a = result[0]
            self.a = np.exp(log_a)
            self.c = 0.0
        else:
            A = log_n.reshape(-1, 1)
            result = np.linalg.lstsq(A, log_l, rcond=None)
            self.b = result[0][0]
            self.a = 1.0
            self.c = 0.0

        self._fitted = True
        return self

    def predict(self, n: float) -> float:
        """Predict loss for a given N."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.a * (n ** self.b) + self.c

    def optimal_allocation(self, total_flops: float) -> Tuple[float, float]:
        """Estimate optimal (N_params, D_tokens) for a given FLOP budget.

        Uses Chinchilla scaling: N_opt ≈ sqrt(FLOPs / 6), D_opt ≈ sqrt(FLOPs * 6).
        """
        n_opt = math.sqrt(total_flops / 6.0)
        d_opt = total_flops / (6.0 * n_opt)
        return n_opt, d_opt

    def extrapolate_loss(self, current_n: float, target_n: float, current_loss: float) -> float:
        """Extrapolate expected loss at target_n given current (n, loss)."""
        if self.b is not None:
            return current_loss * (target_n / current_n) ** self.b
        # Fallback: assume exponent of -0.076 (Kaplan 2020)
        return current_loss * (target_n / current_n) ** (-0.076)


class ScalingLawPredictor:
    """Predict model performance at scale using fitted scaling laws.

    Combines:
    - Compute-optimal scaling (Chinchilla)
    - Data scaling exponents
    - Architecture-specific adjustments
    """

    def __init__(
        self,
        compute_exponent: float = -0.050,
        data_exponent: float = -0.095,
        param_exponent: float = -0.076,
        irreducible_loss: float = 1.69,
    ):
        self.compute_exponent = compute_exponent
        self.data_exponent = data_exponent
        self.param_exponent = param_exponent
        self.irreducible_loss = irreducible_loss

    def loss_from_params(self, n_params: float, n_tokens: float) -> float:
        """Predict cross-entropy loss from parameter count and training tokens."""
        param_term = (5.4e13 / n_params) ** (-self.param_exponent)
        data_term = (1.8e13 / n_tokens) ** (-self.data_exponent)
        return self.irreducible_loss + param_term + data_term

    def loss_from_flops(self, flops: float) -> float:
        """Predict loss from total training FLOPs."""
        n_opt, d_opt = PowerLawFitter().optimal_allocation(flops)
        return self.loss_from_params(n_opt, d_opt)

    def compute_budget_for_loss(self, target_loss: float) -> float:
        """Estimate compute (FLOPs) required to achieve target_loss."""
        from scipy.optimize import brentq
        try:
            def objective(log_flops):
                return self.loss_from_flops(math.exp(log_flops)) - target_loss
            log_flops = brentq(objective, 20, 40)
            return math.exp(log_flops)
        except Exception:
            return float("nan")

    def params_for_target_loss(self, target_loss: float, n_tokens: float) -> float:
        """Estimate parameter count needed for target_loss given token budget."""
        from scipy.optimize import brentq
        try:
            def objective(log_n):
                return self.loss_from_params(math.exp(log_n), n_tokens) - target_loss
            log_n = brentq(objective, 10, 30)
            return math.exp(log_n)
        except Exception:
            return float("nan")


# =============================================================================
# SECTION: Model Size Calculator
# =============================================================================

class TransformerSizeCalculator:
    """Calculate parameter counts and FLOPs for transformer architectures."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_ratio: float = 4.0,
        max_seq_len: int = 2048,
        tie_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_ratio = ffn_ratio
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.d_ff = int(d_model * ffn_ratio)
        self.d_head = d_model // n_heads

    def embedding_params(self) -> int:
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        return token_emb + pos_emb

    def attention_params_per_layer(self) -> int:
        qkv = 3 * self.d_model * self.d_model
        out_proj = self.d_model * self.d_model
        return qkv + out_proj

    def ffn_params_per_layer(self) -> int:
        fc1 = self.d_model * self.d_ff
        fc2 = self.d_ff * self.d_model
        bias1 = self.d_ff
        bias2 = self.d_model
        return fc1 + fc2 + bias1 + bias2

    def layernorm_params_per_layer(self) -> int:
        return 4 * self.d_model  # 2 layer norms, each with weight + bias

    def total_params(self) -> int:
        emb = self.embedding_params()
        if self.tie_embeddings:
            lm_head = 0
        else:
            lm_head = self.vocab_size * self.d_model

        per_layer = (
            self.attention_params_per_layer()
            + self.ffn_params_per_layer()
            + self.layernorm_params_per_layer()
        )
        return emb + self.n_layers * per_layer + lm_head

    def flops_per_token(self) -> int:
        """Estimate FLOPs per token for a forward pass (approximate)."""
        attn_flops = 4 * self.d_model * self.d_model  # QKV + out proj
        ffn_flops = 8 * self.d_model * self.d_ff
        return self.n_layers * (attn_flops + ffn_flops)

    def training_flops(self, n_tokens: int) -> int:
        """Total training FLOPs ≈ 6 * N_params * N_tokens (Chinchilla approximation)."""
        return 6 * self.total_params() * n_tokens

    def memory_footprint_gb(self, batch_size: int = 1, precision: str = "fp32") -> float:
        """Estimate GPU memory for activations + parameters + gradients."""
        bytes_per_param = 4 if precision == "fp32" else 2
        param_bytes = self.total_params() * bytes_per_param
        grad_bytes = param_bytes  # same size for gradients
        # Activations: rough estimate
        act_bytes = (
            batch_size
            * self.max_seq_len
            * self.d_model
            * self.n_layers
            * 2  # forward + backward
            * bytes_per_param
        )
        return (param_bytes + grad_bytes + act_bytes) / (1024 ** 3)

    def summary(self) -> dict:
        return {
            "total_params": self.total_params(),
            "total_params_M": self.total_params() / 1e6,
            "embedding_params": self.embedding_params(),
            "params_per_layer": (
                self.attention_params_per_layer()
                + self.ffn_params_per_layer()
                + self.layernorm_params_per_layer()
            ),
            "flops_per_token": self.flops_per_token(),
            "training_flops_100B": self.training_flops(100_000_000_000),
            "memory_fp32_gb": self.memory_footprint_gb(precision="fp32"),
            "memory_fp16_gb": self.memory_footprint_gb(precision="fp16"),
        }


# =============================================================================
# SECTION: Dynamic Architecture Scaling
# =============================================================================

class DepthScaler:
    """Scale model depth (number of layers) dynamically.

    Supports:
    - Progressive layer dropping (training regularization)
    - Layer freezing for staged fine-tuning
    - Layer-wise adaptive learning rates
    """

    def __init__(self, model: nn.Module, layer_attr: str = "layers"):
        self.model = model
        self.layer_attr = layer_attr
        self._frozen_layers = set()
        self._drop_rates = {}

    def _get_layers(self) -> nn.ModuleList:
        return getattr(self.model, self.layer_attr)

    def freeze_bottom_k(self, k: int):
        """Freeze the bottom k layers."""
        layers = self._get_layers()
        for i in range(min(k, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad = False
            self._frozen_layers.add(i)

    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer."""
        layers = self._get_layers()
        if layer_idx < len(layers):
            for p in layers[layer_idx].parameters():
                p.requires_grad = True
            self._frozen_layers.discard(layer_idx)

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for p in self.model.parameters():
            p.requires_grad = True
        self._frozen_layers.clear()

    def set_progressive_drop_rate(self, max_drop_rate: float = 0.2):
        """Set stochastic depth drop rates: deeper layers drop more often."""
        layers = self._get_layers()
        n = len(layers)
        for i, layer in enumerate(layers):
            rate = max_drop_rate * i / max(n - 1, 1)
            self._drop_rates[i] = rate
            if hasattr(layer, "drop_rate"):
                layer.drop_rate = rate

    def get_trainable_params(self) -> int:
        """Count currently trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)


class WidthScaler:
    """Scale model width (hidden dimensions) dynamically.

    Supports:
    - Width multipliers for each component
    - Attention head pruning
    - FFN intermediate dimension adjustment
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        width_multiplier: float = 1.0,
    ):
        self.d_model = int(d_model * width_multiplier)
        self.n_heads = n_heads
        self.d_ff = int(d_ff * width_multiplier)
        self.d_head = self.d_model // self.n_heads
        self.width_multiplier = width_multiplier

    def get_config(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "d_head": self.d_head,
            "width_multiplier": self.width_multiplier,
        }

    def scale_to(self, new_multiplier: float) -> "WidthScaler":
        """Return a new WidthScaler with a different multiplier."""
        base_d = int(self.d_model / self.width_multiplier)
        base_ff = int(self.d_ff / self.width_multiplier)
        return WidthScaler(base_d, self.n_heads, base_ff, new_multiplier)


# =============================================================================
# SECTION: Mixture of Depth (MoD)
# =============================================================================

class TokenRouterMoD(nn.Module):
    """Token routing for Mixture of Depth (MoD).

    Selects which tokens pass through a given layer vs. being skipped.
    Based on Raposo et al. (2024): each layer only processes the top-k tokens
    by routing weight; the rest are passed through via residual connection.

    This allows variable compute per token, similar to MoE but along depth.
    """

    def __init__(self, d_model: int, capacity_factor: float = 0.5):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> torch.Tensor:
        """Route tokens through layer: top-k processed, rest skip."""
        B, T, D = x.shape
        k = max(1, int(T * self.capacity_factor))

        scores = self.router(x).squeeze(-1)  # [B, T]
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)

        # Gather selected tokens
        expanded_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        selected = x.gather(1, expanded_idx)  # [B, k, D]

        # Process selected tokens through layer
        processed = layer(selected)

        # Scatter back
        out = x.clone()
        weights = torch.sigmoid(topk_vals).unsqueeze(-1)
        out.scatter_add_(1, expanded_idx, processed * weights - selected * weights)

        return out


class MixtureOfDepthTransformer(nn.Module):
    """Transformer with Mixture of Depth routing.

    Alternates between full layers (all tokens) and MoD layers (top-k tokens).
    Reduces total FLOPs while maintaining model capacity.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        capacity_factor: float = 0.5,
        mod_every_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.is_mod_layer = []

        for i in range(n_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.layers.append(layer)
            use_mod = (i % mod_every_n == 1)
            self.is_mod_layer.append(use_mod)
            if use_mod:
                self.routers.append(TokenRouterMoD(d_model, capacity_factor))
            else:
                self.routers.append(None)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        for layer, router, is_mod in zip(self.layers, self.routers, self.is_mod_layer):
            if is_mod and router is not None:
                x = router(x, layer)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def estimate_effective_flops(self, seq_len: int) -> float:
        """Estimate effective FLOPs considering MoD routing."""
        n_full = sum(1 for b in self.is_mod_layer if not b)
        n_mod = sum(1 for b in self.is_mod_layer if b)
        cap = self.routers[0].capacity_factor if n_mod > 0 else 1.0

        full_flop = n_full * seq_len
        mod_flop = n_mod * seq_len * cap
        return full_flop + mod_flop


# =============================================================================
# SECTION: Efficient Attention Approximations
# =============================================================================

class LinearAttentionKernel(nn.Module):
    """Linear attention using kernel feature maps (Katharopoulos 2020).

    Approximates softmax attention with O(n) complexity using:
    Attention(Q, K, V) ≈ phi(Q) * (phi(K)^T * V) / (phi(Q) * phi(K)^T * 1)

    where phi(x) = elu(x) + 1 is the feature map.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # Compute KV product: [B, H, Dh, Dh]
        KV = torch.einsum("bhnd,bhnm->bhdm", K, V)
        # Compute QKV: [B, H, T, Dh]
        QKV = torch.einsum("bhnd,bhdm->bhnm", Q, KV)
        # Normalizer
        K_sum = K.sum(dim=2)  # [B, H, Dh]
        Z = torch.einsum("bhnd,bhd->bhn", Q, K_sum).unsqueeze(-1).clamp(min=1e-6)

        out = (QKV / Z).transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.out_proj(out)


class PerformerAttention(nn.Module):
    """PERFORMER: random feature attention for O(n) complexity (Choromanski 2021).

    Uses random orthogonal features to approximate the softmax kernel:
    K(q, k) = E[phi(q) * phi(k)]

    where phi uses sin/cos random projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_random_features: int = 256,
        orthogonal_random: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_random_features = n_random_features
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Random projection matrix
        self._init_random_features(orthogonal_random)

    def _init_random_features(self, orthogonal: bool):
        m = self.n_random_features
        d = self.d_head
        if orthogonal:
            W = torch.zeros(m, d)
            for i in range(0, m, d):
                block = torch.randn(min(d, m - i), d)
                q, _ = torch.linalg.qr(block)
                W[i:i+min(d, m-i)] = q[:min(d, m-i)]
        else:
            W = torch.randn(m, d) / math.sqrt(d)
        self.register_buffer("random_features", W)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Random feature map: phi(x) = exp(-||x||^2/2) * [cos(Wx), sin(Wx)] / sqrt(m)."""
        norms_sq = (x ** 2).sum(dim=-1, keepdim=True)
        projections = x @ self.random_features.T  # [..., m]
        cos_proj = torch.cos(projections)
        sin_proj = torch.sin(projections)
        scale = torch.exp(-norms_sq / 2) / math.sqrt(self.n_random_features)
        return torch.cat([cos_proj, sin_proj], dim=-1) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        m = self.n_random_features * 2

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        Q_feat = self._phi(Q)  # [B, H, T, 2m]
        K_feat = self._phi(K)

        KV = torch.einsum("bhnd,bhnv->bhdv", K_feat, V)  # [B, H, 2m, Dh]
        QKV = torch.einsum("bhnd,bhdv->bhnv", Q_feat, KV)  # [B, H, T, Dh]

        K_sum = K_feat.sum(dim=2)
        Z = torch.einsum("bhnd,bhd->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-6)

        out = (QKV / Z).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# =============================================================================
# SECTION: Gradient Checkpointing Strategies
# =============================================================================

class SelectiveGradientCheckpointing:
    """Selectively apply gradient checkpointing to reduce memory.

    Strategies:
    - every_n: checkpoint every n-th layer
    - memory_budget: checkpoint until memory target is met
    - sensitivity: checkpoint layers with smallest gradient norms
    """

    def __init__(self, model: nn.Module, strategy: str = "every_n", n: int = 2):
        self.model = model
        self.strategy = strategy
        self.n = n
        self._checkpointed = []

    def apply_every_n(self, layer_attr: str = "layers"):
        """Apply checkpointing to every n-th layer."""
        from torch.utils.checkpoint import checkpoint

        layers = getattr(self.model, layer_attr, None)
        if layers is None:
            return

        for i, layer in enumerate(layers):
            if i % self.n == 0:
                original_forward = layer.forward

                def make_checkpointed(orig_fwd):
                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(orig_fwd, *args, use_reentrant=False, **kwargs)
                    return checkpointed_forward

                layer.forward = make_checkpointed(original_forward)
                self._checkpointed.append(i)

    def estimate_memory_savings(self, d_model: int, seq_len: int, n_layers: int) -> dict:
        """Estimate memory savings from gradient checkpointing."""
        bytes_per_activation = 4  # fp32
        full_activation_bytes = n_layers * seq_len * d_model * bytes_per_activation
        n_checkpointed = len(self._checkpointed)
        saved = n_checkpointed * seq_len * d_model * bytes_per_activation
        return {
            "full_activation_mb": full_activation_bytes / 1e6,
            "saved_mb": saved / 1e6,
            "remaining_mb": (full_activation_bytes - saved) / 1e6,
            "checkpointed_layers": self._checkpointed,
        }


# =============================================================================
# SECTION: Knowledge Distillation for Model Compression
# =============================================================================

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets.

    L = alpha * CE(student, hard_labels) + (1 - alpha) * KL(student, teacher) * T^2

    Following Hinton et al. (2015) and DistilBERT (Sanh 2019).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor = None,
    ) -> dict:
        import torch.nn.functional as F

        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)

        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

        if hard_labels is not None and self.alpha > 0:
            ce_loss = F.cross_entropy(student_logits, hard_labels)
            total = self.alpha * ce_loss + (1 - self.alpha) * distill_loss
        else:
            ce_loss = torch.tensor(0.0)
            total = distill_loss

        return {
            "total": total,
            "distillation": distill_loss,
            "classification": ce_loss,
        }


class LayerMatchingDistillation(nn.Module):
    """Intermediate layer matching distillation (PKD, TinyBERT).

    Aligns intermediate hidden states between teacher and student
    using MSE or cosine similarity losses.
    """

    def __init__(
        self,
        student_d_model: int,
        teacher_d_model: int,
        layer_pairs: List[Tuple[int, int]],
        loss_type: str = "mse",
    ):
        super().__init__()
        self.layer_pairs = layer_pairs
        self.loss_type = loss_type

        # Projection layers to match dimensions
        self.projections = nn.ModuleList([
            nn.Linear(student_d_model, teacher_d_model, bias=False)
            for _ in layer_pairs
        ])

    def forward(
        self,
        student_hiddens: List[torch.Tensor],
        teacher_hiddens: List[torch.Tensor],
    ) -> torch.Tensor:
        import torch.nn.functional as F

        total_loss = torch.tensor(0.0, device=student_hiddens[0].device)

        for (s_idx, t_idx), proj in zip(self.layer_pairs, self.projections):
            s_h = proj(student_hiddens[s_idx])
            t_h = teacher_hiddens[t_idx].detach()

            if self.loss_type == "mse":
                loss = F.mse_loss(s_h, t_h)
            elif self.loss_type == "cosine":
                loss = 1.0 - F.cosine_similarity(s_h, t_h, dim=-1).mean()
            elif self.loss_type == "huber":
                loss = F.huber_loss(s_h, t_h)
            else:
                loss = F.mse_loss(s_h, t_h)

            total_loss = total_loss + loss

        return total_loss / max(len(self.layer_pairs), 1)


class AttentionTransferDistillation(nn.Module):
    """Attention map transfer distillation (Zagoruyko & Komodakis 2017).

    Transfers attention patterns from teacher to student using
    sum-of-squares normalization.
    """

    def __init__(self, beta: float = 1000.0):
        super().__init__()
        self.beta = beta

    def _attention_map(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute sum-of-squares attention map: F(A) = ||A||_2 normalized."""
        return torch.nn.functional.normalize(attention.pow(2).sum(dim=1), dim=1)

    def forward(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
    ) -> torch.Tensor:
        import torch.nn.functional as F

        total = torch.tensor(0.0)
        for s_att, t_att in zip(student_attentions, teacher_attentions):
            s_map = self._attention_map(s_att)
            t_map = self._attention_map(t_att).detach()
            total = total + F.mse_loss(s_map, t_map) * self.beta

        return total / max(len(student_attentions), 1)


# =============================================================================
# SECTION: Model Pruning
# =============================================================================

class MagnitudePruner:
    """Prune weights by magnitude (unstructured pruning).

    Creates binary masks for each parameter where small-magnitude
    weights are zeroed out.
    """

    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self._masks = {}

    def compute_masks(self, global_threshold: bool = True):
        """Compute pruning masks based on weight magnitudes."""
        if global_threshold:
            all_weights = torch.cat([
                p.data.abs().flatten()
                for p in self.model.parameters()
                if p.requires_grad and p.ndim >= 2
            ])
            threshold = torch.quantile(all_weights, self.sparsity)

            for name, p in self.model.named_parameters():
                if p.requires_grad and p.ndim >= 2:
                    self._masks[name] = (p.data.abs() > threshold).float()
        else:
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.ndim >= 2:
                    threshold = torch.quantile(p.data.abs(), self.sparsity)
                    self._masks[name] = (p.data.abs() > threshold).float()

    def apply_masks(self):
        """Zero out pruned weights using stored masks."""
        for name, p in self.model.named_parameters():
            if name in self._masks:
                p.data.mul_(self._masks[name])

    def sparsity_stats(self) -> dict:
        """Report actual sparsity per parameter."""
        stats = {}
        for name, mask in self._masks.items():
            total = mask.numel()
            zeros = (mask == 0).sum().item()
            stats[name] = {"sparsity": zeros / total, "total": total, "pruned": zeros}
        return stats

    def global_sparsity(self) -> float:
        """Overall fraction of zeroed weights."""
        total_zeros = sum((m == 0).sum().item() for m in self._masks.values())
        total_params = sum(m.numel() for m in self._masks.values())
        return total_zeros / max(total_params, 1)


class StructuredPruner:
    """Structured pruning: remove entire attention heads or FFN neurons.

    More hardware-friendly than unstructured pruning since it directly
    reduces matrix dimensions.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._head_importance = {}
        self._neuron_importance = {}

    def compute_head_importance(self, dataloader, loss_fn, n_batches: int = 10) -> dict:
        """Estimate attention head importance via gradient norms."""
        importance = {}

        for name, module in self.model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "n_heads"):
                importance[name] = torch.zeros(module.n_heads)

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            self.model.zero_grad()
            out = self.model(**batch)
            loss = loss_fn(out)
            loss.backward()

            for name, module in self.model.named_modules():
                if name in importance and hasattr(module, "q_proj"):
                    grad = module.q_proj.weight.grad
                    if grad is not None:
                        n_heads = module.n_heads
                        d_head = grad.shape[0] // n_heads
                        head_grads = grad.view(n_heads, d_head, -1)
                        importance[name] += head_grads.abs().mean(dim=(1, 2)).detach()

        self._head_importance = importance
        return importance

    def prune_heads(self, heads_to_prune: Dict[str, List[int]]):
        """Prune specific attention heads from modules."""
        for name, head_indices in heads_to_prune.items():
            parts = name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)

            if hasattr(module, "prune_heads"):
                module.prune_heads(head_indices)


# =============================================================================
# SECTION: Adaptive Compute
# =============================================================================

class AdaptiveComputeWrapper(nn.Module):
    """Adaptive computation time (ACT) wrapper for variable-depth processing.

    Based on Graves (2016): each token decides how many steps of computation
    it needs via a halting probability.
    """

    def __init__(
        self,
        layer: nn.Module,
        d_model: int,
        max_steps: int = 10,
        halt_threshold: float = 0.99,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.layer = layer
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.epsilon = epsilon
        self.halting_unit = nn.Linear(d_model, 1)
        self.ponder_cost_weight = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run layer with adaptive computation.

        Returns:
            (output, ponder_cost) where ponder_cost is the mean steps taken.
        """
        B, T, D = x.shape
        halting_probs = torch.zeros(B, T, device=x.device)
        remainders = torch.zeros(B, T, device=x.device)
        n_updates = torch.zeros(B, T, device=x.device)
        cumulative = torch.zeros(B, T, 1, device=x.device)
        output = torch.zeros_like(x)
        state = x.clone()

        for step in range(self.max_steps):
            p = torch.sigmoid(self.halting_unit(state).squeeze(-1))

            still_running = (halting_probs < self.halt_threshold).float()

            new_halted = ((halting_probs + p * still_running) >= self.halt_threshold).float() * still_running
            remainder = (1.0 - halting_probs) * new_halted

            halting_probs = halting_probs + p * still_running
            remainders = remainders + remainder
            n_updates = n_updates + still_running

            update_weights = (p * still_running + remainder).unsqueeze(-1)

            state_new = self.layer(state)
            output = output + update_weights * state_new
            state = state_new

            if (still_running.sum() == 0):
                break

        ponder_cost = (n_updates + remainders).mean()
        return output, ponder_cost * self.ponder_cost_weight


# =============================================================================
# SECTION: Efficient Training Utilities
# =============================================================================

class GradientNormMonitor:
    """Monitor and clip gradient norms during training."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self._history = []

    def clip_and_record(self, parameters) -> float:
        """Clip gradients and record the norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        self._history.append(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
        return total_norm

    def recent_stats(self, window: int = 100) -> dict:
        """Statistics over recent gradient norms."""
        recent = self._history[-window:]
        if not recent:
            return {}
        import statistics
        return {
            "mean": statistics.mean(recent),
            "max": max(recent),
            "min": min(recent),
            "clipped_fraction": sum(1 for n in recent if n > self.max_norm) / len(recent),
        }


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = last_step + 1

    def step(self):
        """Update learning rate for current step."""
        s = self._step
        if s < self.warmup_steps:
            factor = s / max(self.warmup_steps, 1)
        else:
            progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        for lr, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor

        self._step += 1
        return factor

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class CyclicLRScheduler:
    """Cyclic LR with triangular or exp range policy (Smith 2017)."""

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int = 2000,
        mode: str = "triangular",
        gamma: float = 0.999,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self._cycle = 0
        self._step = 0

    def step(self):
        cycle = math.floor(1 + self._step / (2 * self.step_size))
        x = abs(self._step / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale * (self.gamma ** self._step)
        else:
            lr = self.base_lr

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self._step += 1
        return lr


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization (Foret et al. 2021).

    SAM perturbs weights toward higher loss regions, then optimizes
    the perturbed loss to find flat minima. Two-step update:
    1. Gradient step to find perturbation delta
    2. Unperturb and take gradient step at original weights
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute and apply perturbation."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Remove perturbation and take gradient step."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive") else torch.ones_like(p)) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two-step: first_step + second_step")


# =============================================================================
# SECTION: Mixed Precision Training Utilities
# =============================================================================

class DynamicLossScaler:
    """Dynamic loss scaling for mixed precision training.

    Automatically adjusts loss scale based on gradient overflow detection.
    Equivalent to GradScaler but with more control over scaling schedule.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2 ** 24,
        min_scale: float = 1.0,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.min_scale = min_scale
        self._steps_since_overflow = 0
        self._total_overflows = 0
        self._total_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def check_overflow(self, parameters) -> bool:
        """Check for inf/nan gradients."""
        for p in parameters:
            if p.grad is not None:
                if torch.any(torch.isinf(p.grad)) or torch.any(torch.isnan(p.grad)):
                    return True
        return False

    def update(self, overflow: bool):
        """Update scale based on overflow status."""
        self._total_steps += 1
        if overflow:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._steps_since_overflow = 0
            self._total_overflows += 1
        else:
            self._steps_since_overflow += 1
            if self._steps_since_overflow >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._steps_since_overflow = 0

    def state_dict(self) -> dict:
        return {
            "scale": self.scale,
            "steps_since_overflow": self._steps_since_overflow,
            "total_overflows": self._total_overflows,
        }

    def load_state_dict(self, state: dict):
        self.scale = state["scale"]
        self._steps_since_overflow = state.get("steps_since_overflow", 0)
        self._total_overflows = state.get("total_overflows", 0)


# =============================================================================
# SECTION: Scaling Registry
# =============================================================================

_SCALING_REGISTRY = {}


def register_scaling_strategy(name: str):
    def decorator(cls):
        _SCALING_REGISTRY[name] = cls
        return cls
    return decorator


def get_scaling_strategy(name: str):
    if name not in _SCALING_REGISTRY:
        raise KeyError(f"Scaling strategy '{name}' not found. Available: {list(_SCALING_REGISTRY.keys())}")
    return _SCALING_REGISTRY[name]


@register_scaling_strategy("depth")
class DepthScalingStrategy:
    """Scale model by increasing number of layers."""

    def __init__(self, base_n_layers: int = 6, scale_factor: float = 2.0):
        self.base_n_layers = base_n_layers
        self.scale_factor = scale_factor

    def get_config(self, target_scale: float) -> dict:
        return {"n_layers": int(self.base_n_layers * target_scale)}


@register_scaling_strategy("width")
class WidthScalingStrategy:
    """Scale model by increasing hidden dimensions."""

    def __init__(self, base_d_model: int = 256, scale_factor: float = 2.0):
        self.base_d_model = base_d_model
        self.scale_factor = scale_factor

    def get_config(self, target_scale: float) -> dict:
        d = int(self.base_d_model * target_scale)
        return {"d_model": d, "d_ff": d * 4}


@register_scaling_strategy("balanced")
class BalancedScalingStrategy:
    """Balanced depth/width scaling following compound scaling (EfficientNet style)."""

    def __init__(self, depth_coeff: float = 1.2, width_coeff: float = 1.1):
        self.depth_coeff = depth_coeff
        self.width_coeff = width_coeff

    def get_config(self, base_config: dict, target_scale: float) -> dict:
        cfg = dict(base_config)
        cfg["n_layers"] = int(cfg.get("n_layers", 6) * (self.depth_coeff ** target_scale))
        cfg["d_model"] = int(cfg.get("d_model", 256) * (self.width_coeff ** target_scale))
        cfg["d_ff"] = cfg["d_model"] * 4
        return cfg
'''

with open(SCALING_PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess, sys
result = subprocess.run(
    [sys.executable, "-c",
     f"lines = open(r'{SCALING_PATH}').readlines(); print(len(lines))"],
    capture_output=True, text=True
)
print(result.stdout.strip(), SCALING_PATH)
