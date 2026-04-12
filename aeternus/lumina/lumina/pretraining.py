"""
lumina/pretraining.py

Pre-training objectives for Lumina Financial Foundation Model:

  - MaskedReturnModeling (MRM)    : predict masked OHLCV patches
  - NextPatchPrediction (NPP)     : causal next-patch prediction
  - ContrastiveLoss               : InfoNCE / NT-Xent across assets
  - RegimePrediction              : auxiliary market regime classification
  - MultiTaskPretrainingLoss      : combines all objectives with learned weights
  - PretrainingSchedule           : curriculum with objective annealing
  - PretrainingTrainer            : training loop with logging
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PretrainingConfig:
    # Masking
    mask_ratio:            float = 0.15   # fraction of patches to mask
    mask_strategy:         str   = "random"  # "random" | "block" | "temporal"
    block_mask_min:        int   = 4
    block_mask_max:        int   = 16

    # Objectives
    use_mrm:               bool  = True   # masked return modeling
    use_npp:               bool  = True   # next-patch prediction
    use_contrastive:       bool  = True   # contrastive across assets
    use_regime:            bool  = False  # market regime prediction
    use_volatility:        bool  = True   # volatility prediction aux task

    # Loss weights
    mrm_weight:            float = 1.0
    npp_weight:            float = 1.0
    contrastive_weight:    float = 0.5
    regime_weight:         float = 0.3
    volatility_weight:     float = 0.3

    # Contrastive
    temperature:           float = 0.07
    n_negatives:           int   = 64
    contrastive_mode:      str   = "in_batch"  # "in_batch" | "queue"
    queue_size:            int   = 65536

    # Curriculum
    warmup_steps:          int   = 10000
    curriculum_steps:      int   = 50000

    # Training
    max_steps:             int   = 500000
    log_every:             int   = 100
    eval_every:            int   = 5000
    save_every:            int   = 10000
    grad_clip:             float = 1.0
    lr:                    float = 1e-4
    weight_decay:          float = 0.01
    betas:                 Tuple[float, float] = (0.9, 0.95)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

class PatchMaskGenerator(nn.Module):
    """Generates patch masks for masked pre-training."""

    def __init__(self, cfg: PretrainingConfig):
        super().__init__()
        self.cfg = cfg

    def random_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Random i.i.d. mask. True = masked."""
        noise = torch.rand(B, N, device=device)
        return noise < self.cfg.mask_ratio

    def block_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Block masking: mask contiguous spans of patches."""
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            n_masked = 0
            target   = int(N * self.cfg.mask_ratio)
            while n_masked < target:
                span_len = torch.randint(
                    self.cfg.block_mask_min,
                    self.cfg.block_mask_max + 1,
                    (1,)
                ).item()
                start = torch.randint(0, max(1, N - span_len), (1,)).item()
                end   = min(N, start + span_len)
                mask[b, start:end] = True
                n_masked += (end - start)
        return mask

    def temporal_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Temporal masking: mask a contiguous suffix (harder)."""
        mask     = torch.zeros(B, N, dtype=torch.bool, device=device)
        n_mask   = int(N * self.cfg.mask_ratio)
        # Mask the last n_mask patches
        mask[:, N - n_mask:] = True
        return mask

    def forward(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        strategy = self.cfg.mask_strategy
        if strategy == "random":
            return self.random_mask(B, N, device)
        elif strategy == "block":
            return self.block_mask(B, N, device)
        elif strategy == "temporal":
            return self.temporal_mask(B, N, device)
        else:
            return self.random_mask(B, N, device)


# ---------------------------------------------------------------------------
# Masked Return Modeling (MRM)
# ---------------------------------------------------------------------------

class MaskedReturnModelingHead(nn.Module):
    """
    Prediction head for Masked Return Modeling.
    Predicts the original (normalized) patch values at masked positions.

    This is analogous to MLM in BERT but for financial time series:
    the model must reconstruct masked OHLCV patches.
    """

    def __init__(
        self,
        d_model:     int,
        patch_size:  int,
        n_channels:  int = 5,
        hidden_dim:  Optional[int] = None,
    ):
        super().__init__()
        hidden_dim   = hidden_dim or d_model
        out_dim      = patch_size * n_channels

        self.proj    = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.patch_size = patch_size
        self.n_channels = n_channels

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N, D) → predictions (B, N, patch_size * n_channels)."""
        return self.proj(hidden)


class MaskedReturnModelingLoss(nn.Module):
    """
    Compute MRM loss: MSE between predicted and true patch values at masked positions.

    Implements multiple loss variants:
      - MSE (default)
      - Huber (robust to outliers)
      - Log-cosh (smooth Huber)
      - RMSE
    """

    def __init__(
        self,
        loss_type:   str   = "mse",
        huber_delta: float = 1.0,
        normalize:   bool  = True,
    ):
        super().__init__()
        self.loss_type   = loss_type
        self.huber_delta = huber_delta
        self.normalize   = normalize

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        elif self.loss_type == "log_cosh":
            diff = pred - target
            return torch.log(torch.cosh(diff + 1e-12))
        elif self.loss_type == "mae":
            return (pred - target).abs()
        else:
            return F.mse_loss(pred, target, reduction="none")

    def forward(
        self,
        pred:   torch.Tensor,   # (B, N, out_dim) predictions
        target: torch.Tensor,   # (B, N, out_dim) true values
        mask:   torch.Tensor,   # (B, N) True = masked (predict these)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns: (loss scalar, metrics dict)
        """
        # Expand mask to match output dim
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)   # (B, N, out_dim)

        element_loss = self._compute_loss(pred, target)  # (B, N, out_dim)

        # Only compute loss at masked positions
        masked_loss = (element_loss * mask_expanded.float()).sum()
        n_masked    = mask_expanded.float().sum().clamp(min=1)
        loss        = masked_loss / n_masked

        # Metrics
        with torch.no_grad():
            mae = (pred - target).abs()
            mae_masked = (mae * mask_expanded.float()).sum() / n_masked
            # Correlation at masked positions
            p_flat = pred[mask_expanded].detach()
            t_flat = target[mask_expanded].detach()
            if len(p_flat) > 1:
                corr_val = torch.corrcoef(torch.stack([p_flat, t_flat]))[0, 1]
            else:
                corr_val = torch.tensor(0.0)

        metrics = {
            "mrm_loss": loss.item(),
            "mrm_mae":  mae_masked.item(),
            "mrm_corr": corr_val.item() if not corr_val.isnan() else 0.0,
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Next-Patch Prediction (NPP)
# ---------------------------------------------------------------------------

class NextPatchPredictionHead(nn.Module):
    """
    Autoregressive next-patch prediction head.
    Given a sequence of patches, predict the next patch.
    """

    def __init__(
        self,
        d_model:     int,
        patch_size:  int,
        n_channels:  int = 5,
    ):
        super().__init__()
        out_dim  = patch_size * n_channels
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim),
        )
        self.patch_size = patch_size
        self.n_channels = n_channels

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N, D) → predictions (B, N, out_dim)."""
        return self.proj(hidden)


class NextPatchPredictionLoss(nn.Module):
    """
    Compute NPP loss: predict patch (t+1) from representation at position t.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred:   torch.Tensor,   # (B, N, out_dim)
        target: torch.Tensor,   # (B, N, out_dim) — shifted by 1
        mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        pred at position t predicts target at position t+1.
        """
        # Shift: pred[:, :-1] predicts target[:, 1:]
        pred_shifted   = pred[:, :-1, :]
        target_shifted = target[:, 1:, :]

        if mask is not None:
            # Don't compute loss at padding positions
            valid_mask = ~mask[:, 1:]
        else:
            valid_mask = torch.ones(pred_shifted.shape[:2], dtype=torch.bool, device=pred.device)

        if self.loss_type == "mse":
            elem_loss = F.mse_loss(pred_shifted, target_shifted, reduction="none")
        else:
            elem_loss = F.huber_loss(pred_shifted, target_shifted, reduction="none")

        valid_expanded = valid_mask.unsqueeze(-1).expand_as(elem_loss).float()
        loss = (elem_loss * valid_expanded).sum() / (valid_expanded.sum().clamp(1))

        with torch.no_grad():
            mae = (pred_shifted - target_shifted).abs()
            mae_val = (mae * valid_expanded).sum() / valid_expanded.sum().clamp(1)

        return loss, {"npp_loss": loss.item(), "npp_mae": mae_val.item()}


# ---------------------------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss (van den Oord et al. 2018).

    Creates positive pairs from different time windows of the same asset,
    and uses in-batch negatives from other assets.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction   = reduction

    def forward(
        self,
        z1: torch.Tensor,   # (B, D) embeddings from view 1
        z2: torch.Tensor,   # (B, D) embeddings from view 2
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Symmetric InfoNCE loss over the batch.
        Positive pair: (z1[i], z2[i])
        Negatives: all z2[j] for j ≠ i (in-batch)
        """
        B, D = z1.shape

        # Normalize
        z1_n = F.normalize(z1, dim=-1)
        z2_n = F.normalize(z2, dim=-1)

        # Similarity matrix: (B, B)
        sim = torch.matmul(z1_n, z2_n.T) / self.temperature

        # Labels: diagonal is the positive pair
        labels = torch.arange(B, device=z1.device)

        # Cross-entropy loss in both directions
        loss_12 = F.cross_entropy(sim,   labels)
        loss_21 = F.cross_entropy(sim.T, labels)
        loss    = (loss_12 + loss_21) / 2

        # Accuracy (how often is the positive pair the top-1 match?)
        with torch.no_grad():
            acc_12 = (sim.argmax(1) == labels).float().mean()
            acc_21 = (sim.T.argmax(1) == labels).float().mean()
            acc    = (acc_12 + acc_21) / 2

        metrics = {
            "contrastive_loss": loss.item(),
            "contrastive_acc":  acc.item(),
        }
        return loss, metrics


class CrossAssetContrastiveLoss(nn.Module):
    """
    Contrastive loss between correlated asset pairs.
    Encourages assets with similar market regimes to have similar representations.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.info_nce = InfoNCELoss(temperature)

    def forward(
        self,
        embeddings:   torch.Tensor,   # (B, A, D) — B batches, A assets, D features
        correlations: Optional[torch.Tensor] = None,  # (A, A) correlation matrix
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        For each batch item, treat assets with high correlation as positive pairs.
        """
        B, A, D = embeddings.shape

        if correlations is None:
            # Fall back to treating all assets as positive pairs within a window
            # Use the first and last asset as a pseudo-pair
            z1 = embeddings[:, 0, :]
            z2 = embeddings[:, -1, :]
        else:
            # Find the most correlated pair of assets
            corr_no_diag = correlations.clone()
            corr_no_diag.fill_diagonal_(-1)
            best_pair    = corr_no_diag.argmax()
            a1 = best_pair // A
            a2 = best_pair % A
            z1 = embeddings[:, a1, :]
            z2 = embeddings[:, a2, :]

        return self.info_nce(z1, z2)


class MomentumContrastiveQueue(nn.Module):
    """
    MoCo-style momentum contrastive queue for large negative sets.
    Maintains a FIFO queue of negative embeddings.
    """

    def __init__(self, d_model: int, queue_size: int = 65536, temperature: float = 0.07, momentum: float = 0.999):
        super().__init__()
        self.d_model     = d_model
        self.queue_size  = queue_size
        self.temperature = temperature
        self.momentum    = momentum

        # Initialize queue with normalized random vectors
        self.register_buffer("queue",    F.normalize(torch.randn(d_model, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update the queue with new keys. keys: (B, D)."""
        B    = keys.shape[0]
        ptr  = int(self.queue_ptr)
        # Circular buffer
        if ptr + B <= self.queue_size:
            self.queue[:, ptr:ptr + B] = keys.T
        else:
            # Wrap around
            part1 = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:part1].T
            self.queue[:, :B - part1] = keys[part1:].T
        self.queue_ptr[0] = (ptr + B) % self.queue_size

    def forward(
        self,
        q: torch.Tensor,   # (B, D) online encoder queries
        k: torch.Tensor,   # (B, D) momentum encoder keys
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, D = q.shape
        q_n  = F.normalize(q, dim=-1)
        k_n  = F.normalize(k, dim=-1)

        # Positive logits: (B, 1)
        l_pos = torch.bmm(q_n.unsqueeze(1), k_n.unsqueeze(2)).squeeze(2) / self.temperature

        # Negative logits: (B, queue_size)
        l_neg = torch.mm(q_n, self.queue.clone().detach()) / self.temperature

        logits = torch.cat([l_pos, l_neg], dim=1)   # (B, 1 + queue_size)
        labels = torch.zeros(B, dtype=torch.long, device=q.device)  # positives at index 0

        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean()

        self._dequeue_and_enqueue(k_n)

        return loss, {"moco_loss": loss.item(), "moco_acc": acc.item()}


# ---------------------------------------------------------------------------
# Market Regime Prediction
# ---------------------------------------------------------------------------

class RegimePredictionHead(nn.Module):
    """
    Auxiliary head that predicts market regime labels.

    Regimes:
      0: Bull market (trending up)
      1: Bear market (trending down)
      2: Sideways / ranging
      3: High volatility
    """

    def __init__(self, d_model: int, n_regimes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_regimes = n_regimes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_regimes),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """cls_emb: (B, D) → (B, n_regimes) logits."""
        return self.head(cls_emb)


def infer_regime_labels(
    returns:     torch.Tensor,   # (B, T) return series
    vol_window:  int = 20,
    up_threshold: float = 0.02,
    down_threshold: float = -0.02,
) -> torch.Tensor:
    """
    Auto-generate regime labels from return series.
    Returns (B,) integer labels.
    """
    B, T = returns.shape

    # Cumulative return over window
    cum_ret  = returns[:, -vol_window:].sum(1)    # (B,)
    rolling_vol = returns[:, -vol_window:].std(1)  # (B,)

    labels = torch.zeros(B, dtype=torch.long, device=returns.device)
    labels[cum_ret > up_threshold]   = 0   # Bull
    labels[cum_ret < down_threshold] = 1   # Bear
    high_vol_mask = rolling_vol > returns.std(1) * 1.5
    labels[high_vol_mask] = 3            # High vol
    # Remaining = sideways
    neutral = (labels == 0) & (cum_ret.abs() <= up_threshold)
    labels[neutral] = 2

    return labels


# ---------------------------------------------------------------------------
# Volatility Prediction Auxiliary Task
# ---------------------------------------------------------------------------

class VolatilityPredictionHead(nn.Module):
    """Predicts next-period realized volatility."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),   # ensure positive vol
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.head(cls_emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Multi-Task Loss
# ---------------------------------------------------------------------------

class AutomaticLossWeighter(nn.Module):
    """
    Learn task loss weights automatically (Kendall et al. 2018 uncertainty weighting).
    Minimizes sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
        assert len(losses) == self.log_vars.shape[0]
        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total     = total + precision * loss + self.log_vars[i]
            weights.append(precision.item())
        return total, weights


class MultiTaskPretrainingLoss(nn.Module):
    """
    Combines multiple pre-training objectives with configurable weights.

    Objectives:
      - MRM: Masked Return Modeling
      - NPP: Next Patch Prediction
      - Contrastive: InfoNCE across assets
      - Regime: Market regime prediction
      - Volatility: Volatility forecasting

    Supports both fixed and learned (automatic) loss weighting.
    """

    def __init__(
        self,
        cfg:            PretrainingConfig,
        patch_size:     int = 16,
        n_channels:     int = 5,
        d_model:        int = 512,
        n_regimes:      int = 4,
        auto_weight:    bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        # Heads
        if cfg.use_mrm:
            self.mrm_head = MaskedReturnModelingHead(d_model, patch_size, n_channels)
            self.mrm_loss = MaskedReturnModelingLoss()

        if cfg.use_npp:
            self.npp_head = NextPatchPredictionHead(d_model, patch_size, n_channels)
            self.npp_loss = NextPatchPredictionLoss()

        if cfg.use_contrastive:
            self.contrastive_loss = InfoNCELoss(cfg.temperature)

        if cfg.use_regime:
            self.regime_head = RegimePredictionHead(d_model, n_regimes)

        if cfg.use_volatility:
            self.vol_head = VolatilityPredictionHead(d_model)

        # Loss weighting
        self.auto_weight = auto_weight
        if auto_weight:
            n_tasks = sum([cfg.use_mrm, cfg.use_npp, cfg.use_contrastive,
                           cfg.use_regime, cfg.use_volatility])
            self.weighter = AutomaticLossWeighter(n_tasks)

    def forward(
        self,
        hidden:        torch.Tensor,           # (B, N, D) from transformer
        cls_emb:       torch.Tensor,            # (B, D)
        patch_targets: torch.Tensor,            # (B, N, patch_size * n_channels) original patches
        mask:          torch.Tensor,            # (B, N) True = masked
        contrastive_z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (z1, z2)
        regime_labels: Optional[torch.Tensor]  = None,
        vol_targets:   Optional[torch.Tensor]  = None,
        padding_mask:  Optional[torch.Tensor]  = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns: (total_loss, metrics_dict)
        """
        cfg      = self.cfg
        losses   = []
        metrics  = {}
        task_losses = []

        # MRM
        if cfg.use_mrm and hasattr(self, "mrm_head"):
            preds          = self.mrm_head(hidden)
            mrm_l, mrm_m   = self.mrm_loss(preds, patch_targets, mask)
            losses.append(cfg.mrm_weight * mrm_l)
            task_losses.append(mrm_l)
            metrics.update(mrm_m)

        # NPP
        if cfg.use_npp and hasattr(self, "npp_head"):
            npp_preds     = self.npp_head(hidden)
            npp_l, npp_m  = self.npp_loss(npp_preds, patch_targets, padding_mask)
            losses.append(cfg.npp_weight * npp_l)
            task_losses.append(npp_l)
            metrics.update(npp_m)

        # Contrastive
        if cfg.use_contrastive and contrastive_z is not None and hasattr(self, "contrastive_loss"):
            z1, z2          = contrastive_z
            cont_l, cont_m  = self.contrastive_loss(z1, z2)
            losses.append(cfg.contrastive_weight * cont_l)
            task_losses.append(cont_l)
            metrics.update(cont_m)

        # Regime
        if cfg.use_regime and regime_labels is not None and hasattr(self, "regime_head"):
            regime_logits = self.regime_head(cls_emb)
            reg_l         = F.cross_entropy(regime_logits, regime_labels)
            with torch.no_grad():
                reg_acc = (regime_logits.argmax(1) == regime_labels).float().mean()
            losses.append(cfg.regime_weight * reg_l)
            task_losses.append(reg_l)
            metrics["regime_loss"] = reg_l.item()
            metrics["regime_acc"]  = reg_acc.item()

        # Volatility
        if cfg.use_volatility and vol_targets is not None and hasattr(self, "vol_head"):
            vol_pred  = self.vol_head(cls_emb)
            vol_l     = F.mse_loss(vol_pred, vol_targets)
            losses.append(cfg.volatility_weight * vol_l)
            task_losses.append(vol_l)
            metrics["vol_loss"] = vol_l.item()

        if not losses:
            return torch.tensor(0.0), metrics

        if self.auto_weight and hasattr(self, "weighter") and task_losses:
            total_loss, weights = self.weighter(task_losses)
            metrics["task_weights"] = weights
        else:
            total_loss = sum(losses)

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Pretraining Schedule
# ---------------------------------------------------------------------------

class PretrainingSchedule:
    """
    Curriculum learning schedule for pre-training.
    Gradually introduces harder objectives and longer sequences.
    """

    def __init__(self, cfg: PretrainingConfig):
        self.cfg  = cfg
        self.step = 0

    def get_mask_ratio(self) -> float:
        """Start with low mask ratio and increase gradually."""
        base    = self.cfg.mask_ratio
        # Linear warmup from 0.05 to base over curriculum_steps
        frac    = min(1.0, self.step / max(1, self.cfg.curriculum_steps))
        return 0.05 + frac * (base - 0.05)

    def get_sequence_length(self, base_len: int, max_len: int) -> int:
        """Gradually increase sequence length during training."""
        frac = min(1.0, self.step / max(1, self.cfg.curriculum_steps))
        return int(base_len + frac * (max_len - base_len))

    def get_active_objectives(self) -> Dict[str, bool]:
        """Enable objectives progressively."""
        cfg = self.cfg
        active = {"mrm": True}  # MRM always active

        if self.step >= self.cfg.warmup_steps:
            active["npp"]  = cfg.use_npp
        else:
            active["npp"]  = False

        if self.step >= self.cfg.warmup_steps * 2:
            active["contrastive"] = cfg.use_contrastive
            active["volatility"]  = cfg.use_volatility
        else:
            active["contrastive"] = False
            active["volatility"]  = False

        active["regime"] = cfg.use_regime and self.step >= self.cfg.warmup_steps * 3
        return active

    def step_update(self) -> None:
        self.step += 1

    def state_dict(self) -> Dict:
        return {"step": self.step}

    def load_state_dict(self, d: Dict) -> None:
        self.step = d["step"]


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------

class CosineWithWarmupScheduler:
    """Cosine decay learning rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_steps:  int,
        max_steps:     int,
        min_lr_ratio:  float = 0.1,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_count   = 0
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        lr_scale = self._get_lr_scale()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale

    def _get_lr_scale(self) -> float:
        t = self.step_count
        if t < self.warmup_steps:
            return t / max(1, self.warmup_steps)
        progress = (t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine   = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine

    def get_last_lr(self) -> List[float]:
        scale = self._get_lr_scale()
        return [base * scale for base in self.base_lrs]


# ---------------------------------------------------------------------------
# Gradient clipping utilities
# ---------------------------------------------------------------------------

def clip_grad_norm(
    parameters: Any,
    max_norm:   float,
    norm_type:  float = 2.0,
) -> torch.Tensor:
    """Clip gradient norms and return the total gradient norm."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm for monitoring."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


# ---------------------------------------------------------------------------
# Pretraining Trainer
# ---------------------------------------------------------------------------

class PretrainingTrainer:
    """
    Training loop for Lumina pre-training.

    Handles:
      - Forward pass with multi-task loss
      - Gradient accumulation
      - LR scheduling
      - Logging / metric tracking
      - Checkpoint saving
    """

    def __init__(
        self,
        model:         nn.Module,
        loss_module:   MultiTaskPretrainingLoss,
        optimizer:     torch.optim.Optimizer,
        scheduler:     CosineWithWarmupScheduler,
        cfg:           PretrainingConfig,
        device:        torch.device,
        grad_accum:    int = 1,
    ):
        self.model      = model
        self.loss_mod   = loss_module
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.cfg        = cfg
        self.device     = device
        self.grad_accum = grad_accum
        self.step       = 0
        self.metrics_history: List[Dict] = []
        self.schedule   = PretrainingSchedule(cfg)
        self.mask_gen   = PatchMaskGenerator(cfg)

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulate: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run a single forward + backward step."""
        ohlcv        = batch["ohlcv"].to(self.device)          # (B, T, 5)
        patch_target = batch.get("patch_target")
        B            = ohlcv.shape[0]

        # Tokenize + encode
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Get embeddings from model tokenizer
            out = self.model(ohlcv)
            hidden   = out["hidden"]
            cls_emb  = out.get("cls_emb", hidden[:, 0, :])

            N = hidden.shape[1]

            # Generate mask
            mask = self.mask_gen(B, N, self.device)

            # Patch targets
            if patch_target is None:
                patch_target = torch.zeros(
                    B, N, self.loss_mod.mrm_head.patch_size * self.loss_mod.mrm_head.n_channels,
                    device=self.device
                )
            else:
                patch_target = patch_target.to(self.device)

            # Contrastive pair: two different windows of the same asset
            z1 = cls_emb
            z2 = cls_emb + 0.01 * torch.randn_like(cls_emb)   # augmented view

            total_loss, metrics = self.loss_mod(
                hidden, cls_emb, patch_target, mask,
                contrastive_z=(z1, z2),
            )
            total_loss = total_loss / self.grad_accum

        if not accumulate or (self.step % self.grad_accum == self.grad_accum - 1):
            total_loss.backward()
        else:
            total_loss.backward()

        return total_loss * self.grad_accum, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single optimizer step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, metrics = self._forward_step(batch)

        grad_norm = clip_grad_norm(self.model.parameters(), self.cfg.grad_clip)
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        self.optimizer.step()
        self.scheduler.step()
        self.schedule.step_update()
        self.step += 1

        metrics["lr"] = self.scheduler.get_last_lr()[0]
        self.metrics_history.append(metrics)
        return metrics

    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_metrics: Dict[str, float] = {}
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                _, metrics = self._forward_step(batch)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        total_metrics[k] = total_metrics.get(k, 0.0) + v
                count += 1

        return {k: v / count for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step":           self.step,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.step_count,
            "schedule":       self.schedule.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.step_count = ckpt["scheduler_step"]
        self.schedule.load_state_dict(ckpt["schedule"])
        self.step = ckpt["step"]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PretrainingConfig",
    "PatchMaskGenerator",
    "MaskedReturnModelingHead",
    "MaskedReturnModelingLoss",
    "NextPatchPredictionHead",
    "NextPatchPredictionLoss",
    "InfoNCELoss",
    "CrossAssetContrastiveLoss",
    "MomentumContrastiveQueue",
    "RegimePredictionHead",
    "VolatilityPredictionHead",
    "AutomaticLossWeighter",
    "MultiTaskPretrainingLoss",
    "PretrainingSchedule",
    "CosineWithWarmupScheduler",
    "PretrainingTrainer",
    "infer_regime_labels",
    "clip_grad_norm",
    "compute_grad_norm",
]
