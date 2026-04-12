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
# ---------------------------------------------------------------------------
# Contrastive Pre-training Utilities
# ---------------------------------------------------------------------------

class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss for financial time series.

    Creates positive pairs by augmenting the same sequence differently,
    negative pairs from different time windows.

    Specifically designed for financial data:
    - Augmentations: time warping, jitter, amplitude scaling
    - Hard negatives: sequences from different market regimes

    Args:
        temperature:      InfoNCE temperature
        n_negatives:      number of negative pairs per positive
        hard_negative_weight: weight for hard (regime-different) negatives

    Example:
        >>> loss_fn = TemporalContrastiveLoss(temperature=0.07)
        >>> z1 = torch.randn(16, 256)  # augmented view 1
        >>> z2 = torch.randn(16, 256)  # augmented view 2
        >>> loss = loss_fn(z1, z2)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_negatives: int = 256,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute temporal contrastive loss.

        Args:
            z1:     (N, D) embeddings from view 1
            z2:     (N, D) embeddings from view 2
            labels: (N,) optional regime labels for hard negative mining

        Returns:
            loss: scalar contrastive loss
        """
        N, D = z1.shape

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask self-similarities
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])  # (2N,)

        # Hard negative weighting (upweight negatives from same regime)
        if labels is not None:
            labels_2x = torch.cat([labels, labels], dim=0)  # (2N,)
            same_regime = (labels_2x.unsqueeze(0) == labels_2x.unsqueeze(1))  # (2N, 2N)
            # Upweight same-regime negatives
            weight_mask = same_regime & ~mask
            sim = sim + weight_mask.float() * math.log(self.hard_negative_weight)

        loss = F.cross_entropy(sim, pos_idx)
        return loss


class BarcodingLoss(nn.Module):
    """Barcoding / Patch-order prediction loss.

    Predicts the relative temporal order of shuffled patches.
    This objective teaches the model about temporal structure in
    financial data without reconstruction.

    Args:
        n_pairs:    number of random patch pairs to compare per sample
        dropout:    dropout on the classifier head

    Example:
        >>> bcl = BarcodingLoss(n_pairs=32)
        >>> hidden = torch.randn(4, 64, 512)  # (B, T, d)
        >>> shuffle_idx = torch.argsort(torch.rand(4, 64), dim=-1)
        >>> loss = bcl(hidden, shuffle_idx)
    """

    def __init__(self, n_pairs: int = 32, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_pairs = n_pairs
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        original_order: torch.Tensor,
    ) -> torch.Tensor:
        """Compute patch order prediction loss.

        Args:
            hidden:         (B, T, d_model) hidden states of shuffled sequence
            original_order: (B, T) original patch indices before shuffling

        Returns:
            loss: scalar BCE loss
        """
        B, T, D = hidden.shape

        losses = []
        for _ in range(self.n_pairs):
            # Pick random pairs
            i_idx = torch.randint(0, T, (B,), device=hidden.device)
            j_idx = torch.randint(0, T, (B,), device=hidden.device)

            h_i = hidden[torch.arange(B), i_idx, :]  # (B, D)
            h_j = hidden[torch.arange(B), j_idx, :]  # (B, D)

            # True order: original_order[b, i_idx[b]] < original_order[b, j_idx[b]]
            pos_i = original_order[torch.arange(B), i_idx]
            pos_j = original_order[torch.arange(B), j_idx]
            label = (pos_i < pos_j).float()  # (B,)

            # Predict order
            pair_feat = torch.cat([h_i, h_j], dim=-1)  # (B, 2D)
            pred = self.classifier(pair_feat).squeeze(-1)  # (B,)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            losses.append(loss)

        return torch.stack(losses).mean()


class ReturnDistributionModeling(nn.Module):
    """Pre-training objective: model the conditional return distribution.

    Instead of point prediction (MSE), models the full conditional
    distribution of future returns as a mixture of Gaussians.

    P(r_t+1 | h_t) = sum_k pi_k * N(mu_k, sigma_k^2)

    Loss: negative log-likelihood of observed returns under the mixture.

    Args:
        d_model:    hidden state dimension
        n_components: number of Gaussian mixture components

    Example:
        >>> rdm = ReturnDistributionModeling(d_model=512, n_components=3)
        >>> hidden = torch.randn(4, 64, 512)
        >>> targets = torch.randn(4, 64)
        >>> loss = rdm(hidden, targets)
    """

    def __init__(self, d_model: int, n_components: int = 3):
        super().__init__()
        self.n_components = n_components

        self.mu_head = nn.Linear(d_model, n_components)
        self.log_sigma_head = nn.Linear(d_model, n_components)
        self.logit_pi_head = nn.Linear(d_model, n_components)

        nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.log_sigma_head.bias, math.log(0.01))  # init small sigma

    def forward(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute GMM NLL loss.

        Args:
            hidden:  (B, T, d_model)
            targets: (B, T) true return values
            mask:    (B, T) bool, True = compute loss at this position

        Returns:
            loss:   scalar NLL
            extras: dict with mixture parameters
        """
        mu = self.mu_head(hidden)              # (B, T, K)
        log_sigma = self.log_sigma_head(hidden)
        sigma = log_sigma.exp().clamp(min=1e-4)
        log_pi = F.log_softmax(self.logit_pi_head(hidden), dim=-1)

        # Compute log-likelihood for each component
        targets_expanded = targets.unsqueeze(-1)  # (B, T, 1)
        log_prob = (
            log_pi
            - log_sigma
            - 0.5 * ((targets_expanded - mu) / sigma).pow(2)
            - 0.5 * math.log(2 * math.pi)
        )  # (B, T, K)

        # Sum over components via log-sum-exp
        log_likelihood = torch.logsumexp(log_prob, dim=-1)  # (B, T)

        if mask is not None:
            loss = -(log_likelihood * mask.float()).sum() / (mask.float().sum() + 1e-8)
        else:
            loss = -log_likelihood.mean()

        pi = log_pi.exp()
        extras = {
            "mu": mu,
            "sigma": sigma,
            "pi": pi,
            "expected_return": (mu * pi).sum(dim=-1),
        }
        return loss, extras


class CurriculumMasking:
    """Curriculum learning for masked return modeling.

    Gradually increases masking difficulty over training:
    - Early: mask random single patches (easy to predict)
    - Middle: mask consecutive spans (harder)
    - Late: mask large temporal blocks (hardest)

    Args:
        initial_ratio:   initial masking ratio (default 0.15)
        final_ratio:     final masking ratio (default 0.40)
        warmup_steps:    steps to reach max masking ratio
        initial_span:    initial max consecutive mask span
        final_span:      final max consecutive mask span

    Example:
        >>> cm = CurriculumMasking(initial_ratio=0.1, final_ratio=0.4)
        >>> mask_ratio, span = cm.get_params(step=5000)
    """

    def __init__(
        self,
        initial_ratio: float = 0.15,
        final_ratio: float = 0.40,
        warmup_steps: int = 10000,
        initial_span: int = 1,
        final_span: int = 10,
    ):
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps
        self.initial_span = initial_span
        self.final_span = final_span

    def get_params(self, step: int) -> Tuple[float, int]:
        """Get masking parameters for current training step.

        Args:
            step: current training step

        Returns:
            mask_ratio: fraction of tokens to mask
            max_span:   maximum span of consecutive masked tokens
        """
        progress = min(1.0, step / self.warmup_steps)
        mask_ratio = self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
        max_span = int(self.initial_span + progress * (self.final_span - self.initial_span))
        return mask_ratio, max_span


class DataAugmentationForPretraining:
    """Data augmentation strategies for financial pre-training.

    Augmentations designed to be return-preserving (don't change target labels)
    while increasing data diversity:

    1. Price Translation: add constant offset to all prices (doesn't change returns)
    2. Price Scaling:     multiply all prices by constant (preserves return shape)
    3. Time Reversal:     predict returns in reversed time (different regime signal)
    4. Amplitude Jitter:  add small Gaussian noise to prices
    5. Temporal Crop:     randomly crop sub-sequences

    Args:
        jitter_std:       std of Gaussian noise for jitter
        scale_range:      range for scale augmentation
        p_reverse:        probability of time reversal

    Example:
        >>> aug = DataAugmentationForPretraining(jitter_std=0.001)
        >>> x = torch.randn(4, 128, 5)  # OHLCV batches
        >>> x_aug = aug(x)
    """

    def __init__(
        self,
        jitter_std: float = 0.001,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p_reverse: float = 0.1,
        p_crop: float = 0.2,
        min_crop_ratio: float = 0.8,
    ):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.p_reverse = p_reverse
        self.p_crop = p_crop
        self.min_crop_ratio = min_crop_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations.

        Args:
            x: (B, T, C) OHLCV tensor (price channels = 0:4, volume = 4)

        Returns:
            x_aug: (B, T, C) augmented tensor
        """
        x = x.clone()
        B, T, C = x.shape

        # Jitter
        if self.jitter_std > 0:
            noise = torch.randn_like(x) * self.jitter_std
            noise[:, :, 4] = 0  # don't jitter volume
            x = x + noise

        # Scale
        scale = torch.FloatTensor(B, 1, 1).uniform_(*self.scale_range).to(x.device)
        x[:, :, :4] = x[:, :, :4] * scale  # scale price channels only

        # Time reversal (flip sequence)
        if self.p_reverse > 0:
            flip_mask = torch.rand(B) < self.p_reverse
            if flip_mask.any():
                x[flip_mask] = x[flip_mask].flip(dims=[1])

        return x


# ---------------------------------------------------------------------------
# Pre-training Metrics
# ---------------------------------------------------------------------------

class PretrainingMetrics:
    """Track and compute pre-training evaluation metrics.

    Metrics tracked:
    - Masked prediction loss (MSE)
    - Reconstruction accuracy (within threshold)
    - Perplexity (for autoregressive models)
    - Contrastive alignment score
    - Cross-asset correlation prediction accuracy

    Example:
        >>> metrics = PretrainingMetrics()
        >>> metrics.update(pred_loss=0.5, contrastive_loss=0.1)
        >>> report = metrics.compute()
    """

    def __init__(self):
        self._loss_history: List[float] = []
        self._pred_loss: List[float] = []
        self._contrastive_loss: List[float] = []
        self._regime_acc: List[float] = []
        self._n_updates = 0

    def update(
        self,
        pred_loss: Optional[float] = None,
        contrastive_loss: Optional[float] = None,
        regime_acc: Optional[float] = None,
        total_loss: Optional[float] = None,
    ) -> None:
        """Record metrics for a single step."""
        self._n_updates += 1
        if total_loss is not None:
            self._loss_history.append(total_loss)
        if pred_loss is not None:
            self._pred_loss.append(pred_loss)
        if contrastive_loss is not None:
            self._contrastive_loss.append(contrastive_loss)
        if regime_acc is not None:
            self._regime_acc.append(regime_acc)

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics.

        Returns:
            metrics: dict of metric name → value
        """
        def safe_mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else float("nan")

        return {
            "n_updates": self._n_updates,
            "avg_total_loss": safe_mean(self._loss_history),
            "avg_pred_loss": safe_mean(self._pred_loss),
            "avg_contrastive_loss": safe_mean(self._contrastive_loss),
            "avg_regime_accuracy": safe_mean(self._regime_acc),
            "last_total_loss": self._loss_history[-1] if self._loss_history else float("nan"),
        }

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._loss_history.clear()
        self._pred_loss.clear()
        self._contrastive_loss.clear()
        self._regime_acc.clear()
        self._n_updates = 0

    def get_loss_history(self) -> List[float]:
        """Return full loss history."""
        return self._loss_history.copy()


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
    "TemporalContrastiveLoss",
    "BarcodingLoss",
    "ReturnDistributionModeling",
    "CurriculumMasking",
    "DataAugmentationForPretraining",
    "PretrainingMetrics",
    "infer_regime_labels",
    "clip_grad_norm",
    "compute_grad_norm",
]


# =============================================================================
# SECTION: Advanced Self-Supervised Pre-Training Objectives
# =============================================================================

class BYOL_FinancialLoss(nn.Module):
    """Bootstrap Your Own Latent (BYOL) for financial time series.

    BYOL trains without negative samples by using two networks:
    - Online network: learns and updates via gradient
    - Target network: exponential moving average of online network

    Two augmented views of the same time series are created; the online
    network predicts the target network's representation.

    Reference: Grill et al., "Bootstrap Your Own Latent" NeurIPS 2020

    Args:
        d_model: Representation dimension
        projection_dim: Projection head output dimension
        prediction_dim: Prediction head hidden dimension
        ema_decay: EMA decay rate for target network (typically 0.99-0.999)
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int = 256,
        prediction_dim: int = 4096,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        # Projector for online network
        self.online_projector = nn.Sequential(
            nn.Linear(d_model, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim),
        )
        # Predictor (only on online path)
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim),
        )
        # Target projector (no gradients, EMA of online_projector)
        self.target_projector = nn.Sequential(
            nn.Linear(d_model, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim),
        )
        # Disable gradients for target
        for param in self.target_projector.parameters():
            param.requires_grad = False
        # Initialize target = online
        self._sync_target()

    def _sync_target(self) -> None:
        """Copy online projector weights to target."""
        for op, tp in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            tp.data.copy_(op.data)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target network from online network."""
        for op, tp in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1: Online network representation from view 1 (B, D)
            z2: Online network representation from view 2 (B, D)
        Returns:
            BYOL loss scalar
        """
        # Online path: project then predict
        p1 = self.online_predictor(self.online_projector(z1))  # (B, proj_dim)
        p2 = self.online_predictor(self.online_projector(z2))

        # Target path: project only (EMA, no gradient)
        with torch.no_grad():
            t1 = self.target_projector(z1)
            t2 = self.target_projector(z2)

        # L2 normalize
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        t1 = F.normalize(t1, dim=-1)
        t2 = F.normalize(t2, dim=-1)

        # Symmetric loss: predict each view's target from the other
        loss = 2 - 2 * (p1 * t2).sum(dim=-1).mean() - 2 * (p2 * t1).sum(dim=-1).mean()
        loss = loss / 2  # Symmetric average
        return loss


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization for SSL.

    VICReg avoids representation collapse without negative samples
    or a momentum network by explicitly optimizing:
    - Variance: Keep std dev above threshold (prevent collapse)
    - Invariance: Minimize MSE between embeddings of two views
    - Covariance: Minimize off-diagonal covariance (decorrelate features)

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning" (ICLR 2022)

    Args:
        sim_coeff: Invariance loss weight (lambda)
        std_coeff: Variance loss weight (mu)
        cov_coeff: Covariance loss weight (nu)
        eps: Variance floor
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Variance regularization: keep std > 1 per dimension."""
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return F.relu(1 - std).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Covariance regularization: off-diagonal terms -> 0."""
        B, D = z.shape
        z = z - z.mean(dim=0)
        cov = torch.matmul(z.T, z) / (B - 1)  # (D, D)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: Embeddings from view 1 (B, D)
            z2: Embeddings from view 2 (B, D)
        Returns:
            Dict with 'loss', 'sim_loss', 'std_loss', 'cov_loss'
        """
        sim_loss = F.mse_loss(z1, z2)
        std_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)
        loss = (self.sim_coeff * sim_loss +
                self.std_coeff * std_loss +
                self.cov_coeff * cov_loss)
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss,
        }


class SwAVLoss(nn.Module):
    """SwAV: Swapping Assignments between Views.

    Uses online clustering with prototypes instead of direct feature
    comparison. Each view's features are assigned to prototypes,
    and the assignments from one view must predict prototypes for
    the other view.

    Reference: Caron et al., "Unsupervised Learning of Visual Features
    by Contrasting Cluster Assignments" NeurIPS 2020.

    Args:
        d_model: Feature dimension
        num_prototypes: Number of cluster prototypes
        temperature: Softmax temperature
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations
        epsilon: Sinkhorn regularization
    """

    def __init__(
        self,
        d_model: int,
        num_prototypes: int = 3000,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        # Prototype vectors
        self.prototypes = nn.Linear(d_model, num_prototypes, bias=False)
        nn.init.orthogonal_(self.prototypes.weight)

    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm for soft cluster assignments."""
        Q = torch.exp(scores / self.epsilon)
        B = Q.shape[0]
        K = Q.shape[1]
        Q = Q / Q.sum()
        for _ in range(self.sinkhorn_iterations):
            # Normalize columns
            Q = Q / (Q.sum(dim=0, keepdim=True) * K)
            # Normalize rows
            Q = Q / (Q.sum(dim=1, keepdim=True) * B)
        return Q

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: L2-normalized features (B, D)
        Returns:
            SwAV loss scalar
        """
        # Compute prototype assignments (soft)
        scores1 = self.prototypes(z1)  # (B, K)
        scores2 = self.prototypes(z2)
        q1 = self._sinkhorn(scores1)  # (B, K) - assignments for view 1
        q2 = self._sinkhorn(scores2)
        # Cross-prediction: use one view's assignment to predict the other's features
        p1 = scores1 / self.temperature
        p2 = scores2 / self.temperature
        loss = -(q2 * F.log_softmax(p1, dim=-1)).sum(dim=-1).mean()
        loss = loss - (q1 * F.log_softmax(p2, dim=-1)).sum(dim=-1).mean()
        return loss / 2


class MAELoss(nn.Module):
    """Masked Autoencoder (MAE) reconstruction loss.

    Given masked patches of financial time series, reconstructs
    the missing values. Operates in normalized patch space.

    Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners"
    CVPR 2022 (adapted for financial time series).

    Args:
        patch_size: Number of timesteps per patch
        num_features: Number of features per timestep
        norm_pix_loss: Normalize targets per patch before computing loss
    """

    def __init__(
        self,
        patch_size: int = 16,
        num_features: int = 5,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_features = num_features
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, C) to (B, num_patches, patch_size * C)."""
        B, T, C = x.shape
        P = self.patch_size
        num_patches = T // P
        # Reshape
        x = x[:, :num_patches * P, :]  # Truncate to multiple of P
        x = x.view(B, num_patches, P, C)
        x = x.view(B, num_patches, P * C)
        return x

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Reconstructed patches (B, num_patches, patch_dim)
            target: Original patches (B, num_patches, patch_dim)
            mask: Boolean mask (B, num_patches), True = masked/removed
        Returns:
            MSE loss on masked patches only
        """
        if self.norm_pix_loss:
            # Normalize each patch independently
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, num_patches)
        # Only compute loss on masked patches
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
        return loss


class FinancialPretrainingObjective(nn.Module):
    """Composite pre-training objective for financial foundation models.

    Combines multiple self-supervised objectives:
    1. Masked Return Modeling (MRM): predict masked future returns
    2. Temporal Contrastive Learning: same regime = similar, different = different
    3. Next-Patch Prediction (NPP): predict statistics of next patch
    4. Volatility Regime Prediction: predict current volatility regime

    Args:
        d_model: Model output dimension
        num_regimes: Number of volatility regimes
        forecast_horizons: Return horizons for MRM
        contrastive_temp: Temperature for contrastive loss
        mrm_weight: Weight for masked return modeling loss
        npp_weight: Weight for next-patch prediction loss
        contrastive_weight: Weight for contrastive loss
        regime_weight: Weight for regime classification loss
    """

    def __init__(
        self,
        d_model: int,
        num_regimes: int = 5,
        forecast_horizons: Optional[List[int]] = None,
        contrastive_temp: float = 0.1,
        mrm_weight: float = 1.0,
        npp_weight: float = 0.5,
        contrastive_weight: float = 0.5,
        regime_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.contrastive_temp = contrastive_temp
        self.mrm_weight = mrm_weight
        self.npp_weight = npp_weight
        self.contrastive_weight = contrastive_weight
        self.regime_weight = regime_weight

        # MRM head: predict returns at each horizon
        self.mrm_head = nn.Linear(d_model, len(self.forecast_horizons))
        # NPP head: predict next patch mean and std
        self.npp_head = nn.Linear(d_model, 2)  # mean, log_std
        # Regime classification head
        self.regime_head = nn.Linear(d_model, num_regimes)
        # Projection for contrastive
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
        )

    def mrm_loss(
        self,
        representations: torch.Tensor,
        target_returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked return modeling loss.

        Args:
            representations: (B, T, D) encoded representations
            target_returns: (B, T, num_horizons) forward returns
            mask: (B, T) True for masked positions
        """
        pred = self.mrm_head(representations)  # (B, T, H)
        loss = F.huber_loss(pred, target_returns, reduction="none")  # (B, T, H)
        loss = loss.mean(dim=-1)  # (B, T)
        return (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)

    def npp_loss(
        self,
        representations: torch.Tensor,
        next_patch_mean: torch.Tensor,
        next_patch_std: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Next-patch prediction loss (Gaussian NLL).

        Args:
            representations: (B, T, D)
            next_patch_mean: (B, T) mean return of next patch
            next_patch_std: (B, T) std of next patch (>0)
            mask: (B, T) True for valid prediction positions
        """
        pred = self.npp_head(representations)  # (B, T, 2)
        pred_mean = pred[:, :, 0]
        pred_log_std = pred[:, :, 1].clamp(-3, 3)
        pred_std = pred_log_std.exp() + 1e-6
        # Gaussian NLL
        nll = 0.5 * ((next_patch_mean - pred_mean) / pred_std) ** 2 + pred_log_std
        return (nll * mask.float()).sum() / (mask.float().sum() + 1e-6)

    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Temporal contrastive loss (InfoNCE / NT-Xent).

        Args:
            z1: First view embeddings (B, D)
            z2: Second view embeddings (B, D)
            labels: Optional same-class labels for supervised contrastive
        """
        z1 = F.normalize(self.proj(z1), dim=-1)
        z2 = F.normalize(self.proj(z2), dim=-1)
        B = z1.size(0)
        # Concat and compute pairwise similarities
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.matmul(z, z.T) / self.contrastive_temp  # (2B, 2B)
        # Self-similarities -> -inf
        sim.fill_diagonal_(float("-inf"))
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(z.device)
        loss = F.cross_entropy(sim, pos_idx)
        return loss

    def regime_loss(
        self,
        representations: torch.Tensor,
        regime_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Regime classification auxiliary loss.

        Args:
            representations: (B, T, D)
            regime_labels: (B, T) long tensor
            mask: (B, T) valid positions
        """
        logits = self.regime_head(representations)  # (B, T, R)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), regime_labels.view(-1), reduction="none")
        loss = loss.view(representations.size(0), -1)  # (B, T)
        if mask is not None:
            return (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
        return loss.mean()

    def forward(
        self,
        representations: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        target_returns: Optional[torch.Tensor] = None,
        next_patch_mean: Optional[torch.Tensor] = None,
        next_patch_std: Optional[torch.Tensor] = None,
        regime_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Combined pre-training loss.

        Args:
            representations: (B, T, D) from encoder
            z1, z2: Pooled representations from two augmented views (B, D)
            target_returns: (B, T, num_horizons) optional
            next_patch_mean: (B, T) optional
            next_patch_std: (B, T) optional
            regime_labels: (B, T) long optional
            mask: (B, T) bool, True=masked
        Returns:
            Dict with total loss and component losses
        """
        if mask is None:
            mask = torch.ones(representations.shape[:2], dtype=torch.bool, device=representations.device)

        losses = {}
        total = torch.tensor(0.0, device=representations.device)

        if target_returns is not None:
            l = self.mrm_loss(representations, target_returns, mask)
            losses["mrm"] = l
            total = total + self.mrm_weight * l

        if next_patch_mean is not None and next_patch_std is not None:
            l = self.npp_loss(representations, next_patch_mean, next_patch_std, ~mask)
            losses["npp"] = l
            total = total + self.npp_weight * l

        l_c = self.contrastive_loss(z1, z2)
        losses["contrastive"] = l_c
        total = total + self.contrastive_weight * l_c

        if regime_labels is not None:
            l = self.regime_loss(representations, regime_labels, ~mask)
            losses["regime"] = l
            total = total + self.regime_weight * l

        losses["total"] = total
        return losses


class PretrainingScheduler:
    """Manages curriculum for pre-training objectives.

    Gradually increases:
    - Masking ratio: from low to target over warmup
    - Mask span: from short to long
    - Contrastive difficulty: progressively harder negatives
    - Number of pre-training tasks: start simple, add objectives

    Args:
        total_steps: Total pre-training steps
        mask_ratio_start: Initial masking ratio
        mask_ratio_end: Final masking ratio
        span_start: Initial mask span
        span_end: Final mask span
        warmup_ratio: Fraction of steps for linear warmup
    """

    def __init__(
        self,
        total_steps: int,
        mask_ratio_start: float = 0.15,
        mask_ratio_end: float = 0.75,
        span_start: int = 1,
        span_end: int = 10,
        warmup_ratio: float = 0.3,
    ) -> None:
        self.total_steps = total_steps
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.span_start = span_start
        self.span_end = span_end
        self.warmup_steps = int(total_steps * warmup_ratio)
        self._step = 0

    def step(self) -> None:
        self._step += 1

    @property
    def current_step(self) -> int:
        return self._step

    def get_mask_ratio(self) -> float:
        """Current masking ratio based on training progress."""
        if self._step >= self.warmup_steps:
            return self.mask_ratio_end
        t = self._step / max(1, self.warmup_steps)
        return self.mask_ratio_start + t * (self.mask_ratio_end - self.mask_ratio_start)

    def get_span_length(self) -> int:
        """Current average span length."""
        if self._step >= self.warmup_steps:
            return self.span_end
        t = self._step / max(1, self.warmup_steps)
        span = self.span_start + t * (self.span_end - self.span_start)
        return max(1, int(span))

    def get_state(self) -> Dict[str, float]:
        return {
            "step": self._step,
            "mask_ratio": self.get_mask_ratio(),
            "span_length": self.get_span_length(),
            "progress": self._step / self.total_steps,
        }

    def load_state(self, state: Dict) -> None:
        self._step = state["step"]


class AugmentationPipeline(nn.Module):
    """Financial time series augmentation pipeline for contrastive learning.

    Applies stochastic augmentations to create diverse views:
    1. Time jitter: small temporal shifts
    2. Amplitude scaling: scale returns by random factor
    3. Gaussian noise injection
    4. Time masking: block out contiguous segments
    5. Frequency masking: remove frequency bands via FFT
    6. Temporal flipping: reverse time (with sign flipping for returns)
    7. Random cropping: take random contiguous subsequence
    8. Mixup: blend two time series with weight lambda

    Args:
        time_jitter_prob: Probability of time jitter
        scale_prob: Probability of amplitude scaling
        noise_prob: Probability of Gaussian noise
        mask_prob: Probability of time masking
        flip_prob: Probability of temporal flipping
        crop_prob: Probability of random crop
        noise_std: Standard deviation of Gaussian noise
        scale_range: (min, max) amplitude scaling
    """

    def __init__(
        self,
        time_jitter_prob: float = 0.2,
        scale_prob: float = 0.3,
        noise_prob: float = 0.4,
        mask_prob: float = 0.3,
        flip_prob: float = 0.1,
        crop_prob: float = 0.2,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        super().__init__()
        self.time_jitter_prob = time_jitter_prob
        self.scale_prob = scale_prob
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.noise_std = noise_std
        self.scale_range = scale_range

    def time_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Small random circular shift."""
        B, T, C = x.shape
        shifts = torch.randint(-3, 4, (B,))
        out = torch.zeros_like(x)
        for b, s in enumerate(shifts):
            out[b] = torch.roll(x[b], s.item(), dims=0)
        return out

    def amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random per-sample scaling of feature values."""
        B = x.size(0)
        lo, hi = self.scale_range
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(lo, hi)
        return x * scale

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise scaled by sample std."""
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(x) * self.noise_std * std
        return x + noise

    def time_mask(self, x: torch.Tensor, max_mask_frac: float = 0.2) -> torch.Tensor:
        """Block masking: zero out a contiguous time segment."""
        B, T, C = x.shape
        out = x.clone()
        mask_len = int(T * max_mask_frac)
        for b in range(B):
            if mask_len > 0:
                start = torch.randint(0, T - mask_len + 1, (1,)).item()
                out[b, start:start + mask_len, :] = 0.0
        return out

    def temporal_flip(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse time order, negate return features."""
        return x.flip(dims=[1])

    def random_crop(self, x: torch.Tensor, min_frac: float = 0.7) -> torch.Tensor:
        """Take a random sub-sequence and resize to original length."""
        B, T, C = x.shape
        out = torch.zeros_like(x)
        min_len = max(1, int(T * min_frac))
        for b in range(B):
            crop_len = torch.randint(min_len, T + 1, (1,)).item()
            start = torch.randint(0, T - crop_len + 1, (1,)).item()
            cropped = x[b, start:start + crop_len, :]  # (crop_len, C)
            # Resize via interpolation
            cropped = F.interpolate(
                cropped.T.unsqueeze(0), size=T, mode="linear", align_corners=False
            ).squeeze(0).T
            out[b] = cropped
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation pipeline.

        Args:
            x: Input time series (B, T, C)
        Returns:
            Augmented tensor (B, T, C)
        """
        if self.training:
            if torch.rand(1) < self.time_jitter_prob:
                x = self.time_jitter(x)
            if torch.rand(1) < self.scale_prob:
                x = self.amplitude_scale(x)
            if torch.rand(1) < self.noise_prob:
                x = self.add_noise(x)
            if torch.rand(1) < self.mask_prob:
                x = self.time_mask(x)
            if torch.rand(1) < self.flip_prob:
                x = self.temporal_flip(x)
            if torch.rand(1) < self.crop_prob:
                x = self.random_crop(x)
        return x


class SpanMaskingStrategy(nn.Module):
    """Span-based masking strategy for financial time series.

    Instead of independently masking each token, masks contiguous
    spans drawn from a geometric distribution. Tends to produce
    more realistic missing data patterns.

    Args:
        mask_ratio: Target fraction of tokens to mask
        span_lambda: Parameter of geometric distribution for span lengths
        max_span: Maximum span length
        min_span: Minimum span length
        mask_adjacent_prob: Probability of masking adjacent spans
    """

    def __init__(
        self,
        mask_ratio: float = 0.3,
        span_lambda: float = 3.0,
        max_span: int = 10,
        min_span: int = 1,
        mask_adjacent_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.span_lambda = span_lambda
        self.max_span = max_span
        self.min_span = min_span
        self.mask_adjacent_prob = mask_adjacent_prob

    def _sample_span_length(self) -> int:
        """Sample span length from geometric distribution."""
        import math
        # Geometric distribution: P(k) = (1-p)^(k-1) * p
        p = 1.0 / (1.0 + self.span_lambda)
        k = 1
        while k < self.max_span:
            if torch.rand(1).item() < p:
                break
            k += 1
        return max(self.min_span, min(k, self.max_span))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate mask for a sequence.

        Args:
            seq_len: Length of the sequence to mask
            device: Target device
        Returns:
            Boolean mask (seq_len,), True = masked
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        num_to_mask = int(seq_len * self.mask_ratio)
        num_masked = 0
        max_attempts = seq_len * 10

        for _ in range(max_attempts):
            if num_masked >= num_to_mask:
                break
            span = self._sample_span_length()
            start = torch.randint(0, max(1, seq_len - span + 1), (1,)).item()
            end = min(seq_len, start + span)
            mask[start:end] = True
            num_masked = mask.sum().item()

        return mask

    def batch_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate masks for a batch.

        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            device: Target device
        Returns:
            (batch_size, seq_len) bool mask
        """
        return torch.stack([self.forward(seq_len, device) for _ in range(batch_size)])


class CrossModalMaskingStrategy(nn.Module):
    """Cross-modal masking for multi-modal financial data.

    Creates correlated masks across modalities:
    - Temporal alignment masks (mask same time periods across modalities)
    - Modality dropout (randomly drop entire modalities)
    - Conditional masking (mask in one modality conditions another)

    Args:
        num_modalities: Number of input modalities
        modality_dropout_prob: Probability of dropping an entire modality
        temporal_correlation: Degree of temporal mask correlation across modalities
        mask_ratio: Base masking ratio
    """

    def __init__(
        self,
        num_modalities: int,
        modality_dropout_prob: float = 0.1,
        temporal_correlation: float = 0.5,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_dropout_prob = modality_dropout_prob
        self.temporal_correlation = temporal_correlation
        self.mask_ratio = mask_ratio

    def forward(
        self,
        modality_lengths: List[int],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Generate correlated masks for multiple modalities.

        Args:
            modality_lengths: List of sequence lengths per modality
            device: Target device
        Returns:
            masks: List of (seq_len,) bool masks, one per modality
            modality_active: (num_modalities,) bool, True = modality not dropped
        """
        # Modality dropout
        modality_active = torch.rand(self.num_modalities) > self.modality_dropout_prob

        # Shared temporal mask for correlation
        max_len = max(modality_lengths)
        shared_mask = torch.rand(max_len, device=device) < self.mask_ratio

        masks = []
        for m, ml in enumerate(modality_lengths):
            if not modality_active[m]:
                masks.append(torch.ones(ml, dtype=torch.bool, device=device))
                continue
            # Mix shared mask with independent random mask
            indep_mask = torch.rand(ml, device=device) < self.mask_ratio
            shared_m = shared_mask[:ml]
            # Weighted combination
            combined = (
                self.temporal_correlation * shared_m.float() +
                (1 - self.temporal_correlation) * indep_mask.float()
            )
            masks.append(combined > 0.5)

        return masks, modality_active


_NEW_PRETRAINING_EXPORTS = [
    "BYOL_FinancialLoss", "VICRegLoss", "SwAVLoss", "MAELoss",
    "FinancialPretrainingObjective", "PretrainingScheduler",
    "AugmentationPipeline", "SpanMaskingStrategy", "CrossModalMaskingStrategy",
]
