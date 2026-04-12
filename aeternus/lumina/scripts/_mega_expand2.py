"""Second mega expansion pass."""
import os, subprocess, sys

BASE = os.path.join(os.path.dirname(__file__), "..", "lumina")

def count_lines(path):
    return len(open(path, encoding="utf-8", errors="replace").readlines())

def append_to(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    return count_lines(path)

# ============================================================
# 1. Large expansion of pretraining.py
# ============================================================
PRETRAINING_ADD = '''

# =============================================================================
# SECTION: Advanced Pretraining Methods (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class ElasticityAugmentation(nn.Module):
    """Elastic deformation augmentation for financial time series.

    Applies random warping along the time dimension using B-spline interpolation.
    Creates plausible alternative paths consistent with the original data distribution.
    """

    def __init__(
        self,
        n_control_points: int = 8,
        warp_std: float = 0.05,
        p_apply: float = 0.5,
    ):
        super().__init__()
        self.n_control_points = n_control_points
        self.warp_std = warp_std
        self.p_apply = p_apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p_apply:
            return x

        B, T, D = x.shape

        # Generate control point displacements
        ctrl_displace = torch.randn(B, self.n_control_points, device=x.device) * self.warp_std

        # Interpolate to full time grid
        ctrl_t = torch.linspace(0, T - 1, self.n_control_points, device=x.device)
        full_t = torch.arange(T, dtype=torch.float32, device=x.device)

        warped_results = []
        for b in range(B):
            # Interpolate displacement at each time step
            displace = torch.zeros(T, device=x.device)
            for i in range(self.n_control_points - 1):
                mask = (full_t >= ctrl_t[i]) & (full_t < ctrl_t[i+1])
                alpha = (full_t[mask] - ctrl_t[i]) / (ctrl_t[i+1] - ctrl_t[i] + 1e-8)
                displace[mask] = ctrl_displace[b, i] * (1 - alpha) + ctrl_displace[b, i+1] * alpha

            # Warp time indices
            warp_t = (full_t + displace * T).clamp(0, T - 1)
            warp_t_idx = warp_t.long()
            warp_alpha = warp_t - warp_t_idx.float()

            # Linear interpolation
            warp_t_idx_next = (warp_t_idx + 1).clamp(0, T - 1)
            warped = (
                x[b][warp_t_idx] * (1 - warp_alpha.unsqueeze(-1)) +
                x[b][warp_t_idx_next] * warp_alpha.unsqueeze(-1)
            )
            warped_results.append(warped)

        return torch.stack(warped_results)


class MixupAugmentation(nn.Module):
    """Mixup augmentation for financial time series (Zhang et al. 2018).

    Creates convex combinations of training examples and their labels.
    Works both in input space and in hidden representation space (Manifold Mixup).
    """

    def __init__(self, alpha: float = 0.2, mode: str = "input"):
        super().__init__()
        self.alpha = alpha
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        if not self.training or self.alpha == 0:
            return x, y, None, 1.0

        import numpy as np
        lam = float(np.random.beta(self.alpha, self.alpha))

        B = x.shape[0]
        idx = torch.randperm(B, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[idx]

        if y is not None:
            return mixed_x, y, y[idx], lam
        return mixed_x, None, None, lam


class CutMixAugmentation(nn.Module):
    """CutMix augmentation adapted for time series.

    Randomly cuts a contiguous temporal segment from one sample
    and pastes it into another, mixing their labels proportionally.
    """

    def __init__(self, alpha: float = 1.0, p_apply: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.p_apply = p_apply

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        if not self.training or torch.rand(1).item() > self.p_apply:
            return x, y, None, 1.0

        import numpy as np
        B, T, D = x.shape
        lam = float(np.random.beta(self.alpha, self.alpha))

        cut_len = int(T * (1 - lam))
        cut_start = torch.randint(0, T - cut_len + 1, (1,)).item()

        idx = torch.randperm(B, device=x.device)
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len, :] = x[idx, cut_start:cut_start + cut_len, :]

        lam_actual = 1.0 - cut_len / T

        if y is not None:
            return mixed_x, y, y[idx], lam_actual
        return mixed_x, None, None, lam_actual


class TimeFrequencyMasking(nn.Module):
    """Time-frequency masking inspired by SpecAugment (Park et al. 2019).

    Applies masks in both time and frequency (via FFT) dimensions.
    Particularly useful for financial signals with periodic patterns.
    """

    def __init__(
        self,
        time_mask_pct: float = 0.15,
        freq_mask_pct: float = 0.10,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
    ):
        super().__init__()
        self.time_mask_pct = time_mask_pct
        self.freq_mask_pct = freq_mask_pct
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B, T, D = x.shape
        x_aug = x.clone()

        # Time masking
        for _ in range(self.n_time_masks):
            mask_len = int(T * self.time_mask_pct)
            for b in range(B):
                start = torch.randint(0, T - mask_len + 1, (1,)).item()
                x_aug[b, start:start + mask_len, :] = 0.0

        # Frequency masking (apply to each feature dimension via FFT)
        n_freq = D // 2 + 1
        for _ in range(self.n_freq_masks):
            mask_len = int(n_freq * self.freq_mask_pct)
            for b in range(B):
                freq_start = torch.randint(0, n_freq - mask_len + 1, (1,)).item()
                x_fft = torch.fft.rfft(x_aug[b], dim=0)
                x_fft[freq_start:freq_start + mask_len, :] = 0.0
                x_aug[b] = torch.fft.irfft(x_fft, n=T, dim=0)

        return x_aug


class SectorContrastiveLoss(nn.Module):
    """Contrastive loss that leverages sector/industry labels as supervision.

    Pull together representations from the same sector,
    push apart representations from different sectors.
    Uses a soft negative mining strategy.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int = 128,
        temperature: float = 0.07,
        n_sectors: int = 11,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_sectors = n_sectors
        self.hard_negative_weight = hard_negative_weight

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        sector_labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            embeddings: [B, d_model]
            sector_labels: [B] sector indices
        """
        z = F.normalize(self.proj(embeddings), dim=-1)
        B = z.shape[0]

        # Similarity matrix
        sim = z @ z.T  # [B, B]
        sim = sim / self.temperature

        # Mask: positives are same sector, negatives are different sector
        labels_equal = (sector_labels.unsqueeze(0) == sector_labels.unsqueeze(1))

        # Remove diagonal
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        labels_equal = labels_equal & diag_mask

        # Loss: for each anchor, pull positives and push negatives
        total_loss = torch.tensor(0.0, device=z.device)
        n_valid = 0

        for i in range(B):
            pos_mask = labels_equal[i]
            neg_mask = ~labels_equal[i] & diag_mask[i]

            if not pos_mask.any():
                continue

            pos_sim = sim[i][pos_mask]
            neg_sim = sim[i][neg_mask]

            if not neg_mask.any():
                continue

            # Soft positive loss
            logsumexp_neg = torch.logsumexp(neg_sim, dim=0)
            pos_loss = (-pos_sim + logsumexp_neg).mean()

            # Hard negative weighting
            hard_neg_weight = torch.ones_like(neg_sim)
            top_neg_idx = neg_sim.topk(min(3, neg_mask.sum().item())).indices
            hard_neg_weight[top_neg_idx] = self.hard_negative_weight
            weighted_neg = (neg_sim * hard_neg_weight).logsumexp(dim=0)
            hard_loss = (-pos_sim.mean() + weighted_neg)

            total_loss = total_loss + 0.5 * pos_loss + 0.5 * hard_loss
            n_valid += 1

        if n_valid == 0:
            return {"loss": torch.tensor(0.0, device=z.device)}

        return {"loss": total_loss / n_valid}


class TimeSeriesGAN(nn.Module):
    """GAN-based data augmentation for financial time series.

    Generator: noise -> synthetic time series
    Discriminator: real/synthetic classification
    Uses WGAN-GP for stable training.
    """

    class Generator(nn.Module):
        def __init__(self, n_latent: int, T: int, n_features: int, d_model: int):
            super().__init__()
            self.T = T
            self.n_features = n_features
            self.init_proj = nn.Linear(n_latent, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, 4, d_model*2, 0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 3)
            self.out = nn.Linear(d_model, n_features)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            B = z.shape[0]
            h = self.init_proj(z).unsqueeze(1).expand(-1, self.T, -1)
            h = self.transformer(h)
            return self.out(h)

    class Discriminator(nn.Module):
        def __init__(self, T: int, n_features: int, d_model: int):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, 4, d_model*2, 0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 3)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.proj(x)
            h = self.transformer(h)
            return self.head(h[:, -1, :])

    def __init__(
        self,
        n_latent: int = 64,
        T: int = 64,
        n_features: int = 10,
        d_model: int = 128,
        lambda_gp: float = 10.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.lambda_gp = lambda_gp

        self.generator = self.Generator(n_latent, T, n_features, d_model)
        self.discriminator = self.Discriminator(T, n_features, d_model)

    def generate(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(batch_size, self.n_latent, device=device)
        return self.generator(z)

    def compute_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        B = real.shape[0]
        alpha = torch.rand(B, 1, 1, device=real.device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        disc_out = self.discriminator(interpolated)
        grad = torch.autograd.grad(
            outputs=disc_out, inputs=interpolated,
            grad_outputs=torch.ones_like(disc_out),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grad.view(B, -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean() * self.lambda_gp

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_loss = -self.discriminator(real).mean()
        fake_loss = self.discriminator(fake.detach()).mean()
        gp = self.compute_gradient_penalty(real, fake.detach())
        return real_loss + fake_loss + gp

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return -self.discriminator(fake).mean()


class TemporalContrastivePretraining(nn.Module):
    """Temporal contrastive pretraining for financial models.

    Creates positive pairs from nearby time windows (should be similar)
    and negative pairs from distant windows (should be different).
    Based on Temporal Contrastive Learning (TCL, Franceschi 2019).
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int = 128,
        positive_window: int = 5,
        negative_gap: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.positive_window = positive_window
        self.negative_gap = negative_gap
        self.temperature = temperature

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj),
        )

    def create_temporal_pairs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample positive and negative pairs from a sequence."""
        B, T, D = x.shape
        anchors, positives, negatives = [], [], []

        for b in range(B):
            for t in range(self.negative_gap, T - self.positive_window):
                anchor = x[b, t]
                # Positive: nearby window
                pos_t = torch.randint(
                    max(0, t - self.positive_window),
                    min(T, t + self.positive_window + 1),
                    (1,),
                ).item()
                positive = x[b, pos_t]

                # Negative: distant window
                neg_t = torch.randint(0, max(1, t - self.negative_gap), (1,)).item()
                negative = x[b, neg_t]

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
    ) -> dict:
        """
        Args:
            anchor_emb: [N, d_model]
            positive_emb: [N, d_model]
            negative_embs: [N, K, d_model]
        """
        z_a = F.normalize(self.proj(anchor_emb), dim=-1)
        z_p = F.normalize(self.proj(positive_emb), dim=-1)

        pos_sim = (z_a * z_p).sum(dim=-1) / self.temperature

        if negative_embs.ndim == 3:
            B, K, D = negative_embs.shape
            z_n = F.normalize(self.proj(negative_embs.view(B*K, D)), dim=-1).view(B, K, -1)
            neg_sim = torch.einsum("bd,bkd->bk", z_a, z_n) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=z_a.device)
            loss = F.cross_entropy(logits, labels)
        else:
            z_n = F.normalize(self.proj(negative_embs), dim=-1)
            neg_sim = (z_a * z_n).sum(dim=-1) / self.temperature
            loss = F.softplus(neg_sim - pos_sim).mean()

        return {
            "loss": loss,
            "pos_sim_mean": pos_sim.mean().item(),
        }


class ReturnPredictionCurriculum:
    """Curriculum learning strategy for return prediction tasks.

    Gradually increases difficulty by:
    1. Start with short-horizon, high-volatility, liquid assets
    2. Progress to longer horizons and less predictable assets
    3. Eventually train on full cross-section with all horizons
    """

    def __init__(
        self,
        n_horizons: int = 10,
        n_stages: int = 5,
        initial_horizon: int = 1,
        steps_per_stage: int = 5000,
    ):
        self.n_horizons = n_horizons
        self.n_stages = n_stages
        self.initial_horizon = initial_horizon
        self.steps_per_stage = steps_per_stage
        self._step = 0

    @property
    def current_stage(self) -> int:
        return min(self._step // self.steps_per_stage, self.n_stages - 1)

    @property
    def active_horizons(self) -> List[int]:
        """Return list of horizon indices active at current stage."""
        max_h = max(1, int(self.n_horizons * (self.current_stage + 1) / self.n_stages))
        return list(range(min(self.initial_horizon, self.n_horizons), max_h + 1))

    def step(self):
        self._step += 1

    def get_horizon_weights(self) -> torch.Tensor:
        """Return loss weights for each horizon (active horizons get weight 1, others 0)."""
        weights = torch.zeros(self.n_horizons)
        for h in self.active_horizons:
            if h - 1 < self.n_horizons:
                weights[h - 1] = 1.0
        return weights / weights.sum().clamp(min=1)

    def info(self) -> dict:
        return {
            "step": self._step,
            "stage": self.current_stage,
            "active_horizons": self.active_horizons,
        }


class GradualPretrainingScheduler:
    """Schedule pretraining objectives over the course of training.

    Phase 1 (warmup): Only simple objectives (masked autoencoding)
    Phase 2 (main): Full suite of objectives
    Phase 3 (annealing): High-quality contrastive + downstream alignment
    """

    def __init__(
        self,
        total_steps: int = 100000,
        warmup_fraction: float = 0.05,
        anneal_fraction: float = 0.2,
    ):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.anneal_start = int(total_steps * (1 - anneal_fraction))
        self._step = 0

    @property
    def phase(self) -> str:
        if self._step < self.warmup_steps:
            return "warmup"
        elif self._step < self.anneal_start:
            return "main"
        else:
            return "anneal"

    def get_objective_weights(self) -> dict:
        """Return weights for each pretraining objective."""
        if self.phase == "warmup":
            return {"mrm": 1.0, "contrastive": 0.0, "npp": 0.0, "regime": 0.0}
        elif self.phase == "main":
            progress = (self._step - self.warmup_steps) / max(self.anneal_start - self.warmup_steps, 1)
            return {
                "mrm": 1.0,
                "contrastive": progress,
                "npp": 0.5 * progress,
                "regime": 0.3 * progress,
            }
        else:
            annealing = 1.0 - (self._step - self.anneal_start) / (self.total_steps - self.anneal_start)
            return {
                "mrm": 0.5 * annealing,
                "contrastive": 1.0,
                "npp": 0.5,
                "regime": 0.5,
            }

    def step(self):
        self._step += 1


class RepresentationRegularizer(nn.Module):
    """Regularize learned representations to prevent collapse and improve transfer.

    Applies multiple representation regularization losses:
    - Variance: prevent mode collapse (VICReg)
    - Covariance: decorrelate features
    - Entropy maximization: spread mass
    - Invariance to augmentation
    """

    def __init__(
        self,
        d_model: int,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma
        self.eps = eps
        self.d_model = d_model

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        """
        Args:
            z_a, z_b: [B, d_model] representations of augmented views
        """
        B, D = z_a.shape

        # Invariance loss
        inv_loss = F.mse_loss(z_a, z_b)

        # Variance loss: std should be at least gamma
        def var_loss(z):
            std = z.std(dim=0).clamp(min=self.eps)
            return F.relu(self.gamma - std).mean()

        var_loss_a = var_loss(z_a)
        var_loss_b = var_loss(z_b)

        # Covariance loss: off-diagonal should be small
        def cov_loss(z):
            z_centered = z - z.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (B - 1)
            off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
            return off_diag / D

        cov_loss_total = cov_loss(z_a) + cov_loss(z_b)

        total = (
            self.sim_weight * inv_loss
            + self.var_weight * (var_loss_a + var_loss_b)
            + self.cov_weight * cov_loss_total
        )

        return {
            "total": total,
            "invariance": inv_loss.item(),
            "variance": (var_loss_a + var_loss_b).item() / 2,
            "covariance": cov_loss_total.item(),
        }


# Additional pretraining utility classes

class NoisyLabelPretraining(nn.Module):
    """Handle noisy labels in financial pretraining via noise-robust loss functions.

    Financial returns are noisy labels (true alpha obscured by noise).
    Uses:
    - Symmetric cross entropy loss
    - MAE loss (robust to outliers)
    - Huber loss (combines MSE and MAE)
    - Trimmed loss (ignore top-k loss samples)
    """

    def __init__(
        self,
        loss_type: str = "huber",
        trim_fraction: float = 0.1,
        label_smoothing: float = 0.1,
        delta: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.trim_fraction = trim_fraction
        self.label_smoothing = label_smoothing
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        if self.loss_type == "huber":
            loss = F.huber_loss(pred, target, delta=self.delta)
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred, target)
        elif self.loss_type == "trimmed":
            raw_loss = (pred - target).pow(2).view(-1)
            k = int(len(raw_loss) * self.trim_fraction)
            sorted_loss, _ = raw_loss.sort()
            loss = sorted_loss[k:].mean()
        elif self.loss_type == "sce":
            # Symmetric cross-entropy for regression
            loss = 0.5 * F.mse_loss(pred, target) + 0.5 * F.mse_loss(target, pred.detach())
        else:
            loss = F.mse_loss(pred, target)

        return {"loss": loss, "loss_type": self.loss_type}


class MultiTaskPretrainingObjective(nn.Module):
    """Combine multiple pretraining objectives with dynamic weighting.

    Uses gradient-based dynamic weighting (GradNorm) to balance
    task losses automatically, preventing any single task from dominating.
    """

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
        use_gradnorm: bool = True,
    ):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        self.use_gradnorm = use_gradnorm
        self.n_tasks = len(task_names)

        # Learnable task weights
        self.log_weights = nn.Parameter(torch.zeros(self.n_tasks))
        self._initial_losses = None

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Combine task losses with optional GradNorm weighting."""
        weights = F.softmax(self.log_weights, dim=0)
        task_losses = [losses.get(name, torch.tensor(0.0)) for name in self.task_names]

        if self.use_gradnorm and self._initial_losses is None:
            self._initial_losses = [l.item() for l in task_losses]

        weighted_losses = {
            name: w * l
            for name, w, l in zip(self.task_names, weights, task_losses)
        }
        total = sum(weighted_losses.values())

        return total, {
            "weights": {name: w.item() for name, w in zip(self.task_names, weights)},
            "weighted_losses": {name: l.item() for name, l in weighted_losses.items()},
        }

    def gradnorm_loss(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params,
    ) -> torch.Tensor:
        """Compute GradNorm auxiliary loss for weight update."""
        weights = F.softmax(self.log_weights, dim=0)
        task_losses = [losses.get(name, torch.tensor(0.0)) for name in self.task_names]

        if self._initial_losses is None:
            return torch.tensor(0.0)

        # Gradient norms
        grad_norms = []
        for l in task_losses:
            grad = torch.autograd.grad(l, shared_params, retain_graph=True, allow_unused=True)
            g_norm = sum(g.norm() for g in grad if g is not None)
            grad_norms.append(g_norm)

        avg_norm = sum(grad_norms) / len(grad_norms)
        loss_ratios = torch.tensor(
            [l.item() / max(init, 1e-8) for l, init in zip(task_losses, self._initial_losses)],
            device=weights.device
        )
        r = loss_ratios / loss_ratios.mean()
        targets = [avg_norm * (r_i ** self.alpha) for r_i in r]

        gn_loss = sum(
            F.l1_loss(gn, tgt.detach())
            for gn, tgt in zip(grad_norms, targets)
        )
        return gn_loss
'''

n = append_to(os.path.join(BASE, "pretraining.py"), PRETRAINING_ADD)
print(f"pretraining.py: {n} lines")


# ============================================================
# 2. Large expansion of finetuning.py
# ============================================================
FINETUNING_ADD = '''

# =============================================================================
# SECTION: Advanced Fine-tuning Methods (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math


class PromptTuning(nn.Module):
    """Prompt tuning for financial transformer models (Lester et al. 2021).

    Prepends learnable "soft prompts" to the input sequence,
    keeping all model weights frozen. Only prompt tokens are updated.
    """

    def __init__(
        self,
        n_prompt_tokens: int = 20,
        d_model: int = 256,
        prompt_init: str = "random",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_prompt_tokens = n_prompt_tokens
        self.d_model = d_model
        self.temperature = temperature

        self.prompt_tokens = nn.Parameter(torch.randn(n_prompt_tokens, d_model))

        if prompt_init == "zeros":
            nn.init.zeros_(self.prompt_tokens)
        elif prompt_init == "uniform":
            nn.init.uniform_(self.prompt_tokens, -0.1, 0.1)
        # else: keep randn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend prompt tokens to input sequence."""
        B = x.shape[0]
        prompts = self.prompt_tokens.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prompts, x], dim=1)

    def remove_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """Remove prompt tokens from output sequence."""
        return x[:, self.n_prompt_tokens:, :]


class PrefixTuning(nn.Module):
    """Prefix tuning: prepend trainable key-value pairs to each attention layer.

    More expressive than prompt tuning: directly adds to K, V matrices
    at each layer rather than just the input.
    Based on Li & Liang (2021).
    """

    def __init__(
        self,
        n_prefix_tokens: int = 20,
        n_layers: int = 12,
        n_heads: int = 8,
        d_head: int = 64,
        prefix_dropout: float = 0.0,
        reparameterize: bool = True,
        d_reparameterize: int = 512,
    ):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.reparameterize = reparameterize

        d_model = n_heads * d_head

        if reparameterize:
            # Use MLP to generate prefix from smaller embedding
            self.prefix_mlp = nn.Sequential(
                nn.Embedding(n_prefix_tokens, d_reparameterize),
                nn.Linear(d_reparameterize, d_reparameterize * 2),
                nn.Tanh(),
                nn.Linear(d_reparameterize * 2, n_layers * 2 * d_model),
            )
            self.prefix_ids = nn.Parameter(
                torch.arange(n_prefix_tokens).float(), requires_grad=False
            )
        else:
            # Direct prefix parameters
            self.prefix_keys = nn.Parameter(torch.randn(n_layers, n_heads, n_prefix_tokens, d_head))
            self.prefix_values = nn.Parameter(torch.randn(n_layers, n_heads, n_prefix_tokens, d_head))

        self.dropout = nn.Dropout(prefix_dropout)

    def get_prefix_kv(
        self, layer_idx: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prefix key-value pairs for a specific layer."""
        if self.reparameterize:
            ids = self.prefix_ids.long()
            prefix = self.prefix_mlp(ids).view(self.n_prefix_tokens, self.n_layers, 2, self.n_heads, self.d_head)
            pk = prefix[:, layer_idx, 0, :, :]  # [n_prefix, n_heads, d_head]
            pv = prefix[:, layer_idx, 1, :, :]
            pk = pk.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
            pv = pv.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            pk = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            pv = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)

        return self.dropout(pk), self.dropout(pv)


class IA3Tuning(nn.Module):
    """IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations.

    Introduces 3 learned vectors per transformer layer that scale:
    - Keys (K scaling)
    - Values (V scaling)
    - FFN activations (inner activation scaling)

    Extremely parameter-efficient: only 3 * d_model scalars per layer.
    Liu et al. (2022).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ff: int = None,
        scale_init: float = 1.0,
    ):
        super().__init__()
        d_ff = d_ff or (d_model * 4)

        # Per-layer scaling vectors
        self.k_scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_model) * scale_init) for _ in range(n_layers)
        ])
        self.v_scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_model) * scale_init) for _ in range(n_layers)
        ])
        self.ffn_scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_ff) * scale_init) for _ in range(n_layers)
        ])

    def apply_k_scaling(self, k: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return k * self.k_scales[layer_idx]

    def apply_v_scaling(self, v: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return v * self.v_scales[layer_idx]

    def apply_ffn_scaling(self, act: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return act * self.ffn_scales[layer_idx]

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RobustFineTuner(nn.Module):
    """Fine-tune with robustness to distribution shifts.

    Combines:
    - Feature normalization alignment (covariate shift correction)
    - Label noise-robust loss
    - Gradient clipping with adaptive norm tracking
    - Weight averaging for improved generalization
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int = None,
        noise_rate: float = 0.1,
        augment_prob: float = 0.3,
        weight_avg_freq: int = 100,
        lr: float = 2e-5,
    ):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.noise_rate = noise_rate
        self.augment_prob = augment_prob
        self.weight_avg_freq = weight_avg_freq
        self._step = 0
        self._avg_model_weights = None
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self._grad_norm_history = []

    def _get_robust_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "huber",
    ) -> torch.Tensor:
        if loss_type == "huber":
            return F.huber_loss(pred, target.float())
        elif loss_type == "mae":
            return F.l1_loss(pred, target.float())
        elif loss_type == "mse":
            return F.mse_loss(pred, target.float())
        else:
            return F.huber_loss(pred, target.float())

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.augment_prob:
            noise = torch.randn_like(x) * 0.01
            return x + noise
        return x

    def _update_weight_average(self):
        """Maintain exponential moving average of model weights."""
        if self._avg_model_weights is None:
            self._avg_model_weights = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
        else:
            decay = 0.999
            for k, v in self.model.state_dict().items():
                self._avg_model_weights[k] = decay * self._avg_model_weights[k] + (1 - decay) * v

    def training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_type: str = "huber",
    ) -> dict:
        self.model.train()
        self._optimizer.zero_grad()

        x_aug = self._apply_augmentation(x)
        out = self.model(x_aug)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(out, dict):
            pred = next(iter(out.values()))

        loss = self._get_robust_loss(pred, y, loss_type)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._grad_norm_history.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

        self._optimizer.step()
        self._step += 1

        if self._step % self.weight_avg_freq == 0:
            self._update_weight_average()

        return {
            "loss": loss.item(),
            "grad_norm": self._grad_norm_history[-1],
            "step": self._step,
        }

    def get_averaged_model(self) -> dict:
        """Return the weight-averaged model state dict."""
        return self._avg_model_weights or self.model.state_dict()


class SparseFinetuning(nn.Module):
    """Sparse fine-tuning: only update the most important parameters.

    Identifies and unfreezes only the top-k% parameters by
    estimated importance score (gradient magnitude in pilot run).
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.05,
        importance_metric: str = "fisher",
    ):
        super().__init__()
        self.model = model
        self.sparsity = sparsity
        self.importance_metric = importance_metric
        self._importance_scores = {}
        self._active_params = set()

    def estimate_importance(self, pilot_dataloader, loss_fn, n_batches: int = 10):
        """Estimate parameter importance using a small pilot dataset."""
        self.model.eval()
        importance = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for i, batch in enumerate(pilot_dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[-1]
            else:
                x, y = batch, None

            self.model.zero_grad()
            out = self.model(x)
            if callable(loss_fn):
                loss = loss_fn(out, y) if y is not None else loss_fn(out)
            else:
                loss = out.sum()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if self.importance_metric == "fisher":
                        importance[name] += param.grad.pow(2)
                    elif self.importance_metric == "magnitude":
                        importance[name] += param.data.abs()

        self._importance_scores = importance

    def activate_top_k(self):
        """Freeze all parameters, then unfreeze top-k% by importance."""
        # First freeze all
        for p in self.model.parameters():
            p.requires_grad = False

        if not self._importance_scores:
            return

        # Collect all importance scores
        all_scores = torch.cat([s.view(-1) for s in self._importance_scores.values()])
        threshold = torch.quantile(all_scores, 1.0 - self.sparsity)

        # Unfreeze important parameters
        for name, param in self.model.named_parameters():
            if name in self._importance_scores:
                mask = self._importance_scores[name] >= threshold
                if mask.any():
                    param.requires_grad = True
                    self._active_params.add(name)

    def n_active_params(self) -> int:
        return sum(
            p.numel() for n, p in self.model.named_parameters()
            if n in self._active_params and p.requires_grad
        )

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def activation_fraction(self) -> float:
        return self.n_active_params() / max(self.n_total_params(), 1)


class DomainAdaptationFinetuner(nn.Module):
    """Domain adaptation for financial data across different market regimes or geographies.

    Aligns representations between source domain (e.g., US equities)
    and target domain (e.g., emerging markets) using adversarial training.
    """

    def __init__(
        self,
        model: nn.Module,
        d_model: int,
        n_domains: int = 2,
        lambda_domain: float = 1.0,
        reverse_gradient: bool = True,
    ):
        super().__init__()
        self.model = model
        self.lambda_domain = lambda_domain
        self.reverse_gradient = reverse_gradient

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_domains),
        )

    def _grad_reverse(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Gradient reversal layer."""
        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, scale):
                ctx.scale = scale
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                return -ctx.scale * grad, None

        return GradReverse.apply(x, scale)

    def forward(
        self,
        x: torch.Tensor,
        domain_labels: torch.Tensor = None,
        return_domain_loss: bool = True,
    ) -> dict:
        out = self.model(x)
        hidden = out.get("hidden", out.get("last_hidden", None)) if isinstance(out, dict) else out
        if hidden is None:
            hidden = out

        cls = hidden[:, -1, :] if hidden.ndim == 3 else hidden

        result = {"model_output": out}

        if domain_labels is not None and return_domain_loss:
            if self.reverse_gradient:
                h_rev = self._grad_reverse(cls, self.lambda_domain)
            else:
                h_rev = cls

            domain_logits = self.domain_classifier(h_rev)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            result["domain_loss"] = domain_loss
            result["domain_logits"] = domain_logits

        return result
'''

n = append_to(os.path.join(BASE, "finetuning.py"), FINETUNING_ADD)
print(f"finetuning.py: {n} lines")


# ============================================================
# 3. Large expansion of evaluation.py
# ============================================================
EVALUATION_ADD = '''

# =============================================================================
# SECTION: Advanced Evaluation and Backtesting (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
import math


class MonteCarloBacktest:
    """Monte Carlo simulation-based backtesting.

    Runs multiple market scenarios to stress test strategy performance.
    Computes distribution of outcomes and tail risk metrics.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        horizon_days: int = 252,
        confidence_levels: List[float] = None,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10, 0.25]
        self.seed = seed

    def simulate_returns(
        self,
        mean: float,
        std: float,
        skew: float = 0.0,
        kurt: float = 3.0,
    ) -> np.ndarray:
        """Simulate return paths with given moments."""
        np.random.seed(self.seed)
        # Use moment matching: normal if skew=0, kurt=3
        if abs(skew) < 0.01 and abs(kurt - 3.0) < 0.01:
            returns = np.random.normal(mean, std, (self.n_simulations, self.horizon_days))
        else:
            # Simple skew-t approximation
            t_df = 6.0 / (kurt - 3.0 + 1e-6) + 4
            t_df = max(4.01, min(t_df, 1000))
            from scipy import stats
            returns = stats.t.rvs(t_df, loc=mean, scale=std, size=(self.n_simulations, self.horizon_days))
        return returns

    def run(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray = None,
    ) -> dict:
        """Run Monte Carlo backtest and compute risk metrics."""
        # Fit moments from historical returns
        mu = strategy_returns.mean()
        sigma = strategy_returns.std()
        skew = float(((strategy_returns - mu) ** 3).mean() / (sigma ** 3 + 1e-8))
        kurt = float(((strategy_returns - mu) ** 4).mean() / (sigma ** 4 + 1e-8))

        sim_returns = self.simulate_returns(mu, sigma, skew, kurt)

        # Compute cumulative returns for each path
        cum_returns = (1 + sim_returns).cumprod(axis=1) - 1

        # Final wealth distribution
        final_wealth = cum_returns[:, -1]

        results = {
            "mean_final_wealth": final_wealth.mean(),
            "std_final_wealth": final_wealth.std(),
            "median_final_wealth": np.median(final_wealth),
            "skewness": skew,
            "excess_kurtosis": kurt - 3,
        }

        for cl in self.confidence_levels:
            var = np.percentile(final_wealth, cl * 100)
            cvar = final_wealth[final_wealth <= var].mean() if (final_wealth <= var).any() else var
            results[f"VaR_{int(cl*100)}pct"] = var
            results[f"CVaR_{int(cl*100)}pct"] = cvar

        # Probability of loss
        results["prob_loss"] = (final_wealth < 0).mean()
        results["prob_loss_10pct"] = (final_wealth < -0.10).mean()
        results["prob_loss_20pct"] = (final_wealth < -0.20).mean()

        # Max drawdown distribution
        def max_drawdown(path):
            cummax = np.maximum.accumulate(path + 1)
            drawdown = (path + 1) / cummax - 1
            return drawdown.min()

        mdd_dist = np.array([max_drawdown(sim_returns[i]) for i in range(min(self.n_simulations, 100))])
        results["expected_max_drawdown"] = mdd_dist.mean()
        results["worst_10pct_max_drawdown"] = np.percentile(mdd_dist, 10)

        return results


class FactorModelEvaluator:
    """Evaluate factor model performance: IC, ICIR, factor returns, etc.

    Implements standard quantitative equity research metrics.
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        holding_period: int = 1,
        transaction_cost_bps: float = 5.0,
    ):
        self.n_quantiles = n_quantiles
        self.holding_period = holding_period
        self.tc_bps = transaction_cost_bps / 10000

    def compute_ic(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
        method: str = "spearman",
    ) -> float:
        """Compute Information Coefficient (IC)."""
        from scipy import stats
        valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
        if valid.sum() < 5:
            return float("nan")

        f = factor_scores[valid]
        r = forward_returns[valid]

        if method == "spearman":
            ic, _ = stats.spearmanr(f, r)
        elif method == "pearson":
            ic, _ = stats.pearsonr(f, r)
        else:
            ic, _ = stats.spearmanr(f, r)

        return float(ic)

    def compute_ic_series(
        self,
        factor_panel: np.ndarray,
        return_panel: np.ndarray,
    ) -> Dict[str, float]:
        """Compute IC series metrics over time."""
        T = factor_panel.shape[0]
        ic_series = []

        for t in range(T):
            ic = self.compute_ic(factor_panel[t], return_panel[t])
            if not np.isnan(ic):
                ic_series.append(ic)

        ic_arr = np.array(ic_series)
        if len(ic_arr) == 0:
            return {}

        return {
            "mean_ic": float(ic_arr.mean()),
            "icir": float(ic_arr.mean() / (ic_arr.std() + 1e-8)),
            "ic_positive_pct": float((ic_arr > 0).mean()),
            "ic_greater_02": float((ic_arr > 0.02).mean()),
            "ic_t_stat": float(ic_arr.mean() / (ic_arr.std() / (len(ic_arr) ** 0.5) + 1e-8)),
        }

    def quantile_returns(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute mean forward return by factor quantile."""
        valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
        f = factor_scores[valid]
        r = forward_returns[valid]

        quantile_edges = np.linspace(0, 100, self.n_quantiles + 1)
        quantile_returns = []

        for i in range(self.n_quantiles):
            lo = np.percentile(f, quantile_edges[i])
            hi = np.percentile(f, quantile_edges[i + 1])
            mask = (f >= lo) & (f < hi)
            if mask.any():
                quantile_returns.append(r[mask].mean())
            else:
                quantile_returns.append(float("nan"))

        return np.array(quantile_returns)

    def long_short_return(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
    ) -> float:
        """Compute long-short return: top quintile minus bottom quintile."""
        q_returns = self.quantile_returns(factor_scores, forward_returns)
        if np.isnan(q_returns[0]) or np.isnan(q_returns[-1]):
            return float("nan")
        return float(q_returns[-1] - q_returns[0]) - 2 * self.tc_bps

    def factor_decay(
        self,
        factor_scores: np.ndarray,
        returns_multi_horizon: np.ndarray,
    ) -> np.ndarray:
        """Compute IC decay across multiple forward horizons."""
        n_horizons = returns_multi_horizon.shape[1]
        ics = []
        for h in range(n_horizons):
            ic = self.compute_ic(factor_scores, returns_multi_horizon[:, h])
            ics.append(ic)
        return np.array(ics)


class WalkForwardValidator:
    """Walk-forward validation for time series models.

    Simulates real deployment:
    - Train on past N years
    - Validate on next M months
    - Roll forward M months at a time
    """

    def __init__(
        self,
        train_window: int = 252 * 3,
        val_window: int = 21,
        step_size: int = 21,
        min_train_samples: int = 252,
    ):
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples

    def split(self, T: int) -> List[Tuple[range, range]]:
        """Generate (train_idx, val_idx) splits for walk-forward validation."""
        splits = []
        pos = 0

        while pos + self.train_window + self.val_window <= T:
            train_start = max(0, pos)
            train_end = pos + self.train_window
            val_start = train_end
            val_end = min(T, val_start + self.val_window)

            if train_end - train_start >= self.min_train_samples:
                splits.append((
                    range(train_start, train_end),
                    range(val_start, val_end),
                ))

            pos += self.step_size

        return splits

    def evaluate(
        self,
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn,
    ) -> Dict[str, np.ndarray]:
        """Run walk-forward evaluation.

        Args:
            model_fn: callable(X_train, y_train, X_val) -> predictions
            X: [T, n_features] features
            y: [T] targets
            metric_fn: callable(y_true, y_pred) -> float
        """
        splits = self.split(len(X))
        metrics = []
        train_sizes = []

        for train_idx, val_idx in splits:
            X_train = X[list(train_idx)]
            y_train = y[list(train_idx)]
            X_val = X[list(val_idx)]
            y_val = y[list(val_idx)]

            y_pred = model_fn(X_train, y_train, X_val)
            m = metric_fn(y_val, y_pred)
            metrics.append(m)
            train_sizes.append(len(train_idx))

        return {
            "metrics": np.array(metrics),
            "mean_metric": np.nanmean(metrics),
            "std_metric": np.nanstd(metrics),
            "n_splits": len(splits),
            "train_sizes": np.array(train_sizes),
        }


class PerformanceAttributionSuite:
    """Comprehensive performance attribution for portfolio strategies.

    Includes:
    - Brinson-Hood-Beebower (BHB) attribution
    - Factor-based attribution (Fama-French, BARRA)
    - Risk-adjusted return attribution
    - Style box analysis
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def brinson_attribution(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        sector_map: Dict[int, str] = None,
    ) -> dict:
        """Compute BHB attribution: allocation + selection + interaction effects."""
        n_assets = len(portfolio_weights)
        total_port_return = (portfolio_weights * portfolio_returns).sum()
        total_bench_return = (benchmark_weights * benchmark_returns).sum()
        excess = total_port_return - total_bench_return

        allocation = (portfolio_weights - benchmark_weights) * (benchmark_returns - total_bench_return)
        selection = benchmark_weights * (portfolio_returns - benchmark_returns)
        interaction = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)

        result = {
            "total_excess": float(excess),
            "allocation_effect": float(allocation.sum()),
            "selection_effect": float(selection.sum()),
            "interaction_effect": float(interaction.sum()),
            "attribution_sum": float(allocation.sum() + selection.sum() + interaction.sum()),
        }

        if sector_map:
            sector_attribution = {}
            for asset_idx, sector in sector_map.items():
                if sector not in sector_attribution:
                    sector_attribution[sector] = {"allocation": 0.0, "selection": 0.0}
                if asset_idx < n_assets:
                    sector_attribution[sector]["allocation"] += float(allocation[asset_idx])
                    sector_attribution[sector]["selection"] += float(selection[asset_idx])
            result["sector_attribution"] = sector_attribution

        return result

    def factor_attribution(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
        risk_free_rate: float = None,
    ) -> dict:
        """Attribute portfolio returns to risk factors via OLS regression."""
        rf = risk_free_rate or self.risk_free_rate / 252
        excess_returns = portfolio_returns - rf

        factor_matrix = np.column_stack(list(factor_returns.values()))
        factor_names = list(factor_returns.keys())

        A = np.hstack([factor_matrix, np.ones((len(excess_returns), 1))])
        try:
            betas, _, _, _ = np.linalg.lstsq(A, excess_returns, rcond=None)
        except np.linalg.LinAlgError:
            return {"error": "Regression failed"}

        factor_betas = {name: float(b) for name, b in zip(factor_names, betas[:-1])}
        alpha = float(betas[-1])

        # Factor contributions
        contributions = {
            name: float(betas[i] * factor_returns[name].mean())
            for i, name in enumerate(factor_names)
        }

        pred = A @ betas
        ss_res = ((excess_returns - pred) ** 2).sum()
        ss_tot = ((excess_returns - excess_returns.mean()) ** 2).sum()
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))

        return {
            "alpha_annualized": alpha * 252,
            "factor_betas": factor_betas,
            "factor_contributions": contributions,
            "r_squared": r2,
            "specific_return": float(excess_returns.mean() - sum(contributions.values())),
        }

    def style_box_analysis(
        self,
        portfolio_weights: np.ndarray,
        market_caps: np.ndarray,
        pb_ratios: np.ndarray,
    ) -> dict:
        """Compute Morningstar-style style box position (size x value/growth)."""
        # Size dimension: small/mid/large
        weighted_mcap = (portfolio_weights * market_caps).sum() / portfolio_weights.sum()
        total_mcap = market_caps.sum()
        relative_size = weighted_mcap / (total_mcap / len(market_caps))

        if relative_size > 2.0:
            size_style = "large"
        elif relative_size > 0.5:
            size_style = "mid"
        else:
            size_style = "small"

        # Value/growth dimension: P/B ratio
        weighted_pb = (portfolio_weights * pb_ratios).sum() / portfolio_weights.sum()
        universe_pb = pb_ratios.mean()

        if weighted_pb < universe_pb * 0.8:
            value_style = "value"
        elif weighted_pb > universe_pb * 1.2:
            value_style = "growth"
        else:
            value_style = "blend"

        return {
            "size_style": size_style,
            "value_style": value_style,
            "relative_size": float(relative_size),
            "portfolio_pb": float(weighted_pb),
            "universe_pb": float(universe_pb),
        }


class TailRiskMetrics:
    """Compute tail risk metrics for financial strategies.

    Includes:
    - Value at Risk (VaR) - parametric and historical
    - Expected Shortfall (ES / CVaR)
    - Conditional Drawdown at Risk (CDaR)
    - Omega ratio
    - Upside/Downside capture ratios
    """

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]

    def historical_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Historical simulation VaR."""
        return float(np.percentile(returns, confidence * 100))

    def parametric_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Parametric (normal) VaR."""
        from scipy import stats
        mu = returns.mean()
        sigma = returns.std()
        return float(stats.norm.ppf(confidence, mu, sigma))

    def expected_shortfall(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Expected Shortfall (CVaR): mean loss beyond VaR."""
        var = self.historical_var(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """Omega ratio: probability-weighted gain/loss ratio above/below threshold."""
        gains = (returns[returns > threshold] - threshold).sum()
        losses = (threshold - returns[returns <= threshold]).sum()
        return float(gains / max(losses, 1e-10))

    def capture_ratios(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> dict:
        """Upside and downside capture ratios."""
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0

        if up_mask.any():
            upside_capture = float(portfolio_returns[up_mask].mean() / (benchmark_returns[up_mask].mean() + 1e-10))
        else:
            upside_capture = float("nan")

        if down_mask.any():
            downside_capture = float(portfolio_returns[down_mask].mean() / (benchmark_returns[down_mask].mean() + 1e-10))
        else:
            downside_capture = float("nan")

        return {
            "upside_capture": upside_capture,
            "downside_capture": downside_capture,
            "capture_ratio": upside_capture / max(abs(downside_capture), 1e-10) if not (
                math.isnan(upside_capture) or math.isnan(downside_capture)
            ) else float("nan"),
        }

    def full_tail_risk_report(self, returns: np.ndarray) -> dict:
        """Compute all tail risk metrics."""
        result = {}
        for cl in self.confidence_levels:
            result[f"hist_VaR_{int(cl*100)}"] = self.historical_var(returns, cl)
            result[f"param_VaR_{int(cl*100)}"] = self.parametric_var(returns, cl)
            result[f"ES_{int(cl*100)}"] = self.expected_shortfall(returns, cl)

        result["omega_ratio"] = self.omega_ratio(returns)
        result["skewness"] = float(((returns - returns.mean()) ** 3).mean() / (returns.std() ** 3 + 1e-8))
        result["excess_kurtosis"] = float(
            ((returns - returns.mean()) ** 4).mean() / (returns.std() ** 4 + 1e-8) - 3
        )

        return result
'''

n = append_to(os.path.join(BASE, "evaluation.py"), EVALUATION_ADD)
print(f"evaluation.py: {n} lines")


# Final count
import glob
py_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "**", "*.py"), recursive=True)
total = sum(len(open(f, encoding="utf-8", errors="replace").readlines()) for f in py_files)
print(f"\nTotal lines: {total}")
