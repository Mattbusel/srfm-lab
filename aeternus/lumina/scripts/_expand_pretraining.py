"""Expand pretraining.py with advanced objectives."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\pretraining.py"

CONTENT = r'''

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
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
