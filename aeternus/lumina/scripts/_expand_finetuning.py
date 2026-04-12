"""Expand finetuning.py with advanced fine-tuning components."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\finetuning.py"

CONTENT = r'''

# =============================================================================
# SECTION: Advanced Fine-Tuning Strategies
# =============================================================================

class LayerwiseLearningRateDecay:
    """Layer-wise learning rate decay for transformer fine-tuning.

    Multiplies learning rate by `decay_factor` for each transformer layer
    closer to the input. This prevents catastrophic forgetting by
    updating lower layers more conservatively.

    Args:
        optimizer: PyTorch optimizer
        model: Transformer model with named layers
        decay_factor: LR multiplier per layer from output to input
        base_lr: Base (top-layer) learning rate
    """

    def __init__(
        self,
        optimizer,
        model: nn.Module,
        decay_factor: float = 0.8,
        base_lr: float = 1e-4,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.decay_factor = decay_factor
        self.base_lr = base_lr
        self._apply_llrd()

    def _apply_llrd(self) -> None:
        """Apply layer-wise learning rate decay to optimizer param groups."""
        # Collect named layer groups
        named_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.Linear)):
                named_layers.append(name)

        for i, group in enumerate(self.optimizer.param_groups):
            # Exponential decay: deeper layers get lower LR
            depth = max(0, len(named_layers) - 1 - i)
            lr = self.base_lr * (self.decay_factor ** depth)
            group["lr"] = lr


class GradualUnfreezing:
    """Gradually unfreeze transformer layers during fine-tuning.

    Starts with only the top layers trainable, then progressively
    unfreezes lower layers as training proceeds. Prevents early
    overfitting of the task-specific head.

    Args:
        model: Model with attribute 'layers' (ModuleList)
        total_steps: Total fine-tuning steps
        num_stages: Number of unfreezing stages
        initial_layers: Number of layers initially trainable (from top)
    """

    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        num_stages: int = 4,
        initial_layers: int = 1,
    ) -> None:
        self.model = model
        self.total_steps = total_steps
        self.num_stages = num_stages
        self.initial_layers = initial_layers
        self._step = 0
        self._current_unfrozen = initial_layers
        if hasattr(model, "layers"):
            self._num_layers = len(model.layers)
        else:
            self._num_layers = 0
        self._apply_freeze(self._num_layers - initial_layers)

    def _apply_freeze(self, freeze_below: int) -> None:
        """Freeze layers below index freeze_below."""
        if not hasattr(self.model, "layers"):
            return
        for i, layer in enumerate(self.model.layers):
            frozen = i < freeze_below
            for param in layer.parameters():
                param.requires_grad = not frozen

    def step(self) -> bool:
        """Update step; return True if unfreezing occurred."""
        self._step += 1
        stage = int(self._step / self.total_steps * self.num_stages)
        new_unfrozen = min(
            self._num_layers,
            self.initial_layers + stage * max(1, (self._num_layers - self.initial_layers) // self.num_stages)
        )
        if new_unfrozen != self._current_unfrozen:
            self._current_unfrozen = new_unfrozen
            self._apply_freeze(self._num_layers - new_unfrozen)
            return True
        return False

    def get_num_trainable(self) -> int:
        return sum(p.requires_grad for p in self.model.parameters())


class EWCRegularizer(nn.Module):
    """Elastic Weight Consolidation for continual learning.

    EWC prevents catastrophic forgetting by adding a regularization
    term that penalizes changes to weights that were important for
    a previous task.

    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
    in neural networks" PNAS 2017.

    Args:
        model: Neural network model
        ewc_lambda: Strength of the EWC regularization
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0) -> None:
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._params_star: Optional[Dict[str, torch.Tensor]] = None

    def compute_fisher(
        self,
        dataloader,
        loss_fn,
        num_samples: int = 200,
    ) -> None:
        """Compute diagonal Fisher Information Matrix.

        Args:
            dataloader: Dataset to compute Fisher on
            loss_fn: Loss function callable(model, batch) -> loss
            num_samples: Maximum samples to use
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self._params_star = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        n_processed = 0
        for batch in dataloader:
            if n_processed >= num_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            n_processed += 1

        self._fisher = {n: f / max(1, n_processed) for n, f in fisher.items()}

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization term."""
        if self._fisher is None or self._params_star is None:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._params_star[n]).pow(2)).sum()
        return self.ewc_lambda / 2 * loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TaskVectorFineTuner(nn.Module):
    """Task vector arithmetic for model merging and fine-tuning.

    Task vectors = (fine-tuned params - pretrained params).
    These can be added, subtracted, and scaled to combine capabilities.

    Reference: Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023)

    Args:
        pretrained_model: Base pretrained model
        scale: Scaling factor for task vectors when applying
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.pretrained_params = {
            n: p.data.clone() for n, p in pretrained_model.named_parameters()
        }
        self.scale = scale

    def compute_task_vector(
        self,
        finetuned_model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute task vector = finetuned - pretrained.

        Args:
            finetuned_model: Fine-tuned model
        Returns:
            Dict of parameter differences
        """
        tv = {}
        for n, p in finetuned_model.named_parameters():
            if n in self.pretrained_params:
                tv[n] = p.data - self.pretrained_params[n]
        return tv

    def apply_task_vector(
        self,
        model: nn.Module,
        task_vector: Dict[str, torch.Tensor],
        scale: Optional[float] = None,
    ) -> None:
        """Add scaled task vector to model parameters.

        Args:
            model: Model to modify in-place
            task_vector: Dict of parameter differences
            scale: Override instance scale if provided
        """
        s = scale if scale is not None else self.scale
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in task_vector:
                    p.data.add_(s * task_vector[n])

    def merge_task_vectors(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge multiple task vectors with weighted sum.

        Args:
            task_vectors: List of task vector dicts
            weights: Optional per-vector weights (normalized to sum=1)
        Returns:
            Merged task vector
        """
        if weights is None:
            weights = [1.0 / len(task_vectors)] * len(task_vectors)
        merged = {}
        for tv, w in zip(task_vectors, weights):
            for n, delta in tv.items():
                if n not in merged:
                    merged[n] = torch.zeros_like(delta)
                merged[n] = merged[n] + w * delta
        return merged


class MultiDomainFineTuner(nn.Module):
    """Fine-tune on multiple financial domains simultaneously.

    Manages separate task heads and loss functions for different
    financial prediction tasks, with shared backbone.

    Supported task types:
    - regression: MSE/Huber loss
    - classification: Cross-entropy
    - ranking: Pairwise ranking loss
    - quantile: Pinball/quantile loss

    Args:
        backbone: Shared transformer backbone
        task_configs: List of task configurations
        backbone_lr_scale: LR scale for backbone vs heads
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_configs: List[Dict],
        backbone_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_lr_scale = backbone_lr_scale
        d_model = task_configs[0].get("d_model", 512)

        self.task_heads = nn.ModuleDict()
        self.task_types = {}
        self.task_weights = {}

        for cfg in task_configs:
            name = cfg["name"]
            task_type = cfg["type"]
            output_dim = cfg.get("output_dim", 1)
            weight = cfg.get("weight", 1.0)
            self.task_types[name] = task_type
            self.task_weights[name] = weight
            if task_type in ("regression", "ranking"):
                self.task_heads[name] = nn.Linear(d_model, output_dim)
            elif task_type == "classification":
                self.task_heads[name] = nn.Linear(d_model, output_dim)
            elif task_type == "quantile":
                num_quantiles = cfg.get("num_quantiles", 9)
                self.task_heads[name] = nn.Linear(d_model, output_dim * num_quantiles)
                cfg["num_quantiles"] = num_quantiles
        self.task_configs = {cfg["name"]: cfg for cfg in task_configs}

    def _compute_loss(
        self,
        name: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        cfg: Dict,
    ) -> torch.Tensor:
        task_type = self.task_types[name]
        if task_type == "regression":
            return F.huber_loss(pred.squeeze(-1), target)
        elif task_type == "classification":
            return F.cross_entropy(pred, target.long())
        elif task_type == "ranking":
            # Pairwise ranking loss
            B = pred.size(0)
            if B < 2:
                return F.huber_loss(pred.squeeze(-1), target)
            diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
            diff_tgt = (target.unsqueeze(1) - target.unsqueeze(0)).sign()
            ranking_loss = F.relu(1 - diff_pred.squeeze(-1) * diff_tgt)
            mask = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1)
            return (ranking_loss * mask).sum() / (mask.sum() + 1e-6)
        elif task_type == "quantile":
            nq = cfg.get("num_quantiles", 9)
            qs = torch.linspace(0.05, 0.95, nq, device=pred.device)
            pred = pred.view(pred.size(0), -1, nq)
            target_exp = target.unsqueeze(-1).expand_as(pred)
            errors = target_exp - pred
            loss = torch.max((qs - 1) * errors, qs * errors).mean()
            return loss
        return F.huber_loss(pred.squeeze(-1), target)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, T, D) or (B, T, num_features)
            targets: Dict of {task_name: target_tensor}
        Returns:
            Dict with 'predictions', 'losses', 'total_loss'
        """
        # Encode
        if hasattr(self.backbone, "encode"):
            h = self.backbone.encode(x)
        else:
            h = self.backbone(x)

        # Pool to sequence-level if needed
        if h.dim() == 3:
            h = h.mean(dim=1)  # (B, D)

        predictions = {}
        losses = {}

        for name, head in self.task_heads.items():
            pred = head(h)
            predictions[name] = pred
            if targets is not None and name in targets:
                cfg = self.task_configs[name]
                losses[name] = self._compute_loss(name, pred, targets[name], cfg)

        total_loss = torch.tensor(0.0, device=h.device)
        for name, loss in losses.items():
            total_loss = total_loss + self.task_weights[name] * loss

        return {
            "predictions": predictions,
            "losses": losses,
            "total_loss": total_loss,
        }

    def get_optimizer_params(self, head_lr: float = 1e-4) -> List[Dict]:
        """Return parameter groups with different LRs for backbone and heads."""
        return [
            {"params": self.backbone.parameters(), "lr": head_lr * self.backbone_lr_scale},
            {"params": self.task_heads.parameters(), "lr": head_lr},
        ]


class InformationCoefficientOptimizer(nn.Module):
    """IC-based loss for direct optimization of information coefficient.

    Instead of MSE, optimizes for high rank correlation between
    predicted and realized returns, which is the standard
    quantitative finance performance metric.

    IC = Spearman rank correlation between predictions and realized returns.
    ICIR = IC / std(IC) over time.

    Args:
        d_model: Model dimension
        smoothing: Smooth the Spearman correlation approximation
        ic_window: Window for rolling IC computation
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.001,
        ic_window: int = 20,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.ic_window = ic_window
        # Alpha signal prediction head
        self.pred_head = nn.Linear(d_model, 1)
        # Rolling IC tracking
        self._ic_history: List[float] = []

    def soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable soft rank approximation.

        Uses a smooth approximation to the ranking function
        to enable gradient flow.

        Args:
            x: Input tensor (N,)
        Returns:
            Approximate ranks (N,) in [0, 1]
        """
        N = x.size(0)
        # Pairwise differences
        diffs = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N)
        # Smooth step function: sigmoid(diff / smoothing)
        smooth_ranks = torch.sigmoid(diffs / self.smoothing).sum(dim=1)
        # Normalize to [0, 1]
        return smooth_ranks / N

    def spearman_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - Spearman correlation as loss.

        Args:
            pred: Predicted scores (N,)
            target: Realized returns (N,)
        Returns:
            Loss = 1 - SpearmanCorr
        """
        pred_rank = self.soft_rank(pred)
        tgt_rank = self.soft_rank(target)
        # Pearson correlation of ranks = Spearman correlation
        pred_c = pred_rank - pred_rank.mean()
        tgt_c = tgt_rank - tgt_rank.mean()
        corr = (pred_c * tgt_c).sum() / (
            torch.sqrt((pred_c ** 2).sum() * (tgt_c ** 2).sum()) + 1e-8
        )
        return 1.0 - corr

    def forward(
        self,
        h: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: Representations (B, D) for cross-sectional prediction
            returns: Forward returns (B,)
        Returns:
            Dict with 'loss', 'pred', 'ic'
        """
        pred = self.pred_head(h).squeeze(-1)  # (B,)
        loss = self.spearman_loss(pred, returns)
        with torch.no_grad():
            # Compute true Pearson IC
            pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
            ret_z = (returns - returns.mean()) / (returns.std() + 1e-8)
            ic = (pred_z * ret_z).mean().item()
            self._ic_history.append(ic)
            if len(self._ic_history) > self.ic_window:
                self._ic_history.pop(0)
            icir = (sum(self._ic_history) / len(self._ic_history)) / (
                max(1e-8, (sum((x - sum(self._ic_history)/len(self._ic_history))**2
                              for x in self._ic_history) / max(1, len(self._ic_history)-1)) ** 0.5)
            )
        return {"loss": loss, "pred": pred, "ic": ic, "icir": icir}


class LongShortPortfolioHead(nn.Module):
    """Portfolio construction head for long/short equity strategies.

    Takes cross-sectional model predictions and constructs:
    - Top-N long portfolio (highest predicted returns)
    - Bottom-N short portfolio (lowest predicted returns)
    - Dollar-neutral weighting
    - Optional risk parity weighting

    Args:
        d_model: Model dimension
        num_long: Number of long positions
        num_short: Number of short positions
        risk_parity: Whether to use risk parity weighting
        volatility_lookback: Lookback for realized vol estimation
    """

    def __init__(
        self,
        d_model: int,
        num_long: int = 20,
        num_short: int = 20,
        risk_parity: bool = False,
        volatility_lookback: int = 20,
    ) -> None:
        super().__init__()
        self.num_long = num_long
        self.num_short = num_short
        self.risk_parity = risk_parity
        self.volatility_lookback = volatility_lookback
        self.alpha_head = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
        volatilities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: Asset representations (N, D) for N cross-sectional assets
            volatilities: Optional realized vols (N,) for risk parity
        Returns:
            Dict with 'alpha_scores', 'weights', 'long_idx', 'short_idx'
        """
        N, D = h.shape
        alpha = self.alpha_head(h).squeeze(-1)  # (N,)

        # Rank assets
        ranks = alpha.argsort(descending=True)
        long_idx = ranks[:self.num_long]
        short_idx = ranks[-self.num_short:]

        # Construct weights
        if self.risk_parity and volatilities is not None:
            long_vols = volatilities[long_idx]
            short_vols = volatilities[short_idx]
            long_weights = 1.0 / (long_vols + 1e-8)
            short_weights = 1.0 / (short_vols + 1e-8)
            long_weights = long_weights / long_weights.sum()
            short_weights = short_weights / short_weights.sum()
        else:
            long_weights = torch.ones(self.num_long, device=h.device) / self.num_long
            short_weights = torch.ones(self.num_short, device=h.device) / self.num_short

        # Full weight vector (dollar neutral)
        weights = torch.zeros(N, device=h.device)
        weights[long_idx] = long_weights
        weights[short_idx] = -short_weights

        return {
            "alpha_scores": alpha,
            "weights": weights,
            "long_idx": long_idx,
            "short_idx": short_idx,
            "long_weights": long_weights,
            "short_weights": short_weights,
        }


class CalibratedReturnForecaster(nn.Module):
    """Return forecaster with calibrated uncertainty estimates.

    Produces point forecasts and calibrated prediction intervals.
    Calibration is achieved via conformal prediction or temperature
    scaling.

    Args:
        d_model: Model dimension
        num_horizons: Number of forecast horizons
        quantiles: Quantile levels for interval prediction
        conformal_calib: Use conformal prediction for calibration
    """

    def __init__(
        self,
        d_model: int,
        num_horizons: int = 5,
        quantiles: Optional[List[float]] = None,
        conformal_calib: bool = False,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.conformal_calib = conformal_calib
        nq = len(self.quantiles)
        # Point forecast head
        self.point_head = nn.Linear(d_model, num_horizons)
        # Quantile head
        self.quantile_head = nn.Linear(d_model, num_horizons * nq)
        # Calibration temperature (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        # Conformal calibration (stored, not learned)
        self._conformal_alpha: Optional[torch.Tensor] = None

    def quantile_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Pinball loss for quantile regression.

        Args:
            pred: (B, H, Q) predicted quantiles
            target: (B, H) realized returns
        Returns:
            Scalar loss
        """
        q_tensor = torch.tensor(self.quantiles, device=pred.device, dtype=pred.dtype)
        target_exp = target.unsqueeze(-1).expand_as(pred)
        errors = target_exp - pred
        loss = torch.max((q_tensor - 1) * errors, q_tensor * errors)
        return loss.mean()

    def set_conformal_calibration(
        self,
        calibration_scores: torch.Tensor,
        alpha: float = 0.1,
    ) -> None:
        """Set conformal prediction calibration scores.

        Args:
            calibration_scores: Nonconformity scores on calibration set
            alpha: Desired miscoverage rate (1-alpha = coverage)
        """
        n = len(calibration_scores)
        level = int(math.ceil((n + 1) * (1 - alpha)) / n * n)
        level = min(level, n - 1)
        sorted_scores, _ = calibration_scores.sort()
        self._conformal_alpha = sorted_scores[level]

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (B, D) or (B, T, D) representations
            targets: (B, H) optional ground truth returns
        Returns:
            Dict with 'point_pred', 'quantile_pred', 'loss' (if targets provided)
        """
        if h.dim() == 3:
            h = h.mean(dim=1)  # Pool
        B, D = h.shape
        nq = len(self.quantiles)

        point_pred = self.point_head(h)  # (B, H)
        quantile_pred = self.quantile_head(h).view(B, self.num_horizons, nq)

        # Temperature scaling
        quantile_pred = quantile_pred / self.temperature

        out = {
            "point_pred": point_pred,
            "quantile_pred": quantile_pred,
            "temperature": self.temperature.item(),
        }

        if targets is not None:
            point_loss = F.huber_loss(point_pred, targets)
            q_loss = self.quantile_loss(quantile_pred, targets)
            out["loss"] = point_loss + q_loss
            out["point_loss"] = point_loss
            out["quantile_loss"] = q_loss

        return out


class AdversarialRobustnessFinetuner(nn.Module):
    """Adversarial training for robust financial models.

    Adds adversarial examples during training to improve robustness
    to distribution shift (market regime changes, data quality issues).

    Methods:
    - FGSM: Fast Gradient Sign Method (single step)
    - PGD: Projected Gradient Descent (multi-step)
    - FreeAT: Free Adversarial Training (reuse gradient from last step)

    Args:
        model: Financial model to harden
        epsilon: Adversarial perturbation budget
        alpha: Step size for PGD
        num_steps: Number of PGD steps
        method: 'fgsm', 'pgd', or 'free'
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_steps: int = 7,
        method: str = "pgd",
    ) -> None:
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.method = method

    def fgsm_attack(
        self,
        x: torch.Tensor,
        loss_fn,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """FGSM attack: x_adv = x + epsilon * sign(gradient).

        Args:
            x: Input tensor (B, T, C)
            loss_fn: Loss callable(output, target)
            target: Ground truth tensor
        Returns:
            Adversarial examples (B, T, C)
        """
        x_adv = x.clone().detach().requires_grad_(True)
        output = self.model(x_adv)
        loss = loss_fn(output, target)
        loss.backward()
        with torch.no_grad():
            x_adv = x + self.epsilon * x_adv.grad.sign()
        return x_adv.detach()

    def pgd_attack(
        self,
        x: torch.Tensor,
        loss_fn,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """PGD attack: iterative FGSM with projection.

        Args:
            x: Input tensor (B, T, C)
            loss_fn: Loss callable
            target: Ground truth
        Returns:
            Adversarial examples (B, T, C)
        """
        x_adv = x.clone().detach()
        # Random initialization within epsilon ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)

        for _ in range(self.num_steps):
            x_adv = x_adv.requires_grad_(True)
            output = self.model(x_adv)
            loss = loss_fn(output, target)
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project back to epsilon ball
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = x + delta
                x_adv = x_adv.detach()
        return x_adv

    def forward(
        self,
        x: torch.Tensor,
        loss_fn,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute adversarial training loss.

        Returns:
            Dict with 'clean_loss', 'adv_loss', 'total_loss', 'adv_x'
        """
        # Clean forward pass
        clean_out = self.model(x)
        clean_loss = loss_fn(clean_out, target)

        # Generate adversarial examples
        if self.method == "fgsm":
            x_adv = self.fgsm_attack(x, loss_fn, target)
        elif self.method == "pgd":
            x_adv = self.pgd_attack(x, loss_fn, target)
        else:
            x_adv = self.fgsm_attack(x, loss_fn, target)

        # Adversarial forward pass
        adv_out = self.model(x_adv)
        adv_loss = loss_fn(adv_out, target)

        total_loss = (clean_loss + adv_loss) / 2

        return {
            "clean_loss": clean_loss,
            "adv_loss": adv_loss,
            "total_loss": total_loss,
            "adv_x": x_adv,
            "clean_out": clean_out,
            "adv_out": adv_out,
        }


_NEW_FINETUNING_EXPORTS = [
    "LayerwiseLearningRateDecay", "GradualUnfreezing", "EWCRegularizer",
    "TaskVectorFineTuner", "MultiDomainFineTuner", "InformationCoefficientOptimizer",
    "LongShortPortfolioHead", "CalibratedReturnForecaster", "AdversarialRobustnessFinetuner",
]
'''

import math  # noqa for the content above

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
