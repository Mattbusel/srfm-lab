"""
lumina/finetuning.py

Fine-tuning pipelines for Lumina Financial Foundation Model:

  - DirectionClassificationHead   : up/down/flat prediction
  - VolatilityForecastingHead     : realized vol prediction
  - RegimePredictionHead          : bull/bear/sideways classification
  - PortfolioOptimizationHead     : direct portfolio weight output
  - FineTuningConfig              : configuration
  - FineTuner                     : universal fine-tuning trainer
  - TaskSpecificLoss              : per-task loss functions
  - FineTuningDataset             : labeled dataset wrapper
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FineTuningConfig:
    # Task
    task:               str   = "direction"   # "direction" | "volatility" | "regime" | "portfolio" | "return"
    n_classes:          int   = 3             # for classification: up/down/flat
    horizon:            int   = 5             # prediction horizon in bars
    n_assets:           int   = 1

    # Architecture
    d_model:            int   = 512
    use_lora:           bool  = True
    lora_r:             int   = 16
    lora_alpha:         float = 32.0
    freeze_backbone:    bool  = True
    unfreeze_last_n:    int   = 2             # unfreeze last N transformer layers

    # Training
    lr:                 float = 1e-4
    weight_decay:       float = 0.01
    warmup_ratio:       float = 0.1
    max_epochs:         int   = 20
    batch_size:         int   = 64
    dropout:            float = 0.1
    label_smoothing:    float = 0.1
    max_grad_norm:      float = 1.0

    # Evaluation
    eval_metric:        str   = "accuracy"
    early_stopping:     bool  = True
    patience:           int   = 5

    # Portfolio
    portfolio_mode:     str   = "long_only"   # "long_only" | "long_short" | "softmax"
    transaction_cost:   float = 0.001
    risk_aversion:      float = 1.0


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class DirectionClassificationHead(nn.Module):
    """
    Predicts price direction: up / down / flat (3-class).
    Uses CLS token embedding from transformer.
    """

    def __init__(
        self,
        d_model:   int,
        n_classes: int   = 3,
        dropout:   float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.n_classes = n_classes
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """cls_emb: (B, D) → (B, n_classes) logits."""
        return self.net(cls_emb)

    def predict(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.forward(cls_emb).argmax(dim=-1)

    def predict_proba(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return F.softmax(self.forward(cls_emb), dim=-1)


class MultiHorizonDirectionHead(nn.Module):
    """Direction classifier that simultaneously predicts at multiple horizons."""

    def __init__(self, d_model: int, horizons: List[int], n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.horizons = horizons
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(d_model // 2, n_classes) for _ in horizons
        ])

    def forward(self, cls_emb: torch.Tensor) -> Dict[int, torch.Tensor]:
        shared = self.shared(cls_emb)
        return {h: head(shared) for h, head in zip(self.horizons, self.heads)}


class VolatilityForecastingHead(nn.Module):
    """
    Predicts next-period realized volatility.
    Returns a positive scalar (annualized vol estimate).
    """

    def __init__(
        self,
        d_model:       int,
        dropout:       float = 0.1,
        predict_dist:  bool  = False,   # if True, predict (mean, std) of vol distribution
    ):
        super().__init__()
        self.predict_dist = predict_dist
        out_dim = 2 if predict_dist else 1

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        if not predict_dist:
            self.output_act = nn.Softplus()
        else:
            self.output_act = nn.Identity()

    def forward(self, cls_emb: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.net(cls_emb)
        if self.predict_dist:
            mu  = F.softplus(out[:, 0])
            std = F.softplus(out[:, 1]) + 1e-6
            return mu, std
        return self.output_act(out).squeeze(-1)


class RealizedVolLoss(nn.Module):
    """Loss functions for volatility forecasting."""

    def __init__(self, mode: str = "mse"):
        super().__init__()
        self.mode = mode

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "mse":
            return F.mse_loss(pred, target)
        elif self.mode == "qlike":
            # QLIKE loss: common in volatility forecasting
            return (target / (pred + 1e-8) - torch.log(target / (pred + 1e-8)) - 1).mean()
        elif self.mode == "log_mse":
            return F.mse_loss(torch.log(pred + 1e-8), torch.log(target + 1e-8))
        elif self.mode == "mae":
            return F.l1_loss(pred, target)
        else:
            return F.mse_loss(pred, target)


class RegimeClassificationHead(nn.Module):
    """
    Multi-class market regime classifier.

    Regimes: bull / bear / sideways / high-vol
    """

    def __init__(self, d_model: int, n_regimes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_regimes),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.net(cls_emb)


class ReturnPredictionHead(nn.Module):
    """Predicts raw return (or z-scored return) as a scalar."""

    def __init__(
        self,
        d_model:  int,
        dropout:  float = 0.1,
        n_output: int   = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_output),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.net(cls_emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Portfolio Optimization Head
# ---------------------------------------------------------------------------

class PortfolioOptimizationHead(nn.Module):
    """
    Outputs portfolio weights directly from the model.

    Supports:
      - Long-only (softmax weights summing to 1)
      - Long-short (tanh, then rescale)
      - Mean-variance efficient frontier (differentiable)
    """

    def __init__(
        self,
        d_model:     int,
        n_assets:    int,
        mode:        str   = "long_only",
        dropout:     float = 0.1,
        risk_aversion: float = 1.0,
    ):
        super().__init__()
        self.n_assets     = n_assets
        self.mode         = mode
        self.risk_aversion = risk_aversion

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_assets),
        )

    def _apply_constraints(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply portfolio weight constraints."""
        if self.mode == "long_only":
            return F.softmax(raw, dim=-1)
        elif self.mode == "long_short":
            weights = torch.tanh(raw)
            # Normalize to sum to 0 (dollar-neutral)
            weights = weights - weights.mean(-1, keepdim=True)
            # Scale to unit L1 norm
            weights = weights / (weights.abs().sum(-1, keepdim=True) + 1e-8)
            return weights
        elif self.mode == "softmax":
            return F.softmax(raw, dim=-1)
        else:
            return raw

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """cls_emb: (B, D) → (B, n_assets) portfolio weights."""
        raw = self.net(cls_emb)
        return self._apply_constraints(raw)

    def compute_portfolio_return(
        self,
        weights:     torch.Tensor,   # (B, n_assets)
        asset_returns: torch.Tensor, # (B, n_assets)
    ) -> torch.Tensor:
        """Portfolio return = sum(weights * asset_returns)."""
        return (weights * asset_returns).sum(-1)   # (B,)

    def compute_portfolio_variance(
        self,
        weights:   torch.Tensor,   # (B, n_assets)
        cov_matrix: torch.Tensor,  # (B, n_assets, n_assets)
    ) -> torch.Tensor:
        """Portfolio variance = w^T Sigma w."""
        return torch.bmm(weights.unsqueeze(1),
                         torch.bmm(cov_matrix, weights.unsqueeze(2))).squeeze(-1).squeeze(-1)


class SharpeMaximizationLoss(nn.Module):
    """
    Loss function that maximizes negative Sharpe ratio.
    L = -E[R_p] / std(R_p) where R_p = portfolio return.
    """

    def __init__(self, risk_free: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.risk_free = risk_free
        self.eps       = eps

    def forward(
        self,
        portfolio_returns: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        excess  = portfolio_returns - self.risk_free
        mean    = excess.mean()
        std     = excess.std().clamp(min=self.eps)
        sharpe  = mean / std
        return -sharpe   # minimize negative Sharpe = maximize Sharpe


class MeanVarianceLoss(nn.Module):
    """
    Mean-variance portfolio optimization loss.
    L = -E[R_p] + lambda * Var(R_p)
    """

    def __init__(self, risk_aversion: float = 1.0):
        super().__init__()
        self.risk_aversion = risk_aversion

    def forward(
        self,
        weights:     torch.Tensor,   # (B, n_assets)
        returns:     torch.Tensor,   # (B, n_assets) future returns
        cov:         Optional[torch.Tensor] = None,  # (B, n_assets, n_assets)
    ) -> torch.Tensor:
        port_ret = (weights * returns).sum(-1)   # (B,)
        mean_ret = port_ret.mean()

        if cov is not None:
            port_var = torch.bmm(
                weights.unsqueeze(1), torch.bmm(cov, weights.unsqueeze(2))
            ).squeeze(-1).squeeze(-1).mean()
        else:
            port_var = port_ret.var()

        return -mean_ret + self.risk_aversion * port_var


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, n_steps: int = 1000) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=n_steps)
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        return F.cross_entropy(self.forward(logits), labels).item()


# ---------------------------------------------------------------------------
# Task-specific loss functions
# ---------------------------------------------------------------------------

class TaskSpecificLoss(nn.Module):
    """Unified loss function for all fine-tuning tasks."""

    def __init__(self, cfg: FineTuningConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.task == "direction" or cfg.task == "regime":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        elif cfg.task == "volatility":
            self.loss_fn = RealizedVolLoss("qlike")
        elif cfg.task == "return":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        elif cfg.task == "portfolio":
            self.loss_fn = SharpeMaximizationLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = self.loss_fn(pred, target)

        metrics: Dict[str, float] = {"loss": loss.item()}

        if self.cfg.task in ["direction", "regime"]:
            acc = (pred.argmax(-1) == target).float().mean()
            metrics["accuracy"] = acc.item()

        elif self.cfg.task == "volatility":
            mae = F.l1_loss(pred, target)
            metrics["mae"] = mae.item()

        elif self.cfg.task == "portfolio":
            # Assumes target is (B, n_assets) returns
            port_ret = (pred * target).sum(-1).mean()
            metrics["portfolio_return"] = port_ret.item()

        return loss, metrics


# ---------------------------------------------------------------------------
# Fine-Tuning Trainer
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing and class weighting."""

    def __init__(self, n_classes: int, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll       = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Smooth: uniform over all classes
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss

        if self.weight is not None:
            w    = self.weight[targets]
            loss = (loss * w).mean()
        else:
            loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for imbalanced class distributions."""

    def __init__(self, gamma: float = 2.0, n_classes: int = 3):
        super().__init__()
        self.gamma    = gamma
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs      = F.softmax(logits, dim=-1)
        log_probs  = F.log_softmax(logits, dim=-1)
        pt         = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        log_pt     = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        focal_w    = (1 - pt) ** self.gamma
        return -(focal_w * log_pt).mean()


class FineTuner:
    """
    Universal fine-tuning trainer for Lumina.
    Supports classification, regression, and portfolio optimization tasks.
    """

    def __init__(
        self,
        backbone:    nn.Module,
        task_head:   nn.Module,
        cfg:         FineTuningConfig,
        device:      torch.device,
    ):
        self.backbone  = backbone
        self.task_head = task_head
        self.cfg       = cfg
        self.device    = device
        self.loss_fn   = TaskSpecificLoss(cfg)

        # Freeze backbone if requested
        if cfg.freeze_backbone:
            self._freeze_backbone()
        if cfg.unfreeze_last_n > 0:
            self._unfreeze_last_layers(cfg.unfreeze_last_n)

        # Optimizer: only train task head + unfrozen backbone params
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        trainable += list(task_head.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        self.best_val_metric = float('-inf') if cfg.eval_metric in ['accuracy', 'sharpe'] else float('inf')
        self.patience_count  = 0
        self.step            = 0
        self.history: List[Dict] = []

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_layers(self, n: int) -> None:
        # Try to unfreeze the last N transformer blocks
        layers = None
        if hasattr(self.backbone, 'backbone'):
            bb = self.backbone.backbone
        else:
            bb = self.backbone

        if hasattr(bb, 'transformer') and hasattr(bb.transformer, 'layers'):
            layers = bb.transformer.layers
        elif hasattr(bb, 'layers'):
            layers = bb.layers

        if layers is not None:
            for layer in list(layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def train_epoch(
        self,
        loader: Any,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, float]:
        self.backbone.train()
        self.task_head.train()

        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                enc = self.backbone(x)
                if isinstance(enc, dict):
                    cls_emb = enc.get("cls_emb", enc["hidden"][:, 0])
                else:
                    cls_emb = enc

                pred = self.task_head(cls_emb)
                loss, metrics = self.loss_fn(pred, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.task_head.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += metrics["loss"]
            if "accuracy" in metrics:
                total_acc += metrics["accuracy"]
            n_batches  += 1
            self.step  += 1

        epoch_metrics = {
            "train_loss": total_loss / n_batches,
            "train_acc":  total_acc  / n_batches if total_acc > 0 else 0.0,
        }
        self.history.append(epoch_metrics)
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, loader: Any) -> Dict[str, float]:
        self.backbone.eval()
        self.task_head.eval()

        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            enc = self.backbone(x)
            if isinstance(enc, dict):
                cls_emb = enc.get("cls_emb", enc["hidden"][:, 0])
            else:
                cls_emb = enc

            pred = self.task_head(cls_emb)
            loss, _ = self.loss_fn(pred, y)
            total_loss += loss.item()

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        metrics: Dict[str, float] = {"val_loss": total_loss / max(1, len(all_preds))}

        if self.cfg.task in ["direction", "regime"]:
            acc = (preds.argmax(-1) == targets).float().mean().item()
            metrics["val_accuracy"] = acc
            # Per-class accuracy
            for c in range(self.cfg.n_classes):
                mask = targets == c
                if mask.sum() > 0:
                    metrics[f"val_acc_class_{c}"] = (
                        preds[mask].argmax(-1) == targets[mask]
                    ).float().mean().item()

        elif self.cfg.task == "volatility":
            mae  = F.l1_loss(preds.squeeze(), targets).item()
            rmse = ((preds.squeeze() - targets) ** 2).mean().sqrt().item()
            metrics["val_mae"]  = mae
            metrics["val_rmse"] = rmse

        elif self.cfg.task == "portfolio":
            port_ret = (preds * targets).sum(-1).mean().item()
            port_std = (preds * targets).sum(-1).std().item()
            metrics["val_portfolio_return"] = port_ret
            metrics["val_sharpe"] = port_ret / (port_std + 1e-8) * math.sqrt(252)

        return metrics

    def check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Returns True if training should stop."""
        if not self.cfg.early_stopping:
            return False

        metric_key = f"val_{self.cfg.eval_metric}"
        current    = val_metrics.get(metric_key, 0.0)

        improved = (
            current > self.best_val_metric
            if self.cfg.eval_metric in ['accuracy', 'sharpe']
            else current < self.best_val_metric
        )

        if improved:
            self.best_val_metric = current
            self.patience_count  = 0
            return False
        else:
            self.patience_count += 1
            return self.patience_count >= self.cfg.patience

    def save_checkpoint(self, path: str, epoch: int) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch":          epoch,
            "step":           self.step,
            "backbone":       self.backbone.state_dict(),
            "task_head":      self.task_head.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "best_metric":    self.best_val_metric,
            "cfg":            self.cfg,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone"], strict=False)
        self.task_head.load_state_dict(ckpt["task_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_val_metric = ckpt["best_metric"]
        self.step            = ckpt["step"]
        return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Specialized fine-tuning dataset
# ---------------------------------------------------------------------------

class FinancialFineTuningDataset:
    """
    Dataset wrapper for financial fine-tuning.
    Takes raw OHLCV series and computes labels based on future returns.
    """

    def __init__(
        self,
        ohlcv:        np.ndarray,    # (T, 5)
        cfg:          FineTuningConfig,
        timestamps:   Optional[np.ndarray] = None,
        asset_names:  Optional[List[str]]   = None,
    ):
        self.ohlcv       = ohlcv
        self.cfg         = cfg
        self.timestamps  = timestamps
        self.asset_names = asset_names
        self.labels      = self._compute_labels()
        self.window_size = 256   # lookback window

    def _compute_labels(self) -> np.ndarray:
        T       = len(self.ohlcv)
        close   = self.ohlcv[:, 3]
        horizon = self.cfg.horizon

        if self.cfg.task == "direction":
            future_ret = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-8)
            labels     = np.zeros(T, dtype=np.int64)
            labels[:T - horizon][future_ret > 0.005]  = 0  # up
            labels[:T - horizon][future_ret < -0.005] = 1  # down
            labels[:T - horizon][np.abs(future_ret) <= 0.005] = 2  # flat
            return labels

        elif self.cfg.task == "volatility":
            returns   = np.diff(np.log(close + 1e-8))
            vol       = np.zeros(T)
            for t in range(horizon, T):
                vol[t] = returns[max(0, t - horizon):t].std() * np.sqrt(252)
            return vol

        elif self.cfg.task == "return":
            ret = np.zeros(T)
            ret[:T - horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-8)
            return ret

        elif self.cfg.task == "regime":
            returns    = np.diff(np.log(close + 1e-8))
            labels     = np.zeros(T, dtype=np.int64)
            win        = 20
            for t in range(win, T):
                seg = returns[t - win:t]
                cum = seg.sum()
                vol = seg.std()
                if cum > 0.02:
                    labels[t] = 0  # bull
                elif cum < -0.02:
                    labels[t] = 1  # bear
                elif vol > returns[:t].std() * 1.5:
                    labels[t] = 3  # high vol
                else:
                    labels[t] = 2  # sideways
            return labels

        return np.zeros(T)

    def __len__(self) -> int:
        return max(0, len(self.ohlcv) - self.window_size - self.cfg.horizon)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.ohlcv[idx: idx + self.window_size]
        y = self.labels[idx + self.window_size]

        x_t = torch.from_numpy(x).float()
        y_t = (torch.tensor(y).long() if self.cfg.task in ["direction", "regime"]
               else torch.tensor(y).float())

        return {"x": x_t, "y": y_t}


# ---------------------------------------------------------------------------
# IC (Information Coefficient) based evaluation
# ---------------------------------------------------------------------------

def information_coefficient(
    forecasts: torch.Tensor,
    returns:   torch.Tensor,
) -> torch.Tensor:
    """
    Compute Spearman rank correlation (IC) between forecasts and realized returns.
    """
    import scipy.stats as stats
    fc = forecasts.detach().cpu().numpy()
    rt = returns.detach().cpu().numpy()

    ic_values = []
    for t in range(fc.shape[0]):
        if len(fc.shape) == 1:
            ic, _ = stats.spearmanr(fc, rt)
        else:
            ic, _ = stats.spearmanr(fc[t], rt[t])
        ic_values.append(ic)

    return torch.tensor(np.nanmean(ic_values))


def rank_ic(forecasts: np.ndarray, returns: np.ndarray) -> float:
    """Compute mean rank IC over time."""
    from scipy.stats import spearmanr
    T = forecasts.shape[0]
    ics = []
    for t in range(T):
        ic, _ = spearmanr(forecasts[t], returns[t])
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Alpha Signal Generation
# ---------------------------------------------------------------------------

class AlphaSignalHead(nn.Module):
    """Generate trading alpha signals from model hidden states.

    Produces raw alpha scores that can be used for portfolio construction.
    Outputs are in the form of cross-sectional return predictions.

    The signal goes through:
    1. Hidden → raw score (Linear)
    2. Cross-sectional normalization (z-score across assets)
    3. Optional: sigmoid for long-only, tanh for long-short

    Args:
        d_model:       input hidden dimension
        n_assets:      number of assets (for cross-sectional normalization)
        signal_type:   "zscore" | "rank" | "sigmoid" | "tanh"
        dropout:       dropout before head

    Example:
        >>> head = AlphaSignalHead(d_model=512, n_assets=100)
        >>> hidden = torch.randn(4, 100, 512)  # (B, n_assets, d_model)
        >>> signals = head(hidden)  # (4, 100) cross-sectional alphas
    """

    def __init__(
        self,
        d_model: int,
        n_assets: int = 100,
        signal_type: str = "zscore",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets
        self.signal_type = signal_type

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def _normalize_signal(self, raw: torch.Tensor) -> torch.Tensor:
        """Normalize raw signal cross-sectionally."""
        if self.signal_type == "zscore":
            mu = raw.mean(dim=-1, keepdim=True)
            std = raw.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (raw - mu) / std
        elif self.signal_type == "rank":
            # Rank normalization
            ranks = raw.argsort(dim=-1).argsort(dim=-1).float()
            n = raw.shape[-1]
            return (ranks / (n - 1) - 0.5) * 2  # normalize to [-1, 1]
        elif self.signal_type == "sigmoid":
            return torch.sigmoid(raw)
        elif self.signal_type == "tanh":
            return torch.tanh(raw / raw.std(dim=-1, keepdim=True).clamp(min=1e-6))
        else:
            return raw

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, N, d_model) where N = n_assets

        Returns:
            signals: (B, N) alpha scores
        """
        x = self.dropout(hidden)
        raw = self.proj(x).squeeze(-1)  # (B, N)
        return self._normalize_signal(raw)


# ---------------------------------------------------------------------------
# Risk Parity Portfolio Head
# ---------------------------------------------------------------------------

class RiskParityHead(nn.Module):
    """Portfolio construction via risk parity.

    Estimates per-asset volatility from model outputs and constructs
    a portfolio where each asset contributes equally to total portfolio risk.

    Risk parity weights: w_i ∝ 1 / sigma_i

    Then normalized so sum(w) = 1 (long-only) or constrained long-short.

    Args:
        d_model:     hidden state dimension
        n_assets:    number of assets
        min_weight:  minimum portfolio weight per asset
        max_weight:  maximum portfolio weight per asset
        long_only:   if True, weights are non-negative

    Example:
        >>> rp = RiskParityHead(d_model=512, n_assets=20)
        >>> hidden = torch.randn(4, 20, 512)
        >>> weights, pred_vols = rp(hidden)
        >>> # weights: (4, 20), pred_vols: (4, 20)
    """

    def __init__(
        self,
        d_model: int,
        n_assets: int = 20,
        min_weight: float = 0.0,
        max_weight: float = 0.3,
        long_only: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.long_only = long_only

        # Predict log-volatility
        self.vol_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute risk-parity portfolio weights.

        Args:
            hidden: (B, N, d_model)

        Returns:
            weights:   (B, N) portfolio weights
            pred_vols: (B, N) predicted volatilities
        """
        log_vol = self.vol_head(hidden).squeeze(-1)  # (B, N)
        pred_vols = log_vol.exp()  # (B, N) always positive

        # Risk parity: w_i = 1/sigma_i, then normalize
        inv_vol = 1.0 / (pred_vols + 1e-8)

        if self.long_only:
            weights = inv_vol / inv_vol.sum(dim=-1, keepdim=True)
            # Clip to max weight
            weights = weights.clamp(max=self.max_weight)
            # Renormalize after clipping
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            # Long-short: center at zero
            weights = inv_vol - inv_vol.mean(dim=-1, keepdim=True)
            # Normalize by L1 norm
            weights = weights / (weights.abs().sum(dim=-1, keepdim=True) + 1e-8)

        return weights, pred_vols


# ---------------------------------------------------------------------------
# Multi-Task Finetuning Framework
# ---------------------------------------------------------------------------

class MultiTaskFineTuner(nn.Module):
    """Multi-task fine-tuning framework for financial models.

    Supports simultaneous training on multiple tasks:
    - Direction prediction (classification)
    - Return prediction (regression)
    - Volatility forecasting (regression)
    - Regime classification (classification)
    - Portfolio optimization (differentiable)

    Uses soft parameter sharing (shared backbone, task-specific heads).

    Args:
        backbone:    pre-trained LuminaModel
        tasks:       list of task names to enable
        task_weights:per-task loss weights (None = equal weighting)
        d_model:     backbone hidden dimension
        freeze_backbone_steps: steps to freeze backbone (head warmup)

    Example:
        >>> backbone = LuminaModel(config)
        >>> finetuner = MultiTaskFineTuner(
        ...     backbone=backbone,
        ...     tasks=["direction", "volatility", "regime"],
        ...     d_model=512,
        ... )
        >>> outputs = finetuner(x, targets)
    """

    SUPPORTED_TASKS = {
        "direction",
        "return",
        "volatility",
        "regime",
        "portfolio",
        "alpha",
        "risk_parity",
    }

    def __init__(
        self,
        backbone: nn.Module,
        tasks: List[str],
        task_weights: Optional[Dict[str, float]] = None,
        d_model: int = 512,
        n_classes_direction: int = 3,
        n_classes_regime: int = 5,
        n_assets: int = 20,
        freeze_backbone_steps: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.tasks = tasks
        self.d_model = d_model
        self.freeze_backbone_steps = freeze_backbone_steps
        self._step = 0

        # Validate task names
        for t in tasks:
            if t not in self.SUPPORTED_TASKS:
                raise ValueError(f"Unknown task: {t}. Supported: {self.SUPPORTED_TASKS}")

        # Build task heads
        self.heads = nn.ModuleDict()
        if "direction" in tasks:
            self.heads["direction"] = DirectionClassificationHead(
                d_model, n_classes=n_classes_direction
            )
        if "return" in tasks:
            self.heads["return"] = ReturnPredictionHead(d_model)
        if "volatility" in tasks:
            self.heads["volatility"] = VolatilityForecastingHead(d_model)
        if "regime" in tasks:
            self.heads["regime"] = RegimeClassificationHead(
                d_model, n_classes=n_classes_regime
            )
        if "portfolio" in tasks:
            self.heads["portfolio"] = PortfolioOptimizationHead(d_model, n_assets=n_assets)
        if "alpha" in tasks:
            self.heads["alpha"] = AlphaSignalHead(d_model, n_assets=n_assets)
        if "risk_parity" in tasks:
            self.heads["risk_parity"] = RiskParityHead(d_model, n_assets=n_assets)

        # Task weights
        if task_weights is None:
            self.task_weights = {t: 1.0 for t in tasks}
        else:
            self.task_weights = task_weights

    def _maybe_freeze_backbone(self) -> None:
        """Freeze/unfreeze backbone based on current step."""
        if self._step < self.freeze_backbone_steps:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(
        self,
        token_embeddings: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            token_embeddings: (B, T, d_token) input token embeddings
            targets:          dict of task name → target tensor
            attention_mask:   (B, T) bool
            timestamps:       (B, T) float

        Returns:
            outputs: dict with task predictions and optional losses
        """
        self._step += 1
        self._maybe_freeze_backbone()

        # Backbone forward
        backbone_out = self.backbone(
            token_embeddings,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
        hidden = backbone_out["hidden"]   # (B, T, d_model)
        pooled = backbone_out.get("pooled", hidden.mean(dim=1))  # (B, d_model)

        outputs = {}
        total_loss = torch.tensor(0.0, device=hidden.device)

        for task in self.tasks:
            head = self.heads[task]
            weight = self.task_weights.get(task, 1.0)

            if task == "direction":
                logits = head(pooled)
                outputs["direction_logits"] = logits
                if targets and "direction" in targets:
                    loss = F.cross_entropy(logits, targets["direction"])
                    outputs["direction_loss"] = loss
                    total_loss = total_loss + weight * loss

            elif task == "return":
                pred = head(pooled)
                outputs["return_pred"] = pred
                if targets and "returns" in targets:
                    loss = F.mse_loss(pred.squeeze(-1), targets["returns"].float())
                    outputs["return_loss"] = loss
                    total_loss = total_loss + weight * loss

            elif task == "volatility":
                pred_vol, log_vol = head(pooled)
                outputs["vol_pred"] = pred_vol
                if targets and "volatility" in targets:
                    loss = F.mse_loss(log_vol.squeeze(-1), targets["volatility"].float().log())
                    outputs["vol_loss"] = loss
                    total_loss = total_loss + weight * loss

            elif task == "regime":
                logits = head(pooled)
                outputs["regime_logits"] = logits
                if targets and "regime" in targets:
                    loss = F.cross_entropy(logits, targets["regime"])
                    outputs["regime_loss"] = loss
                    total_loss = total_loss + weight * loss

            elif task == "alpha":
                signals = head(hidden)  # (B, N, d_model) → (B, N)
                outputs["alpha_signals"] = signals

            elif task == "risk_parity":
                weights, pred_vols = head(hidden)
                outputs["portfolio_weights"] = weights
                outputs["predicted_vols"] = pred_vols

        outputs["total_loss"] = total_loss
        outputs["backbone_aux_loss"] = backbone_out.get("aux_loss", torch.tensor(0.0))
        return outputs

    def get_task_params(self, task: str) -> List[nn.Parameter]:
        """Get parameters for a specific task head."""
        if task not in self.heads:
            return []
        return list(self.heads[task].parameters())


# ---------------------------------------------------------------------------
# Import helpers (avoid circular imports)
# ---------------------------------------------------------------------------

def _import_head(name: str):
    """Lazy import of head classes from within module."""
    import importlib
    mod = importlib.import_module("lumina.finetuning")
    return getattr(mod, name, None)


# ---------------------------------------------------------------------------
# Fine-tuning Data Utilities
# ---------------------------------------------------------------------------

class TargetLabelGenerator:
    """Generate target labels for various fine-tuning tasks.

    Computes various label types from raw OHLCV data:
    - Direction labels: {-1, 0, 1} or {0, 1, 2}
    - Return targets: continuous forward returns
    - Volatility targets: realized volatility over horizon
    - Regime labels: trend/volatility regime classification

    Args:
        horizon:         prediction horizon in bars
        direction_threshold: minimum return to classify as up/down (deadband)
        vol_window:     window for realized volatility computation

    Example:
        >>> gen = TargetLabelGenerator(horizon=5)
        >>> ohlcv = np.random.randn(500, 5)
        >>> close = ohlcv[:, 3]
        >>> labels = gen.get_direction(close)  # (500,) int
        >>> returns = gen.get_returns(close)   # (500,) float
    """

    def __init__(
        self,
        horizon: int = 1,
        direction_threshold: float = 0.0,
        vol_window: int = 20,
    ):
        self.horizon = horizon
        self.direction_threshold = direction_threshold
        self.vol_window = vol_window

    def get_returns(self, close: np.ndarray) -> np.ndarray:
        """Compute forward log returns.

        Args:
            close: (T,) closing prices

        Returns:
            returns: (T,) float, returns[t] = log(close[t+h] / close[t])
        """
        T = len(close)
        returns = np.full(T, np.nan)
        for i in range(T - self.horizon):
            if close[i] > 0 and close[i + self.horizon] > 0:
                returns[i] = np.log(close[i + self.horizon] / close[i])
        return returns

    def get_direction(self, close: np.ndarray) -> np.ndarray:
        """Compute direction labels {0=down, 1=flat, 2=up}.

        Args:
            close: (T,) closing prices

        Returns:
            labels: (T,) int, 0=down, 1=neutral, 2=up
        """
        returns = self.get_returns(close)
        labels = np.ones(len(close), dtype=int)  # Default: flat/neutral
        labels[returns > self.direction_threshold] = 2  # up
        labels[returns < -self.direction_threshold] = 0  # down
        labels[np.isnan(returns)] = 1  # unknown → neutral
        return labels

    def get_volatility(
        self,
        close: np.ndarray,
        annualization_factor: float = 252.0,
    ) -> np.ndarray:
        """Compute forward realized volatility.

        Args:
            close:                (T,) closing prices
            annualization_factor: factor to annualize (252 for daily)

        Returns:
            vol: (T,) annualized realized volatility
        """
        T = len(close)
        log_returns = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0)
        vol = np.full(T, np.nan)

        for i in range(T - self.horizon):
            window = log_returns[i:i + self.horizon]
            if len(window) > 0:
                vol[i] = window.std() * np.sqrt(annualization_factor)

        return vol

    def get_multi_horizon_returns(
        self, close: np.ndarray, horizons: List[int]
    ) -> np.ndarray:
        """Compute returns at multiple horizons.

        Args:
            close:    (T,) closing prices
            horizons: list of horizons in bars

        Returns:
            returns: (T, len(horizons)) multi-horizon return matrix
        """
        T = len(close)
        returns = np.full((T, len(horizons)), np.nan)
        for h_idx, h in enumerate(horizons):
            for i in range(T - h):
                if close[i] > 0 and close[i + h] > 0:
                    returns[i, h_idx] = np.log(close[i + h] / close[i])
        return returns


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "FineTuningConfig",
    "DirectionClassificationHead",
    "MultiHorizonDirectionHead",
    "VolatilityForecastingHead",
    "RealizedVolLoss",
    "RegimeClassificationHead",
    "ReturnPredictionHead",
    "PortfolioOptimizationHead",
    "SharpeMaximizationLoss",
    "MeanVarianceLoss",
    "TemperatureScaling",
    "TaskSpecificLoss",
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "FineTuner",
    "FinancialFineTuningDataset",
    "AlphaSignalHead",
    "RiskParityHead",
    "MultiTaskFineTuner",
    "TargetLabelGenerator",
    "information_coefficient",
    "rank_ic",
]


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
