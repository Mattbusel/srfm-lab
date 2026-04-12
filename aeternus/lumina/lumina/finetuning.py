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
    "information_coefficient",
    "rank_ic",
]
