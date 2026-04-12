"""
residual_error_compensator.py — Residual Error Compensator (REC).

Lightweight 3-layer residual MLP that sits on top of agent action output.
Trained to predict and correct the delta between simulated execution outcomes
and live market impact.

Architecture:
- Input: (action, spread, depth, vol, time_of_day, regime_id)
- Output: corrected action (adjusted bid/ask/size) + uncertainty (mean + variance)
- Online training on (simulated_outcome, actual_outcome) pairs
- Shadow mode: run alongside agent, only activate after confidence threshold
- Uncertainty-aware: high uncertainty → reduce position size
"""

from __future__ import annotations

import math
import time
import logging
import collections
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RECMode(Enum):
    SHADOW = auto()         # log corrections, don't apply
    ACTIVE = auto()         # apply corrections
    DISABLED = auto()       # REC not running


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RECNetworkConfig:
    """3-layer residual MLP configuration."""
    action_dim: int = 4                 # [bid_offset, ask_offset, bid_size, ask_size]
    context_dim: int = 8                # [spread, depth_bid, depth_ask, vol, tod, regime_id, lambda, fill_rate]
    hidden_dim: int = 64
    output_dim: int = 4                 # corrected action (same dim as action_dim)
    dropout: float = 0.1
    use_layer_norm: bool = True
    residual_connections: bool = True
    num_hidden_layers: int = 3          # total hidden layers
    activation: str = "relu"            # relu, elu, gelu


@dataclass
class RECTrainingConfig:
    """Online training configuration."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    update_every_n_steps: int = 50
    replay_buffer_size: int = 5_000
    min_samples: int = 64
    grad_clip: float = 0.5
    weight_decay: float = 1e-4
    warm_start_steps: int = 100
    loss_type: str = "huber"            # huber, mse, mae
    huber_delta: float = 1.0
    bootstrap_steps: int = 500          # steps of purely offline training before activating
    kl_reg_weight: float = 1e-4         # KL regularization for uncertainty head


@dataclass
class RECUncertaintyConfig:
    """Uncertainty estimation configuration."""
    enabled: bool = True
    num_mc_samples: int = 20            # Monte Carlo dropout samples
    uncertainty_threshold: float = 0.3  # reduce size if uncertainty > threshold
    size_reduction_factor: float = 0.5  # factor to multiply size by under uncertainty
    confidence_window: int = 200        # rolling window for confidence tracking
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    confidence_ema_alpha: float = 0.05


@dataclass
class RECShadowConfig:
    """Shadow mode configuration."""
    shadow_steps: int = 500             # steps to run in shadow before considering activation
    activation_confidence_threshold: float = 0.7
    activation_accuracy_threshold: float = 0.6  # fraction of corrections that improved outcome
    deactivation_threshold: float = 0.3  # deactivate if accuracy falls below
    log_corrections: bool = True
    log_window: int = 1000


@dataclass
class RECConfig:
    """Master REC configuration."""
    network: RECNetworkConfig = field(default_factory=RECNetworkConfig)
    training: RECTrainingConfig = field(default_factory=RECTrainingConfig)
    uncertainty: RECUncertaintyConfig = field(default_factory=RECUncertaintyConfig)
    shadow: RECShadowConfig = field(default_factory=RECShadowConfig)

    initial_mode: RECMode = RECMode.SHADOW
    seed: Optional[int] = None
    device: str = "cpu"
    enabled: bool = True

    # Context feature names (must match context_dim)
    context_feature_names: List[str] = field(default_factory=lambda: [
        "spread", "depth_bid", "depth_ask", "vol",
        "time_of_day", "regime_id", "kyle_lambda", "fill_rate",
    ])


# ---------------------------------------------------------------------------
# REC Network
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class RECResidualBlock(nn.Module):
        """Single residual block with layer norm and dropout."""

        def __init__(
            self,
            dim: int,
            dropout: float = 0.1,
            use_ln: bool = True,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
            self.drop = nn.Dropout(dropout)
            self._act_name = activation

        def _act(self, x: torch.Tensor) -> torch.Tensor:
            if self._act_name == "elu":
                return F.elu(x)
            elif self._act_name == "gelu":
                return F.gelu(x)
            return F.relu(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self._act(self.ln(self.fc1(x)))
            x = self.drop(x)
            x = self.fc2(x)
            return self._act(x + residual)

    class RECNetwork(nn.Module):
        """
        3-layer residual MLP for action correction.

        Outputs:
        - mean: corrected action (action_dim)
        - log_var: log variance of correction (action_dim) [uncertainty]
        """

        def __init__(self, cfg: RECNetworkConfig) -> None:
            super().__init__()
            in_dim = cfg.action_dim + cfg.context_dim
            h = cfg.hidden_dim

            # Input projection
            self.input_proj = nn.Linear(in_dim, h)
            self.input_ln = nn.LayerNorm(h) if cfg.use_layer_norm else nn.Identity()

            # Residual blocks
            self.blocks = nn.ModuleList([
                RECResidualBlock(h, cfg.dropout, cfg.use_layer_norm, cfg.activation)
                for _ in range(cfg.num_hidden_layers)
            ])

            # Output heads
            self.mean_head = nn.Linear(h, cfg.output_dim)
            self.log_var_head = nn.Linear(h, cfg.output_dim)

            self._cfg = cfg
            self._act_name = cfg.activation

        def _act(self, x: torch.Tensor) -> torch.Tensor:
            if self._act_name == "elu":
                return F.elu(x)
            elif self._act_name == "gelu":
                return F.gelu(x)
            return F.relu(x)

        def forward(
            self,
            action: torch.Tensor,
            context: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                mean: corrected action (B, output_dim)
                log_var: log variance (B, output_dim)
            """
            x = torch.cat([action, context], dim=-1)
            x = self._act(self.input_ln(self.input_proj(x)))

            for block in self.blocks:
                x = block(x)

            mean = self.mean_head(x)
            log_var = self.log_var_head(x)
            # Clamp log_var for stability
            log_var = torch.clamp(log_var, -10.0, 2.0)

            return mean, log_var


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class RECReplayBuffer:
    """Circular replay buffer for REC training data."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._actions: collections.deque = collections.deque(maxlen=capacity)
        self._contexts: collections.deque = collections.deque(maxlen=capacity)
        self._deltas: collections.deque = collections.deque(maxlen=capacity)   # actual - simulated
        self._weights: collections.deque = collections.deque(maxlen=capacity)  # importance weights
        self._n: int = 0

    def push(
        self,
        action: np.ndarray,
        context: np.ndarray,
        delta: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        self._actions.append(action.copy())
        self._contexts.append(context.copy())
        self._deltas.append(delta.copy())
        self._weights.append(weight)
        self._n += 1

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._actions)
        batch_size = min(batch_size, n)
        indices = np.random.choice(n, size=batch_size, replace=False)
        actions = np.array([list(self._actions)[i] for i in indices])
        contexts = np.array([list(self._contexts)[i] for i in indices])
        deltas = np.array([list(self._deltas)[i] for i in indices])
        weights = np.array([list(self._weights)[i] for i in indices])
        return actions, contexts, deltas, weights

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def total_pushed(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Correction logger (shadow mode)
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRecord:
    step: int
    original_action: np.ndarray
    corrected_action: np.ndarray
    context: np.ndarray
    uncertainty: np.ndarray
    applied: bool
    outcome_improved: Optional[bool] = None


class CorrectionLogger:
    """Logs corrections in shadow mode for analysis."""

    def __init__(self, window: int = 1000) -> None:
        self._records: collections.deque = collections.deque(maxlen=window)
        self._accuracy_buf: collections.deque = collections.deque(maxlen=window)
        self._total: int = 0
        self._improved: int = 0

    def log(self, record: CorrectionRecord) -> None:
        self._records.append(record)
        self._total += 1
        if record.outcome_improved is not None:
            self._accuracy_buf.append(1.0 if record.outcome_improved else 0.0)
            if record.outcome_improved:
                self._improved += 1

    def get_accuracy(self) -> float:
        if not self._accuracy_buf:
            return 0.5
        return float(np.mean(list(self._accuracy_buf)))

    def get_correction_magnitude(self) -> float:
        if not self._records:
            return 0.0
        mags = [
            float(np.mean(np.abs(r.corrected_action - r.original_action)))
            for r in self._records
        ]
        return float(np.mean(mags))

    def get_uncertainty_mean(self) -> float:
        if not self._records:
            return 0.0
        return float(np.mean([
            float(np.mean(r.uncertainty)) for r in self._records
        ]))

    @property
    def num_corrections(self) -> int:
        return self._total

    def to_summary(self) -> Dict[str, Any]:
        return {
            "total_corrections": self._total,
            "accuracy": self.get_accuracy(),
            "mean_correction_magnitude": self.get_correction_magnitude(),
            "mean_uncertainty": self.get_uncertainty_mean(),
            "window_size": len(self._records),
        }


# ---------------------------------------------------------------------------
# Confidence tracker
# ---------------------------------------------------------------------------

class ConfidenceTracker:
    """Tracks REC confidence over time using EWMA."""

    def __init__(self, config: RECUncertaintyConfig) -> None:
        self.cfg = config
        self._confidence: float = 0.0
        self._ema: float = 0.0
        self._step: int = 0
        self._history: collections.deque = collections.deque(
            maxlen=config.confidence_window
        )

    def update(self, prediction_error: float, uncertainty: float) -> float:
        """
        Update confidence based on prediction error and uncertainty.

        Returns current confidence score in [0, 1].
        """
        # Confidence = 1 - normalized_error, modulated by uncertainty
        max_error = 2.0  # normalize
        norm_error = min(1.0, prediction_error / max_error)
        norm_uncertainty = min(1.0, uncertainty / self.cfg.uncertainty_threshold)
        raw_confidence = max(0.0, 1.0 - 0.7 * norm_error - 0.3 * norm_uncertainty)

        alpha = self.cfg.confidence_ema_alpha
        if self._step == 0:
            self._ema = raw_confidence
        else:
            self._ema = (1 - alpha) * self._ema + alpha * raw_confidence

        self._confidence = float(np.clip(
            self._ema, self.cfg.min_confidence, self.cfg.max_confidence
        ))
        self._history.append(self._confidence)
        self._step += 1
        return self._confidence

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def trend(self) -> float:
        """Return confidence trend (positive = increasing)."""
        h = list(self._history)
        if len(h) < 20:
            return 0.0
        recent = np.mean(h[-10:])
        older = np.mean(h[-20:-10])
        return float(recent - older)

    def reset(self) -> None:
        self._confidence = 0.0
        self._ema = 0.0
        self._step = 0
        self._history.clear()


# ---------------------------------------------------------------------------
# Residual Error Compensator (main class)
# ---------------------------------------------------------------------------

class ResidualErrorCompensator:
    """
    Residual Error Compensator (REC).

    Sits on top of agent action output. Predicts and corrects the delta
    between simulated execution and live market impact.

    Lifecycle:
    1. SHADOW mode: collect data, log corrections, don't apply
    2. After confidence threshold: switch to ACTIVE mode
    3. If performance degrades: switch back to SHADOW or DISABLED
    """

    def __init__(self, config: Optional[RECConfig] = None) -> None:
        self.cfg = config or RECConfig()
        self._mode = self.cfg.initial_mode
        self._step = 0
        self._shadow_step = 0
        self._lock = threading.Lock()

        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            if _TORCH_AVAILABLE:
                torch.manual_seed(self.cfg.seed)

        # Build network
        self._net: Optional[Any] = None
        self._optimizer: Optional[Any] = None
        if _TORCH_AVAILABLE and self.cfg.enabled:
            self._net = RECNetwork(self.cfg.network).to(self.cfg.device)
            self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
            )

        # Sub-modules
        self._buffer = RECReplayBuffer(self.cfg.training.replay_buffer_size)
        self._logger = CorrectionLogger(self.cfg.shadow.log_window)
        self._confidence = ConfidenceTracker(self.cfg.uncertainty)

        # Metrics
        self._loss_history: collections.deque = collections.deque(maxlen=500)
        self._last_update_step: int = 0
        self._activation_step: Optional[int] = None
        self._total_corrections_applied: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def correct_action(
        self,
        action: np.ndarray,
        context: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, RECMode]:
        """
        Compute corrected action.

        Args:
            action: original agent action (action_dim,)
            context: context features [spread, depth_bid, depth_ask, vol, tod, regime_id, lambda, fill_rate]

        Returns:
            (corrected_action, uncertainty, confidence, mode)
        """
        if not self.cfg.enabled or self._mode == RECMode.DISABLED:
            dummy_unc = np.zeros(self.cfg.network.action_dim)
            return action.copy(), dummy_unc, 0.0, self._mode

        mean_correction, uncertainty = self._forward(action, context)
        confidence = self._confidence.confidence

        if self._mode == RECMode.SHADOW:
            # Log but don't apply
            corrected = action + mean_correction
            record = CorrectionRecord(
                step=self._step,
                original_action=action.copy(),
                corrected_action=corrected,
                context=context.copy(),
                uncertainty=uncertainty,
                applied=False,
            )
            self._logger.log(record)
            # Check if we should activate
            self._check_activation()
            return action.copy(), uncertainty, confidence, self._mode

        elif self._mode == RECMode.ACTIVE:
            corrected = action + mean_correction
            # Uncertainty-aware size reduction
            corrected = self._apply_uncertainty_reduction(corrected, uncertainty)
            self._total_corrections_applied += 1
            record = CorrectionRecord(
                step=self._step,
                original_action=action.copy(),
                corrected_action=corrected,
                context=context.copy(),
                uncertainty=uncertainty,
                applied=True,
            )
            self._logger.log(record)
            return corrected, uncertainty, confidence, self._mode

        return action.copy(), np.zeros(self.cfg.network.action_dim), 0.0, self._mode

    def _forward(
        self,
        action: np.ndarray,
        context: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (mean_correction, std_correction)."""
        if not _TORCH_AVAILABLE or self._net is None:
            return np.zeros_like(action), np.ones_like(action)

        cfg = self.cfg.uncertainty
        device = self.cfg.device

        with torch.no_grad():
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            ctx_t = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

            if cfg.enabled and cfg.num_mc_samples > 1:
                # MC Dropout for uncertainty
                self._net.train()  # keep dropout active
                means_list = []
                for _ in range(cfg.num_mc_samples):
                    m, _ = self._net(act_t, ctx_t)
                    means_list.append(m.cpu().numpy())
                means_arr = np.array(means_list).squeeze(1)  # (mc, action_dim)
                mean_correction = means_arr.mean(axis=0)
                std_correction = means_arr.std(axis=0)
                self._net.eval()
            else:
                self._net.eval()
                mean_t, log_var_t = self._net(act_t, ctx_t)
                mean_correction = mean_t.squeeze(0).cpu().numpy()
                std_correction = np.exp(0.5 * log_var_t.squeeze(0).cpu().numpy())

        return mean_correction, std_correction

    def _apply_uncertainty_reduction(
        self, corrected: np.ndarray, uncertainty: np.ndarray
    ) -> np.ndarray:
        """Reduce position size components if uncertainty is high."""
        cfg = self.cfg.uncertainty
        if not cfg.enabled:
            return corrected

        mean_unc = float(np.mean(uncertainty))
        if mean_unc > cfg.uncertainty_threshold:
            # Size dimensions are typically indices 2 and 3 (bid_size, ask_size)
            result = corrected.copy()
            n = self.cfg.network.action_dim
            size_dims = list(range(n // 2, n))  # convention: first half = prices, second = sizes
            for d in size_dims:
                result[d] *= cfg.size_reduction_factor
            return result
        return corrected

    # ------------------------------------------------------------------
    # Online training
    # ------------------------------------------------------------------

    def push_outcome(
        self,
        action: np.ndarray,
        context: np.ndarray,
        simulated_outcome: np.ndarray,
        actual_outcome: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        """
        Push a (simulated, actual) outcome pair into the replay buffer.

        delta = actual_outcome - simulated_outcome (the correction to learn)
        """
        self._step += 1
        delta = actual_outcome - simulated_outcome

        # Update confidence tracker
        pred_correction, uncertainty = self._forward(action, context)
        pred_error = float(np.mean(np.abs(pred_correction - delta)))
        mean_unc = float(np.mean(uncertainty))
        self._confidence.update(pred_error, mean_unc)

        # Update shadow mode outcome tracking
        if self._mode == RECMode.ACTIVE:
            outcome_improved = pred_error < float(np.mean(np.abs(delta)))
            # Retroactively log outcome
            if self._logger._records:
                rec = self._logger._records[-1]
                rec.outcome_improved = outcome_improved

        self._buffer.push(action, context, delta, weight)

        # Trigger update
        if self._should_update():
            self._update()

    def _should_update(self) -> bool:
        cfg = self.cfg.training
        if len(self._buffer) < cfg.min_samples:
            return False
        if self._step < cfg.warm_start_steps:
            return False
        return (self._step - self._last_update_step) >= cfg.update_every_n_steps

    def _update(self) -> Optional[float]:
        """Run mini-batch gradient update. Returns loss."""
        if not _TORCH_AVAILABLE or self._net is None or self._optimizer is None:
            return None

        cfg = self.cfg.training
        self._last_update_step = self._step

        actions, contexts, deltas, weights = self._buffer.sample(cfg.batch_size)

        device = self.cfg.device
        act_t = torch.tensor(actions, dtype=torch.float32, device=device)
        ctx_t = torch.tensor(contexts, dtype=torch.float32, device=device)
        tgt_t = torch.tensor(deltas, dtype=torch.float32, device=device)
        w_t = torch.tensor(weights, dtype=torch.float32, device=device)

        self._net.train()
        self._optimizer.zero_grad()

        mean_t, log_var_t = self._net(act_t, ctx_t)

        # Negative log-likelihood under Gaussian
        var_t = torch.exp(log_var_t) + 1e-8
        nll = 0.5 * (log_var_t + (tgt_t - mean_t) ** 2 / var_t)

        # Optional: Huber loss on mean
        if cfg.loss_type == "huber":
            rec_loss = F.huber_loss(mean_t, tgt_t, delta=cfg.huber_delta, reduction="none")
        elif cfg.loss_type == "mae":
            rec_loss = torch.abs(mean_t - tgt_t)
        else:
            rec_loss = (mean_t - tgt_t) ** 2

        # KL-like regularization on uncertainty
        kl_reg = cfg.kl_reg_weight * torch.mean(log_var_t ** 2)

        loss = torch.mean(w_t.unsqueeze(1) * (rec_loss + 0.1 * nll)) + kl_reg
        loss.backward()

        nn.utils.clip_grad_norm_(self._net.parameters(), cfg.grad_clip)
        self._optimizer.step()

        loss_val = float(loss.item())
        self._loss_history.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def _check_activation(self) -> None:
        """Check if REC should transition from SHADOW to ACTIVE."""
        cfg = self.cfg.shadow
        self._shadow_step += 1

        if self._shadow_step < cfg.shadow_steps:
            return
        if len(self._buffer) < self.cfg.training.bootstrap_steps:
            return

        accuracy = self._logger.get_accuracy()
        confidence = self._confidence.confidence

        if (
            accuracy >= cfg.activation_accuracy_threshold
            and confidence >= cfg.activation_confidence_threshold
        ):
            logger.info(
                "REC activating: accuracy=%.3f confidence=%.3f",
                accuracy, confidence,
            )
            self._mode = RECMode.ACTIVE
            self._activation_step = self._step

    def check_deactivation(self) -> bool:
        """Check if REC should deactivate due to poor performance."""
        if self._mode != RECMode.ACTIVE:
            return False
        accuracy = self._logger.get_accuracy()
        if accuracy < self.cfg.shadow.deactivation_threshold:
            logger.warning(
                "REC deactivating due to low accuracy: %.3f", accuracy
            )
            self._mode = RECMode.SHADOW
            self._shadow_step = 0
            return True
        return False

    def force_activate(self) -> None:
        """Force REC into ACTIVE mode."""
        self._mode = RECMode.ACTIVE
        self._activation_step = self._step

    def force_shadow(self) -> None:
        """Force REC into SHADOW mode."""
        self._mode = RECMode.SHADOW

    def disable(self) -> None:
        """Disable REC entirely."""
        self._mode = RECMode.DISABLED

    # ------------------------------------------------------------------
    # State and metrics
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "mode": self._mode.name,
            "step": self._step,
            "shadow_step": self._shadow_step,
            "activation_step": self._activation_step,
            "buffer_size": len(self._buffer),
            "confidence": self._confidence.confidence,
            "confidence_trend": self._confidence.trend,
            "mean_loss": float(np.mean(list(self._loss_history))) if self._loss_history else 0.0,
            "shadow_accuracy": self._logger.get_accuracy(),
            "total_corrections_applied": self._total_corrections_applied,
            "correction_summary": self._logger.to_summary(),
        }

    def reset(self) -> None:
        """Reset REC to initial state (keep network weights)."""
        self._step = 0
        self._shadow_step = 0
        self._confidence.reset()

    def full_reset(self) -> None:
        """Full reset including network weights and buffer."""
        self.reset()
        self._buffer = RECReplayBuffer(self.cfg.training.replay_buffer_size)
        self._logger = CorrectionLogger(self.cfg.shadow.log_window)
        self._mode = self.cfg.initial_mode
        self._activation_step = None
        self._total_corrections_applied = 0
        self._loss_history.clear()
        if _TORCH_AVAILABLE and self._net is not None:
            # Re-initialize weights
            for m in self._net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    @property
    def mode(self) -> RECMode:
        return self._mode

    @property
    def is_active(self) -> bool:
        return self._mode == RECMode.ACTIVE

    @property
    def confidence_score(self) -> float:
        return self._confidence.confidence

    def get_network_state(self) -> Optional[Dict[str, Any]]:
        if _TORCH_AVAILABLE and self._net is not None:
            return self._net.state_dict()
        return None

    def load_network_state(self, state: Dict[str, Any]) -> None:
        if _TORCH_AVAILABLE and self._net is not None:
            self._net.load_state_dict(state)


# ---------------------------------------------------------------------------
# Multi-asset REC
# ---------------------------------------------------------------------------

class MultiAssetREC:
    """
    Per-asset REC instances with shared context features.

    Each asset gets its own REC network, but they share a common context
    representation pipeline.
    """

    def __init__(
        self,
        num_assets: int,
        config: Optional[RECConfig] = None,
    ) -> None:
        self.num_assets = num_assets
        base_cfg = config or RECConfig()
        self._recs: List[ResidualErrorCompensator] = [
            ResidualErrorCompensator(base_cfg) for _ in range(num_assets)
        ]

    def correct_all_actions(
        self,
        actions: np.ndarray,
        contexts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Correct actions for all assets.

        Args:
            actions: (num_assets, action_dim)
            contexts: (num_assets, context_dim)

        Returns:
            (corrected_actions, uncertainties, confidences)
        """
        corrected = []
        uncertainties = []
        confidences = []
        for i in range(self.num_assets):
            ca, unc, conf, _ = self._recs[i].correct_action(actions[i], contexts[i])
            corrected.append(ca)
            uncertainties.append(unc)
            confidences.append(conf)
        return np.array(corrected), np.array(uncertainties), np.array(confidences)

    def push_outcomes(
        self,
        actions: np.ndarray,
        contexts: np.ndarray,
        simulated: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        for i in range(self.num_assets):
            self._recs[i].push_outcome(actions[i], contexts[i], simulated[i], actual[i])

    def get_states(self) -> List[Dict[str, Any]]:
        return [r.get_state() for r in self._recs]

    def force_activate_all(self) -> None:
        for r in self._recs:
            r.force_activate()

    def reset_all(self) -> None:
        for r in self._recs:
            r.reset()

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._recs if r.is_active)


# ---------------------------------------------------------------------------
# Context feature builder
# ---------------------------------------------------------------------------

class ContextFeatureBuilder:
    """
    Builds the REC context vector from raw market state.

    Standardizes and normalizes features before passing to REC.
    """

    FEATURE_NAMES = [
        "spread", "depth_bid", "depth_ask", "vol",
        "time_of_day", "regime_id", "kyle_lambda", "fill_rate",
    ]

    def __init__(self, num_assets: int = 4) -> None:
        self.num_assets = num_assets
        self._means: Dict[str, float] = {f: 0.0 for f in self.FEATURE_NAMES}
        self._stds: Dict[str, float] = {f: 1.0 for f in self.FEATURE_NAMES}
        self._history: Dict[str, collections.deque] = {
            f: collections.deque(maxlen=1000) for f in self.FEATURE_NAMES
        }
        self._n: int = 0

    def build(
        self,
        spread: float,
        depth_bid: float,
        depth_ask: float,
        vol: float,
        time_of_day: float,
        regime_id: int,
        kyle_lambda: float = 0.001,
        fill_rate: float = 0.9,
        normalize: bool = True,
    ) -> np.ndarray:
        """Build and optionally normalize the context vector."""
        raw = np.array([
            spread, depth_bid, depth_ask, vol,
            time_of_day, float(regime_id), kyle_lambda, fill_rate,
        ], dtype=np.float32)

        if normalize and self._n > 100:
            for i, name in enumerate(self.FEATURE_NAMES):
                std = self._stds.get(name, 1.0)
                mean = self._means.get(name, 0.0)
                raw[i] = (raw[i] - mean) / max(std, 1e-8)

        return raw

    def update_stats(
        self,
        spread: float,
        depth_bid: float,
        depth_ask: float,
        vol: float,
        time_of_day: float,
        regime_id: int,
        kyle_lambda: float = 0.001,
        fill_rate: float = 0.9,
    ) -> None:
        """Update running statistics for normalization."""
        vals = [spread, depth_bid, depth_ask, vol, time_of_day, float(regime_id), kyle_lambda, fill_rate]
        for name, val in zip(self.FEATURE_NAMES, vals):
            self._history[name].append(val)
        self._n += 1

        if self._n % 50 == 0:
            for name in self.FEATURE_NAMES:
                h = list(self._history[name])
                if h:
                    self._means[name] = float(np.mean(h))
                    self._stds[name] = float(np.std(h)) if len(h) > 1 else 1.0


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_rec(
    action_dim: int = 4,
    context_dim: int = 8,
    hidden_dim: int = 64,
    device: str = "cpu",
    initial_mode: RECMode = RECMode.SHADOW,
    seed: Optional[int] = None,
) -> ResidualErrorCompensator:
    """Create a REC with common defaults."""
    cfg = RECConfig(
        network=RECNetworkConfig(
            action_dim=action_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        ),
        initial_mode=initial_mode,
        device=device,
        seed=seed,
    )
    return ResidualErrorCompensator(cfg)


def make_aggressive_rec(
    action_dim: int = 4,
    context_dim: int = 8,
    device: str = "cpu",
) -> ResidualErrorCompensator:
    """REC that activates quickly, more aggressive corrections."""
    cfg = RECConfig(
        network=RECNetworkConfig(
            action_dim=action_dim,
            context_dim=context_dim,
            hidden_dim=128,
            num_hidden_layers=4,
        ),
        training=RECTrainingConfig(
            learning_rate=3e-3,
            batch_size=64,
            update_every_n_steps=20,
        ),
        shadow=RECShadowConfig(
            shadow_steps=200,
            activation_confidence_threshold=0.5,
            activation_accuracy_threshold=0.5,
        ),
        uncertainty=RECUncertaintyConfig(
            num_mc_samples=10,
            uncertainty_threshold=0.5,
        ),
        initial_mode=RECMode.SHADOW,
        device=device,
    )
    return ResidualErrorCompensator(cfg)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "RECMode",
    # Configs
    "RECConfig",
    "RECNetworkConfig",
    "RECTrainingConfig",
    "RECUncertaintyConfig",
    "RECShadowConfig",
    # Sub-components
    "RECReplayBuffer",
    "CorrectionLogger",
    "CorrectionRecord",
    "ConfidenceTracker",
    "ContextFeatureBuilder",
    # Main
    "ResidualErrorCompensator",
    "MultiAssetREC",
    # Factories
    "make_rec",
    "make_aggressive_rec",
    # Extended
    "RECEnsemble",
    "AdaptiveRECController",
    "RECDiagnosticsPanel",
    "OnlineRECCalibrator",
    "RECPerformanceMonitor",
    "ActionSmoother",
    "UncertaintyAwareSelector",
    "RECStateSerializer",
    "ContextualBanditREC",
    "RECWarmstarter",
]


# ---------------------------------------------------------------------------
# RECEnsemble — ensemble of REC instances with weighted voting
# ---------------------------------------------------------------------------

class RECEnsemble:
    """Ensemble of ResidualErrorCompensator instances; corrections are averaged
    weighted by inverse recent MSE."""

    def __init__(self, num_members: int = 3, context_dim: int = 8,
                 action_dim: int = 3, **kwargs):
        self.members = [
            ResidualErrorCompensator(context_dim=context_dim,
                                      action_dim=action_dim, **kwargs)
            for _ in range(num_members)
        ]
        self._weights = [1.0 / num_members] * num_members
        self._mse_ema = [0.0] * num_members
        self._ema_alpha = 0.05

    # ------------------------------------------------------------------
    def correct_action(self, action: np.ndarray, context: np.ndarray
                       ) -> np.ndarray:
        corrections = [m.correct_action(action, context) for m in self.members]
        w = np.array(self._weights, dtype=np.float32)
        w /= w.sum()
        return sum(c * wi for c, wi in zip(corrections, w))

    # ------------------------------------------------------------------
    def push_outcome(self, action: np.ndarray, context: np.ndarray,
                     real_delta: np.ndarray) -> None:
        for i, m in enumerate(self.members):
            pred = m.correct_action(action, context) - action
            err = float(np.mean((pred - real_delta) ** 2))
            self._mse_ema[i] = (1 - self._ema_alpha) * self._mse_ema[i] + \
                                self._ema_alpha * err
            m.push_outcome(action, context, real_delta)
        # Recompute weights: inverse MSE, avoid div-by-zero
        inv = [1.0 / (e + 1e-8) for e in self._mse_ema]
        total = sum(inv)
        self._weights = [v / total for v in inv]

    # ------------------------------------------------------------------
    def set_mode(self, mode: RECMode) -> None:
        for m in self.members:
            m.mode = mode

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "members": [m.state_dict() if hasattr(m, "state_dict") else {}
                        for m in self.members],
            "weights": self._weights,
            "mse_ema": self._mse_ema,
        }


# ---------------------------------------------------------------------------
# AdaptiveRECController — dynamic enable/disable based on regime
# ---------------------------------------------------------------------------

class AdaptiveRECController:
    """Wraps a REC and a regime signal; disables REC during calm regimes where
    sim-to-real gap is presumed small."""

    CALM_REGIMES = {"CALM", "TRENDING_UP", "TRENDING_DOWN"}

    def __init__(self, rec: ResidualErrorCompensator,
                 calm_threshold: float = 0.3):
        self.rec = rec
        self.calm_threshold = calm_threshold
        self._regime_vol: float = 1.0
        self._enabled: bool = True

    # ------------------------------------------------------------------
    def update_regime(self, regime_name: str, vol_multiplier: float) -> None:
        self._regime_vol = vol_multiplier
        self._enabled = (regime_name not in self.CALM_REGIMES or
                         vol_multiplier > self.calm_threshold * 3)

    # ------------------------------------------------------------------
    def correct_action(self, action: np.ndarray, context: np.ndarray
                       ) -> np.ndarray:
        if not self._enabled:
            return action.copy()
        return self.rec.correct_action(action, context)

    # ------------------------------------------------------------------
    def push_outcome(self, action: np.ndarray, context: np.ndarray,
                     real_delta: np.ndarray) -> None:
        self.rec.push_outcome(action, context, real_delta)

    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# RECDiagnosticsPanel — collects per-step diagnostics
# ---------------------------------------------------------------------------

class RECDiagnosticsPanel:
    """Records corrections, uncertainties, and outcomes for post-hoc analysis."""

    def __init__(self, max_records: int = 10_000):
        self.max_records = max_records
        self._records: list = []

    # ------------------------------------------------------------------
    def record(self, step: int, raw_action: np.ndarray,
               corrected_action: np.ndarray, uncertainty: float,
               mode: str, context: Optional[np.ndarray] = None) -> None:
        if len(self._records) >= self.max_records:
            self._records.pop(0)
        self._records.append({
            "step": step,
            "delta_norm": float(np.linalg.norm(corrected_action - raw_action)),
            "uncertainty": uncertainty,
            "mode": mode,
            "context_norm": float(np.linalg.norm(context)) if context is not None else 0.0,
        })

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        if not self._records:
            return {}
        deltas = [r["delta_norm"] for r in self._records]
        uncerts = [r["uncertainty"] for r in self._records]
        return {
            "num_records": len(self._records),
            "mean_delta_norm": float(np.mean(deltas)),
            "max_delta_norm": float(np.max(deltas)),
            "mean_uncertainty": float(np.mean(uncerts)),
            "max_uncertainty": float(np.max(uncerts)),
            "active_fraction": sum(1 for r in self._records
                                   if r["mode"] == "ACTIVE") / len(self._records),
        }

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# OnlineRECCalibrator — periodically recalibrates REC hyper-parameters
# ---------------------------------------------------------------------------

class OnlineRECCalibrator:
    """Monitors REC correction quality and adjusts uncertainty threshold and
    learning rate online."""

    def __init__(self, rec: ResidualErrorCompensator,
                 recalibrate_every: int = 500):
        self.rec = rec
        self.recalibrate_every = recalibrate_every
        self._step = 0
        self._recent_errors: list = []

    # ------------------------------------------------------------------
    def push_error(self, prediction_mse: float) -> None:
        self._recent_errors.append(prediction_mse)
        self._step += 1
        if self._step % self.recalibrate_every == 0:
            self._recalibrate()

    # ------------------------------------------------------------------
    def _recalibrate(self) -> None:
        if len(self._recent_errors) < 10:
            return
        recent = self._recent_errors[-self.recalibrate_every:]
        mean_err = float(np.mean(recent))
        # If mean error is large, lower confidence threshold to stay in shadow longer
        if mean_err > 0.1 and hasattr(self.rec, "_confidence_threshold"):
            self.rec._confidence_threshold = min(
                0.95, self.rec._confidence_threshold + 0.02)
        elif mean_err < 0.01 and hasattr(self.rec, "_confidence_threshold"):
            self.rec._confidence_threshold = max(
                0.5, self.rec._confidence_threshold - 0.01)


# ---------------------------------------------------------------------------
# RECPerformanceMonitor — tracks correction benefit vs overhead
# ---------------------------------------------------------------------------

class RECPerformanceMonitor:
    """Compares episode returns with and without REC active."""

    def __init__(self):
        self._returns_with: list = []
        self._returns_without: list = []

    # ------------------------------------------------------------------
    def record_episode(self, total_return: float, rec_active: bool) -> None:
        if rec_active:
            self._returns_with.append(total_return)
        else:
            self._returns_without.append(total_return)

    # ------------------------------------------------------------------
    def benefit(self) -> float:
        """Return mean improvement from using REC. Positive = beneficial."""
        if not self._returns_with or not self._returns_without:
            return 0.0
        return float(np.mean(self._returns_with) - np.mean(self._returns_without))

    # ------------------------------------------------------------------
    def report(self) -> dict:
        return {
            "benefit": self.benefit(),
            "n_with": len(self._returns_with),
            "n_without": len(self._returns_without),
            "mean_with": float(np.mean(self._returns_with)) if self._returns_with else 0.0,
            "mean_without": float(np.mean(self._returns_without)) if self._returns_without else 0.0,
        }


# ---------------------------------------------------------------------------
# ActionSmoother — temporal smoothing of corrected actions
# ---------------------------------------------------------------------------

class ActionSmoother:
    """Applies exponential moving average to REC-corrected actions to prevent
    high-frequency oscillation."""

    def __init__(self, alpha: float = 0.7, action_dim: int = 3):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def smooth(self, action: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = action.copy()
            return action.copy()
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._prev = None


# ---------------------------------------------------------------------------
# UncertaintyAwareSelector — selects between raw/corrected action by uncertainty
# ---------------------------------------------------------------------------

class UncertaintyAwareSelector:
    """Falls back to raw agent action when REC uncertainty is too high."""

    def __init__(self, uncertainty_ceiling: float = 0.8):
        self.uncertainty_ceiling = uncertainty_ceiling
        self._fallbacks = 0
        self._total = 0

    # ------------------------------------------------------------------
    def select(self, raw_action: np.ndarray, corrected_action: np.ndarray,
               uncertainty: float) -> np.ndarray:
        self._total += 1
        if uncertainty > self.uncertainty_ceiling:
            self._fallbacks += 1
            return raw_action.copy()
        return corrected_action.copy()

    # ------------------------------------------------------------------
    @property
    def fallback_rate(self) -> float:
        return self._fallbacks / max(1, self._total)


# ---------------------------------------------------------------------------
# RECStateSerializer — save / load REC network weights
# ---------------------------------------------------------------------------

class RECStateSerializer:
    """Utility to save and load REC state to/from a dict or file path."""

    @staticmethod
    def to_dict(rec: ResidualErrorCompensator) -> dict:
        state: dict = {
            "mode": rec.mode.name,
            "context_dim": rec.context_dim,
            "action_dim": rec.action_dim,
        }
        if _TORCH_AVAILABLE and rec.network is not None:
            import io, torch
            buf = io.BytesIO()
            torch.save(rec.network.state_dict(), buf)
            state["network_bytes"] = buf.getvalue()
        return state

    @staticmethod
    def from_dict(state: dict) -> ResidualErrorCompensator:
        rec = ResidualErrorCompensator(
            context_dim=state["context_dim"],
            action_dim=state["action_dim"],
        )
        rec.mode = RECMode[state["mode"]]
        if _TORCH_AVAILABLE and "network_bytes" in state and rec.network is not None:
            import io, torch
            buf = io.BytesIO(state["network_bytes"])
            rec.network.load_state_dict(torch.load(buf, map_location="cpu"))
        return rec


# ---------------------------------------------------------------------------
# ContextualBanditREC — UCB-style selection among correction magnitudes
# ---------------------------------------------------------------------------

class ContextualBanditREC:
    """Uses UCB bandit to select among discrete correction scale factors
    {0.0, 0.25, 0.5, 0.75, 1.0} applied to the base REC correction."""

    SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(self, rec: ResidualErrorCompensator, ucb_c: float = 1.0):
        self.rec = rec
        self.ucb_c = ucb_c
        self._counts = np.ones(len(self.SCALES), dtype=np.float32)
        self._rewards = np.zeros(len(self.SCALES), dtype=np.float32)
        self._last_arm: int = len(self.SCALES) - 1  # start with full correction
        self._last_raw: Optional[np.ndarray] = None
        self._last_base_correction: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def correct_action(self, action: np.ndarray, context: np.ndarray
                       ) -> np.ndarray:
        base_corrected = self.rec.correct_action(action, context)
        base_correction = base_corrected - action
        t = float(self._counts.sum())
        ucb_vals = self._rewards / self._counts + \
                   self.ucb_c * np.sqrt(np.log(t + 1) / self._counts)
        arm = int(np.argmax(ucb_vals))
        self._last_arm = arm
        self._last_raw = action.copy()
        self._last_base_correction = base_correction.copy()
        return action + self.SCALES[arm] * base_correction

    # ------------------------------------------------------------------
    def push_reward(self, reward: float) -> None:
        arm = self._last_arm
        self._counts[arm] += 1.0
        # EMA reward update
        n = self._counts[arm]
        self._rewards[arm] += (reward - self._rewards[arm]) / n


# ---------------------------------------------------------------------------
# RECWarmstarter — pre-trains REC on synthetic residuals before live trading
# ---------------------------------------------------------------------------

class RECWarmstarter:
    """Generates synthetic (context, residual) pairs by adding Gaussian noise
    and pre-trains the REC network offline."""

    def __init__(self, rec: ResidualErrorCompensator,
                 noise_scale: float = 0.05, num_steps: int = 2000):
        self.rec = rec
        self.noise_scale = noise_scale
        self.num_steps = num_steps

    # ------------------------------------------------------------------
    def warmstart(self, context_dim: int, action_dim: int,
                  rng: Optional[np.random.Generator] = None) -> int:
        """Generate synthetic data and train. Returns number of steps trained."""
        if rng is None:
            rng = np.random.default_rng()
        trained = 0
        for _ in range(self.num_steps):
            ctx = rng.standard_normal(context_dim).astype(np.float32)
            action = rng.standard_normal(action_dim).astype(np.float32)
            # Synthetic residual: small noise correlated with context magnitude
            residual = (self.noise_scale * rng.standard_normal(action_dim) *
                        (1 + np.abs(ctx[:action_dim]))).astype(np.float32)
            self.rec.push_outcome(action, ctx, residual)
            trained += 1
        # After warmstart, keep in shadow mode — let live data decide activation
        self.rec.mode = RECMode.SHADOW
        return trained
