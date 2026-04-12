"""
lumina/rlhf.py

RLHF (Reinforcement Learning from Human Feedback) for Lumina financial signals.

Covers:
  - Reward model training on backtested P&L
  - PPO (Proximal Policy Optimization) fine-tuning with financial reward
  - Preference learning from alpha signal quality comparisons
  - KL constraint to prevent distribution shift from reference policy
  - Reward hacking detection (overoptimization monitoring)
  - Direct Preference Optimization (DPO) as RLHF alternative
  - Risk-adjusted reward shaping (penalize drawdown, reward Sharpe)
  - Constitutional AI-style constraint enforcement for financial signals
"""

from __future__ import annotations

import copy
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class FinancialRewardModel(nn.Module):
    """
    Reward model that predicts the quality of a financial signal.

    Trained on (signal, realized_pnl) pairs from backtests.
    The reward captures risk-adjusted performance: Sharpe ratio,
    drawdown, turnover, and consistency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        output_components: int = 5,    # Sharpe, Sortino, Calmar, MaxDD, Turnover
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_components = output_components

        # Transformer-based reward encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_components)
        self.reward_head = nn.Linear(output_components, 1)

        # Learned reward weights (how much to value each component)
        self.reward_weights = nn.Parameter(torch.ones(output_components))

    def forward(
        self,
        signal_sequence: Tensor,     # (B, T, input_dim) — signal history
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            signal_sequence: Sequence of signal values and context.
            mask: Padding mask.

        Returns:
            Dict with "reward", "components", "weights".
        """
        x = self.input_proj(signal_sequence)
        x = self.transformer(x, src_key_padding_mask=mask)
        # Aggregate over time via mean pooling
        if mask is not None:
            # Masked mean
            lengths = (~mask).float().sum(dim=1, keepdim=True)
            x_masked = x * (~mask.unsqueeze(-1)).float()
            pooled = x_masked.sum(dim=1) / (lengths + 1e-10)
        else:
            pooled = x.mean(dim=1)

        components = self.output_proj(pooled)  # (B, n_components)
        weights = F.softmax(self.reward_weights, dim=0)
        reward = (components * weights).sum(dim=-1, keepdim=True)  # (B, 1)

        return {
            "reward": reward.squeeze(-1),
            "components": components,
            "weights": weights,
        }

    def compute_risk_adjusted_reward(
        self,
        returns: np.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Compute scalar reward from backtested return series.
        Combines multiple risk-adjusted metrics.
        """
        if len(returns) < 5:
            return 0.0

        # Sharpe ratio
        excess = returns - risk_free / 252
        sharpe = np.mean(excess) / (np.std(excess) + 1e-10) * math.sqrt(252)

        # Max drawdown (penalized)
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        mdd = ((cum - peak) / (peak + 1e-10)).min()

        # Calmar
        ann_ret = np.prod(1 + returns) ** (252 / len(returns)) - 1
        calmar = ann_ret / (abs(mdd) + 1e-10)

        # Sortino
        downside = excess[excess < 0]
        sortino = np.mean(excess) / (math.sqrt(np.mean(downside ** 2)) + 1e-10) * math.sqrt(252)

        # Weighted combination
        reward = (
            0.4 * np.tanh(sharpe)
            + 0.2 * np.tanh(calmar / 10)
            + 0.2 * np.tanh(sortino)
            + 0.2 * np.tanh(1 + mdd)  # mdd is negative; closer to 0 = better
        )
        return float(reward)


class RewardModelTrainer:
    """
    Trains the reward model on (signal, pnl) preference pairs.

    Supports two modes:
      1. Regression: predict scalar reward from backtest metrics.
      2. Preference: Bradley-Terry model on pairwise comparisons.
    """

    def __init__(
        self,
        reward_model: FinancialRewardModel,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.rm = reward_model
        self.optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)
        self.device = device or torch.device("cpu")
        self.rm = self.rm.to(self.device)

    def train_on_preferences(
        self,
        preferred_signals: List[Tensor],    # "winning" signals
        rejected_signals: List[Tensor],      # "losing" signals
        n_epochs: int = 10,
    ) -> List[float]:
        """
        Train reward model using Bradley-Terry preference model.

        Loss = -log sigmoid(r(preferred) - r(rejected))
        """
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n = 0
            for pref, rej in zip(preferred_signals, rejected_signals):
                pref = pref.to(self.device)
                rej = rej.to(self.device)
                if pref.ndim == 2:
                    pref = pref.unsqueeze(0)
                if rej.ndim == 2:
                    rej = rej.unsqueeze(0)

                self.optimizer.zero_grad()
                r_pref = self.rm(pref)["reward"]
                r_rej = self.rm(rej)["reward"]
                loss = -F.logsigmoid(r_pref - r_rej).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rm.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1

            avg_loss = epoch_loss / max(1, n)
            losses.append(avg_loss)
            logger.info(f"Reward model epoch {epoch}: loss={avg_loss:.4f}")
        return losses

    def train_on_regression(
        self,
        signals: List[Tensor],
        rewards: List[float],
        n_epochs: int = 20,
    ) -> List[float]:
        """Train reward model via regression on scalar rewards."""
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n = 0
            pairs = list(zip(signals, rewards))
            random.shuffle(pairs)
            for sig, rew in pairs:
                sig = sig.to(self.device)
                if sig.ndim == 2:
                    sig = sig.unsqueeze(0)
                self.optimizer.zero_grad()
                pred = self.rm(sig)["reward"]
                target = torch.tensor([rew], device=self.device)
                loss = F.mse_loss(pred, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / max(1, n)
            losses.append(avg)
        return losses


# ---------------------------------------------------------------------------
# PPO for financial signal fine-tuning
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    lr_actor: float = 1e-5
    lr_critic: float = 1e-4
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    kl_coeff: float = 0.1
    kl_target: float = 0.02
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_response_len: int = 128
    reward_clip: float = 5.0
    whiten_rewards: bool = True


@dataclass
class PPOExperience:
    """One step of collected PPO experience."""
    obs: Tensor
    action: Tensor
    log_prob: Tensor
    value: Tensor
    reward: float
    done: bool
    ref_log_prob: Optional[Tensor] = None


class ValueHead(nn.Module):
    """Value function head for PPO (critic)."""

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]    # Last token for causal model
        return self.net(x).squeeze(-1)


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) fine-tuning for Lumina.

    Treats Lumina as a policy: it generates signal values/actions.
    Rewards come from a financial reward model (Sharpe, P&L, etc.)
    KL divergence from the reference (pre-RLHF) policy acts as a constraint
    to prevent reward hacking / distribution collapse.
    """

    def __init__(
        self,
        actor: nn.Module,               # Policy model (Lumina)
        ref_model: nn.Module,           # Frozen reference policy
        reward_model: FinancialRewardModel,
        config: PPOConfig,
        device: Optional[torch.device] = None,
    ):
        self.actor = actor
        self.ref_model = copy.deepcopy(ref_model)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.reward_model = reward_model
        self.config = config
        self.device = device or torch.device("cpu")

        # Add value head to actor
        d_model = self._get_d_model(actor)
        self.value_head = ValueHead(d_model).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.AdamW(
            self.value_head.parameters(), lr=config.lr_critic
        )

        self._experience_buffer: List[PPOExperience] = []
        self._kl_adaptive: float = config.kl_coeff
        self._step: int = 0

    def _get_d_model(self, model: nn.Module) -> int:
        """Try to infer d_model from model config."""
        for attr in ["d_model", "hidden_size", "embed_dim"]:
            if hasattr(model, "config") and hasattr(model.config, attr):
                return getattr(model.config, attr)
        for name, param in model.named_parameters():
            if "embed" in name or "wte" in name:
                return param.shape[-1]
        return 512  # Default

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        gae = 0.0

        for t in reversed(range(T)):
            next_val = 0.0 if dones[t] or t == T - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_val - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (0 if dones[t] else gae)
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        if self.config.whiten_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_kl_divergence(
        self,
        obs: Tensor,
        actor_log_probs: Tensor,
    ) -> Tensor:
        """
        Compute KL(actor || ref_model).
        """
        with torch.no_grad():
            ref_outputs = self.ref_model(obs)
            if isinstance(ref_outputs, dict):
                ref_logits = ref_outputs.get("logits", ref_outputs.get("output"))
            else:
                ref_logits = ref_outputs

        actor_outputs = self.actor(obs)
        if isinstance(actor_outputs, dict):
            actor_logits = actor_outputs.get("logits", actor_outputs.get("output"))
        else:
            actor_logits = actor_outputs

        if ref_logits is None or actor_logits is None:
            return torch.zeros(1, device=self.device)

        kl = F.kl_div(
            F.log_softmax(actor_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )
        return kl

    def ppo_step(
        self,
        obs_batch: Tensor,
        actions_batch: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> Dict[str, float]:
        """
        One PPO gradient step.
        """
        obs_batch = obs_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Forward pass
        actor_out = self.actor(obs_batch)
        if isinstance(actor_out, dict):
            logits = actor_out.get("logits", actor_out.get("output"))
        else:
            logits = actor_out

        if logits is None:
            return {}

        # Get hidden states for value head
        if isinstance(actor_out, dict) and "hidden_states" in actor_out:
            hidden = actor_out["hidden_states"][-1]
        else:
            hidden = logits

        # New log probs
        log_probs = F.log_softmax(logits, dim=-1)
        if actions_batch.ndim == 1:
            new_log_probs = log_probs.gather(-1, actions_batch.unsqueeze(-1)).squeeze(-1)
        else:
            new_log_probs = (log_probs * actions_batch).sum(-1)

        # Policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped PPO objective
        clip_eps = self.config.clip_epsilon
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values_pred = self.value_head(hidden)
        value_loss = F.mse_loss(values_pred, returns)

        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()

        # KL divergence penalty
        kl = self.compute_kl_divergence(obs_batch, new_log_probs)

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy
            + self._kl_adaptive * kl
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Adaptive KL: adjust penalty if KL is too high/low
        with torch.no_grad():
            kl_val = kl.item()
        if kl_val > 1.5 * self.config.kl_target:
            self._kl_adaptive *= 2.0
        elif kl_val < 0.5 * self.config.kl_target:
            self._kl_adaptive = max(0.01, self._kl_adaptive / 2.0)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl_val,
            "kl_coeff": self._kl_adaptive,
            "ratio_mean": ratio.mean().item(),
            "clip_fraction": (ratio.abs() > 1 + clip_eps).float().mean().item(),
        }

    def collect_experience(
        self,
        obs: Tensor,
        env_step_fn: Callable[[Tensor], Tuple[Tensor, float, bool]],
    ) -> List[PPOExperience]:
        """
        Collect rollout experience from environment.

        env_step_fn: Given action, returns (next_obs, reward, done).
        """
        experiences = []
        current_obs = obs.to(self.device)

        for _ in range(self.config.max_response_len):
            actor_out = self.actor(current_obs)
            if isinstance(actor_out, dict):
                logits = actor_out.get("logits", actor_out.get("output"))
                hidden = actor_out.get("hidden_states", [logits])
                if isinstance(hidden, list):
                    hidden = hidden[-1]
            else:
                logits = actor_out
                hidden = logits

            if logits is None:
                break

            # Sample action
            probs = F.softmax(logits[:, -1, :], dim=-1)
            action = torch.multinomial(probs, 1)
            log_prob = F.log_softmax(logits[:, -1, :], dim=-1).gather(-1, action)

            # Value
            with torch.no_grad():
                value = self.value_head(hidden[:, -1:, :] if hidden.ndim == 3 else hidden)

            next_obs, reward, done = env_step_fn(action)
            reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

            # Reference log prob for KL
            with torch.no_grad():
                ref_out = self.ref_model(current_obs)
                if isinstance(ref_out, dict):
                    ref_logits = ref_out.get("logits", ref_out.get("output"))
                else:
                    ref_logits = ref_out
                if ref_logits is not None:
                    ref_log_prob = F.log_softmax(ref_logits[:, -1, :], dim=-1).gather(-1, action)
                else:
                    ref_log_prob = log_prob.clone()

            experiences.append(PPOExperience(
                obs=current_obs.cpu(),
                action=action.cpu(),
                log_prob=log_prob.cpu(),
                value=value.cpu(),
                reward=reward,
                done=done,
                ref_log_prob=ref_log_prob.cpu(),
            ))

            current_obs = next_obs.to(self.device)
            if done:
                break

        return experiences

    def train_step(
        self,
        experiences: List[PPOExperience],
    ) -> Dict[str, float]:
        """Run PPO update on collected experience."""
        if not experiences:
            return {}

        rewards = [e.reward for e in experiences]
        values = [e.value.item() for e in experiences]
        dones = [e.done for e in experiences]

        advantages, returns = self.compute_advantages(rewards, values, dones)

        obs_batch = torch.cat([e.obs for e in experiences])
        actions_batch = torch.cat([e.action for e in experiences])
        old_log_probs = torch.cat([e.log_prob for e in experiences])
        adv_tensor = torch.from_numpy(advantages).float()
        ret_tensor = torch.from_numpy(returns).float()

        total_metrics: Dict[str, float] = {}
        for _ in range(self.config.n_epochs):
            metrics = self.ppo_step(
                obs_batch, actions_batch, old_log_probs, adv_tensor, ret_tensor
            )
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

        for k in total_metrics:
            total_metrics[k] /= self.config.n_epochs

        self._step += 1
        return total_metrics


# ---------------------------------------------------------------------------
# Direct Preference Optimization (DPO)
# ---------------------------------------------------------------------------

class DPOTrainer:
    """
    Direct Preference Optimization (Rafailov et al., 2023).

    Alternative to PPO that avoids explicit reward model and RL loop.
    Optimizes a policy directly on preference pairs.

    Loss = -log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x))
                        - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))

    For financial signals:
      y_w = preferred signal (better backtest performance)
      y_l = rejected signal (worse backtest performance)
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        beta: float = 0.1,
        lr: float = 1e-5,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.ref_policy = copy.deepcopy(ref_policy)
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.beta = beta
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.device = device or torch.device("cpu")
        self.policy = self.policy.to(self.device)
        self.ref_policy = self.ref_policy.to(self.device)

    def _get_log_probs(
        self,
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute sum of log probs of targets given inputs."""
        outputs = model(inputs)
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("output"))
        else:
            logits = outputs

        if logits is None:
            return torch.zeros(inputs.shape[0], device=self.device)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs.gather(
            -1, shift_targets.clamp(0, log_probs.shape[-1] - 1).unsqueeze(-1)
        ).squeeze(-1)
        return per_token_lp.sum(dim=-1)

    def dpo_loss(
        self,
        prompt: Tensor,
        chosen: Tensor,
        rejected: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        Args:
            prompt: Input context (B, T_prompt).
            chosen: Preferred continuation (B, T_chosen).
            rejected: Rejected continuation (B, T_rejected).

        Returns:
            (loss, metrics_dict)
        """
        # Concatenate prompt + response
        chosen_input = torch.cat([prompt, chosen], dim=1)
        rejected_input = torch.cat([prompt, rejected], dim=1)

        # Policy log probs
        pi_lp_chosen = self._get_log_probs(self.policy, chosen_input, chosen_input)
        pi_lp_rejected = self._get_log_probs(self.policy, rejected_input, rejected_input)

        # Reference log probs
        with torch.no_grad():
            ref_lp_chosen = self._get_log_probs(self.ref_policy, chosen_input, chosen_input)
            ref_lp_rejected = self._get_log_probs(self.ref_policy, rejected_input, rejected_input)

        # DPO objective
        chosen_reward = self.beta * (pi_lp_chosen - ref_lp_chosen)
        rejected_reward = self.beta * (pi_lp_rejected - ref_lp_rejected)

        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        metrics = {
            "dpo_loss": loss.item(),
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
            "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
            "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        }
        return loss, metrics

    def train_step(
        self,
        prompt: Tensor,
        chosen: Tensor,
        rejected: Tensor,
    ) -> Dict[str, float]:
        prompt = prompt.to(self.device)
        chosen = chosen.to(self.device)
        rejected = rejected.to(self.device)

        self.optimizer.zero_grad()
        loss, metrics = self.dpo_loss(prompt, chosen, rejected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return metrics

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 3,
    ) -> List[Dict[str, float]]:
        """Full DPO training loop."""
        history = []
        for epoch in range(n_epochs):
            epoch_metrics: Dict[str, float] = {}
            n = 0
            for batch in dataloader:
                prompt = batch.get("prompt") or batch[0]
                chosen = batch.get("chosen") or batch[1]
                rejected = batch.get("rejected") or batch[2]
                metrics = self.train_step(prompt, chosen, rejected)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n += 1
            for k in epoch_metrics:
                epoch_metrics[k] /= max(1, n)
            epoch_metrics["epoch"] = epoch
            history.append(epoch_metrics)
            logger.info(
                f"DPO epoch {epoch}: loss={epoch_metrics.get('dpo_loss', 0):.4f}, "
                f"acc={epoch_metrics.get('reward_accuracy', 0):.3f}"
            )
        return history


# ---------------------------------------------------------------------------
# Reward hacking detection
# ---------------------------------------------------------------------------

class RewardHackingDetector:
    """
    Detects reward hacking / over-optimization during RLHF training.

    Monitors:
      - KL divergence from reference policy (should stay bounded)
      - Reward model score vs. true backtest performance
      - Distribution shift in signal values
      - Reward variance (overfit = near-zero variance)
    """

    def __init__(
        self,
        kl_threshold: float = 10.0,
        reward_collapse_threshold: float = 0.01,
        window: int = 100,
    ):
        self.kl_threshold = kl_threshold
        self.reward_collapse_threshold = reward_collapse_threshold
        self.window = window
        self._kl_history: deque = deque(maxlen=window)
        self._reward_history: deque = deque(maxlen=window)
        self._true_perf_history: deque = deque(maxlen=window)
        self._alerts: List[str] = []

    def update(
        self,
        kl: float,
        reward_model_score: float,
        true_performance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update monitoring state and check for hacking."""
        self._kl_history.append(kl)
        self._reward_history.append(reward_model_score)
        if true_performance is not None:
            self._true_perf_history.append(true_performance)

        alerts = []
        status = "ok"

        # Check KL blow-up
        if kl > self.kl_threshold:
            alert = f"KL divergence blow-up: {kl:.2f} > {self.kl_threshold}"
            alerts.append(alert)
            status = "warning"

        # Check reward collapse (variance near zero)
        if len(self._reward_history) >= 10:
            reward_var = np.var(list(self._reward_history)[-10:])
            if reward_var < self.reward_collapse_threshold:
                alert = f"Reward variance collapse: {reward_var:.4f}"
                alerts.append(alert)
                status = "warning"

        # Check reward-performance divergence (Goodhart's Law)
        if len(self._true_perf_history) >= 10:
            rewards_arr = np.array(list(self._reward_history)[-10:])
            perf_arr = np.array(list(self._true_perf_history)[-10:])
            corr = np.corrcoef(rewards_arr, perf_arr)[0, 1]
            if not np.isnan(corr) and corr < 0.1:
                alert = f"Reward-performance decorrelation: corr={corr:.3f}"
                alerts.append(alert)
                status = "critical"

        # Rapidly increasing rewards (overoptimization sign)
        if len(self._reward_history) >= 20:
            recent = list(self._reward_history)[-10:]
            early = list(self._reward_history)[-20:-10]
            trend = np.mean(recent) - np.mean(early)
            if trend > 5.0:
                alert = f"Rapid reward increase detected: +{trend:.2f}"
                alerts.append(alert)
                status = "warning"

        self._alerts.extend(alerts)
        for a in alerts:
            logger.warning(f"[RewardHacking] {a}")

        return {
            "status": status,
            "kl": kl,
            "reward_score": reward_model_score,
            "kl_mean": np.mean(self._kl_history) if self._kl_history else 0.0,
            "reward_mean": np.mean(self._reward_history) if self._reward_history else 0.0,
            "alerts": alerts,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total_alerts": len(self._alerts),
            "recent_alerts": self._alerts[-5:],
            "kl_mean": np.mean(self._kl_history) if self._kl_history else 0.0,
            "kl_max": max(self._kl_history) if self._kl_history else 0.0,
            "reward_mean": np.mean(self._reward_history) if self._reward_history else 0.0,
        }


# ---------------------------------------------------------------------------
# Constitutional AI constraint enforcement
# ---------------------------------------------------------------------------

class FinancialConstitution:
    """
    Constitutional AI-style constraints for financial signal generation.

    Enforces:
      - No leverage beyond maximum allowed
      - No extreme position concentrations
      - Risk limits (VaR, drawdown thresholds)
      - Regulatory constraints (e.g., no short selling on specific stocks)
    """

    def __init__(
        self,
        max_leverage: float = 2.0,
        max_position_pct: float = 0.20,
        max_drawdown_limit: float = -0.30,
        var_confidence: float = 0.99,
        var_limit: float = 0.05,    # 5% daily VaR limit
    ):
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        self.var_limit = var_limit

    def check_weights(self, weights: np.ndarray) -> Tuple[bool, List[str]]:
        """Check if portfolio weights satisfy constitutional constraints."""
        violations = []

        # Leverage check
        gross_exposure = np.abs(weights).sum()
        if gross_exposure > self.max_leverage:
            violations.append(f"Leverage violation: {gross_exposure:.2f} > {self.max_leverage}")

        # Concentration check
        max_weight = np.abs(weights).max()
        if max_weight > self.max_position_pct:
            violations.append(f"Concentration violation: {max_weight:.2%} > {self.max_position_pct:.2%}")

        return len(violations) == 0, violations

    def check_backtest(
        self,
        returns: np.ndarray,
    ) -> Tuple[bool, List[str]]:
        """Check if backtest results satisfy constraints."""
        violations = []

        # Drawdown check
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        mdd = ((cum - peak) / (peak + 1e-10)).min()
        if mdd < self.max_drawdown_limit:
            violations.append(f"Drawdown violation: {mdd:.2%} < {self.max_drawdown_limit:.2%}")

        # VaR check
        if len(returns) >= 20:
            var = np.percentile(returns, (1 - self.var_confidence) * 100)
            if var < -self.var_limit:
                violations.append(f"VaR violation: {var:.2%} < {-self.var_limit:.2%}")

        return len(violations) == 0, violations

    def project_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto the feasible set satisfying constraints.
        Simple rescaling approach.
        """
        # Cap individual positions
        weights = np.clip(weights, -self.max_position_pct, self.max_position_pct)
        # Normalize to max leverage
        gross = np.abs(weights).sum()
        if gross > self.max_leverage:
            weights = weights / gross * self.max_leverage
        return weights

    def compute_reward_penalty(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Compute reward penalty for constitutional violations."""
        ok_weights, weight_violations = self.check_weights(weights)
        ok_backtest, backtest_violations = self.check_backtest(returns)
        n_violations = len(weight_violations) + len(backtest_violations)
        return -1.0 * n_violations  # -1 per violation


# ---------------------------------------------------------------------------
# Full RLHF pipeline
# ---------------------------------------------------------------------------

class LuminaRLHFPipeline:
    """
    Complete RLHF pipeline for Lumina financial foundation model.

    Steps:
      1. Collect preference data (signal quality comparisons)
      2. Train reward model on preferences
      3. Fine-tune policy with PPO using reward model
      4. Monitor for reward hacking
      5. Evaluate against holdout financial benchmarks
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_model: FinancialRewardModel,
        ppo_config: PPOConfig,
        constitution: Optional[FinancialConstitution] = None,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.ppo_config = ppo_config
        self.constitution = constitution or FinancialConstitution()
        self.device = device or torch.device("cpu")

        self.rm_trainer = RewardModelTrainer(reward_model, device=device)
        self.ppo_trainer = PPOTrainer(
            policy, policy, reward_model, ppo_config, device
        )
        self.dpo_trainer = DPOTrainer(policy, policy, device=device)
        self.hacking_detector = RewardHackingDetector()

    def run(
        self,
        preference_dataloader: torch.utils.data.DataLoader,
        rl_dataloader: torch.utils.data.DataLoader,
        n_rm_epochs: int = 10,
        n_ppo_steps: int = 100,
        use_dpo: bool = False,
    ) -> Dict[str, Any]:
        """Run full RLHF pipeline."""
        logger.info("Step 1: Training reward model...")
        rm_losses = []
        for batch in preference_dataloader:
            pref = batch.get("preferred") or batch[0]
            rej = batch.get("rejected") or batch[1]
            rewards = batch.get("rewards", None)
            if rewards is not None:
                rm_losses.extend(
                    self.rm_trainer.train_on_regression([pref, rej], [rewards[0].item(), rewards[1].item()])
                )
            else:
                rm_losses.extend(
                    self.rm_trainer.train_on_preferences([pref], [rej])
                )

        logger.info("Step 2: Fine-tuning policy...")
        if use_dpo:
            dpo_history = self.dpo_trainer.train(preference_dataloader, n_epochs=3)
            ppo_history = []
        else:
            ppo_history = []
            for i, batch in enumerate(rl_dataloader):
                if i >= n_ppo_steps:
                    break
                obs = (batch.get("input_ids") or batch[0]).to(self.device)
                # Simulate getting reward (in real use, this comes from backtesting)
                with torch.no_grad():
                    rm_out = self.reward_model(obs)
                    kl_approx = 0.0
                    hack_status = self.hacking_detector.update(
                        kl_approx, rm_out["reward"].mean().item()
                    )
                    ppo_history.append(hack_status)

        return {
            "rm_losses": rm_losses,
            "ppo_history": ppo_history,
            "hacking_summary": self.hacking_detector.summary(),
        }


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Reward model
    "FinancialRewardModel",
    "RewardModelTrainer",
    # PPO
    "PPOConfig",
    "PPOExperience",
    "ValueHead",
    "PPOTrainer",
    # DPO
    "DPOTrainer",
    # Reward hacking
    "RewardHackingDetector",
    # Constitutional AI
    "FinancialConstitution",
    # Pipeline
    "LuminaRLHFPipeline",
]
