

# ============================================================
# Extended RLHF Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 16
    n_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95  # GAE lambda
    normalize_advantages: bool = True
    kl_target: float = 0.02
    kl_coef: float = 0.1
    adaptive_kl: bool = True


class RewardModel(nn.Module):
    """Reward model for RLHF: maps (prompt, response) -> scalar reward."""

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        pooling: str = "last_token",  # last_token | mean | cls
    ):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.base_model(input_ids)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.pooling == "last_token":
            if attention_mask is not None:
                last_pos = attention_mask.sum(-1) - 1
                pooled = hidden[torch.arange(hidden.shape[0]), last_pos]
            else:
                pooled = hidden[:, -1]
        elif self.pooling == "mean":
            if attention_mask is not None:
                pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            else:
                pooled = hidden.mean(1)
        else:  # cls
            pooled = hidden[:, 0]

        return self.value_head(pooled).squeeze(-1)


class PreferenceDataset:
    """Dataset of (prompt, chosen, rejected) preference pairs."""

    def __init__(self):
        self.data: List[Dict[str, torch.Tensor]] = []

    def add(self, prompt: torch.Tensor, chosen: torch.Tensor, rejected: torch.Tensor):
        self.data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class BradleyTerryLoss(nn.Module):
    """Bradley-Terry preference loss for reward model training."""

    def forward(self, reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
        """reward_chosen, reward_rejected: (B,) scalars"""
        # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
        return loss


class ListwiseLoss(nn.Module):
    """Listwise ranking loss using softmax over rewards."""

    def forward(self, rewards: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """rewards: (B, K), labels: (B, K) with 1 for best."""
        log_probs = F.log_softmax(rewards, dim=-1)
        loss = -(labels * log_probs).sum(-1).mean()
        return loss


class GeneralizedAdvantagEstimation:
    """Computes GAE (Schulman et al. 2016) returns and advantages."""

    @staticmethod
    def compute(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards: (T,), values: (T+1,), dones: (T,)
        Returns: returns (T,), advantages (T,)
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=rewards.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:T]
        return returns, advantages


class PPOTrainer:
    """PPO trainer for RLHF with KL penalty against reference policy."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        value_model: nn.Module,
        config: PPOConfig,
    ):
        self.policy = policy_model
        self.ref_policy = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.config = config

        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)
        for param in self.reward_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_model.parameters()),
            lr=config.learning_rate,
        )
        self.kl_coef = config.kl_coef
        self._step = 0

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward = reward_model(x, y) - kl_coef * KL(policy || ref)."""
        full_ids = torch.cat([input_ids, response_ids], dim=1)

        with torch.no_grad():
            rm_reward = self.reward_model(full_ids, attention_mask)

            # KL divergence: sum over response tokens
            policy_logits = self.policy(full_ids)
            ref_logits = self.ref_policy(full_ids)

            if isinstance(policy_logits, tuple):
                policy_logits = policy_logits[0]
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]

            # Only KL over response tokens
            T_prompt = input_ids.shape[1]
            policy_log_probs = F.log_softmax(policy_logits[:, T_prompt:-1], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, T_prompt:-1], dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - policy_log_probs)).sum(-1).sum(-1)

        return rm_reward - self.kl_coef * kl

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Single PPO update step."""
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        T_prompt = input_ids.shape[1]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.config.n_epochs):
            # Policy forward
            logits = self.policy(full_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            response_logits = logits[:, T_prompt - 1:-1]  # shift for next-token prediction
            log_probs = F.log_softmax(response_logits, dim=-1)
            # Gather token log probs
            new_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

            # PPO clipped surrogate
            ratio = (new_log_probs - old_log_probs).exp()
            adv = advantages
            if self.config.normalize_advantages:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value_model(full_ids[:, :T_prompt])
            if isinstance(values, tuple):
                values = values[0]
            if values.dim() > 1:
                values = values.squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(-1).mean()

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Adaptive KL
        with torch.no_grad():
            ref_logits = self.ref_policy(full_ids)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            p = F.softmax(logits[:, T_prompt - 1:-1], dim=-1)
            q = F.softmax(ref_logits[:, T_prompt - 1:-1], dim=-1)
            kl = (p * (p.log() - q.log())).sum(-1).mean().item()

        if self.config.adaptive_kl:
            if kl > 2 * self.config.kl_target:
                self.kl_coef *= 1.5
            elif kl < 0.5 * self.config.kl_target:
                self.kl_coef *= 0.75

        self._step += 1
        n = self.config.n_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "kl": kl,
            "kl_coef": self.kl_coef,
        }


class DirectPreferenceOptimization(nn.Module):
    """DPO (Rafailov et al. 2023): directly optimize preference without reward model."""

    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, beta: float = 0.1):
        super().__init__()
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta

        for param in self.ref.parameters():
            param.requires_grad_(False)

    def _log_prob(self, model: nn.Module, input_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probability of response given input."""
        full = torch.cat([input_ids, response_ids], dim=1)
        logits = model(full)
        if isinstance(logits, tuple):
            logits = logits[0]
        T = input_ids.shape[1]
        response_logits = logits[:, T - 1:-1]
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(-1)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss."""
        # Policy log probs
        pi_chosen = self._log_prob(self.policy, prompt_ids, chosen_ids)
        pi_rejected = self._log_prob(self.policy, prompt_ids, rejected_ids)

        # Reference log probs
        with torch.no_grad():
            ref_chosen = self._log_prob(self.ref, prompt_ids, chosen_ids)
            ref_rejected = self._log_prob(self.ref, prompt_ids, rejected_ids)

        # DPO objective
        logits = self.beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        loss = -F.logsigmoid(logits).mean()

        reward_chosen = (pi_chosen - ref_chosen).mean().item()
        reward_rejected = (pi_rejected - ref_rejected).mean().item()
        accuracy = (logits > 0).float().mean().item()

        return loss, {
            "loss": loss.item(),
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
            "reward_margin": reward_chosen - reward_rejected,
            "accuracy": accuracy,
        }


class IdentityPreferenceOptimization(nn.Module):
    """IPO (Azar et al. 2024): fixes DPO overfitting by using squared loss."""

    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, tau: float = 0.1):
        super().__init__()
        self.policy = policy_model
        self.ref = ref_model
        self.tau = tau

        for param in self.ref.parameters():
            param.requires_grad_(False)

    def _log_prob(self, model, input_ids, response_ids):
        full = torch.cat([input_ids, response_ids], dim=1)
        logits = model(full)
        if isinstance(logits, tuple):
            logits = logits[0]
        T = input_ids.shape[1]
        lp = F.log_softmax(logits[:, T - 1:-1], dim=-1)
        return lp.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

    def forward(self, prompt_ids, chosen_ids, rejected_ids):
        pi_w = self._log_prob(self.policy, prompt_ids, chosen_ids)
        pi_l = self._log_prob(self.policy, prompt_ids, rejected_ids)
        with torch.no_grad():
            ref_w = self._log_prob(self.ref, prompt_ids, chosen_ids)
            ref_l = self._log_prob(self.ref, prompt_ids, rejected_ids)

        h = (pi_w - ref_w) - (pi_l - ref_l)
        loss = ((h - 1 / (2 * self.tau)) ** 2).mean()
        return loss, {"loss": loss.item(), "h": h.mean().item()}


class RewardModelTrainer:
    """Trains reward model from human preference data."""

    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        loss_type: str = "bradley_terry",  # bradley_terry | listwise
    ):
        self.reward_model = reward_model
        self.loss_type = loss_type

        if loss_type == "bradley_terry":
            self.loss_fn = BradleyTerryLoss()
        else:
            self.loss_fn = ListwiseLoss()

        self.optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.step = 0
        self.losses: List[float] = []

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()

        full_chosen = torch.cat([prompt_ids, chosen_ids], dim=1)
        full_rejected = torch.cat([prompt_ids, rejected_ids], dim=1)

        r_chosen = self.reward_model(full_chosen)
        r_rejected = self.reward_model(full_rejected)

        loss = self.loss_fn(r_chosen, r_rejected)
        loss.backward()
        nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        self.losses.append(loss.item())
        return loss.item()

    def evaluate(self, dataset: PreferenceDataset, device: str = "cpu") -> Dict[str, float]:
        self.reward_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for item in dataset.data:
                full_c = torch.cat([item["prompt"], item["chosen"]], dim=0).unsqueeze(0).to(device)
                full_r = torch.cat([item["prompt"], item["rejected"]], dim=0).unsqueeze(0).to(device)
                r_c = self.reward_model(full_c)
                r_r = self.reward_model(full_r)
                correct += (r_c > r_r).item()
                total += 1
        self.reward_model.train()
        return {"preference_accuracy": correct / max(total, 1), "n_evaluated": total}


class ConstitutionalAIFilter(nn.Module):
    """Constitutional AI (Bai et al. 2022) critique-revision filter."""

    def __init__(self, critique_model: nn.Module, revision_model: nn.Module):
        super().__init__()
        self.critique = critique_model
        self.revision = revision_model
        self.principles: List[str] = [
            "Be helpful, harmless, and honest.",
            "Avoid harmful financial advice.",
            "Do not make false predictions about markets.",
            "Acknowledge uncertainty in financial forecasts.",
        ]

    def critique_response(self, prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Generate critique of response w.r.t. principles."""
        combined = torch.cat([prompt, response], dim=1)
        return self.critique(combined)

    def revise_response(
        self,
        prompt: torch.Tensor,
        response: torch.Tensor,
        critique: torch.Tensor,
    ) -> torch.Tensor:
        """Revise response based on critique."""
        combined = torch.cat([prompt, response, critique], dim=1)
        return self.revision(combined)

    def forward(self, prompt: torch.Tensor, response: torch.Tensor, n_revisions: int = 1) -> torch.Tensor:
        current = response
        for _ in range(n_revisions):
            critique = self.critique_response(prompt, current)
            current = self.revise_response(prompt, current, critique)
        return current
