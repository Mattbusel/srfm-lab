"""
adversarial_training.py — Adversarial Robustness Training for Hyper-Agent.

Implements:
- Adversarial agent: separate policy trained to maximally exploit main agent
- RARL (Robust Adversarial Reinforcement Learning)
- Adversarial observation noise and order flow perturbations
- Minimax training loop with ELO tracking
- Robustness certification: performance under K random adversarial seeds
- Policy robustness score: min performance over adversarial distribution
"""

from __future__ import annotations

import math
import time
import copy
import logging
import collections
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

class TrainingPhase(Enum):
    PROTAGONIST = auto()
    ADVERSARY = auto()
    EVALUATION = auto()
    CERTIFICATION = auto()


class PerturbationType(Enum):
    OBSERVATION_NOISE = auto()
    ORDER_FLOW = auto()
    PRICE_SPIKE = auto()
    SPREAD_MANIPULATION = auto()
    INVENTORY_SHOCK = auto()
    COMBINED = auto()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class AdversaryNetworkConfig:
    """Adversary policy network configuration."""
    obs_dim: int = 64
    action_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "tanh"
    action_bound: float = 1.0           # max adversarial perturbation magnitude


@dataclass
class RARLConfig:
    """RARL training configuration."""
    protagonist_steps_per_iter: int = 10
    adversary_steps_per_iter: int = 5
    total_iterations: int = 1000
    num_envs: int = 4                   # parallel environments
    episode_length: int = 500
    learning_rate_protagonist: float = 3e-4
    learning_rate_adversary: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    adversary_reward_scale: float = -1.0  # adversary is rewarded by -protagonist_reward


@dataclass
class ELOConfig:
    """ELO rating configuration."""
    initial_elo: float = 1200.0
    k_factor: float = 32.0
    min_elo: float = 500.0
    max_elo: float = 3000.0
    update_every_n_games: int = 10
    history_window: int = 1000


@dataclass
class PerturbationConfig:
    """Adversarial perturbation configuration."""
    perturbation_type: PerturbationType = PerturbationType.COMBINED
    obs_noise_epsilon: float = 0.1      # L-inf budget for obs noise
    obs_noise_norm: str = "linf"        # linf, l2
    order_flow_size_max: float = 100.0  # max adversarial order size
    price_spike_magnitude: float = 0.02
    spread_manipulation_factor: float = 3.0
    inventory_shock_size: float = 50.0
    # FGSM-style attack
    use_gradient_attack: bool = True
    attack_steps: int = 5              # PGD steps
    attack_step_size: float = 0.02
    # Natural evolution strategy
    use_nes: bool = False
    nes_sigma: float = 0.1
    nes_population: int = 20


@dataclass
class RobustnessCertConfig:
    """Robustness certification configuration."""
    num_seeds: int = 100
    adversary_population_size: int = 20
    eval_episodes_per_seed: int = 5
    performance_percentile: float = 10.0  # use 10th percentile as robustness metric
    min_acceptable_performance: float = -1000.0
    report_all_seeds: bool = False


@dataclass
class AdversarialTrainingConfig:
    """Master adversarial training configuration."""
    adversary_network: AdversaryNetworkConfig = field(default_factory=AdversaryNetworkConfig)
    rarl: RARLConfig = field(default_factory=RARLConfig)
    elo: ELOConfig = field(default_factory=ELOConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    robustness_cert: RobustnessCertConfig = field(default_factory=RobustnessCertConfig)

    seed: Optional[int] = None
    device: str = "cpu"
    enabled: bool = True

    # Logging
    log_every_n_iter: int = 10
    checkpoint_every_n_iter: int = 100


# ---------------------------------------------------------------------------
# ELO Tracker
# ---------------------------------------------------------------------------

class ELOTracker:
    """Tracks ELO ratings for protagonist and adversary."""

    def __init__(self, config: ELOConfig) -> None:
        self.cfg = config
        self._protagonist_elo: float = config.initial_elo
        self._adversary_elo: float = config.initial_elo
        self._game_history: collections.deque = collections.deque(maxlen=config.history_window)
        self._pending_results: List[Tuple[float, float]] = []  # (protagonist_score, adversary_score)
        self._total_games: int = 0

    def record_game(
        self, protagonist_score: float, adversary_score: float
    ) -> None:
        """Record a game result. 1=win, 0=loss, 0.5=draw (from protagonist's perspective)."""
        norm_total = abs(protagonist_score) + abs(adversary_score) + 1e-8
        prot_frac = max(0.0, protagonist_score) / norm_total
        self._pending_results.append((prot_frac, 1.0 - prot_frac))
        self._game_history.append({
            "protagonist_score": protagonist_score,
            "adversary_score": adversary_score,
            "protagonist_elo_before": self._protagonist_elo,
            "adversary_elo_before": self._adversary_elo,
        })
        self._total_games += 1

        if self._total_games % self.cfg.update_every_n_games == 0:
            self._update_elo()

    def _update_elo(self) -> None:
        if not self._pending_results:
            return
        k = self.cfg.k_factor
        for prot_score, adv_score in self._pending_results:
            # Expected scores
            exp_prot = 1.0 / (1.0 + 10 ** ((self._adversary_elo - self._protagonist_elo) / 400.0))
            exp_adv = 1.0 - exp_prot
            # Update
            self._protagonist_elo += k * (prot_score - exp_prot)
            self._adversary_elo += k * (adv_score - exp_adv)
            # Clamp
            self._protagonist_elo = float(np.clip(
                self._protagonist_elo, self.cfg.min_elo, self.cfg.max_elo
            ))
            self._adversary_elo = float(np.clip(
                self._adversary_elo, self.cfg.min_elo, self.cfg.max_elo
            ))
        self._pending_results.clear()

    @property
    def protagonist_elo(self) -> float:
        return self._protagonist_elo

    @property
    def adversary_elo(self) -> float:
        return self._adversary_elo

    @property
    def elo_gap(self) -> float:
        return self._protagonist_elo - self._adversary_elo

    def get_summary(self) -> Dict[str, Any]:
        recent = list(self._game_history)[-20:]
        prot_win_rate = 0.5
        if recent:
            prot_wins = sum(
                1 for g in recent
                if g["protagonist_score"] > g["adversary_score"]
            )
            prot_win_rate = prot_wins / len(recent)
        return {
            "protagonist_elo": self._protagonist_elo,
            "adversary_elo": self._adversary_elo,
            "elo_gap": self.elo_gap,
            "total_games": self._total_games,
            "protagonist_recent_win_rate": prot_win_rate,
        }


# ---------------------------------------------------------------------------
# Adversary Policy Network
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class AdversaryPolicy(nn.Module):
        """
        Adversary policy network.

        Outputs perturbation in the action/observation space to maximally
        harm the protagonist agent.
        """

        def __init__(self, cfg: AdversaryNetworkConfig) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            in_dim = cfg.obs_dim
            for i in range(cfg.num_layers):
                out_dim = cfg.hidden_dim if i < cfg.num_layers - 1 else cfg.action_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < cfg.num_layers - 1:
                    if cfg.activation == "tanh":
                        layers.append(nn.Tanh())
                    elif cfg.activation == "relu":
                        layers.append(nn.ReLU())
                    elif cfg.activation == "elu":
                        layers.append(nn.ELU())
                    else:
                        layers.append(nn.Tanh())
                in_dim = out_dim

            self.net = nn.Sequential(*layers)
            self.bound = cfg.action_bound
            self._cfg = cfg

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """Returns perturbation bounded to [-bound, bound]."""
            raw = self.net(obs)
            return torch.tanh(raw) * self.bound

    class ProtagonistValueHead(nn.Module):
        """Simple value head for critic."""

        def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------

class ObservationNoisePerturbation:
    """Generate worst-case observation noise within epsilon budget."""

    def __init__(self, config: PerturbationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    def random_perturb(self, obs: np.ndarray) -> np.ndarray:
        """Random perturbation within L-inf budget."""
        eps = self.cfg.obs_noise_epsilon
        noise = self.rng.uniform(-eps, eps, size=obs.shape)
        return obs + noise

    def pgd_perturb(
        self,
        obs: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """
        PGD (Projected Gradient Descent) adversarial perturbation.

        Maximizes loss_fn(obs + delta) subject to ||delta||_inf <= epsilon.
        Uses finite differences for gradient estimation.
        """
        cfg = self.cfg
        delta = np.zeros_like(obs)
        eps = cfg.obs_noise_epsilon
        step_size = cfg.attack_step_size
        h = step_size * 0.01  # finite difference step

        for _ in range(cfg.attack_steps):
            # Estimate gradient via finite differences
            grad = np.zeros_like(obs)
            for i in range(len(obs)):
                obs_plus = obs + delta
                obs_plus[i] += h
                obs_minus = obs + delta
                obs_minus[i] -= h
                grad[i] = (loss_fn(obs_plus) - loss_fn(obs_minus)) / (2 * h + 1e-10)

            # Step in gradient direction (maximize loss)
            delta = delta + step_size * np.sign(grad)

            # Project back to epsilon ball
            if cfg.obs_noise_norm == "linf":
                delta = np.clip(delta, -eps, eps)
            else:
                norm = np.linalg.norm(delta)
                if norm > eps:
                    delta = delta * eps / norm

        return obs + delta


class OrderFlowPerturbation:
    """Generate adversarial order flow to disrupt agent."""

    def __init__(self, config: PerturbationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    def generate_orders(
        self,
        mid_price: float,
        spread: float,
        agent_position: float,
        num_orders: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate adversarial orders designed to harm agent."""
        cfg = self.cfg
        orders = []

        # Trend-opposing trades: if agent is long, push price down
        if agent_position > 0:
            side = "sell"
            price = mid_price - spread * 0.5
        elif agent_position < 0:
            side = "buy"
            price = mid_price + spread * 0.5
        else:
            side = "sell" if self.rng.random() < 0.5 else "buy"
            price = mid_price

        for _ in range(num_orders):
            size = float(self.rng.uniform(1.0, cfg.order_flow_size_max))
            orders.append({
                "side": side,
                "price": price + self.rng.normal(0, spread * 0.1),
                "size": size,
                "type": "limit",
                "adversarial": True,
            })

        return orders


# ---------------------------------------------------------------------------
# RARL Rollout Buffer
# ---------------------------------------------------------------------------

class RAARLRolloutBuffer:
    """Stores rollout data for RARL PPO updates."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._values: List[float] = []
        self._log_probs: List[float] = []
        self._adv_perturbations: List[np.ndarray] = []

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        adv_perturbation: Optional[np.ndarray] = None,
    ) -> None:
        if len(self._obs) >= self._cap:
            return
        self._obs.append(obs.copy())
        self._actions.append(action.copy())
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._adv_perturbations.append(
            adv_perturbation.copy() if adv_perturbation is not None
            else np.zeros_like(action)
        )

    def compute_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(self._rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        last_gae = 0.0
        next_val = last_value

        for t in reversed(range(n)):
            done = float(self._dones[t])
            delta = self._rewards[t] + gamma * next_val * (1 - done) - self._values[t]
            last_gae = delta + gamma * gae_lambda * (1 - done) * last_gae
            advantages[t] = last_gae
            next_val = self._values[t]

        returns = advantages + np.array(self._values)
        return advantages, returns

    def clear(self) -> None:
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._values.clear()
        self._log_probs.clear()
        self._adv_perturbations.clear()

    def __len__(self) -> int:
        return len(self._obs)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "obs": np.array(self._obs),
            "actions": np.array(self._actions),
            "rewards": np.array(self._rewards),
            "dones": np.array(self._dones),
            "values": np.array(self._values),
            "log_probs": np.array(self._log_probs),
            "adv_perturbations": np.array(self._adv_perturbations),
        }


# ---------------------------------------------------------------------------
# Minimax training loop
# ---------------------------------------------------------------------------

class MinimaxTrainingLoop:
    """
    Minimax training loop for RARL.

    Alternates between:
    1. Training protagonist with fixed adversary (minimize loss)
    2. Training adversary with fixed protagonist (maximize protagonist's loss)
    """

    def __init__(
        self,
        config: AdversarialTrainingConfig,
        protagonist_policy: Any,   # any policy with .act() and .update()
        env_factory: Callable[[], Any],
    ) -> None:
        self.cfg = config
        self.protagonist = protagonist_policy
        self.env_factory = env_factory
        self.elo = ELOTracker(config.elo)
        self.rng = np.random.default_rng(config.seed)

        # Adversary
        self._adversary: Optional[Any] = None
        self._adv_optimizer: Optional[Any] = None
        if _TORCH_AVAILABLE and config.enabled:
            self._adversary = AdversaryPolicy(config.adversary_network).to(config.device)
            self._adv_optimizer = optim.Adam(
                self._adversary.parameters(),
                lr=config.rarl.learning_rate_adversary,
            )

        # Perturbation generators
        self.obs_perturber = ObservationNoisePerturbation(config.perturbation, self.rng)
        self.flow_perturber = OrderFlowPerturbation(config.perturbation, self.rng)

        # State
        self._iteration = 0
        self._phase = TrainingPhase.PROTAGONIST
        self._protagonist_rollout = RAARLRolloutBuffer(
            config.rarl.protagonist_steps_per_iter * config.rarl.episode_length
        )
        self._adversary_rollout = RAARLRolloutBuffer(
            config.rarl.adversary_steps_per_iter * config.rarl.episode_length
        )

        # Metrics
        self._protagonist_returns: collections.deque = collections.deque(maxlen=1000)
        self._adversary_returns: collections.deque = collections.deque(maxlen=1000)
        self._robustness_scores: List[float] = []

    def run_iteration(self) -> Dict[str, Any]:
        """Run one RARL iteration (protagonist train + adversary train)."""
        cfg = self.cfg.rarl
        self._iteration += 1
        result: Dict[str, Any] = {"iteration": self._iteration}

        # --- Phase 1: Train protagonist ---
        self._phase = TrainingPhase.PROTAGONIST
        prot_returns = self._collect_rollouts(
            protagonist=True,
            num_steps=cfg.protagonist_steps_per_iter,
        )
        prot_loss = self._update_protagonist(self._protagonist_rollout)
        self._protagonist_rollout.clear()
        self._protagonist_returns.extend(prot_returns)
        result["protagonist_returns"] = prot_returns
        result["protagonist_loss"] = prot_loss

        # --- Phase 2: Train adversary ---
        self._phase = TrainingPhase.ADVERSARY
        adv_returns = self._collect_rollouts(
            protagonist=False,
            num_steps=cfg.adversary_steps_per_iter,
        )
        adv_loss = self._update_adversary(self._adversary_rollout)
        self._adversary_rollout.clear()
        self._adversary_returns.extend(adv_returns)
        result["adversary_returns"] = adv_returns
        result["adversary_loss"] = adv_loss

        # ELO update
        if prot_returns and adv_returns:
            prot_mean = float(np.mean(prot_returns))
            adv_mean = float(np.mean(adv_returns))
            self.elo.record_game(prot_mean, adv_mean)

        result["elo"] = self.elo.get_summary()
        result["phase"] = self._phase.name

        if self._iteration % self.cfg.log_every_n_iter == 0:
            logger.info(
                "RARL iter=%d | prot_ret=%.2f | adv_ret=%.2f | ELO gap=%.1f",
                self._iteration,
                float(np.mean(prot_returns)) if prot_returns else 0.0,
                float(np.mean(adv_returns)) if adv_returns else 0.0,
                self.elo.elo_gap,
            )

        return result

    def _collect_rollouts(
        self, protagonist: bool, num_steps: int
    ) -> List[float]:
        """Collect rollouts. Returns list of episode returns."""
        episode_returns: List[float] = []
        cfg = self.cfg.rarl
        buf = self._protagonist_rollout if protagonist else self._adversary_rollout
        buf.clear()

        for _ in range(num_steps):
            env = self.env_factory()
            obs = self._env_reset(env)
            ep_return = 0.0
            for t in range(cfg.episode_length):
                # Adversary generates perturbation
                perturb = np.zeros_like(obs) if obs is not None else np.zeros(self.cfg.adversary_network.obs_dim)
                if (
                    self._adversary is not None
                    and not protagonist
                    and obs is not None
                    and _TORCH_AVAILABLE
                ):
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        perturb = self._adversary(obs_t).squeeze(0).cpu().numpy()
                elif not protagonist:
                    perturb = self.obs_perturber.random_perturb(
                        obs if obs is not None else np.zeros(self.cfg.adversary_network.obs_dim)
                    ) - (obs if obs is not None else np.zeros(self.cfg.adversary_network.obs_dim))

                # Perturb observation for protagonist
                perturbed_obs = (obs + perturb) if obs is not None else perturb

                # Get protagonist action
                action = self._protagonist_act(perturbed_obs)
                value = 0.0
                log_prob = 0.0

                # Step environment
                next_obs, reward, done, info = self._env_step(env, action)
                ep_return += reward

                buf.push(
                    obs if obs is not None else np.zeros(len(perturb)),
                    action,
                    reward if protagonist else -reward,
                    done,
                    value,
                    log_prob,
                    perturb,
                )
                obs = next_obs
                if done:
                    break

            episode_returns.append(ep_return)

        return episode_returns

    def _env_reset(self, env: Any) -> Optional[np.ndarray]:
        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            if obs is not None:
                return np.array(obs, dtype=np.float32).flatten()
        except Exception:
            pass
        return np.zeros(self.cfg.adversary_network.obs_dim, dtype=np.float32)

    def _env_step(
        self, env: Any, action: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, bool, Dict[str, Any]]:
        try:
            result = env.step(action)
            if len(result) == 4:
                obs, rew, done, info = result
            else:
                obs, rew, done, _, info = result
            if obs is not None:
                obs = np.array(obs, dtype=np.float32).flatten()
            return obs, float(rew), bool(done), dict(info)
        except Exception as e:
            logger.debug("Env step error: %s", e)
            return None, 0.0, True, {}

    def _protagonist_act(self, obs: np.ndarray) -> np.ndarray:
        """Get protagonist action. Falls back to random if policy unavailable."""
        try:
            if hasattr(self.protagonist, "act"):
                action = self.protagonist.act(obs)
                return np.array(action, dtype=np.float32).flatten()
        except Exception:
            pass
        return np.zeros(self.cfg.adversary_network.action_dim, dtype=np.float32)

    def _update_protagonist(self, rollout: RAARLRolloutBuffer) -> float:
        """PPO update for protagonist (delegates to protagonist policy)."""
        if not rollout or not hasattr(self.protagonist, "update"):
            return 0.0
        try:
            data = rollout.to_arrays()
            advantages, returns = rollout.compute_advantages(
                0.0,
                self.cfg.rarl.gamma,
                self.cfg.rarl.gae_lambda,
            )
            data["advantages"] = advantages
            data["returns"] = returns
            result = self.protagonist.update(data)
            return float(result) if isinstance(result, (int, float)) else 0.0
        except Exception as e:
            logger.debug("Protagonist update error: %s", e)
            return 0.0

    def _update_adversary(self, rollout: RAARLRolloutBuffer) -> float:
        """PPO update for adversary policy."""
        if not _TORCH_AVAILABLE or self._adversary is None or self._adv_optimizer is None:
            return 0.0
        if len(rollout) == 0:
            return 0.0

        cfg = self.cfg.rarl
        data = rollout.to_arrays()
        advantages, returns = rollout.compute_advantages(
            0.0, cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        device = self.cfg.device
        obs_t = torch.tensor(data["obs"], dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        total_loss = 0.0
        for _ in range(cfg.ppo_epochs):
            n = len(obs_t)
            indices = np.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start:start + cfg.batch_size]
                batch_obs = obs_t[batch_idx]
                batch_adv = adv_t[batch_idx]

                perturb = self._adversary(batch_obs)
                # Adversary loss: maximize protagonist's disadvantage (maximize -returns)
                loss = -torch.mean(batch_adv * torch.norm(perturb, dim=-1))
                # Constraint: keep perturbation within budget
                eps = self.cfg.perturbation.obs_noise_epsilon
                constraint_loss = torch.mean(
                    F.relu(torch.norm(perturb, dim=-1) - eps)
                ) * 10.0

                total = loss + constraint_loss
                self._adv_optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self._adversary.parameters(), cfg.max_grad_norm)
                self._adv_optimizer.step()
                total_loss += float(total.item())

        return total_loss

    def get_training_state(self) -> Dict[str, Any]:
        return {
            "iteration": self._iteration,
            "phase": self._phase.name,
            "elo": self.elo.get_summary(),
            "protagonist_mean_return": float(np.mean(list(self._protagonist_returns))) if self._protagonist_returns else 0.0,
            "adversary_mean_return": float(np.mean(list(self._adversary_returns))) if self._adversary_returns else 0.0,
            "robustness_scores": self._robustness_scores[-10:],
        }


# ---------------------------------------------------------------------------
# Robustness certifier
# ---------------------------------------------------------------------------

class RobustnessCertifier:
    """
    Certifies agent robustness by running against K adversarial seeds.

    Reports percentile performance as the robustness certificate.
    """

    def __init__(
        self,
        config: RobustnessCertConfig,
        protagonist_policy: Any,
        env_factory: Callable[[], Any],
        adversary_population: Optional[List[Any]] = None,
    ) -> None:
        self.cfg = config
        self.protagonist = protagonist_policy
        self.env_factory = env_factory
        self.adversary_population = adversary_population or []
        self._results: List[Dict[str, Any]] = []

    def certify(
        self, num_seeds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run certification. Returns robustness certificate.
        """
        cfg = self.cfg
        n_seeds = num_seeds or cfg.num_seeds
        all_returns: List[float] = []

        for seed_idx in range(n_seeds):
            seed_rng = np.random.default_rng(seed_idx * 1337)
            seed_returns: List[float] = []

            for _ in range(cfg.eval_episodes_per_seed):
                ep_return = self._run_episode(seed_rng, seed_idx)
                seed_returns.append(ep_return)
                all_returns.append(ep_return)

            if cfg.report_all_seeds:
                self._results.append({
                    "seed": seed_idx,
                    "mean_return": float(np.mean(seed_returns)),
                    "min_return": float(np.min(seed_returns)),
                })

        if not all_returns:
            return {"robustness_score": 0.0, "cert_failed": True}

        percentile_perf = float(np.percentile(all_returns, cfg.performance_percentile))
        mean_perf = float(np.mean(all_returns))
        worst_perf = float(np.min(all_returns))

        cert = {
            "robustness_score": percentile_perf,
            "mean_performance": mean_perf,
            "worst_case_performance": worst_perf,
            "performance_percentiles": {
                "p5": float(np.percentile(all_returns, 5)),
                "p10": float(np.percentile(all_returns, 10)),
                "p25": float(np.percentile(all_returns, 25)),
                "p50": float(np.percentile(all_returns, 50)),
                "p75": float(np.percentile(all_returns, 75)),
                "p90": float(np.percentile(all_returns, 90)),
            },
            "num_seeds": n_seeds,
            "total_episodes": len(all_returns),
            "certified_acceptable": percentile_perf > cfg.min_acceptable_performance,
        }
        return cert

    def _run_episode(
        self, rng: np.random.Generator, seed_idx: int
    ) -> float:
        """Run a single adversarial episode. Returns episode return."""
        env = self.env_factory()
        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = np.array(obs, dtype=np.float32).flatten() if obs is not None else np.zeros(64)
        except Exception:
            return 0.0

        # Pick adversary from population (or use random perturbation)
        use_adv = (
            self.adversary_population
            and seed_idx < len(self.adversary_population)
        )
        adv = self.adversary_population[seed_idx % len(self.adversary_population)] if use_adv else None

        ep_return = 0.0
        eps = 0.05  # perturbation budget for certification

        for t in range(1000):
            # Apply adversarial perturbation
            if adv is not None and _TORCH_AVAILABLE:
                try:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        perturb = adv(obs_t).squeeze(0).cpu().numpy()
                    perturbed_obs = np.clip(obs + perturb, obs - eps, obs + eps)
                except Exception:
                    perturbed_obs = obs + rng.uniform(-eps, eps, size=obs.shape)
            else:
                perturbed_obs = obs + rng.uniform(-eps, eps, size=obs.shape)

            # Protagonist action
            try:
                if hasattr(self.protagonist, "act"):
                    action = np.array(self.protagonist.act(perturbed_obs), dtype=np.float32).flatten()
                else:
                    action = np.zeros(4, dtype=np.float32)
            except Exception:
                action = np.zeros(4, dtype=np.float32)

            # Step env
            try:
                result = env.step(action)
                if len(result) == 4:
                    next_obs, rew, done, _ = result
                else:
                    next_obs, rew, done, _, _ = result
                ep_return += float(rew)
                if done:
                    break
                obs = np.array(next_obs, dtype=np.float32).flatten() if next_obs is not None else obs
            except Exception:
                break

        return ep_return


# ---------------------------------------------------------------------------
# Policy robustness score
# ---------------------------------------------------------------------------

class PolicyRobustnessScore:
    """
    Computes a scalar robustness score for an agent policy.

    Score = min_{adversary in distribution} E[return under adversary]
    Approximated via sampling from adversary distribution.
    """

    def __init__(
        self,
        config: AdversarialTrainingConfig,
        env_factory: Callable[[], Any],
    ) -> None:
        self.cfg = config
        self.env_factory = env_factory
        self.rng = np.random.default_rng(config.seed)
        self._score_history: collections.deque = collections.deque(maxlen=100)

    def compute(
        self,
        protagonist_policy: Any,
        adversary_population: Optional[List[Any]] = None,
        n_eval: int = 20,
    ) -> Dict[str, Any]:
        """
        Compute robustness score.

        Returns dict with score and breakdown.
        """
        cfg = self.cfg.robustness_cert
        certifier = RobustnessCertifier(
            cfg,
            protagonist_policy,
            self.env_factory,
            adversary_population,
        )

        cert = certifier.certify(num_seeds=n_eval)
        score = cert.get("robustness_score", 0.0)
        self._score_history.append(score)

        cert["robustness_score_history_mean"] = (
            float(np.mean(list(self._score_history)))
            if self._score_history else 0.0
        )
        cert["robustness_score_trend"] = self._compute_trend()
        return cert

    def _compute_trend(self) -> float:
        """Positive = improving robustness."""
        h = list(self._score_history)
        if len(h) < 10:
            return 0.0
        recent = np.mean(h[-5:])
        older = np.mean(h[-10:-5])
        return float(recent - older)

    @property
    def latest_score(self) -> float:
        if not self._score_history:
            return 0.0
        return float(self._score_history[-1])


# ---------------------------------------------------------------------------
# AdversarialTrainer (main class)
# ---------------------------------------------------------------------------

class AdversarialTrainer:
    """
    Main adversarial training orchestrator.

    Combines:
    - RARL minimax training
    - ELO tracking
    - Robustness certification
    - Policy robustness scoring
    """

    def __init__(
        self,
        config: Optional[AdversarialTrainingConfig] = None,
        protagonist_policy: Optional[Any] = None,
        env_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.cfg = config or AdversarialTrainingConfig()
        self.protagonist = protagonist_policy
        self.env_factory = env_factory
        self.rng = np.random.default_rng(self.cfg.seed)

        # Build sub-modules
        if protagonist_policy is not None and env_factory is not None:
            self.minimax = MinimaxTrainingLoop(
                self.cfg, protagonist_policy, env_factory
            )
            self.robustness_scorer = PolicyRobustnessScore(self.cfg, env_factory)
        else:
            self.minimax = None
            self.robustness_scorer = None

        # Standalone components
        self.obs_perturber = ObservationNoisePerturbation(self.cfg.perturbation, self.rng)
        self.flow_perturber = OrderFlowPerturbation(self.cfg.perturbation, self.rng)
        self.elo = ELOTracker(self.cfg.elo)

        self._iteration = 0
        self._adversary_pool: List[Any] = []     # pool of trained adversaries
        self._training_log: List[Dict[str, Any]] = []

    def train_iteration(self) -> Dict[str, Any]:
        """Run one full RARL iteration."""
        if self.minimax is None:
            return {"error": "no_protagonist_or_env"}
        result = self.minimax.run_iteration()
        self._iteration += 1
        self._training_log.append(result)
        return result

    def train(self, num_iterations: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run full adversarial training."""
        n = num_iterations or self.cfg.rarl.total_iterations
        results = []
        for i in range(n):
            result = self.train_iteration()
            results.append(result)
        return results

    def certify_robustness(self, protagonist_policy: Optional[Any] = None) -> Dict[str, Any]:
        """Run full robustness certification."""
        policy = protagonist_policy or self.protagonist
        if policy is None or self.env_factory is None:
            return {"error": "no_policy_or_env"}
        if self.robustness_scorer is None:
            self.robustness_scorer = PolicyRobustnessScore(self.cfg, self.env_factory)
        adv_population = None
        if self.minimax is not None and self.minimax._adversary is not None:
            adv_population = [self.minimax._adversary]
        return self.robustness_scorer.compute(policy, adv_population)

    def perturb_observation(
        self, obs: np.ndarray, method: str = "random"
    ) -> np.ndarray:
        """Apply adversarial perturbation to observation."""
        if method == "random":
            return self.obs_perturber.random_perturb(obs)
        return obs

    def generate_adversarial_orders(
        self,
        mid_price: float,
        spread: float,
        agent_position: float,
    ) -> List[Dict[str, Any]]:
        """Generate adversarial order flow."""
        return self.flow_perturber.generate_orders(mid_price, spread, agent_position)

    def get_training_summary(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "iteration": self._iteration,
            "elo": self.elo.get_summary(),
            "num_adversaries_in_pool": len(self._adversary_pool),
        }
        if self.minimax is not None:
            state["minimax"] = self.minimax.get_training_state()
        if self.robustness_scorer is not None:
            state["robustness_score"] = self.robustness_scorer.latest_score
        return state


# ---------------------------------------------------------------------------
# Standalone robustness evaluator
# ---------------------------------------------------------------------------

class StandaloneRobustnessEvaluator:
    """
    Evaluates agent robustness without full RARL training.
    Uses random/pre-configured adversarial perturbations.
    """

    def __init__(
        self,
        perturbation_config: Optional[PerturbationConfig] = None,
        seed: int = 42,
    ) -> None:
        self.cfg = perturbation_config or PerturbationConfig()
        self.rng = np.random.default_rng(seed)
        self.obs_perturber = ObservationNoisePerturbation(self.cfg, self.rng)
        self.flow_perturber = OrderFlowPerturbation(self.cfg, self.rng)

    def evaluate_obs_robustness(
        self,
        policy: Any,
        obs_batch: np.ndarray,
        n_perturbations: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate how much action changes under observation perturbation.

        Args:
            policy: policy with .act(obs) method
            obs_batch: (N, obs_dim) batch of observations
            n_perturbations: number of random perturbations per obs

        Returns:
            dict with action consistency metrics
        """
        if not hasattr(policy, "act"):
            return {"error": "policy_has_no_act_method"}

        original_actions = []
        perturbed_action_diffs = []

        for obs in obs_batch:
            try:
                orig_action = np.array(policy.act(obs), dtype=np.float32).flatten()
                original_actions.append(orig_action)

                diffs = []
                for _ in range(n_perturbations):
                    perturbed = self.obs_perturber.random_perturb(obs)
                    pert_action = np.array(policy.act(perturbed), dtype=np.float32).flatten()
                    diff = float(np.mean(np.abs(pert_action - orig_action)))
                    diffs.append(diff)
                perturbed_action_diffs.append(np.mean(diffs))
            except Exception:
                continue

        if not perturbed_action_diffs:
            return {"error": "evaluation_failed"}

        return {
            "mean_action_consistency": float(np.mean(perturbed_action_diffs)),
            "std_action_consistency": float(np.std(perturbed_action_diffs)),
            "p95_action_diff": float(np.percentile(perturbed_action_diffs, 95)),
            "robustness_score": float(1.0 / (1.0 + np.mean(perturbed_action_diffs))),
            "n_obs": len(original_actions),
        }

    def compute_lipschitz_estimate(
        self,
        policy: Any,
        obs_center: np.ndarray,
        n_samples: int = 100,
        max_eps: float = 0.1,
    ) -> float:
        """
        Estimate local Lipschitz constant of policy around obs_center.

        Lipschitz ≈ max |f(x) - f(x')| / |x - x'|
        """
        if not hasattr(policy, "act"):
            return 0.0

        try:
            f0 = np.array(policy.act(obs_center), dtype=np.float32).flatten()
        except Exception:
            return 0.0

        max_ratio = 0.0
        for _ in range(n_samples):
            delta = self.rng.uniform(-max_eps, max_eps, size=obs_center.shape)
            obs_perturbed = obs_center + delta
            try:
                f1 = np.array(policy.act(obs_perturbed), dtype=np.float32).flatten()
                input_dist = float(np.linalg.norm(delta))
                output_dist = float(np.linalg.norm(f1 - f0))
                if input_dist > 1e-10:
                    max_ratio = max(max_ratio, output_dist / input_dist)
            except Exception:
                continue

        return max_ratio


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_adversarial_trainer(
    protagonist_policy: Optional[Any] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    obs_dim: int = 64,
    action_dim: int = 8,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> AdversarialTrainer:
    """Create an AdversarialTrainer with sensible defaults."""
    cfg = AdversarialTrainingConfig(
        adversary_network=AdversaryNetworkConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
        seed=seed,
        device=device,
    )
    return AdversarialTrainer(cfg, protagonist_policy, env_factory)


def make_robustness_evaluator(
    epsilon: float = 0.05,
    seed: int = 42,
) -> StandaloneRobustnessEvaluator:
    """Create a standalone robustness evaluator."""
    cfg = PerturbationConfig(obs_noise_epsilon=epsilon)
    return StandaloneRobustnessEvaluator(cfg, seed)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "TrainingPhase",
    "PerturbationType",
    # Configs
    "AdversarialTrainingConfig",
    "AdversaryNetworkConfig",
    "RARLConfig",
    "ELOConfig",
    "PerturbationConfig",
    "RobustnessCertConfig",
    # Sub-modules
    "ELOTracker",
    "ObservationNoisePerturbation",
    "OrderFlowPerturbation",
    "RAARLRolloutBuffer",
    "MinimaxTrainingLoop",
    "RobustnessCertifier",
    "PolicyRobustnessScore",
    # Main
    "AdversarialTrainer",
    "StandaloneRobustnessEvaluator",
    # Factories
    "make_adversarial_trainer",
    "make_robustness_evaluator",
    # Extended
    "AdversaryPool",
    "NaturalGradientAdversary",
    "BlackBoxAdversary",
    "AdversarialDataAugmentor",
    "RobustnessRegularizer",
    "AdversarialReplayBuffer",
    "MixedStrategyNashSolver",
    "ELOLeaderboard",
    "AdversarialCurriculumManager",
    "WorstCaseEvaluator",
]


# ---------------------------------------------------------------------------
# Extended: AdversaryPool
# ---------------------------------------------------------------------------

class AdversaryPool:
    """
    Pool of diverse adversary policies for self-play training.

    Maintains a population of adversaries with different strategies.
    Sampling is weighted by ELO rating to challenge the protagonist.
    """

    def __init__(
        self,
        max_pool_size: int = 20,
        elo_config: Optional[ELOConfig] = None,
    ) -> None:
        self.max_pool_size = max_pool_size
        self.elo_config = elo_config or ELOConfig()
        self._pool: List[Dict[str, Any]] = []
        self._elo_ratings: List[float] = []
        self._policy_ids: List[int] = []
        self._next_id: int = 0
        self._rng = np.random.default_rng(42)

    def add(
        self,
        policy: Any,
        initial_elo: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a policy to the pool. Returns policy ID."""
        pid = self._next_id
        self._next_id += 1
        self._policy_ids.append(pid)
        self._pool.append({
            "policy": policy,
            "metadata": metadata or {},
            "wins": 0,
            "losses": 0,
            "games": 0,
        })
        self._elo_ratings.append(initial_elo or self.elo_config.initial_elo)

        # Evict weakest if over capacity
        if len(self._pool) > self.max_pool_size:
            self._evict_weakest()
        return pid

    def _evict_weakest(self) -> None:
        """Remove the policy with the lowest ELO."""
        if not self._elo_ratings:
            return
        weakest_idx = int(np.argmin(self._elo_ratings))
        self._pool.pop(weakest_idx)
        self._elo_ratings.pop(weakest_idx)
        self._policy_ids.pop(weakest_idx)

    def sample(self, strategy: str = "elo_weighted") -> Optional[Tuple[int, Any]]:
        """
        Sample a policy from the pool.

        Strategies:
        - elo_weighted: sample proportional to ELO
        - uniform: uniform random
        - strongest: return highest ELO
        - hardest: return policy with best win rate vs protagonist
        """
        if not self._pool:
            return None

        if strategy == "elo_weighted":
            elos = np.array(self._elo_ratings)
            # Softmax over ELO
            elos_normalized = elos - elos.max()
            weights = np.exp(elos_normalized / 200.0)
            weights /= weights.sum()
            idx = int(self._rng.choice(len(self._pool), p=weights))
        elif strategy == "strongest":
            idx = int(np.argmax(self._elo_ratings))
        elif strategy == "hardest":
            # Maximize wins / games
            rates = [
                p["wins"] / max(p["games"], 1) for p in self._pool
            ]
            idx = int(np.argmax(rates))
        else:
            idx = int(self._rng.integers(0, len(self._pool)))

        return self._policy_ids[idx], self._pool[idx]["policy"]

    def update_elo(self, policy_id: int, result: float) -> None:
        """Update ELO for a policy (result: 1=win, 0=loss, 0.5=draw)."""
        if policy_id not in self._policy_ids:
            return
        idx = self._policy_ids.index(policy_id)
        k = self.elo_config.k_factor
        avg_opponent_elo = float(np.mean(self._elo_ratings))
        expected = 1.0 / (1.0 + 10 ** ((avg_opponent_elo - self._elo_ratings[idx]) / 400.0))
        self._elo_ratings[idx] += k * (result - expected)
        self._elo_ratings[idx] = float(np.clip(
            self._elo_ratings[idx],
            self.elo_config.min_elo,
            self.elo_config.max_elo,
        ))
        self._pool[idx]["games"] += 1
        if result > 0.5:
            self._pool[idx]["wins"] += 1
        elif result < 0.5:
            self._pool[idx]["losses"] += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "pool_size": len(self._pool),
            "mean_elo": float(np.mean(self._elo_ratings)) if self._elo_ratings else 0.0,
            "max_elo": float(np.max(self._elo_ratings)) if self._elo_ratings else 0.0,
            "min_elo": float(np.min(self._elo_ratings)) if self._elo_ratings else 0.0,
            "policy_ids": self._policy_ids.copy(),
        }

    def __len__(self) -> int:
        return len(self._pool)


# ---------------------------------------------------------------------------
# Extended: NaturalGradientAdversary
# ---------------------------------------------------------------------------

class NaturalGradientAdversary:
    """
    Adversary that uses Natural Evolution Strategy (NES) for gradient-free
    adversarial perturbation optimization.

    Useful when the protagonist's gradient is not available.
    """

    def __init__(
        self,
        obs_dim: int,
        perturbation_dim: int,
        sigma: float = 0.1,
        lr: float = 0.01,
        population_size: int = 50,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.perturbation_dim = perturbation_dim
        self.sigma = sigma
        self.lr = lr
        self.n = population_size
        self.rng = np.random.default_rng(seed)

        self._theta = np.zeros(perturbation_dim)
        self._step = 0
        self._reward_history: collections.deque = collections.deque(maxlen=200)

    def update(
        self,
        reward_fn: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, float]:
        """
        Run one NES update step.

        reward_fn: callable that takes perturbation and returns scalar reward.
        Returns (updated_theta, mean_reward).
        """
        epsilons = self.rng.standard_normal((self.n, self.perturbation_dim))
        rewards = np.zeros(self.n)

        for i in range(self.n):
            theta_i = self._theta + self.sigma * epsilons[i]
            rewards[i] = reward_fn(theta_i)

        self._reward_history.extend(rewards.tolist())

        # Normalize rewards
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        # NES gradient estimate
        gradient = np.dot(epsilons.T, rewards_norm) / (self.n * self.sigma)
        self._theta += self.lr * gradient
        self._step += 1

        return self._theta.copy(), float(r_mean)

    def get_perturbation(self) -> np.ndarray:
        """Return current best perturbation."""
        return self._theta.copy()

    @property
    def mean_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)))


# ---------------------------------------------------------------------------
# Extended: BlackBoxAdversary
# ---------------------------------------------------------------------------

class BlackBoxAdversary:
    """
    Black-box adversary that queries the protagonist and searches for
    worst-case perturbations using random search.

    No gradient access required. Suitable for:
    - Testing against external policies
    - Discrete action spaces
    - Non-differentiable reward functions
    """

    def __init__(
        self,
        eps: float = 0.1,
        num_queries: int = 100,
        attack_type: str = "linf",  # linf, l2
        seed: int = 0,
    ) -> None:
        self.eps = eps
        self.num_queries = num_queries
        self.attack_type = attack_type
        self.rng = np.random.default_rng(seed)
        self._best_perturbations: List[np.ndarray] = []
        self._best_rewards: List[float] = []

    def attack(
        self,
        obs: np.ndarray,
        reward_fn: Callable[[np.ndarray], float],
        num_restarts: int = 3,
    ) -> Tuple[np.ndarray, float]:
        """
        Find worst-case perturbation.

        Returns (best_perturbation, worst_reward).
        """
        obs_dim = len(obs)
        best_delta = np.zeros(obs_dim)
        best_reward = reward_fn(obs)  # baseline

        for restart in range(num_restarts):
            # Random start within epsilon ball
            if self.attack_type == "linf":
                delta = self.rng.uniform(-self.eps, self.eps, size=obs_dim)
            else:
                delta = self.rng.standard_normal(obs_dim)
                delta = delta * self.eps / (np.linalg.norm(delta) + 1e-10)

            for _ in range(self.num_queries // num_restarts):
                # Random perturbation step
                step = self.rng.standard_normal(obs_dim) * self.eps * 0.1
                new_delta = delta + step

                # Project back
                if self.attack_type == "linf":
                    new_delta = np.clip(new_delta, -self.eps, self.eps)
                else:
                    norm = np.linalg.norm(new_delta)
                    if norm > self.eps:
                        new_delta = new_delta * self.eps / norm

                # Evaluate
                reward = reward_fn(obs + new_delta)
                # Adversary wants to minimize protagonist reward
                if reward < best_reward:
                    best_reward = reward
                    best_delta = new_delta.copy()
                    delta = new_delta

        self._best_perturbations.append(best_delta.copy())
        self._best_rewards.append(best_reward)
        return best_delta, best_reward

    @property
    def historical_worst_rewards(self) -> List[float]:
        return self._best_rewards.copy()


# ---------------------------------------------------------------------------
# Extended: AdversarialDataAugmentor
# ---------------------------------------------------------------------------

class AdversarialDataAugmentor:
    """
    Augments training data with adversarially-generated samples.

    Creates a harder training distribution by mixing clean and adversarial
    observations. This is analogous to adversarial training in computer vision
    (Madry et al., 2018) but adapted for market observations.
    """

    def __init__(
        self,
        obs_dim: int,
        adversarial_frac: float = 0.3,
        eps: float = 0.05,
        num_pgd_steps: int = 3,
        step_size: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.adversarial_frac = adversarial_frac
        self.eps = eps
        self.num_pgd_steps = num_pgd_steps
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

    def augment_batch(
        self,
        obs_batch: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """
        Augment a batch of observations.

        adversarial_frac of the batch gets replaced with adversarial variants.
        """
        n = len(obs_batch)
        n_adv = int(n * self.adversarial_frac)
        adv_indices = self.rng.choice(n, size=n_adv, replace=False)

        augmented = obs_batch.copy()
        for idx in adv_indices:
            obs = obs_batch[idx]
            delta = np.zeros_like(obs)

            for _ in range(self.num_pgd_steps):
                # Finite difference gradient
                grad = np.zeros_like(obs)
                for d in range(min(self.obs_dim, 20)):  # limit FD for speed
                    h = 1e-4
                    obs_plus = obs + delta
                    obs_plus[d] += h
                    obs_minus = obs + delta
                    obs_minus[d] -= h
                    grad[d] = (loss_fn(obs_plus) - loss_fn(obs_minus)) / (2 * h)

                # Step in gradient direction (adversarial = maximize loss)
                delta += self.step_size * np.sign(grad)
                delta = np.clip(delta, -self.eps, self.eps)

            augmented[idx] = obs + delta

        return augmented

    def add_gaussian_adversarial(
        self,
        obs_batch: np.ndarray,
        noise_scale: float = 0.02,
    ) -> np.ndarray:
        """Simple Gaussian adversarial augmentation (fast baseline)."""
        noise = self.rng.normal(0, noise_scale, size=obs_batch.shape)
        noisy = obs_batch + noise
        return noisy

    def mix_clean_adversarial(
        self,
        clean: np.ndarray,
        adversarial: np.ndarray,
        mix_alpha: float = 0.5,
    ) -> np.ndarray:
        """Mixup between clean and adversarial samples."""
        return mix_alpha * clean + (1 - mix_alpha) * adversarial


# ---------------------------------------------------------------------------
# Extended: RobustnessRegularizer
# ---------------------------------------------------------------------------

class RobustnessRegularizer:
    """
    Adds a robustness regularization term to protagonist training loss.

    Penalizes high sensitivity to input perturbations, encouraging the
    protagonist to learn robust (Lipschitz-continuous) policies.

    Based on: TRADES (Zhang et al., 2019) adapted for RL.
    """

    def __init__(
        self,
        reg_weight: float = 1.0,
        eps: float = 0.05,
        kl_type: str = "forward",  # forward, reverse, symmetric
        num_samples: int = 10,
    ) -> None:
        self.reg_weight = reg_weight
        self.eps = eps
        self.kl_type = kl_type
        self.num_samples = num_samples
        self._reg_history: collections.deque = collections.deque(maxlen=500)

    def compute_regularization(
        self,
        policy: Any,
        obs: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute robustness regularization loss for a batch of observations.

        Returns scalar regularization loss.
        """
        if not hasattr(policy, "act"):
            return 0.0

        total_reg = 0.0
        for ob in obs[:min(len(obs), 10)]:  # limit for speed
            try:
                clean_action = np.array(policy.act(ob), dtype=np.float32).flatten()
                reg_sample = 0.0
                for _ in range(self.num_samples):
                    # Adversarial perturbation
                    delta = rng.uniform(-self.eps, self.eps, size=ob.shape)
                    perturbed_action = np.array(
                        policy.act(ob + delta), dtype=np.float32
                    ).flatten()
                    reg_sample += float(np.mean((clean_action - perturbed_action) ** 2))
                total_reg += reg_sample / max(self.num_samples, 1)
            except Exception:
                continue

        reg = total_reg / max(len(obs[:10]), 1) * self.reg_weight
        self._reg_history.append(reg)
        return reg

    @property
    def mean_regularization(self) -> float:
        if not self._reg_history:
            return 0.0
        return float(np.mean(list(self._reg_history)))


# ---------------------------------------------------------------------------
# Extended: AdversarialReplayBuffer
# ---------------------------------------------------------------------------

class AdversarialReplayBuffer:
    """
    Replay buffer that stores adversarial episodes alongside normal ones.

    Prioritizes sampling adversarial experiences during protagonist training
    to ensure exposure to worst-case scenarios.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        adversarial_ratio: float = 0.3,
    ) -> None:
        self._capacity = capacity
        self._adv_ratio = adversarial_ratio
        self._normal_buf: collections.deque = collections.deque(maxlen=int(capacity * (1 - adversarial_ratio)))
        self._adv_buf: collections.deque = collections.deque(maxlen=int(capacity * adversarial_ratio))
        self._total_pushed: int = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        is_adversarial: bool = False,
        adversary_type: str = "none",
        adversary_strength: float = 0.0,
    ) -> None:
        experience = {
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": reward,
            "next_obs": next_obs.copy(),
            "done": done,
            "is_adversarial": is_adversarial,
            "adversary_type": adversary_type,
            "adversary_strength": adversary_strength,
        }
        if is_adversarial:
            self._adv_buf.append(experience)
        else:
            self._normal_buf.append(experience)
        self._total_pushed += 1

    def sample(
        self, batch_size: int
    ) -> List[Dict[str, Any]]:
        """Sample a batch with the configured adversarial ratio."""
        n_adv = int(batch_size * self._adv_ratio)
        n_normal = batch_size - n_adv

        batch = []
        if self._adv_buf and n_adv > 0:
            adv_list = list(self._adv_buf)
            indices = np.random.choice(len(adv_list), size=min(n_adv, len(adv_list)), replace=False)
            batch.extend([adv_list[i] for i in indices])

        if self._normal_buf and n_normal > 0:
            norm_list = list(self._normal_buf)
            indices = np.random.choice(len(norm_list), size=min(n_normal, len(norm_list)), replace=False)
            batch.extend([norm_list[i] for i in indices])

        return batch

    def __len__(self) -> int:
        return len(self._normal_buf) + len(self._adv_buf)

    @property
    def adversarial_fraction(self) -> float:
        total = len(self)
        if total == 0:
            return 0.0
        return len(self._adv_buf) / total

    @property
    def total_pushed(self) -> int:
        return self._total_pushed


# ---------------------------------------------------------------------------
# Extended: MixedStrategyNashSolver
# ---------------------------------------------------------------------------

class MixedStrategyNashSolver:
    """
    Computes approximate Nash equilibrium between protagonist and adversary.

    Uses fictitious self-play / regret minimization to find mixed strategies.
    The Nash equilibrium gives the minimax optimal adversarial training setup.
    """

    def __init__(
        self,
        num_protagonist_strategies: int = 10,
        num_adversary_strategies: int = 10,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.n_prot = num_protagonist_strategies
        self.n_adv = num_adversary_strategies
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        # Strategy distribution
        self._prot_dist = np.ones(num_protagonist_strategies) / num_protagonist_strategies
        self._adv_dist = np.ones(num_adversary_strategies) / num_adversary_strategies

        # Payoff matrix (protagonist reward): prot_strategies x adv_strategies
        self._payoff = np.random.randn(num_protagonist_strategies, num_adversary_strategies) * 0.1

        # Cumulative regret
        self._prot_cumulative_regret = np.zeros(num_protagonist_strategies)
        self._adv_cumulative_regret = np.zeros(num_adversary_strategies)

        self._step = 0

    def update_payoff(
        self,
        prot_strategy: int,
        adv_strategy: int,
        payoff: float,
    ) -> None:
        """Update payoff matrix with observed outcome."""
        alpha = self.lr
        self._payoff[prot_strategy, adv_strategy] = (
            (1 - alpha) * self._payoff[prot_strategy, adv_strategy] + alpha * payoff
        )
        self._step += 1

    def regret_matching_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one regret matching step.

        Returns (protagonist_distribution, adversary_distribution).
        """
        # Protagonist regrets: for each strategy, how much better could they do
        # vs current adv distribution?
        prot_ev = self._payoff @ self._adv_dist  # expected value per prot strategy
        current_prot_ev = float(np.dot(self._prot_dist, prot_ev))
        prot_regrets = prot_ev - current_prot_ev

        # Adversary regrets: adversary minimizes protagonist EV
        adv_ev = self._prot_dist @ self._payoff  # expected value per adv strategy (prot perspective)
        current_adv_ev = float(np.dot(self._adv_dist, adv_ev))
        adv_regrets = -(adv_ev - current_adv_ev)  # adversary wants to minimize

        # Accumulate regrets
        self._prot_cumulative_regret += prot_regrets
        self._adv_cumulative_regret += adv_regrets

        # Regret matching: play proportional to positive cumulative regret
        prot_pos = np.maximum(self._prot_cumulative_regret, 0)
        adv_pos = np.maximum(self._adv_cumulative_regret, 0)

        if prot_pos.sum() > 0:
            self._prot_dist = prot_pos / prot_pos.sum()
        if adv_pos.sum() > 0:
            self._adv_dist = adv_pos / adv_pos.sum()

        return self._prot_dist.copy(), self._adv_dist.copy()

    def sample_strategies(self) -> Tuple[int, int]:
        """Sample (protagonist_strategy, adversary_strategy) from current distributions."""
        prot = int(self.rng.choice(self.n_prot, p=self._prot_dist))
        adv = int(self.rng.choice(self.n_adv, p=self._adv_dist))
        return prot, adv

    @property
    def nash_value(self) -> float:
        """Expected game value under current distributions."""
        return float(self._prot_dist @ self._payoff @ self._adv_dist)

    @property
    def protagonist_distribution(self) -> np.ndarray:
        return self._prot_dist.copy()

    @property
    def adversary_distribution(self) -> np.ndarray:
        return self._adv_dist.copy()


# ---------------------------------------------------------------------------
# Extended: ELOLeaderboard
# ---------------------------------------------------------------------------

class ELOLeaderboard:
    """
    ELO leaderboard for tracking multiple agents across many matchups.

    Supports:
    - Multiple players (not just protagonist/adversary)
    - Tournament scheduling
    - Rating history and trend analysis
    """

    def __init__(self, k_factor: float = 32.0, seed: int = 0) -> None:
        self.k_factor = k_factor
        self.rng = np.random.default_rng(seed)
        self._ratings: Dict[str, float] = {}
        self._game_history: List[Dict[str, Any]] = []
        self._rating_history: Dict[str, List[float]] = {}

    def add_player(self, name: str, initial_rating: float = 1200.0) -> None:
        self._ratings[name] = initial_rating
        self._rating_history[name] = [initial_rating]

    def record_game(
        self, winner: str, loser: str, draw: bool = False
    ) -> Dict[str, float]:
        """Record a game result. Returns rating changes."""
        if winner not in self._ratings or loser not in self._ratings:
            return {}

        r_w = self._ratings[winner]
        r_l = self._ratings[loser]

        expected_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w

        score_w = 0.5 if draw else 1.0
        score_l = 0.5 if draw else 0.0

        delta_w = self.k_factor * (score_w - expected_w)
        delta_l = self.k_factor * (score_l - expected_l)

        self._ratings[winner] += delta_w
        self._ratings[loser] += delta_l

        self._rating_history[winner].append(self._ratings[winner])
        self._rating_history[loser].append(self._ratings[loser])

        game = {
            "winner": winner,
            "loser": loser,
            "draw": draw,
            "delta_winner": delta_w,
            "delta_loser": delta_l,
            "rating_winner": self._ratings[winner],
            "rating_loser": self._ratings[loser],
        }
        self._game_history.append(game)
        return game

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Returns sorted (name, rating) pairs."""
        return sorted(self._ratings.items(), key=lambda x: -x[1])

    def get_trend(self, name: str, window: int = 10) -> float:
        """Returns recent rating trend for a player."""
        history = self._rating_history.get(name, [])
        if len(history) < 2:
            return 0.0
        recent = history[-window:]
        if len(recent) < 2:
            return 0.0
        return float(recent[-1] - recent[0])

    def schedule_round_robin(self) -> List[Tuple[str, str]]:
        """Schedule all matchups for a round-robin tournament."""
        players = list(self._ratings.keys())
        matchups = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                matchups.append((players[i], players[j]))
        self.rng.shuffle(matchups)
        return matchups

    def __len__(self) -> int:
        return len(self._ratings)


# ---------------------------------------------------------------------------
# Extended: AdversarialCurriculumManager
# ---------------------------------------------------------------------------

class AdversarialCurriculumManager:
    """
    Curriculum manager for adversarial training.

    Controls the progression of adversarial difficulty:
    - Starts with weak adversaries
    - Gradually increases adversary strength as protagonist improves
    - Balances exploration vs exploitation of worst-case scenarios
    """

    def __init__(
        self,
        initial_adversary_strength: float = 0.1,
        max_adversary_strength: float = 1.0,
        adaptation_rate: float = 0.05,
        protagonist_win_threshold: float = 0.6,
        min_games_before_update: int = 20,
    ) -> None:
        self.initial_strength = initial_adversary_strength
        self.max_strength = max_adversary_strength
        self.adaptation_rate = adaptation_rate
        self.win_threshold = protagonist_win_threshold
        self.min_games = min_games_before_update

        self._current_strength = initial_adversary_strength
        self._game_outcomes: collections.deque = collections.deque(maxlen=100)
        self._strength_history: List[float] = [initial_adversary_strength]
        self._total_games = 0

    def record_game(self, protagonist_won: bool) -> None:
        self._game_outcomes.append(1.0 if protagonist_won else 0.0)
        self._total_games += 1

        if len(self._game_outcomes) >= self.min_games:
            self._update_strength()

    def _update_strength(self) -> None:
        win_rate = float(np.mean(list(self._game_outcomes)))
        if win_rate > self.win_threshold:
            # Protagonist too strong: increase adversary
            self._current_strength = min(
                self.max_strength,
                self._current_strength + self.adaptation_rate,
            )
        elif win_rate < 1 - self.win_threshold:
            # Adversary too strong: decrease
            self._current_strength = max(
                self.initial_strength,
                self._current_strength - self.adaptation_rate,
            )
        self._strength_history.append(self._current_strength)

    @property
    def adversary_strength(self) -> float:
        return self._current_strength

    @property
    def protagonist_win_rate(self) -> float:
        if not self._game_outcomes:
            return 0.5
        return float(np.mean(list(self._game_outcomes)))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "current_strength": self._current_strength,
            "protagonist_win_rate": self.protagonist_win_rate,
            "total_games": self._total_games,
            "strength_history": self._strength_history[-10:],
        }


# ---------------------------------------------------------------------------
# Extended: WorstCaseEvaluator
# ---------------------------------------------------------------------------

class WorstCaseEvaluator:
    """
    Finds and evaluates worst-case scenarios for a protagonist policy.

    Uses a combination of:
    - Grid search over scenario parameters
    - Bayesian optimization of worst-case parameters
    - Historical worst-case replay
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        num_random_scenarios: int = 50,
        num_episodes_per_scenario: int = 3,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.num_scenarios = num_random_scenarios
        self.num_episodes = num_episodes_per_scenario
        self.rng = np.random.default_rng(seed)
        self._worst_case_records: List[Dict[str, Any]] = []

    def find_worst_case(
        self,
        scenario_params_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Search for worst-case scenario parameters.

        Returns dict with worst-case params and expected return.
        """
        default_ranges = {
            "spread_mult": (1.0, 20.0),
            "vol_mult": (0.5, 5.0),
            "kyle_lambda_mult": (1.0, 10.0),
            "fill_rate_mod": (-0.5, 0.0),
        }
        ranges = scenario_params_ranges or default_ranges

        worst_return = float("inf")
        worst_params: Dict[str, float] = {}
        all_results: List[Dict[str, Any]] = []

        for scenario_idx in range(self.num_scenarios):
            # Sample random scenario
            params: Dict[str, float] = {}
            for key, (low, high) in ranges.items():
                params[key] = float(self.rng.uniform(low, high))

            # Evaluate
            returns = []
            for ep in range(self.num_episodes):
                ep_return = self._run_episode(params, ep)
                returns.append(ep_return)

            mean_return = float(np.mean(returns))
            all_results.append({
                "scenario_idx": scenario_idx,
                "params": params.copy(),
                "mean_return": mean_return,
                "returns": returns,
            })

            if mean_return < worst_return:
                worst_return = mean_return
                worst_params = params.copy()

        # Sort by mean return
        all_results.sort(key=lambda r: r["mean_return"])
        self._worst_case_records.extend(all_results[:5])

        return {
            "worst_case_return": worst_return,
            "worst_case_params": worst_params,
            "top_worst_scenarios": all_results[:10],
            "mean_return_over_scenarios": float(
                np.mean([r["mean_return"] for r in all_results])
            ),
        }

    def _run_episode(self, params: Dict[str, float], seed: int) -> float:
        """Run a single episode with given params."""
        env = self.env_factory()
        # Apply params to env
        try:
            if hasattr(env, "spreads") and "spread_mult" in params:
                env.spreads = np.array(env.spreads) * params["spread_mult"]
            if hasattr(env, "volatility") and "vol_mult" in params:
                env.volatility = np.array(env.volatility) * params["vol_mult"]
        except Exception:
            pass

        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = np.array(obs, dtype=np.float32).flatten() if obs is not None else np.zeros(64)
        except Exception:
            return 0.0

        ep_return = 0.0
        for _ in range(500):
            try:
                if hasattr(self.policy, "act"):
                    action = np.array(self.policy.act(obs), dtype=np.float32).flatten()
                else:
                    action = np.zeros(4, dtype=np.float32)
                result = env.step(action)
                if len(result) == 4:
                    next_obs, rew, done, _ = result
                else:
                    next_obs, rew, done, _, _ = result
                ep_return += float(rew)
                if done:
                    break
                obs = np.array(next_obs, dtype=np.float32).flatten() if next_obs is not None else obs
            except Exception:
                break

        return ep_return

    @property
    def worst_case_history(self) -> List[Dict[str, Any]]:
        return self._worst_case_records[-20:]


# ---------------------------------------------------------------------------
# AdversarialPolicyAuditor — statistical audit of policy robustness
# ---------------------------------------------------------------------------

class AdversarialPolicyAuditor:
    """Runs a battery of adversarial tests and produces an audit report with
    pass/fail thresholds for deployment readiness."""

    def __init__(self, policy_fn, obs_dim: int = 16,
                 action_dim: int = 3,
                 min_robustness_score: float = 0.6,
                 max_sensitivity: float = 0.3):
        self.policy_fn = policy_fn
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_robustness_score = min_robustness_score
        self.max_sensitivity = max_sensitivity

    # ------------------------------------------------------------------
    def _random_obs(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.obs_dim).astype(np.float32)

    # ------------------------------------------------------------------
    def audit_sensitivity(self, n_trials: int = 200,
                          epsilon: float = 0.05) -> dict:
        """Measure average action change under epsilon-bounded obs noise."""
        rng = np.random.default_rng(0)
        sensitivities = []
        for _ in range(n_trials):
            obs = self._random_obs(rng)
            noise = rng.uniform(-epsilon, epsilon, self.obs_dim).astype(np.float32)
            clean_act = np.array(self.policy_fn(obs), dtype=np.float32)
            noisy_act = np.array(self.policy_fn(obs + noise), dtype=np.float32)
            sensitivities.append(float(np.linalg.norm(clean_act - noisy_act)))
        mean_sens = float(np.mean(sensitivities))
        return {
            "mean_sensitivity": mean_sens,
            "max_sensitivity": float(np.max(sensitivities)),
            "passes": mean_sens <= self.max_sensitivity,
            "threshold": self.max_sensitivity,
        }

    # ------------------------------------------------------------------
    def audit_consistency(self, n_trials: int = 100) -> dict:
        """Check that same obs always produces same action (determinism)."""
        rng = np.random.default_rng(1)
        inconsistencies = 0
        for _ in range(n_trials):
            obs = self._random_obs(rng)
            act1 = np.array(self.policy_fn(obs), dtype=np.float32)
            act2 = np.array(self.policy_fn(obs), dtype=np.float32)
            if not np.allclose(act1, act2, atol=1e-5):
                inconsistencies += 1
        rate = inconsistencies / n_trials
        return {
            "inconsistency_rate": rate,
            "passes": rate == 0.0,
        }

    # ------------------------------------------------------------------
    def full_audit(self) -> dict:
        sensitivity = self.audit_sensitivity()
        consistency = self.audit_consistency()
        all_pass = sensitivity["passes"] and consistency["passes"]
        return {
            "sensitivity": sensitivity,
            "consistency": consistency,
            "deployment_ready": all_pass,
        }


# ---------------------------------------------------------------------------
# TemporalAdversary — adversary that plans multi-step perturbation sequences
# ---------------------------------------------------------------------------

class TemporalAdversary:
    """Plans a sequence of adversarial perturbations over H steps to maximally
    disrupt policy performance (open-loop adversarial planning)."""

    def __init__(self, obs_dim: int = 16, horizon: int = 5,
                 epsilon: float = 0.03, n_rollouts: int = 10):
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.epsilon = epsilon
        self.n_rollouts = n_rollouts
        self._planned_seq: Optional[np.ndarray] = None
        self._seq_idx: int = 0

    # ------------------------------------------------------------------
    def plan(self, initial_obs: np.ndarray, policy_fn,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Random-search best perturbation sequence over horizon steps."""
        if rng is None:
            rng = np.random.default_rng()
        best_seq = np.zeros((self.horizon, self.obs_dim), dtype=np.float32)
        best_score = -np.inf
        for _ in range(self.n_rollouts):
            seq = rng.uniform(-self.epsilon, self.epsilon,
                              (self.horizon, self.obs_dim)).astype(np.float32)
            score = 0.0
            obs = initial_obs.copy()
            for h in range(self.horizon):
                perturbed = obs + seq[h]
                act = np.array(policy_fn(perturbed), dtype=np.float32)
                # Adversary wants to maximize action magnitude (heuristic disruption)
                score -= float(np.linalg.norm(act))
            if score > best_score:
                best_score = score
                best_seq = seq.copy()
        self._planned_seq = best_seq
        self._seq_idx = 0
        return best_seq

    # ------------------------------------------------------------------
    def perturb(self, obs: np.ndarray) -> np.ndarray:
        """Apply next step in planned sequence."""
        if self._planned_seq is None or self._seq_idx >= self.horizon:
            return obs.copy()
        pert = obs + self._planned_seq[self._seq_idx]
        self._seq_idx += 1
        return pert

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._planned_seq = None
        self._seq_idx = 0


# ---------------------------------------------------------------------------
# AdversarialTrainingScheduler — adapts adversary budget over training
# ---------------------------------------------------------------------------

class AdversarialTrainingScheduler:
    """Manages adversarial perturbation budget (epsilon) using a curriculum
    that starts small and grows as the protagonist improves."""

    def __init__(self, initial_epsilon: float = 0.01,
                 max_epsilon: float = 0.1,
                 growth_rate: float = 1.001,
                 protagonist_win_threshold: float = 0.6):
        self.epsilon = initial_epsilon
        self.max_epsilon = max_epsilon
        self.growth_rate = growth_rate
        self.protagonist_win_threshold = protagonist_win_threshold
        self._protagonist_wins = 0
        self._total_episodes = 0

    # ------------------------------------------------------------------
    def record_episode(self, protagonist_won: bool) -> None:
        self._total_episodes += 1
        if protagonist_won:
            self._protagonist_wins += 1
        # Grow epsilon if protagonist is winning comfortably
        win_rate = self._protagonist_wins / max(1, self._total_episodes)
        if win_rate > self.protagonist_win_threshold:
            self.epsilon = min(self.max_epsilon,
                               self.epsilon * self.growth_rate)

    # ------------------------------------------------------------------
    @property
    def current_epsilon(self) -> float:
        return self.epsilon

    # ------------------------------------------------------------------
    @property
    def protagonist_win_rate(self) -> float:
        return self._protagonist_wins / max(1, self._total_episodes)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "protagonist_wins": self._protagonist_wins,
            "total_episodes": self._total_episodes,
        }
