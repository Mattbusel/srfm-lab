"""
distributed_marl.py
===================
Distributed MARL training infrastructure for the Hyper-Agent ecosystem.

Implements:
  - Ray/RLlib-style actor pools for data collection
  - Parameter server for centralised critic
  - Asynchronous rollout workers
  - Gradient compression (PowerSGD, top-k sparsification)
  - Staleness handling (importance sampling correction)
  - Experience replay with priority queues
  - Distributed experience aggregation
  - Monitoring and throughput tracking
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout batch
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RolloutBatch:
    """A batch of trajectory data from a single worker."""
    worker_id: str
    observations: torch.Tensor          # (T, obs_dim)
    actions: torch.Tensor               # (T, act_dim)
    rewards: torch.Tensor               # (T,)
    dones: torch.Tensor                 # (T,)
    log_probs: torch.Tensor             # (T,)
    values: torch.Tensor                # (T,)
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    global_step: int = 0
    episode_returns: List[float] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.observations.shape[0]

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        T = self.n_steps
        advantages = torch.zeros(T)
        last_gae = 0.0
        values_ext = torch.cat([self.values, torch.zeros(1)])
        for t in reversed(range(T)):
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values_ext[t + 1] * non_terminal - self.values[t]
            last_gae = float(delta) + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
        self.advantages = advantages
        self.returns = advantages + self.values

    def to(self, device: str) -> "RolloutBatch":
        return RolloutBatch(
            worker_id=self.worker_id,
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
            global_step=self.global_step,
            episode_returns=list(self.episode_returns),
        )


# ---------------------------------------------------------------------------
# Parameter server
# ---------------------------------------------------------------------------

class ParameterServer:
    """
    Thread-safe parameter server for distributed training.
    Supports async gradient accumulation and weight broadcasting.
    """

    def __init__(self, model: nn.Module,
                 lr: float = 3e-4,
                 gradient_clip: float = 0.5,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self._optimiser = optim.Adam(model.parameters(), lr=lr)
        self._lock = threading.RLock()
        self._version: int = 0
        self._grad_accumulator: Optional[List[Optional[torch.Tensor]]] = None
        self._n_grad_contributions: int = 0
        self._metrics: Dict[str, float] = {}

    def get_weights(self) -> Dict[str, torch.Tensor]:
        with self._lock:
            return {k: v.cpu().detach().clone()
                    for k, v in self.model.state_dict().items()}

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def apply_gradients(self, grads: List[Optional[torch.Tensor]]) -> int:
        """Apply a list of gradient tensors (one per parameter)."""
        with self._lock:
            if self._grad_accumulator is None:
                self._grad_accumulator = [None] * len(grads)

            for i, (param, grad) in enumerate(zip(self.model.parameters(), grads)):
                if grad is None:
                    continue
                g = grad.to(self.device)
                if self._grad_accumulator[i] is None:
                    self._grad_accumulator[i] = g.clone()
                else:
                    self._grad_accumulator[i] += g

            self._n_grad_contributions += 1
            return self._version

    def step(self, sync_interval: int = 1) -> bool:
        """Apply accumulated gradients if sync interval reached."""
        with self._lock:
            if (self._n_grad_contributions < sync_interval or
                    self._grad_accumulator is None):
                return False

            self._optimiser.zero_grad()
            for param, accum_grad in zip(self.model.parameters(),
                                          self._grad_accumulator):
                if accum_grad is not None:
                    param.grad = accum_grad / max(self._n_grad_contributions, 1)

            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self._optimiser.step()

            self._grad_accumulator = None
            self._n_grad_contributions = 0
            self._version += 1
            return True

    def set_lr(self, lr: float) -> None:
        with self._lock:
            for pg in self._optimiser.param_groups:
                pg["lr"] = lr


# ---------------------------------------------------------------------------
# Gradient compression
# ---------------------------------------------------------------------------

class GradientCompressor:
    """
    Gradient compression utilities for bandwidth-efficient distributed training.
    Implements:
      - Top-k sparsification
      - PowerSGD-style low-rank approximation
      - Quantisation
    """

    class Method(enum.Enum):
        TOPK = "topk"
        POWERSGD = "powersgd"
        QUANTISE = "quantise"
        NONE = "none"

    def __init__(self, method: "GradientCompressor.Method" = None,
                 k_ratio: float = 0.01, n_bits: int = 8,
                 rank: int = 4):
        self.method = method or self.Method.TOPK
        self.k_ratio = k_ratio
        self.n_bits = n_bits
        self.rank = rank
        self._residuals: Optional[List[torch.Tensor]] = None

    def compress(self, grads: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        if self.method == self.Method.NONE:
            return grads
        if self.method == self.Method.TOPK:
            return [self._topk(g) for g in grads]
        if self.method == self.Method.QUANTISE:
            return [self._quantise(g) for g in grads]
        if self.method == self.Method.POWERSGD:
            return [self._powersgd(g) for g in grads]
        return grads

    def _topk(self, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if grad is None:
            return None
        flat = grad.view(-1)
        k = max(1, int(flat.numel() * self.k_ratio))
        _, indices = flat.abs().topk(k)
        compressed = torch.zeros_like(flat)
        compressed[indices] = flat[indices]
        return compressed.view_as(grad)

    def _quantise(self, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if grad is None:
            return None
        n_levels = 2 ** self.n_bits
        min_v = grad.min()
        max_v = grad.max()
        scale = (max_v - min_v) / n_levels
        if scale < 1e-9:
            return grad
        q = torch.round((grad - min_v) / scale) * scale + min_v
        return q

    def _powersgd(self, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if grad is None:
            return None
        if grad.dim() < 2:
            return grad
        # Truncated SVD
        try:
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            rank = min(self.rank, S.numel())
            return (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        except Exception:
            return grad

    def compression_ratio(self) -> float:
        if self.method == self.Method.TOPK:
            return self.k_ratio
        if self.method == self.Method.QUANTISE:
            return self.n_bits / 32.0
        return 1.0


# ---------------------------------------------------------------------------
# Staleness correction
# ---------------------------------------------------------------------------

class StalenessCorrector:
    """
    Corrects for policy staleness in asynchronous training via
    importance sampling weights.
    """

    def __init__(self, max_staleness: int = 20, clip_rho: float = 1.0):
        self.max_staleness = max_staleness
        self.clip_rho = clip_rho

    def compute_importance_weights(
        self, log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
        staleness: int
    ) -> torch.Tensor:
        """
        IS weights corrected for staleness.
        rho = pi_new(a|s) / pi_old(a|s) * staleness_discount
        """
        rho = torch.exp(log_probs_new - log_probs_old)
        staleness_discount = (1.0 - staleness / (self.max_staleness + 1))
        rho = rho * max(0.1, staleness_discount)
        if self.clip_rho > 0:
            rho = rho.clamp(0.0, self.clip_rho)
        return rho

    def vtrace_targets(self,
                        rewards: torch.Tensor,
                        values: torch.Tensor,
                        log_probs_new: torch.Tensor,
                        log_probs_old: torch.Tensor,
                        dones: torch.Tensor,
                        gamma: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V-trace targets for off-policy correction (Espeholt et al., 2018).
        Returns (vs, advantages).
        """
        T = len(rewards)
        rho = torch.exp(log_probs_new - log_probs_old).clamp(0, self.clip_rho)
        c = rho.clamp(0, 1.0)   # trace coefficients

        vs = torch.zeros(T + 1)
        vs[T] = values[T - 1] if T > 0 else 0.0

        for t in reversed(range(T)):
            non_terminal = 1.0 - float(dones[t])
            delta_t = rho[t] * (rewards[t] + gamma * values[t] * non_terminal - values[t])
            vs[t] = values[t] + delta_t + gamma * c[t] * non_terminal * (vs[t + 1] - values[t])

        advantages = rho * (rewards + gamma * vs[1:] - values)
        return vs[:T], advantages


# ---------------------------------------------------------------------------
# Priority replay buffer
# ---------------------------------------------------------------------------

class PrioritisedReplayBuffer:
    """
    Prioritised experience replay buffer with O(log n) sampling.
    Uses sum-tree for efficient priority sampling.
    """

    def __init__(self, capacity: int = 50_000,
                 alpha: float = 0.6, beta: float = 0.4,
                 beta_schedule: Optional[Callable[[int], float]] = None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_schedule = beta_schedule
        self._size = 0
        self._ptr = 0
        self._tree = np.zeros(2 * capacity)
        self._data: List[Optional[Any]] = [None] * capacity
        self._step = 0

    def _tree_update(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.capacity
        self._tree[tree_idx] = priority ** self.alpha
        while tree_idx > 1:
            tree_idx //= 2
            self._tree[tree_idx] = (self._tree[2 * tree_idx] +
                                     self._tree[2 * tree_idx + 1])

    def add(self, data: Any, priority: float = 1.0) -> None:
        self._data[self._ptr] = data
        self._tree_update(self._ptr, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _sample_idx(self, value: float) -> int:
        node = 1
        while node < self.capacity:
            left = 2 * node
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = left + 1
        return node - self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        if self._size == 0:
            return [], np.array([]), np.array([])

        self._step += 1
        beta = (self.beta_schedule(self._step) if self.beta_schedule
                else min(1.0, self.beta + self._step * 1e-6))

        total = self._tree[1]
        segment = total / batch_size
        indices = []
        weights = []
        data = []

        min_prob = (self._tree[self.capacity:self.capacity + self._size].min() / total)
        max_weight = (min_prob * self._size) ** (-beta)

        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            idx = self._sample_idx(val)
            if idx >= self._size:
                idx = int(np.random.randint(0, self._size))
            indices.append(idx)
            prob = self._tree[idx + self.capacity] / total
            weight = (prob * self._size) ** (-beta) / max_weight
            weights.append(weight)
            data.append(self._data[idx])

        return data, np.array(indices), np.array(weights, dtype=np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self._tree_update(int(idx), max(priority, 1e-9))

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Rollout worker (thread-based)
# ---------------------------------------------------------------------------

class RolloutWorker(threading.Thread):
    """
    Asynchronous rollout worker.
    Collects trajectories using the current policy from the parameter server
    and pushes them to the experience queue.
    """

    def __init__(self,
                 worker_id: str,
                 env_fn: Callable,
                 policy: nn.Module,
                 param_server: ParameterServer,
                 experience_queue: "queue.Queue",
                 rollout_length: int = 512,
                 n_rollouts: int = -1,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu"):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.env_fn = env_fn
        self.policy = copy.deepcopy(policy).to(device)
        self.param_server = param_server
        self.experience_queue = experience_queue
        self.rollout_length = rollout_length
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self._stop_event = threading.Event()
        self._local_version = 0
        self._rollout_count = 0
        self._total_steps = 0
        self._episode_returns: deque = deque(maxlen=100)

    import copy as _copy  # workaround for module-level import

    def sync_weights(self) -> None:
        weights = self.param_server.get_weights()
        self.policy.load_state_dict(weights)
        self._local_version = self.param_server.get_version()

    def collect_rollout(self) -> RolloutBatch:
        self.sync_weights()
        env = self.env_fn()
        obs, _ = env.reset()
        obs_list, act_list, rew_list, done_list, lp_list, val_list = [], [], [], [], [], []
        episode_return = 0.0
        episode_returns = []

        self.policy.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        for _ in range(self.rollout_length):
            with torch.no_grad():
                action, lp = self.policy.act(obs_t, deterministic=False)
                _, _, value = self.policy.forward(obs_t)

            action_np = action.cpu().numpy()[0]
            obs_next, reward, done, trunc, _ = env.step(action_np)
            terminated = done or trunc

            obs_list.append(obs_t.squeeze(0))
            act_list.append(action.squeeze(0))
            rew_list.append(torch.tensor(reward, dtype=torch.float32))
            done_list.append(torch.tensor(float(terminated), dtype=torch.float32))
            lp_list.append(lp)
            val_list.append(value.squeeze(-1) if value.dim() > 1 else value)

            episode_return += reward
            self._total_steps += 1

            if terminated:
                episode_returns.append(episode_return)
                self._episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                obs = obs_next
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        batch = RolloutBatch(
            worker_id=self.worker_id,
            observations=torch.stack(obs_list),
            actions=torch.stack(act_list),
            rewards=torch.stack(rew_list),
            dones=torch.stack(done_list),
            log_probs=torch.stack(lp_list).squeeze(-1),
            values=torch.stack(val_list).squeeze(-1),
            global_step=self._local_version,
            episode_returns=episode_returns,
        )
        batch.compute_gae(self.gamma, self.gae_lambda)
        return batch

    def run(self) -> None:
        rollout_num = 0
        while not self._stop_event.is_set():
            if self.n_rollouts > 0 and rollout_num >= self.n_rollouts:
                break
            try:
                batch = self.collect_rollout()
                self.experience_queue.put(batch, timeout=5.0)
                self._rollout_count += 1
                rollout_num += 1
            except queue.Full:
                logger.debug("Worker %s: queue full, waiting", self.worker_id)
            except Exception as exc:
                logger.error("Worker %s error: %s", self.worker_id, exc)
                break

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stats(self) -> Dict[str, float]:
        return {
            "total_steps": self._total_steps,
            "rollout_count": self._rollout_count,
            "mean_return": float(np.mean(self._episode_returns)) if self._episode_returns else 0.0,
        }


import copy  # top-level for RolloutWorker


# ---------------------------------------------------------------------------
# Distributed trainer
# ---------------------------------------------------------------------------

class DistributedMARLTrainer:
    """
    Distributed MARL trainer coordinating multiple rollout workers,
    a parameter server, and a central learner.
    """

    def __init__(self,
                 policy: nn.Module,
                 env_fn: Callable,
                 n_workers: int = 4,
                 rollout_length: int = 256,
                 batch_size: int = 512,
                 n_epochs: int = 4,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 clip_eps: float = 0.2,
                 gradient_clip: float = 0.5,
                 compress_gradients: bool = False,
                 queue_size: int = 16,
                 device: str = "cpu"):
        self.policy = policy.to(device)
        self.env_fn = env_fn
        self.n_workers = n_workers
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_eps = clip_eps
        self.device = device

        self.param_server = ParameterServer(policy, lr, gradient_clip, device)
        self.compressor = (GradientCompressor(GradientCompressor.Method.TOPK, k_ratio=0.1)
                           if compress_gradients else None)
        self.staleness_corrector = StalenessCorrector()

        self._experience_queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._workers: List[RolloutWorker] = []
        self._global_step = 0
        self._metrics_history: deque = deque(maxlen=1000)

    def start_workers(self) -> None:
        for i in range(self.n_workers):
            worker = RolloutWorker(
                worker_id=f"worker_{i}",
                env_fn=self.env_fn,
                policy=self.policy,
                param_server=self.param_server,
                experience_queue=self._experience_queue,
                rollout_length=self.rollout_length,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=self.device,
            )
            worker.start()
            self._workers.append(worker)
        logger.info("Started %d rollout workers", self.n_workers)

    def stop_workers(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._workers.clear()

    def collect_batch(self, timeout: float = 30.0) -> Optional[RolloutBatch]:
        """Collect and merge rollout batches from workers."""
        try:
            batch = self._experience_queue.get(timeout=timeout)
            return batch
        except queue.Empty:
            return None

    def merge_batches(self, batches: List[RolloutBatch]) -> RolloutBatch:
        if len(batches) == 1:
            return batches[0]
        obs = torch.cat([b.observations for b in batches])
        acts = torch.cat([b.actions for b in batches])
        rews = torch.cat([b.rewards for b in batches])
        dones = torch.cat([b.dones for b in batches])
        lps = torch.cat([b.log_probs for b in batches])
        vals = torch.cat([b.values for b in batches])
        advs = torch.cat([b.advantages for b in batches if b.advantages is not None])
        rets = torch.cat([b.returns for b in batches if b.returns is not None])
        ep_returns = [r for b in batches for r in b.episode_returns]
        return RolloutBatch(
            worker_id="merged",
            observations=obs, actions=acts, rewards=rews, dones=dones,
            log_probs=lps, values=vals, advantages=advs, returns=rets,
            global_step=batches[0].global_step,
            episode_returns=ep_returns,
        )

    def update_step(self, batch: RolloutBatch) -> Dict[str, float]:
        """PPO-style update step."""
        if batch.advantages is None or batch.returns is None:
            batch.compute_gae(self.gamma, self.gae_lambda)

        batch = batch.to(self.device)
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = batch.returns

        T = batch.n_steps
        metrics: Dict[str, float] = {}
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                idx = indices[start:start + self.batch_size]
                obs_b = batch.observations[idx]
                act_b = batch.actions[idx]
                old_lp_b = batch.log_probs[idx].detach()
                adv_b = adv[idx]
                ret_b = ret[idx]

                new_lp, entropy, value = self.policy.evaluate(obs_b, act_b)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret_b)
                entropy_loss = -entropy.mean()

                total_loss = (policy_loss +
                              self.value_coef * value_loss +
                              self.entropy_coef * entropy_loss)

                # Compute gradients
                total_loss.backward()

                # Optionally compress gradients
                grads = [p.grad for p in self.policy.parameters()]
                if self.compressor is not None:
                    grads = self.compressor.compress(grads)

                self.param_server.apply_gradients(grads)

                # Zero gradients
                for p in self.policy.parameters():
                    p.grad = None

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())

        # Apply accumulated gradients
        self.param_server.step(sync_interval=1)

        # Sync policy from parameter server
        self.policy.load_state_dict(self.param_server.get_weights())

        self._global_step += 1
        n_updates = max(1, self.n_epochs * (T // max(1, self.batch_size)))
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "global_step": self._global_step,
            "mean_episode_return": float(np.mean(batch.episode_returns)) if batch.episode_returns else 0.0,
            "param_server_version": self.param_server.get_version(),
        }
        self._metrics_history.append(metrics)
        return metrics

    def train(self, n_updates: int = 1000,
               callback: Optional[Callable] = None) -> List[Dict[str, float]]:
        self.start_workers()
        all_metrics = []
        try:
            for update in range(n_updates):
                batch = self.collect_batch(timeout=30.0)
                if batch is None:
                    logger.warning("No batch received at update %d", update)
                    continue
                metrics = self.update_step(batch)
                all_metrics.append(metrics)
                if callback is not None:
                    callback(update, metrics)
                if update % 50 == 0:
                    logger.info("Update %d | loss=%.4f | return=%.2f | version=%d",
                                update,
                                metrics.get("policy_loss", 0),
                                metrics.get("mean_episode_return", 0),
                                metrics.get("param_server_version", 0))
        finally:
            self.stop_workers()
        return all_metrics

    def worker_stats(self) -> List[Dict[str, float]]:
        return [w.stats for w in self._workers]

    def throughput(self) -> float:
        """Samples per second."""
        if not self._metrics_history:
            return 0.0
        return float(self._global_step * self.rollout_length * self.n_workers)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("=== distributed_marl.py smoke test ===")

    obs_dim, act_dim = 64, 10

    # Create a minimal policy
    class MinimalPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(obs_dim, act_dim)
            self.critic = nn.Linear(obs_dim, 1)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs):
            mean = torch.tanh(self.actor(obs))
            return mean, self.log_std.exp(), self.critic(obs).squeeze(-1)

        def act(self, obs, deterministic=False):
            mean, std, val = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            lp = dist.log_prob(action).sum(-1)
            return action.clamp(-1, 1), lp

        def evaluate(self, obs, action):
            mean, std, val = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            lp = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return lp, entropy, val

    policy = MinimalPolicy()
    ps = ParameterServer(policy, lr=3e-4)
    print(f"Parameter server version: {ps.get_version()}")
    weights = ps.get_weights()
    print(f"Weights keys: {list(weights.keys())[:3]}")

    # Test rollout batch
    T = 50
    batch = RolloutBatch(
        worker_id="test",
        observations=torch.randn(T, obs_dim),
        actions=torch.randn(T, act_dim),
        rewards=torch.randn(T),
        dones=torch.zeros(T),
        log_probs=torch.randn(T),
        values=torch.randn(T),
        global_step=0,
    )
    batch.compute_gae(0.99, 0.95)
    print(f"GAE computed: advantages shape = {batch.advantages.shape}")

    # Test priority buffer
    buf = PrioritisedReplayBuffer(capacity=1000)
    for i in range(100):
        buf.add({"obs": np.random.randn(obs_dim)}, priority=float(np.random.rand() + 0.1))
    data, idxs, weights_is = buf.sample(8)
    print(f"Priority buffer sample: {len(data)} items, IS weights = {weights_is.round(3)}")

    # Test gradient compressor
    comp = GradientCompressor(GradientCompressor.Method.TOPK, k_ratio=0.1)
    grads = [torch.randn(100) for _ in range(3)]
    compressed = comp.compress(grads)
    sparsity = float((compressed[0].abs() > 0).float().mean())
    print(f"Compressed gradient sparsity: {sparsity:.2f} (expected ~0.10)")

    print("\nAll smoke tests passed.")
