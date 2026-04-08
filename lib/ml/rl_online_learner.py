"""
RL Exit Optimizer — Online Learning with Prior Blending (T2-3)
Extends the Double DQN exit optimizer with live market data retraining.

Avoids distribution mismatch between synthetic training and live markets.
Safety: blends live updates with frozen synthetic-trained prior.
  q_effective = α * q_live + (1-α) * q_prior
  α starts at 0.1, grows as live data accumulates (max 0.7)

Usage:
    learner = RLOnlineLearner(qtable_path="config/rl_exit_qtable.json")

    # On each bar (state observation):
    learner.observe(state_key="2,1,3,4,2", action="hold", reward=0.002, next_state_key="2,2,3,4,2")

    # After 500 transitions, auto-updates the live Q-table
    # Get Q-values:
    hold_q, exit_q = learner.get_q_values("2,1,3,4,2")
"""
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class RLOnlineConfig:
    qtable_path: str = "config/rl_exit_qtable.json"
    live_qtable_path: str = "config/rl_exit_qtable_live.json"
    replay_buffer_size: int = 10_000
    update_every: int = 500         # update after N new transitions
    batch_size: int = 32
    learning_rate: float = 0.01
    gamma: float = 0.95             # discount factor
    alpha_init: float = 0.10        # initial live blend weight
    alpha_max: float = 0.70         # maximum live blend weight
    alpha_grow_per_update: float = 0.02  # alpha grows with each update batch
    min_transitions_before_blend: int = 200  # don't blend until we have enough data

@dataclass
class Transition:
    state_key: str
    action: str      # "hold" or "exit"
    reward: float
    next_state_key: str
    done: bool = False

class RLOnlineLearner:
    """
    Online Q-learning layer that blends live experience with the prior Q-table.

    The prior Q-table (from synthetic training) is frozen. The live Q-table
    is updated via online TD learning. The effective Q-table is a blend.
    """

    def __init__(self, cfg: RLOnlineConfig = None):
        self.cfg = cfg or RLOnlineConfig()
        self._prior_qtable: dict[str, list[float]] = {}
        self._live_qtable: dict[str, list[float]] = {}
        self._replay_buffer: deque[Transition] = deque(maxlen=self.cfg.replay_buffer_size)
        self._n_transitions: int = 0
        self._n_updates: int = 0
        self._alpha: float = self.cfg.alpha_init

        self._load_prior()
        self._load_live()

    def get_q_values(self, state_key: str) -> tuple[float, float]:
        """
        Returns (hold_q, exit_q) for a state key.
        Uses blended prior + live Q-values if enough live data; prior only otherwise.
        """
        prior_qs = self._prior_qtable.get(state_key, [0.0, 0.0])

        if self._n_transitions < self.cfg.min_transitions_before_blend:
            return float(prior_qs[0]), float(prior_qs[1])

        live_qs = self._live_qtable.get(state_key, prior_qs)

        alpha = self._alpha
        hold_q = alpha * float(live_qs[0]) + (1 - alpha) * float(prior_qs[0])
        exit_q = alpha * float(live_qs[1]) + (1 - alpha) * float(prior_qs[1])
        return hold_q, exit_q

    def observe(
        self,
        state_key: str,
        action: str,
        reward: float,
        next_state_key: str,
        done: bool = False,
    ):
        """
        Record a transition from the live trading environment.
        Triggers update when buffer has enough new transitions.
        """
        t = Transition(
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
        )
        self._replay_buffer.append(t)
        self._n_transitions += 1

        if self._n_transitions % self.cfg.update_every == 0:
            self._update()

    def _update(self):
        """Run a mini-batch TD update on the live Q-table."""
        if len(self._replay_buffer) < self.cfg.batch_size:
            return

        batch = random.sample(list(self._replay_buffer), self.cfg.batch_size)

        for t in batch:
            # Current Q-values (from live table, defaulting to prior)
            current_qs = list(self._live_qtable.get(t.state_key,
                                                      list(self._prior_qtable.get(t.state_key, [0.0, 0.0]))))

            # Next state max Q (greedy)
            next_qs = self._live_qtable.get(t.next_state_key,
                                            self._prior_qtable.get(t.next_state_key, [0.0, 0.0]))
            max_next_q = max(float(next_qs[0]), float(next_qs[1]))

            # TD target
            if t.done:
                td_target = t.reward
            else:
                td_target = t.reward + self.cfg.gamma * max_next_q

            # Update the Q-value for the taken action
            action_idx = 0 if t.action == "hold" else 1
            current_q = current_qs[action_idx]
            new_q = current_q + self.cfg.learning_rate * (td_target - current_q)
            current_qs[action_idx] = new_q

            self._live_qtable[t.state_key] = current_qs

        # Grow blend weight
        self._alpha = min(self.cfg.alpha_max, self._alpha + self.cfg.alpha_grow_per_update)
        self._n_updates += 1

        log.info(
            "RLOnline: update #%d, α=%.3f, live_states=%d, buffer=%d",
            self._n_updates, self._alpha, len(self._live_qtable), len(self._replay_buffer)
        )

        # Persist live Q-table
        self._save_live()

    def _load_prior(self):
        p = Path(self.cfg.qtable_path)
        if p.exists():
            try:
                with open(p) as f:
                    self._prior_qtable = json.load(f)
                log.info("RLOnline: loaded prior Q-table with %d states", len(self._prior_qtable))
            except Exception as e:
                log.warning("RLOnline: failed to load prior Q-table: %s", e)

    def _load_live(self):
        p = Path(self.cfg.live_qtable_path)
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                self._live_qtable = data.get("qtable", {})
                self._n_transitions = data.get("n_transitions", 0)
                self._n_updates = data.get("n_updates", 0)
                self._alpha = data.get("alpha", self.cfg.alpha_init)
                log.info("RLOnline: loaded live Q-table with %d states (α=%.3f)",
                         len(self._live_qtable), self._alpha)
            except Exception as e:
                log.debug("RLOnline: no live Q-table found (fresh start): %s", e)

    def _save_live(self):
        p = Path(self.cfg.live_qtable_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({
                    "qtable": self._live_qtable,
                    "n_transitions": self._n_transitions,
                    "n_updates": self._n_updates,
                    "alpha": self._alpha,
                    "updated_at": time.time(),
                }, f)
        except Exception as e:
            log.warning("RLOnline: save failed: %s", e)
