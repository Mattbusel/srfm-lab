"""
tools/train_rl_exit.py
======================
Q-learning trainer for the RLExitPolicy used in live_trader_alpaca.py.

Generates config/rl_exit_qtable.json -- a mapping of discretized state keys
to [q_hold, q_exit] value pairs consumed by RLExitPolicy._state_key().

State features (5 dimensions, 5 bins each):
  0  position_pnl_pct  bins: [-inf, -0.02, -0.005, 0.005, 0.02, inf]
  1  bars_held         bins: [0, 3, 8, 16, 32, inf]
  2  bh_mass           bins: [0, 0.5, 1.0, 1.92, 3.0, inf]
  3  bh_active         bool -- mapped to bins 0 or 4 (sparse two-class)
  4  atr_ratio         bins: [0, 0.5, 0.8, 1.2, 2.0, inf]

Actions: 0 = HOLD, 1 = EXIT

Usage:
  python tools/train_rl_exit.py [--epochs 50] [--synthetic 1000] [--db path]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[1]
_DB_PATH   = _REPO_ROOT / "execution" / "live_trades.db"
_QTABLE_PATH = _REPO_ROOT / "config" / "rl_exit_qtable.json"

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("train_rl_exit")


# ---------------------------------------------------------------------------
# ExitAction constants
# ---------------------------------------------------------------------------
class ExitAction:
    HOLD = 0
    EXIT = 1


# ---------------------------------------------------------------------------
# ExitState dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExitState:
    """One bar's worth of state for the RL exit policy."""
    pnl_pct:   float   # current position P&L as a fraction (0.02 = +2%)
    bars_held: int     # how many bars the position has been open
    bh_mass:   float   # current Black-Hole mass (0 .. ~4)
    bh_active: bool    # whether BH conviction is active
    atr_ratio: float   # current ATR / entry ATR  (>1 = expanding volatility)


# ---------------------------------------------------------------------------
# StateDiscretizer
# ---------------------------------------------------------------------------
class StateDiscretizer:
    """
    Maps a continuous ExitState to a discrete 5-tuple index.

    Bin edges follow the live trader's feature resolution exactly so the
    trained table is immediately compatible without re-encoding.
    """

    # Upper bin edges (exclusive) -- [bins[i-1], bins[i]) => index i-1
    # 5 bins => 6 edges including -inf / +inf sentinels
    PNL_EDGES  = [-0.02, -0.005, 0.005, 0.02]          # 4 interior edges => 5 bins
    BARS_EDGES = [3,  8,  16, 32]
    MASS_EDGES = [0.5, 1.0, 1.92, 3.0]
    ATR_EDGES  = [0.5, 0.8, 1.2, 2.0]

    # bh_active is bool: False -> bin 0, True -> bin 4 (spread out for max Q contrast)
    _BH_ACTIVE_FALSE = 0
    _BH_ACTIVE_TRUE  = 4

    @staticmethod
    def _bin(value: float, edges: list[float]) -> int:
        """Return 0-based bin index for value given interior edges."""
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)  # last bin

    def encode(self, state: ExitState) -> tuple:
        """Return a 5-tuple of bin indices."""
        f0 = self._bin(state.pnl_pct,   self.PNL_EDGES)
        f1 = self._bin(state.bars_held,  self.BARS_EDGES)
        f2 = self._bin(state.bh_mass,    self.MASS_EDGES)
        f3 = self._BH_ACTIVE_TRUE if state.bh_active else self._BH_ACTIVE_FALSE
        f4 = self._bin(state.atr_ratio,  self.ATR_EDGES)
        return (f0, f1, f2, f3, f4)

    def decode(self, key: tuple) -> str:
        """Human-readable description of a discretized state key."""
        pnl_labels  = ["<-2%", "-2%--0.5%", "-0.5%-+0.5%", "+0.5%-+2%", ">+2%"]
        bars_labels = ["0-2", "3-7", "8-15", "16-31", "32+"]
        mass_labels = ["0-0.5", "0.5-1.0", "1.0-1.92", "1.92-3.0", "3.0+"]
        bh_labels   = ["inactive", "inactive", "inactive", "inactive", "active"]
        atr_labels  = ["<0.5x", "0.5-0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]

        f0, f1, f2, f3, f4 = key
        return (
            f"pnl={pnl_labels[f0]} bars={bars_labels[f1]} "
            f"mass={mass_labels[f2]} bh={bh_labels[f3]} "
            f"atr={atr_labels[f4]}"
        )

    def state_key_str(self, key: tuple) -> str:
        """Encode a tuple key to the comma-separated string used in JSON."""
        return ",".join(str(k) for k in key)

    def parse_key_str(self, s: str) -> tuple:
        return tuple(int(x) for x in s.split(","))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular experience replay buffer storing (state_key, action, reward,
    next_state_key, done) tuples.

    Max capacity: 100_000 transitions.
    """

    CAPACITY = 100_000

    def __init__(self) -> None:
        self._buf: list[tuple] = []
        self._idx: int = 0

    def push(
        self,
        state_key:      tuple,
        action:         int,
        reward:         float,
        next_state_key: tuple,
        done:           bool,
    ) -> None:
        entry = (state_key, action, reward, next_state_key, done)
        if len(self._buf) < self.CAPACITY:
            self._buf.append(entry)
        else:
            self._buf[self._idx] = entry
        self._idx = (self._idx + 1) % self.CAPACITY

    def sample(self, batch_size: int) -> list[tuple]:
        """Random sample without replacement (or full buffer if smaller)."""
        n = min(batch_size, len(self._buf))
        return random.sample(self._buf, n)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# QLearningTrainer
# ---------------------------------------------------------------------------
class QLearningTrainer:
    """
    Tabular Q-learning trainer.

    Q-table maps state keys (5-tuples) to [q_hold, q_exit].
    Training runs episodically: each episode is one complete trade lifecycle
    (entry to exit) represented as a list of per-bar state dicts.

    Hyperparameters
    ---------------
    alpha         : learning rate (Bellman update step size)
    gamma         : discount factor
    epsilon       : initial exploration rate (epsilon-greedy)
    epsilon_decay : multiplicative decay per episode step
    epsilon_min   : floor on exploration rate
    """

    def __init__(
        self,
        alpha:         float = 0.10,
        gamma:         float = 0.95,
        epsilon:       float = 0.30,
        epsilon_decay: float = 0.999,
        epsilon_min:   float = 0.05,
    ) -> None:
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table: key -> [q_hold, q_exit]
        self._qtable: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._disc = StateDiscretizer()
        self._buf  = ReplayBuffer()

        # Training stats
        self._episode_rewards: list[float] = []
        self._updates_total   = 0

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _q(self, key: tuple) -> list[float]:
        return self._qtable[key]

    def get_action(self, state_key: tuple, explore: bool = False) -> int:
        """
        Return action (HOLD=0 or EXIT=1).
        If explore=True uses epsilon-greedy, else pure greedy.
        """
        if explore and random.random() < self.epsilon:
            return random.choice([ExitAction.HOLD, ExitAction.EXIT])
        qs = self._q(state_key)
        return ExitAction.EXIT if qs[1] > qs[0] else ExitAction.HOLD

    def update_q(
        self,
        state_key: tuple,
        action:    int,
        reward:    float,
        next_key:  tuple,
        done:      bool,
    ) -> None:
        """
        Standard Q-learning Bellman update:
          Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        Terminal states have no bootstrap target.
        """
        qs   = self._q(state_key)
        next_max = 0.0 if done else max(self._q(next_key))
        td_target = reward + self.gamma * next_max
        qs[action] += self.alpha * (td_target - qs[action])
        self._updates_total += 1

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        state:       ExitState,
        next_state:  Optional[ExitState],
        action:      int,
        done:        bool,
    ) -> float:
        """
        Reward = P&L improvement this bar
               - holding cost (opportunity cost per bar)
               - BH-inactive stagnation penalty (applied after bar 8)

        Terminal EXIT reward is boosted by realized final P&L to encourage
        the policy to exit at peaks rather than letting trades decay.
        """
        HOLD_COST          = 0.001
        STAGNATION_PENALTY = 0.005

        pnl_change = 0.0
        if next_state is not None:
            pnl_change = next_state.pnl_pct - state.pnl_pct

        reward = pnl_change - HOLD_COST

        # Penalize holding a BH-inactive trade past the early-exit window
        if not state.bh_active and state.bars_held > 8:
            reward -= STAGNATION_PENALTY

        if action == ExitAction.EXIT and done:
            # Give the agent a direct signal about the quality of the exit
            # A positive pnl exit is rewarded; negative is additionally penalized
            reward += state.pnl_pct * 2.0

        return reward

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def train_episode(self, trade_history: list[dict]) -> float:
        """
        Run one training episode through a trade's bar-by-bar history.

        Each element of trade_history is a dict with keys:
          pnl_pct, bars_held, bh_mass, bh_active, atr_ratio

        Returns total undiscounted episode reward.
        """
        if not trade_history:
            return 0.0

        total_reward = 0.0
        states = [
            ExitState(
                pnl_pct=   bar["pnl_pct"],
                bars_held= bar["bars_held"],
                bh_mass=   bar["bh_mass"],
                bh_active= bool(bar["bh_active"]),
                atr_ratio= bar["atr_ratio"],
            )
            for bar in trade_history
        ]

        for i, state in enumerate(states):
            key  = self._disc.encode(state)
            done = i == len(states) - 1

            action = self.get_action(key, explore=True)

            next_state = states[i + 1] if not done else None
            next_key   = self._disc.encode(next_state) if next_state else key

            # Force exit on last bar of episode
            if done:
                action = ExitAction.EXIT

            reward = self._compute_reward(state, next_state, action, done)
            total_reward += reward

            self._buf.push(key, action, reward, next_key, done)
            self.update_q(key, action, reward, next_key, done)

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if action == ExitAction.EXIT:
                break  # episode ends on exit

        # Batch replay updates from buffer (stabilizes learning)
        if len(self._buf) >= 64:
            for transition in self._buf.sample(min(64, len(self._buf))):
                sk, a, r, nk, d = transition
                self.update_q(sk, a, r, nk, d)

        self._episode_rewards.append(total_reward)
        return total_reward

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, episodes: list[list[dict]], n_epochs: int = 50) -> dict:
        """
        Train over all episodes for n_epochs passes.

        Parameters
        ----------
        episodes : list of trade histories (each a list of bar dicts)
        n_epochs : number of full passes over the episode dataset

        Returns
        -------
        dict with training statistics
        """
        if not episodes:
            log.warning("No episodes to train on.")
            return {}

        log.info(
            "Starting Q-learning: %d episodes x %d epochs = %d total episode runs",
            len(episodes), n_epochs, len(episodes) * n_epochs,
        )

        epoch_rewards: list[float] = []

        for epoch in range(1, n_epochs + 1):
            random.shuffle(episodes)
            ep_rew: list[float] = []
            for ep in episodes:
                r = self.train_episode(ep)
                ep_rew.append(r)
            mean_r = float(np.mean(ep_rew)) if ep_rew else 0.0
            epoch_rewards.append(mean_r)

            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d/%d  mean_reward=%.4f  epsilon=%.4f  "
                    "states=%d  updates=%d",
                    epoch, n_epochs, mean_r, self.epsilon,
                    len(self._qtable), self._updates_total,
                )

        # Convergence delta (last 10 epochs)
        tail = epoch_rewards[-10:]
        conv_delta = float(np.std(tail)) if len(tail) >= 2 else float("nan")

        stats = {
            "n_episodes":    len(episodes),
            "n_epochs":      n_epochs,
            "n_states":      len(self._qtable),
            "total_updates": self._updates_total,
            "final_epsilon": round(self.epsilon, 6),
            "mean_reward_final_epoch": round(epoch_rewards[-1], 6) if epoch_rewards else 0.0,
            "convergence_std_last10":  round(conv_delta, 6),
            "epoch_rewards": [round(r, 6) for r in epoch_rewards],
        }
        log.info(
            "Training complete -- states=%d  mean_reward=%.4f  conv_std=%.4f",
            stats["n_states"],
            stats["mean_reward_final_epoch"],
            stats["convergence_std_last10"],
        )
        return stats

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_qtable(self, path: Path | str) -> None:
        """
        Serialize Q-table to JSON.

        Format:
          { "f0,f1,f2,f3,f4": [q_hold, q_exit], ... }

        This matches the format consumed by RLExitPolicy in live_trader_alpaca.py.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        disc = self._disc
        out: dict[str, list[float]] = {}
        for key, qs in self._qtable.items():
            str_key = disc.state_key_str(key)
            out[str_key] = [round(qs[0], 8), round(qs[1], 8)]

        path.write_text(json.dumps(out, indent=2))
        log.info("Q-table saved: %s  (%d states)", path, len(out))

    def load_qtable(self, path: Path | str) -> None:
        """Load a previously saved Q-table from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table not found: {path}")
        data = json.loads(path.read_text())
        disc = self._disc
        self._qtable.clear()
        for str_key, qs in data.items():
            key = disc.parse_key_str(str_key)
            self._qtable[key] = list(qs)
        log.info("Q-table loaded: %s  (%d states)", path, len(self._qtable))


# ---------------------------------------------------------------------------
# TradeDataLoader
# ---------------------------------------------------------------------------
class TradeDataLoader:
    """
    Loads historical trade data from the live_trades SQLite database and
    converts it to bar-by-bar episode format for Q-learning.

    When real data is sparse (< min_real_trades), supplements with synthetic
    GBM-with-regime-switching episodes.
    """

    MIN_REAL_TRADES = 20

    def __init__(self, db_path: Path | str = _DB_PATH) -> None:
        self._db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Real data loading
    # ------------------------------------------------------------------

    def load_real_episodes(self) -> list[list[dict]]:
        """
        Query trade_pnl table and reconstruct bar-by-bar trajectories.

        Each completed trade (entry + exit rows) is converted to a synthetic
        bar-by-bar walk with linearly interpolated P&L, simulating what the
        RL agent would have seen if applied bar-by-bar.
        """
        if not self._db_path.exists():
            log.warning("DB not found: %s -- skipping real episode load", self._db_path)
            return []

        episodes: list[list[dict]] = []

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT symbol, entry_time, exit_time, entry_price, exit_price,
                       qty, pnl, hold_bars
                FROM trade_pnl
                WHERE hold_bars IS NOT NULL AND hold_bars > 0
                ORDER BY entry_time
                """
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception as exc:
            log.warning("DB query failed: %s", exc)
            return []

        for row in rows:
            n_bars   = max(1, int(row["hold_bars"]))
            pnl_frac = float(row["pnl"]) / (
                abs(float(row["entry_price"]) * float(row["qty"])) + 1e-9
            )
            ep = self._interpolate_trajectory(pnl_frac, n_bars)
            if ep:
                episodes.append(ep)

        log.info("Loaded %d real trade episodes from DB", len(episodes))
        return episodes

    @staticmethod
    def _interpolate_trajectory(
        final_pnl: float,
        n_bars:    int,
        bh_mass_start: float = 1.5,
    ) -> list[dict]:
        """
        Construct a plausible bar-by-bar trajectory for a real trade using
        linear interpolation of P&L + Brownian noise.
        """
        if n_bars < 1:
            return []

        rng    = np.random.default_rng()
        noise  = rng.normal(0, 0.003, n_bars)
        trend  = np.linspace(0, final_pnl, n_bars)
        pnls   = trend + np.cumsum(noise) - np.cumsum(noise)[0]

        # BH mass decays gradually -- assume active first half, fading second
        half   = n_bars // 2
        masses = [
            max(0.0, bh_mass_start - 0.05 * i) for i in range(n_bars)
        ]
        actives = [masses[i] > 1.0 for i in range(n_bars)]

        # ATR ratio -- mild expansion mid-trade
        atrs = [1.0 + 0.3 * math.sin(math.pi * i / max(n_bars, 1)) for i in range(n_bars)]

        bars = []
        for i in range(n_bars):
            bars.append({
                "pnl_pct":   float(pnls[i]),
                "bars_held": i,
                "bh_mass":   float(masses[i]),
                "bh_active": bool(actives[i]),
                "atr_ratio": float(atrs[i]),
            })
        return bars

    # ------------------------------------------------------------------
    # Synthetic episode generation
    # ------------------------------------------------------------------

    def generate_synthetic_episodes(self, n: int = 1000) -> list[list[dict]]:
        """
        Generate n synthetic trade episodes using GBM with regime switching.

        Three scenario types (proportional):
          40% -- BH-active trending (momentum continuation)
          30% -- BH-inactive mean-reverting (choppy)
          30% -- losing trades (various exit timing scenarios)

        Returns list of episode histories (each is a list of bar dicts).
        """
        rng = np.random.default_rng(42)
        episodes: list[list[dict]] = []

        n_trend  = int(n * 0.40)
        n_chop   = int(n * 0.30)
        n_loss   = n - n_trend - n_chop

        episodes.extend(self._gen_trending(n_trend, rng))
        episodes.extend(self._gen_mean_reverting(n_chop, rng))
        episodes.extend(self._gen_losing(n_loss, rng))

        random.shuffle(episodes)
        log.info("Generated %d synthetic episodes (trend=%d chop=%d loss=%d)",
                 len(episodes), n_trend, n_chop, n_loss)
        return episodes

    def _gen_trending(self, n: int, rng: np.random.Generator) -> list[list[dict]]:
        """BH-active trending trades: prices drift up, BH mass builds then decays."""
        episodes = []
        for _ in range(n):
            n_bars   = int(rng.integers(8, 50))
            drift    = rng.uniform(0.0005, 0.003)    # positive drift per bar
            vol      = rng.uniform(0.003, 0.010)
            returns  = rng.normal(drift, vol, n_bars)
            prices   = np.cumprod(1.0 + returns)
            pnls     = prices - 1.0

            # BH mass: rises with price then decays
            peak_bar = int(rng.integers(n_bars // 3, n_bars))
            masses   = []
            for i in range(n_bars):
                if i <= peak_bar:
                    masses.append(min(3.5, 0.8 + 2.5 * i / max(peak_bar, 1)))
                else:
                    masses.append(max(0.0, masses[-1] * 0.95))

            atr_base = rng.uniform(0.8, 1.3)
            bars = []
            for i in range(n_bars):
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(masses[i]),
                    "bh_active": masses[i] > 1.0,
                    "atr_ratio": float(atr_base + rng.normal(0, 0.1)),
                })
            episodes.append(bars)
        return episodes

    def _gen_mean_reverting(self, n: int, rng: np.random.Generator) -> list[list[dict]]:
        """BH-inactive mean-reverting: OU process around zero, BH mass low."""
        episodes = []
        theta = 0.15   # mean reversion speed
        sigma = 0.006
        for _ in range(n):
            n_bars = int(rng.integers(6, 35))
            pnl    = 0.0
            pnls: list[float] = []
            for _ in range(n_bars):
                pnl += theta * (0.0 - pnl) + rng.normal(0, sigma)
                pnls.append(pnl)

            mass_base = rng.uniform(0.1, 0.8)
            atr_base  = rng.uniform(0.6, 1.0)
            bars = []
            for i in range(n_bars):
                noise_mass = max(0.0, mass_base + rng.normal(0, 0.05))
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(noise_mass),
                    "bh_active": False,
                    "atr_ratio": float(atr_base + rng.normal(0, 0.05)),
                })
            episodes.append(bars)
        return episodes

    def _gen_losing(self, n: int, rng: np.random.Generator) -> list[list[dict]]:
        """
        Losing trades with three sub-scenarios:
          - Fast blowup (stop-loss territory within 5 bars)
          - Slow grind lower (BH fades out)
          - False breakout (pump then dump)
        """
        episodes = []
        per_type = n // 3 + 1

        # Fast blowup
        for _ in range(per_type):
            n_bars = int(rng.integers(3, 12))
            returns = rng.normal(-0.005, 0.008, n_bars)
            pnls = np.cumprod(1.0 + returns) - 1.0
            mass_start = rng.uniform(1.0, 2.0)
            bars = []
            for i in range(n_bars):
                mass = max(0.0, mass_start - 0.3 * i)
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(mass),
                    "bh_active": mass > 1.0,
                    "atr_ratio": float(rng.uniform(1.2, 2.5)),
                })
            episodes.append(bars)

        # Slow grind lower
        for _ in range(per_type):
            n_bars = int(rng.integers(12, 40))
            returns = rng.normal(-0.001, 0.005, n_bars)
            pnls = np.cumprod(1.0 + returns) - 1.0
            bars = []
            for i in range(n_bars):
                mass = max(0.0, 1.5 - 0.06 * i + rng.normal(0, 0.05))
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(mass),
                    "bh_active": mass > 1.0,
                    "atr_ratio": float(rng.uniform(0.9, 1.4)),
                })
            episodes.append(bars)

        # False breakout
        for _ in range(min(per_type, n - len(episodes))):
            n_up   = int(rng.integers(2, 8))
            n_down = int(rng.integers(5, 20))
            n_bars = n_up + n_down
            up_rets   = rng.normal(0.004, 0.005, n_up)
            down_rets = rng.normal(-0.006, 0.007, n_down)
            returns   = np.concatenate([up_rets, down_rets])
            pnls      = np.cumprod(1.0 + returns) - 1.0
            bars = []
            for i in range(n_bars):
                mass = 1.8 if i < n_up else max(0.0, 1.8 - 0.2 * (i - n_up))
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(mass),
                    "bh_active": mass > 1.0,
                    "atr_ratio": float(1.0 + rng.normal(0, 0.2)),
                })
            episodes.append(bars)

        # Trim to requested count
        return episodes[:n]

    # ------------------------------------------------------------------
    # Combined loader
    # ------------------------------------------------------------------

    def get_all_episodes(
        self,
        n_synthetic: int = 1000,
    ) -> list[list[dict]]:
        """
        Return real episodes supplemented with synthetic ones when real data
        is below MIN_REAL_TRADES.
        """
        real_eps = self.load_real_episodes()
        n_synth  = max(0, n_synthetic - len(real_eps)) if len(real_eps) >= self.MIN_REAL_TRADES else n_synthetic
        synth_eps = self.generate_synthetic_episodes(n_synth)
        all_eps   = real_eps + synth_eps
        log.info("Total episodes: %d real + %d synthetic = %d",
                 len(real_eps), len(synth_eps), len(all_eps))
        return all_eps


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Q-learning exit policy and save to config/rl_exit_qtable.json"
    )
    p.add_argument("--epochs",    type=int,   default=50,     help="Number of training epochs")
    p.add_argument("--synthetic", type=int,   default=1000,   help="Number of synthetic episodes")
    p.add_argument("--db",        type=str,   default=str(_DB_PATH), help="Path to live_trades.db")
    p.add_argument("--output",    type=str,   default=str(_QTABLE_PATH), help="Output Q-table path")
    p.add_argument("--alpha",     type=float, default=0.10,   help="Learning rate")
    p.add_argument("--gamma",     type=float, default=0.95,   help="Discount factor")
    p.add_argument("--epsilon",   type=float, default=0.30,   help="Initial exploration rate")
    p.add_argument("--seed",      type=int,   default=0,      help="Random seed (0=no seed)")
    p.add_argument("--resume",    action="store_true",        help="Load existing Q-table and continue training")
    p.add_argument("--stats",     action="store_true",        help="Print Q-table state distribution after training")
    return p.parse_args()


def _print_qtable_stats(trainer: QLearningTrainer) -> None:
    """Print distribution of HOLD vs EXIT preference across Q-table states."""
    disc   = trainer._disc
    n_hold = 0
    n_exit = 0
    q_diff_sum = 0.0

    for key, qs in trainer._qtable.items():
        if qs[1] > qs[0]:
            n_exit += 1
        else:
            n_hold += 1
        q_diff_sum += qs[1] - qs[0]

    n_total = n_hold + n_exit
    log.info(
        "Q-table state distribution: %d states -- HOLD preferred: %d (%.1f%%)  "
        "EXIT preferred: %d (%.1f%%)  mean Q-diff(exit-hold): %.4f",
        n_total,
        n_hold, 100.0 * n_hold / max(n_total, 1),
        n_exit, 100.0 * n_exit / max(n_total, 1),
        q_diff_sum / max(n_total, 1),
    )

    # Print a few representative states
    sample_keys = list(trainer._qtable.keys())[:10]
    for key in sample_keys:
        qs  = trainer._qtable[key]
        desc = disc.decode(key)
        action = "EXIT" if qs[1] > qs[0] else "HOLD"
        log.info("  %s => %s  [Q_hold=%.4f, Q_exit=%.4f]", desc, action, qs[0], qs[1])


def main() -> None:
    args = _parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    loader  = TradeDataLoader(db_path=args.db)
    trainer = QLearningTrainer(
        alpha=   args.alpha,
        gamma=   args.gamma,
        epsilon= args.epsilon,
    )

    if args.resume:
        output_path = Path(args.output)
        if output_path.exists():
            trainer.load_qtable(output_path)
            log.info("Resumed from existing Q-table (%d states)", len(trainer._qtable))
        else:
            log.warning("--resume: no existing Q-table found at %s, starting fresh", output_path)

    episodes = loader.get_all_episodes(n_synthetic=args.synthetic)
    if not episodes:
        log.error("No episodes available -- cannot train. Exiting.")
        sys.exit(1)

    stats = trainer.train(episodes=episodes, n_epochs=args.epochs)

    if args.stats:
        _print_qtable_stats(trainer)

    trainer.save_qtable(args.output)

    # Write training stats alongside the Q-table
    stats_path = Path(args.output).with_suffix(".training_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Training stats saved: %s", stats_path)


if __name__ == "__main__":
    main()
