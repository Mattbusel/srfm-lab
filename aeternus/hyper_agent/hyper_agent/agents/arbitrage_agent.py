"""
ArbitrageAgent — Exploits temporary mispricings across synthetic cointegrated assets.

Features:
  - Engle-Granger cointegration test at initialization
  - Z-score based entry/exit with RL-learned thresholds
  - Reward: roundtrip PnL on arb trades
  - Adaptive threshold learning when market structure shifts
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from hyper_agent.agents._base_compat import BaseAgent, compute_gae


# ============================================================
# Cointegration utilities
# ============================================================

def engle_granger_test(
    y1: np.ndarray, y2: np.ndarray, max_lag: int = 5
) -> Tuple[float, float, float]:
    """
    Simple Engle-Granger cointegration test.

    Steps:
      1. OLS regression: y1 = α + β*y2 + ε
      2. ADF test on residuals ε
      3. Return (beta, hedge_ratio, adf_stat)

    Returns:
        (beta, hedge_ratio, adf_statistic)
        adf_stat < -3.34 → cointegrated at 5% level (approx)
    """
    if len(y1) < 20 or len(y2) < 20:
        return 1.0, 1.0, 0.0

    # Step 1: OLS
    y2_const = np.column_stack([np.ones_like(y2), y2])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(y2_const, y1, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0, 1.0, 0.0

    alpha, beta = float(coeffs[0]), float(coeffs[1])
    residuals   = y1 - alpha - beta * y2

    # Step 2: ADF test on residuals
    adf_stat = _simple_adf(residuals, max_lag)
    return beta, alpha, adf_stat


def _simple_adf(series: np.ndarray, max_lag: int = 5) -> float:
    """
    Simplified Augmented Dickey-Fuller statistic.
    H0: unit root; more negative → stronger rejection.
    """
    n = len(series)
    if n < max_lag + 5:
        return 0.0

    diff_y = np.diff(series)
    lagged_y = series[:-1]

    # Regress Δy on y_{t-1} and lagged differences
    X_cols = [lagged_y[max_lag:]]
    for lag in range(1, max_lag + 1):
        X_cols.append(diff_y[max_lag - lag: -lag or None])
    X = np.column_stack(X_cols)
    y = diff_y[max_lag:]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat  = X @ coeffs
        resid  = y - y_hat
        sse    = (resid ** 2).sum()
        se_sq  = sse / max(len(y) - X.shape[1], 1)
        cov    = se_sq * np.linalg.pinv(X.T @ X)
        se_rho = math.sqrt(max(cov[0, 0], 1e-12))
        t_stat = coeffs[0] / se_rho
    except (np.linalg.LinAlgError, FloatingPointError):
        return 0.0

    return float(t_stat)


class SpreadTracker:
    """
    Tracks spread between two cointegrated synthetic assets.

    Maintains rolling Z-score: z = (spread - μ) / σ
    """

    def __init__(self, window: int = 60, beta: float = 1.0) -> None:
        self.window = window
        self.beta   = beta  # hedge ratio from cointegration
        self._spread_history: deque = deque(maxlen=window * 3)
        self._rolling_mu:  float = 0.0
        self._rolling_var: float = 1.0
        self._count:       int   = 0

    def update(self, price1: float, price2: float) -> float:
        """
        Compute and return current spread Z-score.
        spread = price1 - beta * price2
        """
        spread = price1 - self.beta * price2
        self._spread_history.append(spread)

        # Welford online update
        n = len(self._spread_history)
        if n >= 5:
            arr = np.array(list(self._spread_history)[-self.window:])
            self._rolling_mu  = float(arr.mean())
            self._rolling_var = float(arr.var()) + 1e-8
        self._count += 1
        return self.z_score(spread)

    def z_score(self, spread: Optional[float] = None) -> float:
        if spread is None and self._spread_history:
            spread = self._spread_history[-1]
        if spread is None:
            return 0.0
        return float((spread - self._rolling_mu) / math.sqrt(self._rolling_var))

    def half_life(self) -> float:
        """Estimate mean-reversion half-life via AR(1) coefficient."""
        if len(self._spread_history) < 20:
            return float("inf")
        s = np.array(list(self._spread_history))
        try:
            rho = np.corrcoef(s[:-1], s[1:])[0, 1]
            if rho >= 1.0 or rho <= 0.0:
                return float("inf")
            return float(-math.log(2.0) / math.log(abs(rho)))
        except Exception:
            return float("inf")

    def refit(self, price1_history: np.ndarray, price2_history: np.ndarray) -> float:
        """Refit hedge ratio using recent data. Returns new beta."""
        beta, _, adf = engle_granger_test(price1_history, price2_history)
        self.beta    = beta
        return adf


# ============================================================
# Arb Threshold Actor (RL policy)
# ============================================================

class ArbActor(nn.Module):
    """
    Actor that learns optimal entry/exit Z-score thresholds.

    Outputs:
      - entry_threshold: z-score to enter arb (typically ~2.0, adaptive)
      - exit_threshold:  z-score to exit  (typically ~0.5)
      - position_size:   fraction of max capital to allocate
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # Entry threshold: softplus → (0, ∞), init ~2.0
        self.entry_head = nn.Linear(hidden_dim // 2, 1)
        # Exit threshold: softplus → (0, ∞), init ~0.5
        self.exit_head  = nn.Linear(hidden_dim // 2, 1)
        # Size: sigmoid → (0, 1)
        self.size_head  = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Bias entry head to produce ~2.0 initially
        with torch.no_grad():
            self.entry_head.bias.fill_(math.log(math.exp(2.0) - 1))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        entry = F.softplus(self.entry_head(h)).squeeze(-1) + 0.1
        exit_ = F.softplus(self.exit_head(h)).squeeze(-1)  + 0.05
        size  = torch.sigmoid(self.size_head(h)).squeeze(-1)
        return entry, exit_, size


# ============================================================
# ArbitrageAgent
# ============================================================

class ArbitrageAgent(BaseAgent):
    """
    Statistical arbitrage agent that trades mean-reverting spread.

    Uses synthetic pair (asset 1 vs beta * asset 2) created from
    price history of the primary market.
    """

    AGENT_TYPE = "arbitrage"

    # Cointegration check interval (steps)
    REFIT_INTERVAL = 200

    def __init__(
        self,
        agent_id:        str,
        obs_dim:         int,
        hidden_dim:      int   = 64,
        lr:              float = 1e-3,
        gamma:           float = 0.99,
        lam:             float = 0.95,
        clip_eps:        float = 0.2,
        entropy_coef:    float = 0.01,
        n_epochs:        int   = 4,
        minibatch_size:  int   = 64,
        rollout_len:     int   = 256,
        spread_window:   int   = 60,
        refit_interval:  int   = 200,
        device:          str   = "cpu",
    ) -> None:
        super().__init__(
            agent_id   = agent_id,
            obs_dim    = obs_dim,
            hidden_dim = hidden_dim,
            lr         = lr,
            gamma      = gamma,
            lam        = lam,
            device     = device,
        )
        self.clip_eps       = clip_eps
        self.entropy_coef   = entropy_coef
        self.n_epochs       = n_epochs
        self.minibatch_size = minibatch_size
        self.rollout_len    = rollout_len
        self.refit_interval = refit_interval

        # Networks
        self.actor  = ArbActor(obs_dim, hidden_dim).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Spread tracker
        self.spread_tracker = SpreadTracker(window=spread_window)

        # Arb position state: +1 (long spread), -1 (short spread), 0 (flat)
        self._arb_position: int = 0
        self._entry_price:  Optional[float] = None
        self._entry_z:      Optional[float] = None

        # Price histories for cointegration refitting
        self._price1_hist: deque = deque(maxlen=500)
        self._price2_hist: deque = deque(maxlen=500)  # synthetic: offset price

        # Rollout storage
        self._rollout: Dict[str, List] = {
            k: [] for k in ["obs", "entry_thres", "exit_thres", "sizes",
                             "log_probs", "rewards", "values", "dones"]
        }

        # Stats
        self.trade_log: List[Dict] = []
        self.adf_history: List[float] = []
        self._steps_since_refit = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> str:
        return self.AGENT_TYPE

    @torch.no_grad()
    def act(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        self.obs_norm.update(obs)
        norm_obs = self.obs_norm.normalize(obs)
        obs_t    = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)

        entry_th, exit_th, size = self.actor(obs_t)
        value = self.critic(obs_t).item()

        et = entry_th.item()
        xt = exit_th.item()
        sz = size.item()

        # Clamp exit below entry
        xt = min(xt, et * 0.8)

        # Determine arb action from current z-score
        z        = self.spread_tracker.z_score()
        direction, log_prob = self._arb_decision(z, et, xt, sz)

        # Store for rollout
        self._rollout["obs"].append(norm_obs)
        self._rollout["entry_thres"].append(et)
        self._rollout["exit_thres"].append(xt)
        self._rollout["sizes"].append(sz)
        self._rollout["log_probs"].append(log_prob)
        self._rollout["values"].append(value)

        # Build standard action array
        if direction > 0:
            logits = np.array([-3.0, 0.0, 3.0 * sz], dtype=np.float32)
        elif direction < 0:
            logits = np.array([3.0 * sz, 0.0, -3.0], dtype=np.float32)
        else:
            logits = np.array([-1.0, 3.0, -1.0], dtype=np.float32)

        action = np.array([*logits, sz], dtype=np.float32)
        return action, log_prob, value

    def _arb_decision(
        self, z: float, entry_th: float, exit_th: float, size: float
    ) -> Tuple[int, float]:
        """
        Translate z-score + learned thresholds into trade direction.

        Returns (direction: -1/0/+1, log_prob).
        """
        direction = 0
        if self._arb_position == 0:
            # Entry conditions
            if z > entry_th:
                direction = -1  # short spread: sell high
            elif z < -entry_th:
                direction = +1  # long spread: buy low
        else:
            # Exit conditions
            if self._arb_position == -1 and z < exit_th:
                direction = 0   # exit short
                self._arb_position = 0
            elif self._arb_position == +1 and z > -exit_th:
                direction = 0   # exit long
                self._arb_position = 0
            else:
                direction = self._arb_position  # hold

        if direction != 0:
            self._arb_position = direction

        # Approximate log prob (deterministic policy with noise)
        log_prob = -0.5  # flat prior; refined in update
        return direction, log_prob

    def observe_prices(self, price: float, step: int) -> float:
        """
        Update spread tracker with new price.

        Uses synthetic "asset 2" as lagged version of asset 1.
        Returns current Z-score.
        """
        self._price1_hist.append(price)
        # Synthetic asset 2: lagged price + noise
        lag = 5
        if len(self._price1_hist) > lag:
            price2 = list(self._price1_hist)[-lag]
        else:
            price2 = price * 0.99

        self._price2_hist.append(price2)
        z = self.spread_tracker.update(price, price2)

        # Periodically refit cointegration
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.refit_interval:
            self._refit_cointegration()
            self._steps_since_refit = 0

        return z

    def _refit_cointegration(self) -> None:
        """Refit hedge ratio using recent price history."""
        p1 = np.array(list(self._price1_hist), dtype=np.float64)
        p2 = np.array(list(self._price2_hist), dtype=np.float64)
        if len(p1) < 30 or len(p2) < 30:
            return
        adf = self.spread_tracker.refit(p1, p2)
        self.adf_history.append(adf)

    def receive_reward(self, reward: float, done: bool) -> float:
        """Store reward in rollout."""
        self._rollout["rewards"].append(reward)
        self._rollout["dones"].append(float(done))
        return reward

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        n = len(self._rollout["rewards"])
        if n < self.minibatch_size:
            return {}

        obs_arr  = np.array(self._rollout["obs"][:n],       dtype=np.float32)
        log_probs= np.array(self._rollout["log_probs"][:n], dtype=np.float32)
        rewards  = np.array(self._rollout["rewards"][:n],   dtype=np.float32)
        values   = np.array(self._rollout["values"][:n],    dtype=np.float32)
        dones    = np.array(self._rollout["dones"][:n],     dtype=np.float32)
        entry_th = np.array(self._rollout["entry_thres"][:n], dtype=np.float32)
        sizes    = np.array(self._rollout["sizes"][:n],     dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, gamma=self.gamma, lam=self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats_list: Dict[str, List[float]] = {"pol_loss": [], "val_loss": []}

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                b = idx[start: start + self.minibatch_size]
                obs_t  = torch.FloatTensor(obs_arr[b]).to(self.device)
                adv_t  = torch.FloatTensor(advantages[b]).to(self.device)
                ret_t  = torch.FloatTensor(returns[b]).to(self.device)
                olp_t  = torch.FloatTensor(log_probs[b]).to(self.device)

                et_t, xt_t, sz_t = self.actor(obs_t)
                # Policy: maximize expected entry_threshold (wider thresholds → less noise)
                # Penalize very wide thresholds to encourage trading
                pol_loss = -(adv_t * sz_t).mean() + 0.01 * (et_t ** 2).mean()

                val      = self.critic(obs_t).squeeze(-1)
                val_loss = F.mse_loss(val, ret_t)

                loss     = pol_loss + 0.5 * val_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
                )
                self.optimizer.step()

                stats_list["pol_loss"].append(pol_loss.item())
                stats_list["val_loss"].append(val_loss.item())

        for k in self._rollout:
            self._rollout[k].clear()

        return {k: float(np.mean(v)) for k, v in stats_list.items()}

    def get_spread_stats(self) -> Dict[str, float]:
        return {
            "z_score":    self.spread_tracker.z_score(),
            "half_life":  self.spread_tracker.half_life(),
            "arb_pos":    float(self._arb_position),
            "last_adf":   self.adf_history[-1] if self.adf_history else 0.0,
        }
