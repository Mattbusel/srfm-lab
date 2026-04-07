"""
ml/rl_agent/environment.py -- OpenAI Gym-compatible trading environment for RL exit policy training.

Provides a discrete-action environment where an agent learns when to exit a
trade position. State space uses 10 continuous features derived from BH (buy-hold)
regime indicators, price dynamics, and position metrics.

Actions:
  0 -- HOLD: do nothing, incur holding cost
  1 -- PARTIAL_EXIT: exit 50% of position
  2 -- FULL_EXIT: exit entire position, episode ends

Designed for compatibility with RLExitPolicy in tools/live_trader_alpaca.py.
Q-table export format matches the existing JSON schema keyed by 5-feature
discretized state strings.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gym compatibility shim -- avoids hard dependency
# ---------------------------------------------------------------------------

try:
    import gym
    from gym import spaces as gym_spaces

    class _Env(gym.Env):
        pass

    class _Discrete(gym_spaces.Discrete):
        pass

    class _Box(gym_spaces.Box):
        pass

    GYM_AVAILABLE = True

except ImportError:
    warnings.warn("gym not installed; using minimal stub. Install gymnasium or gym for full compatibility.")
    GYM_AVAILABLE = False

    class _Env:  # type: ignore
        """Minimal stub matching gym.Env interface."""
        metadata: Dict[str, Any] = {}
        reward_range: Tuple[float, float] = (-float("inf"), float("inf"))
        spec = None
        observation_space = None
        action_space = None

        def reset(self) -> np.ndarray:
            raise NotImplementedError

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
            raise NotImplementedError

        def render(self, mode: str = "human") -> None:
            pass

        def close(self) -> None:
            pass

    class _Discrete:  # type: ignore
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(0, self.n))

        def contains(self, x: int) -> bool:
            return 0 <= int(x) < self.n

    class _Box:  # type: ignore
        def __init__(self, low: np.ndarray, high: np.ndarray, shape: Tuple, dtype=np.float32) -> None:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x: np.ndarray) -> bool:
            return bool(np.all(x >= self.low) and np.all(x <= self.high))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLD = 0
PARTIAL_EXIT = 1
FULL_EXIT = 2

MAX_BARS = 32                   # terminal condition -- force exit after 32 bars
STOP_LOSS_PCT = -0.03           # terminal condition -- force exit at -3% PnL
HOLDING_COST = 0.001            # per-bar cost for HOLD action
TRANSACTION_COST = 0.002        # one-way transaction cost for exits
BH_BONUS = 0.005                # reward bonus for exiting with profit in BH-active regime
BH_INACTIVITY_PENALTY = 0.002  # per-bar penalty for holding when BH inactive

N_FEATURES = 10
N_ACTIONS = 3

# ---------------------------------------------------------------------------
# TradingState dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradingState:
    """
    10-feature continuous state representation for the RL agent.

    Features are normalized to approximately [-1, 1] for network stability.

    Attributes
    ----------
    position_pnl_pct : float
        Unrealized PnL as fraction of entry price (e.g. 0.02 = 2%).
    bars_held : float
        Number of bars the position has been held, normalized to [0, 1] by MAX_BARS.
    bh_mass : float
        BH (buy-hold) oscillator mass value in [0, 1].
    bh_active : float
        Binary indicator: 1.0 if BH regime is active, 0.0 otherwise.
    atr_ratio : float
        Current ATR divided by its rolling mean -- 1.0 = normal volatility.
    hurst_h : float
        Hurst exponent estimate; 0.5 = random walk, >0.5 = trending.
    nav_omega : float
        NAV velocity (recent PnL momentum), normalized.
    vol_percentile : float
        Current volatility percentile in [0, 1].
    time_of_day_sin : float
        Sine encoding of time-of-day in [0, 390] minutes (market hours).
    time_of_day_cos : float
        Cosine encoding of time-of-day.
    """

    position_pnl_pct: float = 0.0
    bars_held: float = 0.0
    bh_mass: float = 0.5
    bh_active: float = 1.0
    atr_ratio: float = 1.0
    hurst_h: float = 0.5
    nav_omega: float = 0.0
    vol_percentile: float = 0.5
    time_of_day_sin: float = 0.0
    time_of_day_cos: float = 1.0

    def to_array(self) -> np.ndarray:
        """Return state as float32 numpy array of shape (10,)."""
        return np.array([
            self.position_pnl_pct,
            self.bars_held,
            self.bh_mass,
            self.bh_active,
            self.atr_ratio,
            self.hurst_h,
            self.nav_omega,
            self.vol_percentile,
            self.time_of_day_sin,
            self.time_of_day_cos,
        ], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "TradingState":
        arr = np.asarray(arr, dtype=np.float32)
        assert arr.shape == (N_FEATURES,), f"Expected shape ({N_FEATURES},), got {arr.shape}"
        return TradingState(
            position_pnl_pct=float(arr[0]),
            bars_held=float(arr[1]),
            bh_mass=float(arr[2]),
            bh_active=float(arr[3]),
            atr_ratio=float(arr[4]),
            hurst_h=float(arr[5]),
            nav_omega=float(arr[6]),
            vol_percentile=float(arr[7]),
            time_of_day_sin=float(arr[8]),
            time_of_day_cos=float(arr[9]),
        )


# ---------------------------------------------------------------------------
# TradeEpisode -- container for a simulated historical trade path
# ---------------------------------------------------------------------------

@dataclass
class TradeEpisode:
    """
    A single simulated trade episode.

    Contains bar-by-bar price and regime data for one trade from entry to
    a fixed maximum horizon.

    Attributes
    ----------
    prices : np.ndarray
        Array of shape (T,) with bar close prices. prices[0] = entry price.
    bh_mass_series : np.ndarray
        BH oscillator mass at each bar, shape (T,).
    bh_active_series : np.ndarray
        Binary BH active flag at each bar, shape (T,).
    atr_series : np.ndarray
        ATR values at each bar, shape (T,).
    hurst_series : np.ndarray
        Hurst exponent estimate at each bar, shape (T,).
    vol_pct_series : np.ndarray
        Volatility percentile at each bar, shape (T,).
    bar_minutes : np.ndarray
        Minutes since market open for each bar, shape (T,).
    episode_type : str
        One of "bh_trending", "bh_inactive", "mixed".
    """

    prices: np.ndarray
    bh_mass_series: np.ndarray
    bh_active_series: np.ndarray
    atr_series: np.ndarray
    hurst_series: np.ndarray
    vol_pct_series: np.ndarray
    bar_minutes: np.ndarray
    episode_type: str = "bh_trending"

    def __len__(self) -> int:
        return len(self.prices)


# ---------------------------------------------------------------------------
# TradeEpisodeGenerator
# ---------------------------------------------------------------------------

class TradeEpisodeGenerator:
    """
    Generates synthetic trade episodes using stochastic processes.

    Three episode types are supported:

    bh_trending -- GBM with positive drift (momentum regime).
        Price tends to trend upward. BH mass high, BH active.

    bh_inactive -- Ornstein-Uhlenbeck mean reversion.
        Price oscillates around entry. BH mass low, BH inactive.

    mixed -- BH active for first N bars, then becomes inactive.
        Tests whether agent can adapt when regime shifts mid-trade.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        episode_type: str = "bh_trending",
        n_bars: int = MAX_BARS + 4,
        entry_price: float = 100.0,
    ) -> TradeEpisode:
        """
        Generate one synthetic trade episode.

        Parameters
        ----------
        episode_type : str
            "bh_trending", "bh_inactive", or "mixed".
        n_bars : int
            Number of bars to simulate (includes entry bar at index 0).
        entry_price : float
            Starting price at bar 0.

        Returns
        -------
        TradeEpisode
        """
        if episode_type == "bh_trending":
            return self._gen_bh_trending(n_bars, entry_price)
        elif episode_type == "bh_inactive":
            return self._gen_bh_inactive(n_bars, entry_price)
        elif episode_type == "mixed":
            return self._gen_mixed(n_bars, entry_price)
        else:
            raise ValueError(f"Unknown episode_type: {episode_type!r}. Use 'bh_trending', 'bh_inactive', or 'mixed'.")

    def _gen_bh_trending(self, n_bars: int, entry_price: float) -> TradeEpisode:
        """GBM with positive drift -- momentum regime."""
        mu = self.rng.uniform(0.001, 0.004)    # per-bar drift
        sigma = self.rng.uniform(0.005, 0.015)  # per-bar vol
        prices = self._gbm(entry_price, mu, sigma, n_bars)

        # BH mass rises from 0.4 to 0.9 over episode
        bh_mass = np.clip(
            np.linspace(0.4, 0.9, n_bars) + self.rng.normal(0, 0.05, n_bars),
            0.0, 1.0,
        )
        bh_active = (bh_mass > 0.55).astype(np.float32)

        atr = self._atr_from_prices(prices, sigma)
        hurst = np.clip(
            np.full(n_bars, 0.65) + self.rng.normal(0, 0.05, n_bars),
            0.5, 0.95,
        )
        vol_pct = np.clip(
            np.linspace(0.5, 0.7, n_bars) + self.rng.normal(0, 0.05, n_bars),
            0.0, 1.0,
        )
        bar_minutes = self._market_bar_minutes(n_bars)
        return TradeEpisode(
            prices=prices,
            bh_mass_series=bh_mass.astype(np.float32),
            bh_active_series=bh_active,
            atr_series=atr.astype(np.float32),
            hurst_series=hurst.astype(np.float32),
            vol_pct_series=vol_pct.astype(np.float32),
            bar_minutes=bar_minutes,
            episode_type="bh_trending",
        )

    def _gen_bh_inactive(self, n_bars: int, entry_price: float) -> TradeEpisode:
        """Ornstein-Uhlenbeck mean reversion -- no momentum."""
        theta = self.rng.uniform(0.05, 0.2)   # mean-reversion speed
        sigma = self.rng.uniform(0.005, 0.012)
        mu_ou = entry_price                    # revert to entry price

        prices = np.zeros(n_bars, dtype=np.float32)
        prices[0] = entry_price
        for t in range(1, n_bars):
            drift = theta * (mu_ou - prices[t - 1])
            noise = sigma * prices[t - 1] * self.rng.standard_normal()
            prices[t] = max(prices[t - 1] + drift + noise, 0.01)

        bh_mass = np.clip(
            np.linspace(0.45, 0.25, n_bars) + self.rng.normal(0, 0.05, n_bars),
            0.0, 1.0,
        )
        bh_active = (bh_mass > 0.55).astype(np.float32)

        atr = self._atr_from_prices(prices, sigma)
        hurst = np.clip(
            np.full(n_bars, 0.35) + self.rng.normal(0, 0.04, n_bars),
            0.1, 0.5,
        )
        vol_pct = np.clip(
            np.linspace(0.4, 0.3, n_bars) + self.rng.normal(0, 0.05, n_bars),
            0.0, 1.0,
        )
        bar_minutes = self._market_bar_minutes(n_bars)
        return TradeEpisode(
            prices=prices,
            bh_mass_series=bh_mass.astype(np.float32),
            bh_active_series=bh_active,
            atr_series=atr.astype(np.float32),
            hurst_series=hurst.astype(np.float32),
            vol_pct_series=vol_pct.astype(np.float32),
            bar_minutes=bar_minutes,
            episode_type="bh_inactive",
        )

    def _gen_mixed(self, n_bars: int, entry_price: float) -> TradeEpisode:
        """BH active for first half of episode, inactive for second half."""
        split = n_bars // 2

        # First half: GBM trending
        mu = self.rng.uniform(0.001, 0.003)
        sigma = self.rng.uniform(0.005, 0.012)
        prices_a = self._gbm(entry_price, mu, sigma, split + 1)

        # Second half: OU reverting from transition price
        theta = self.rng.uniform(0.05, 0.15)
        sigma_ou = self.rng.uniform(0.006, 0.013)
        prices_b = np.zeros(n_bars - split, dtype=np.float32)
        prices_b[0] = prices_a[-1]
        for t in range(1, len(prices_b)):
            drift = theta * (prices_a[-1] - prices_b[t - 1])  # revert to split price
            noise = sigma_ou * prices_b[t - 1] * self.rng.standard_normal()
            prices_b[t] = max(prices_b[t - 1] + drift + noise, 0.01)

        prices = np.concatenate([prices_a[:-1], prices_b]).astype(np.float32)

        bh_mass = np.zeros(n_bars, dtype=np.float32)
        bh_mass[:split] = np.clip(
            np.linspace(0.5, 0.85, split) + self.rng.normal(0, 0.04, split), 0.0, 1.0
        )
        bh_mass[split:] = np.clip(
            np.linspace(0.6, 0.2, n_bars - split) + self.rng.normal(0, 0.04, n_bars - split), 0.0, 1.0
        )
        bh_active = (bh_mass > 0.55).astype(np.float32)

        atr = self._atr_from_prices(prices, sigma)
        hurst_vals = np.zeros(n_bars, dtype=np.float32)
        hurst_vals[:split] = np.clip(0.65 + self.rng.normal(0, 0.04, split), 0.5, 0.95)
        hurst_vals[split:] = np.clip(0.35 + self.rng.normal(0, 0.04, n_bars - split), 0.1, 0.5)

        vol_pct = np.clip(0.5 + self.rng.normal(0, 0.1, n_bars), 0.0, 1.0).astype(np.float32)
        bar_minutes = self._market_bar_minutes(n_bars)
        return TradeEpisode(
            prices=prices,
            bh_mass_series=bh_mass,
            bh_active_series=bh_active,
            atr_series=atr.astype(np.float32),
            hurst_series=hurst_vals,
            vol_pct_series=vol_pct,
            bar_minutes=bar_minutes,
            episode_type="mixed",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gbm(self, s0: float, mu: float, sigma: float, n: int) -> np.ndarray:
        """Simulate Geometric Brownian Motion."""
        dt = 1.0  # 1 bar per step
        prices = np.zeros(n, dtype=np.float32)
        prices[0] = s0
        z = self.rng.standard_normal(n - 1)
        log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
        prices[1:] = s0 * np.exp(np.cumsum(log_returns))
        return prices

    def _atr_from_prices(self, prices: np.ndarray, base_sigma: float) -> np.ndarray:
        """Approximate ATR as rolling std scaled by price, with ratio to mean."""
        n = len(prices)
        atr_abs = np.abs(np.diff(prices, prepend=prices[0]))
        # Normalize to ratio form (1.0 = average)
        mean_atr = np.mean(atr_abs[1:]) + 1e-8
        atr_ratio = atr_abs / mean_atr
        atr_ratio[0] = 1.0
        return atr_ratio.astype(np.float32)

    def _market_bar_minutes(self, n: int) -> np.ndarray:
        """Return simulated bar timestamps as minutes since market open (0-390)."""
        # Randomly pick a starting minute in the trading day
        start = int(self.rng.integers(0, 360))
        return np.array([(start + i * 15) % 390 for i in range(n)], dtype=np.float32)

    def random_episode(self, n_bars: int = MAX_BARS + 4) -> TradeEpisode:
        """Generate a random episode from a uniformly sampled type."""
        ep_type = self.rng.choice(["bh_trending", "bh_inactive", "mixed"])
        entry_price = float(self.rng.uniform(50.0, 500.0))
        return self.generate(ep_type, n_bars=n_bars, entry_price=entry_price)


# ---------------------------------------------------------------------------
# TradingEnvironment -- main Gym-compatible environment
# ---------------------------------------------------------------------------

class TradingEnvironment(_Env):
    """
    OpenAI Gym-compatible trading environment for RL exit policy training.

    The agent controls when to exit a trade position. At each step the agent
    observes a 10-dimensional state vector and chooses from:

      0 -- HOLD: stay in position, pay holding cost + possible BH-inactivity penalty
      1 -- PARTIAL_EXIT: liquidate 50% of position, pay transaction cost
      2 -- FULL_EXIT: liquidate 100%, episode terminates

    Episodes also terminate automatically if:
      - bars_held > MAX_BARS (32)
      - position_pnl_pct < STOP_LOSS_PCT (-3%)
      - FULL_EXIT action is taken

    Parameters
    ----------
    episode_generator : TradeEpisodeGenerator, optional
        Used for synthetic episode generation. Created with default seed if None.
    seed : int, optional
        Random seed for reproducibility.
    reward_scale : float
        Scale factor applied to all rewards.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        episode_generator: Optional[TradeEpisodeGenerator] = None,
        seed: Optional[int] = None,
        reward_scale: float = 100.0,
    ) -> None:
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.episode_gen = episode_generator or TradeEpisodeGenerator(seed=seed)
        self.reward_scale = reward_scale

        # Action and observation spaces
        self.action_space = _Discrete(N_ACTIONS)
        low = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = _Box(low=low, high=high, shape=(N_FEATURES,), dtype=np.float32)

        # Episode state -- initialized by reset()
        self._episode: Optional[TradeEpisode] = None
        self._bar: int = 0
        self._position_fraction: float = 1.0   # 1.0 = full, 0.5 = after partial exit
        self._realized_pnl: float = 0.0
        self._done: bool = True
        self._prev_pnl_pct: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, trade_episode: Optional[TradeEpisode] = None) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        trade_episode : TradeEpisode, optional
            If provided, use this historical/pre-generated episode.
            Otherwise generate a new synthetic episode.

        Returns
        -------
        np.ndarray of shape (10,)
            Initial observation.
        """
        if trade_episode is not None:
            self._episode = trade_episode
        else:
            self._episode = self.episode_gen.random_episode()

        self._bar = 0
        self._position_fraction = 1.0
        self._realized_pnl = 0.0
        self._done = False
        self._prev_pnl_pct = 0.0

        obs = self._get_observation()
        self._prev_pnl_pct = float(obs[0])
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            0 = HOLD, 1 = PARTIAL_EXIT, 2 = FULL_EXIT.

        Returns
        -------
        obs : np.ndarray, shape (10,)
        reward : float
        done : bool
        info : dict
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        ep = self._episode
        assert ep is not None

        current_pnl_pct = self._compute_pnl_pct(self._bar)
        bh_active = bool(ep.bh_active_series[self._bar] > 0.5)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {
            "bar": self._bar,
            "action": action,
            "pnl_pct": current_pnl_pct,
            "position_fraction": self._position_fraction,
            "bh_active": bh_active,
        }

        if action == HOLD:
            # PnL change since last bar
            pnl_change = current_pnl_pct - self._prev_pnl_pct
            # Holding cost (opportunity cost, slippage drag)
            cost = HOLDING_COST
            # Extra penalty if BH has become inactive -- agent should exit sooner
            if not bh_active:
                cost += BH_INACTIVITY_PENALTY
            reward = (pnl_change - cost) * self.reward_scale
            self._bar += 1

        elif action == PARTIAL_EXIT:
            # Realize half the position at current price
            realized_fraction = 0.5 * self._position_fraction
            realized_pnl = realized_fraction * current_pnl_pct
            reward = (realized_pnl - TRANSACTION_COST) * self.reward_scale
            # Bonus: exiting with profit while BH still active
            if bh_active and current_pnl_pct > 0:
                reward += BH_BONUS * self.reward_scale
            self._realized_pnl += realized_pnl
            self._position_fraction *= 0.5
            self._bar += 1
            info["realized_pnl"] = realized_pnl

        elif action == FULL_EXIT:
            # Realize entire remaining position
            realized_pnl = self._position_fraction * current_pnl_pct
            reward = (realized_pnl - TRANSACTION_COST) * self.reward_scale
            if bh_active and current_pnl_pct > 0:
                reward += BH_BONUS * self.reward_scale
            self._realized_pnl += realized_pnl
            self._position_fraction = 0.0
            done = True
            info["realized_pnl"] = realized_pnl
            info["total_pnl"] = self._realized_pnl + realized_pnl

        # Check terminal conditions even after HOLD or PARTIAL_EXIT
        # Re-evaluate PnL at the new bar position (after _bar was advanced)
        if not done:
            bars_held = self._bar
            new_pnl_pct = self._compute_pnl_pct(min(self._bar, len(ep) - 1))
            if bars_held >= MAX_BARS:
                # Force exit at max hold -- treated as FULL_EXIT with no extra cost
                forced_pnl = self._position_fraction * new_pnl_pct
                reward += forced_pnl * self.reward_scale
                self._realized_pnl += forced_pnl
                done = True
                info["terminal"] = "max_bars"
            elif new_pnl_pct < STOP_LOSS_PCT:
                # Hard stop loss -- forced full exit at new bar price
                forced_pnl = self._position_fraction * new_pnl_pct
                reward += forced_pnl * self.reward_scale
                self._realized_pnl += forced_pnl
                done = True
                info["terminal"] = "stop_loss"

        self._done = done
        self._prev_pnl_pct = current_pnl_pct

        if done:
            info["episode_pnl"] = self._realized_pnl
        else:
            obs = self._get_observation()
            self._prev_pnl_pct = float(obs[0])
            return obs, float(reward), done, info

        # If done, return a terminal observation (zeroed out or last valid)
        terminal_obs = self._get_observation_at(min(self._bar, len(ep) - 1))
        return terminal_obs, float(reward), done, info

    def render(self, mode: str = "human") -> Optional[str]:
        ep = self._episode
        if ep is None:
            return None
        pnl = self._compute_pnl_pct(min(self._bar, len(ep) - 1))
        msg = (
            f"Bar {self._bar}/{MAX_BARS} | "
            f"PnL {pnl*100:.2f}% | "
            f"Pos frac {self._position_fraction:.2f} | "
            f"BH active {ep.bh_active_series[min(self._bar, len(ep)-1)] > 0.5}"
        )
        if mode == "human":
            print(msg)
        return msg

    def close(self) -> None:
        self._episode = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pnl_pct(self, bar_idx: int) -> float:
        """Unrealized PnL fraction relative to entry price at bar 0."""
        ep = self._episode
        assert ep is not None
        bar_idx = min(bar_idx, len(ep) - 1)
        entry_price = float(ep.prices[0])
        current_price = float(ep.prices[bar_idx])
        if entry_price == 0:
            return 0.0
        return (current_price - entry_price) / entry_price

    def _get_observation(self) -> np.ndarray:
        bar = min(self._bar, len(self._episode) - 1)  # type: ignore[arg-type]
        return self._get_observation_at(bar)

    def _get_observation_at(self, bar: int) -> np.ndarray:
        ep = self._episode
        assert ep is not None
        bar = min(bar, len(ep) - 1)

        pnl_pct = self._compute_pnl_pct(bar)
        bars_held_norm = float(bar) / MAX_BARS  # [0, 1]

        bh_mass = float(ep.bh_mass_series[bar])
        bh_active = float(ep.bh_active_series[bar])
        atr_ratio = float(ep.atr_series[bar])
        hurst_h = float(ep.hurst_series[bar])
        vol_pct = float(ep.vol_pct_series[bar])

        # NAV omega: recent PnL velocity over last 3 bars
        start_bar = max(0, bar - 3)
        nav_omega = 0.0
        if bar > 0:
            past_pnl = self._compute_pnl_pct(start_bar)
            nav_omega = np.clip((pnl_pct - past_pnl) / 3.0, -1.0, 1.0)

        # Time-of-day encoding
        minutes = float(ep.bar_minutes[bar])
        angle = 2.0 * math.pi * minutes / 390.0
        tod_sin = math.sin(angle)
        tod_cos = math.cos(angle)

        state = TradingState(
            position_pnl_pct=float(np.clip(pnl_pct, -1.0, 1.0)),
            bars_held=bars_held_norm,
            bh_mass=bh_mass,
            bh_active=bh_active,
            atr_ratio=float(np.clip(atr_ratio, 0.0, 5.0)),
            hurst_h=float(np.clip(hurst_h, 0.0, 1.0)),
            nav_omega=float(nav_omega),
            vol_percentile=vol_pct,
            time_of_day_sin=tod_sin,
            time_of_day_cos=tod_cos,
        )
        return state.to_array()

    # ------------------------------------------------------------------
    # Utility: discretize state for Q-table export
    # ------------------------------------------------------------------

    @staticmethod
    def discretize_state_key(
        pnl_pct: float,
        bars_held_norm: float,
        bh_mass: float,
        bh_active: float,
        atr_ratio: float,
        n_bins: int = 5,
    ) -> str:
        """
        Produce the 5-feature discretized state key compatible with RLExitPolicy
        in tools/live_trader_alpaca.py.

        Uses the same scaling as RLExitPolicy._state_key().
        """

        def _disc(v: float, lo: float = -1.0, hi: float = 1.0) -> int:
            clipped = max(lo, min(hi, v))
            idx = int((clipped - lo) / (hi - lo) * n_bins)
            return min(n_bins - 1, max(0, idx))

        f0 = _disc(pnl_pct * 2.0)              # pnl_pct scaled to [-1,1] at +-50%
        f1 = _disc(bars_held_norm * 2.0 - 1.0)  # bars_held_norm -> [-1,1]
        f2 = _disc(bh_mass * 2.0 - 1.0)         # bh_mass [0,1] -> [-1,1]
        f3 = 1 if bh_active > 0.5 else 0
        f4 = _disc(atr_ratio - 1.0)              # atr_ratio centered at 1.0
        return f"{f0},{f1},{f2},{f3},{f4}"
