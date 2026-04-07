"""
research/simulation/market_simulator.py

Realistic market microstructure simulator. Generates synthetic OHLCV bars
with embedded alpha signals, regime switching, correlated multi-asset paths,
and full BH-ACTIVE dynamics compatible with the LARSA BH physics engine.

Design notes:
  - 15-min bar dt = 1/252 / 6.5 / 4  (4 bars/hour, 6.5 trading hours/day)
  - OHLCV built from 5-step intrabar GBM sub-simulation
  - Volume follows log-normal with intraday U-shape (open/close spikes)
  - Regime transitions modelled via Markov chain with configurable probability
  - BH_ACTIVE regime reproduces LARSA mass accumulation dynamics
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 15-min bar: 1 trading day = 252 days, 6.5 hours/day, 4 bars/hour
DT_15M: float = 1.0 / 252.0 / 6.5 / 4.0

# Annualised vol to per-bar vol scaling
BARS_PER_YEAR: float = 252.0 * 6.5 * 4.0   # ~6552 bars/year

# LARSA BH physics defaults (mirror larsa-v16 constants)
BH_FORM_DEFAULT: float = 1.5
BH_COLLAPSE_DEFAULT: float = 1.0
BH_DECAY: float = 0.95
BH_MASS_EMA_FAST: float = 0.03
BH_MASS_EMA_SLOW: float = 0.97

# Regime drift / vol multipliers (annualised)
REGIME_DRIFT = {
    "TRENDING_BULL":    0.05,
    "TRENDING_BEAR":   -0.05,
    "VOLATILE":         0.00,
    "MEAN_REVERTING":   0.00,
    "BLACK_HOLE_ACTIVE": 0.10,
}

REGIME_VOL_MULT = {
    "TRENDING_BULL":    1.0,
    "TRENDING_BEAR":    1.0,
    "VOLATILE":         3.0,
    "MEAN_REVERTING":   0.6,
    "BLACK_HOLE_ACTIVE": 2.0,
}

# Intraday volume U-shape: bar positions within a day (26 bars/day at 15m)
_BARS_PER_DAY = 26  # 6.5 hours * 4 bars/hour


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class MarketRegime(str, Enum):
    """Market regimes recognised by the simulation engine."""
    TRENDING_BULL    = "TRENDING_BULL"
    TRENDING_BEAR    = "TRENDING_BEAR"
    VOLATILE         = "VOLATILE"
    MEAN_REVERTING   = "MEAN_REVERTING"
    BLACK_HOLE_ACTIVE = "BLACK_HOLE_ACTIVE"


@dataclass
class SimConfig:
    """Configuration for a regime-switching simulation run.

    Attributes
    ----------
    n_bars:
        Total number of 15-min bars to generate.
    regime_sequence:
        List of (regime, duration_bars) tuples that define the regime schedule.
        If the total duration is less than n_bars, the last regime is extended.
        If None, a random sequence is generated using regime_transition_prob.
    initial_price:
        Starting price for the simulation.
    annual_vol:
        Annualised volatility (e.g. 0.20 for 20%).
    regime_transition_prob:
        Per-bar probability of a regime transition (used when regime_sequence
        is None to generate a random Markov chain of regimes).
    seed:
        Optional random seed for reproducibility.
    cf:
        Capture-fraction constant mirroring LARSA CF["15m"] (default 0.0003).
    cf_scale:
        Additional scale factor on cf (LARSA sets 3.0 in BULL, 1.0 otherwise).
    """
    n_bars: int = 2000
    regime_sequence: Optional[list[tuple[MarketRegime, int]]] = None
    initial_price: float = 100.0
    annual_vol: float = 0.20
    regime_transition_prob: float = 0.002
    seed: Optional[int] = None
    cf: float = 0.0003
    cf_scale: float = 1.0


# ---------------------------------------------------------------------------
# Geometric Brownian Motion
# ---------------------------------------------------------------------------

class GeometricBrownianMotion:
    """
    Standard and jump-diffusion GBM price path generators.

    All methods return arrays of *price levels* (not returns), starting
    from 1.0 (scale by initial_price at the call site).
    """

    @staticmethod
    def generate(
        n: int,
        mu: float,
        sigma: float,
        dt: float = DT_15M,
        seed: Optional[int] = None,
        initial: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate a GBM price path of length n+1 (includes t=0).

        Parameters
        ----------
        n:
            Number of steps (returns array of length n+1).
        mu:
            Annualised drift.
        sigma:
            Annualised volatility.
        dt:
            Time step in years (default = 15-min bar dt).
        seed:
            Optional random seed.
        initial:
            Starting price level.

        Returns
        -------
        np.ndarray shape (n+1,)
        """
        rng = np.random.default_rng(seed)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)
        z = rng.standard_normal(n)
        log_returns = drift + diffusion * z
        log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
        return initial * np.exp(log_path)

    @staticmethod
    def generate_with_jumps(
        n: int,
        mu: float,
        sigma: float,
        jump_intensity: float = 2.0,
        jump_size_mu: float = -0.02,
        jump_size_sigma: float = 0.04,
        dt: float = DT_15M,
        seed: Optional[int] = None,
        initial: float = 1.0,
    ) -> NDArray[np.float64]:
        """Merton jump-diffusion price path.

        Adds a compound Poisson process to standard GBM.

        Parameters
        ----------
        jump_intensity:
            Expected jumps per year (lambda in Merton model).
        jump_size_mu:
            Mean log-jump size.
        jump_size_sigma:
            Std-dev of log-jump size.
        """
        rng = np.random.default_rng(seed)
        # Jump compensation term so drift is still mu in expectation
        k = math.exp(jump_size_mu + 0.5 * jump_size_sigma ** 2) - 1.0
        drift_comp = (mu - 0.5 * sigma ** 2 - jump_intensity * k) * dt
        diffusion = sigma * math.sqrt(dt)

        z_diff = rng.standard_normal(n)
        # Number of jumps per bar ~ Poisson(lambda * dt)
        n_jumps = rng.poisson(jump_intensity * dt, size=n)
        # Jump sizes -- sum n_jumps[i] normal draws per bar
        # vectorised: draw enough, then accumulate
        max_jumps = max(n_jumps.max(), 1)
        jump_pool = rng.normal(jump_size_mu, jump_size_sigma, size=(n, max_jumps))
        # Mask jumps beyond actual count
        jump_mask = np.arange(max_jumps)[None, :] < n_jumps[:, None]
        jump_log_returns = (jump_pool * jump_mask).sum(axis=1)

        log_returns = drift_comp + diffusion * z_diff + jump_log_returns
        log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
        return initial * np.exp(log_path)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck process
# ---------------------------------------------------------------------------

class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck mean-reverting process.

    Discrete form:
        x[t] = x[t-1] + kappa * (theta - x[t-1]) * dt + sigma * sqrt(dt) * Z
    """

    @staticmethod
    def generate(
        n: int,
        kappa: float,
        theta: float,
        sigma: float,
        x0: Optional[float] = None,
        dt: float = DT_15M,
        seed: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Generate OU path of length n+1 (includes t=0).

        Parameters
        ----------
        kappa:
            Mean reversion speed (per year). Typical range: 2-50.
        theta:
            Long-run mean level.
        sigma:
            Annualised diffusion coefficient.
        x0:
            Starting value. Defaults to theta.

        Returns
        -------
        np.ndarray shape (n+1,)
        """
        rng = np.random.default_rng(seed)
        if x0 is None:
            x0 = theta
        path = np.empty(n + 1)
        path[0] = x0
        sqrt_dt = math.sqrt(dt)
        z = rng.standard_normal(n)
        for i in range(n):
            path[i + 1] = (
                path[i]
                + kappa * (theta - path[i]) * dt
                + sigma * sqrt_dt * z[i]
            )
        return path

    @staticmethod
    def fit(prices: NDArray[np.float64]) -> tuple[float, float, float]:
        """Fit OU parameters via OLS on the discretised SDE.

        Regresses:
            dx = a + b * x[t-1] + noise

        Then recovers: kappa = -b/dt, theta = -a/b, sigma from residuals.

        Parameters
        ----------
        prices:
            1-D array of price levels.

        Returns
        -------
        (kappa, theta, sigma) -- annualised parameters
        """
        dt = DT_15M
        x = prices[:-1]
        dx = np.diff(prices)
        # OLS: dx = a + b*x + eps
        X = np.column_stack([np.ones_like(x), x])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, dx, rcond=None)
        except np.linalg.LinAlgError:
            return (1.0, float(np.mean(prices)), float(np.std(prices)))
        a, b = coeffs
        if b >= 0:
            # Non-mean-reverting: return nominal estimates
            kappa = 0.1
            theta = float(np.mean(prices))
        else:
            kappa = -b / dt
            theta = -a / b if abs(b) > 1e-12 else float(np.mean(prices))
        # Sigma from residual std
        if len(residuals) > 0 and residuals[0] > 0:
            resid_std = math.sqrt(residuals[0] / len(x))
        else:
            resid_std = float(np.std(dx - (a + b * x)))
        sigma = resid_std / math.sqrt(dt)
        return (float(kappa), float(theta), float(sigma))


# ---------------------------------------------------------------------------
# OHLCV bar construction helpers
# ---------------------------------------------------------------------------

def _build_ohlcv_from_sub_path(
    sub_path: NDArray[np.float64],
    open_price: float,
    volume_base: float,
) -> dict:
    """Build a single OHLCV bar from a 6-point intrabar path (5 steps).

    sub_path: shape (6,) -- open to close prices (relative, starting at 1.0)
    """
    prices = sub_path * open_price
    o = prices[0]
    h = float(np.max(prices))
    l = float(np.min(prices))
    c = prices[-1]
    return {"open": o, "high": h, "low": l, "close": c, "volume": volume_base}


def _intraday_volume_factor(bar_idx_in_day: int, n_bars_day: int = _BARS_PER_DAY) -> float:
    """Return a volume multiplier for the U-shape intraday pattern.

    First ~2 bars and last ~2 bars get elevated volume.
    """
    x = bar_idx_in_day / max(n_bars_day - 1, 1)
    # U-shape: high at 0 and 1, low in the middle
    # Uses a simple quadratic: 4*(x - 0.5)^2 + base_level
    factor = 4.0 * (x - 0.5) ** 2 + 0.4
    return float(np.clip(factor, 0.1, 4.0))


# ---------------------------------------------------------------------------
# Regime-switching market generator
# ---------------------------------------------------------------------------

class RegimeSwitchingMarket:
    """
    Generates OHLCV bars with regime switching across all five MarketRegime
    states including BH_ACTIVE dynamics.

    Usage
    -----
    >>> cfg = SimConfig(n_bars=500, initial_price=4500.0, annual_vol=0.18)
    >>> df = RegimeSwitchingMarket.generate(cfg)
    >>> df.columns
    Index(['open', 'high', 'low', 'close', 'volume', 'regime', 'bh_mass'], dtype='object')
    """

    @staticmethod
    def generate(config: SimConfig) -> pd.DataFrame:
        """Generate OHLCV bars with regime-switching dynamics.

        Returns
        -------
        pd.DataFrame with columns:
            open, high, low, close, volume, regime, bh_mass
        Index is a RangeIndex matching bar number.
        """
        rng = np.random.default_rng(config.seed)
        n = config.n_bars
        sigma_annual = config.annual_vol
        p0 = config.initial_price

        # -- Build regime schedule --
        regime_at_bar = RegimeSwitchingMarket._build_regime_schedule(config, rng, n)

        # -- Simulate intrabar sub-paths and build bars --
        records = []
        current_price = p0

        # BH mass state (mirrors LARSA FutureInstrument.update_bh)
        bh_mass = 0.0
        ctl = 0        # consecutive timelike bars
        bh_active = False
        bh_dir = 0
        cf = config.cf * config.cf_scale
        price_window: list[float] = []  # rolling window for BH direction

        # OU state for MEAN_REVERTING regime
        ou_theta = p0
        ou_kappa = 15.0
        ou_sigma = sigma_annual * 0.6

        # BH event state for BLACK_HOLE_ACTIVE regime
        bh_event_bar = 0   # bars since BH_ACTIVE regime started

        for i in range(n):
            regime = regime_at_bar[i]
            bar_in_day = i % _BARS_PER_DAY

            # -- Regime-specific drift and vol --
            drift_ann, vol_ann = RegimeSwitchingMarket._regime_params(
                regime, sigma_annual, bh_event_bar if regime == MarketRegime.BLACK_HOLE_ACTIVE else 0
            )
            dt = DT_15M

            # -- Generate intrabar sub-path (5 steps -> 6 points) --
            sub_path = RegimeSwitchingMarket._intrabar_subpath(
                regime, drift_ann, vol_ann, current_price, ou_theta, ou_kappa, ou_sigma, rng
            )

            # -- Volume --
            vol_base_log = math.log(max(current_price * 1e-3, 1.0)) + 5.0
            vol_noise = rng.normal(0.0, 0.4)
            vol_base = math.exp(vol_base_log + vol_noise)
            # Spike volume at regime transitions
            is_transition = (i > 0 and regime_at_bar[i] != regime_at_bar[i - 1])
            if is_transition:
                vol_base *= rng.uniform(2.0, 5.0)
            intraday_mult = _intraday_volume_factor(bar_in_day)
            final_volume = vol_base * intraday_mult

            # -- Build OHLCV --
            bar = _build_ohlcv_from_sub_path(sub_path, current_price, final_volume)
            current_price = bar["close"]

            # -- Update OU theta for mean-reverting regime --
            if regime == MarketRegime.MEAN_REVERTING:
                window = 50
                if len(price_window) >= window:
                    ou_theta = float(np.mean(price_window[-window:]))
                else:
                    ou_theta = p0

            # -- BH mass update (LARSA update_bh logic) --
            if len(price_window) >= 1:
                prev_close = price_window[-1]
                beta_raw = abs(bar["close"] - prev_close) / (prev_close + 1e-9)
                beta = beta_raw / (cf + 1e-9)
                if beta < 1.0:
                    ctl += 1
                    sb = min(2.0, 1.0 + ctl * 0.1)
                    bh_mass = bh_mass * BH_MASS_EMA_SLOW + BH_MASS_EMA_FAST * 1.0 * sb
                else:
                    ctl = 0
                    bh_mass *= BH_DECAY
                was_active = bh_active
                if not was_active:
                    bh_active = bh_mass > BH_FORM_DEFAULT and ctl >= 3
                else:
                    bh_active = bh_mass > BH_COLLAPSE_DEFAULT and ctl >= 3
                if not was_active and bh_active:
                    lookback = min(20, len(price_window))
                    bh_dir = 1 if bar["close"] > price_window[-lookback] else -1
            else:
                beta = 0.0

            price_window.append(bar["close"])
            if len(price_window) > 50:
                price_window.pop(0)

            # Track bars in BH_ACTIVE regime
            if regime == MarketRegime.BLACK_HOLE_ACTIVE:
                bh_event_bar += 1
            else:
                bh_event_bar = 0

            bar["regime"] = regime.value
            bar["bh_mass"] = bh_mass
            bar["bh_active"] = bh_active
            bar["bh_dir"] = bh_dir
            records.append(bar)

        df = pd.DataFrame(records)
        df.index.name = "bar"
        return df

    @staticmethod
    def _build_regime_schedule(
        config: SimConfig,
        rng: np.random.Generator,
        n: int,
    ) -> list[MarketRegime]:
        """Return list of length n with regime at each bar."""
        if config.regime_sequence is not None:
            schedule: list[MarketRegime] = []
            for regime, dur in config.regime_sequence:
                schedule.extend([regime] * dur)
            # Extend or truncate to n
            if len(schedule) < n:
                schedule.extend([schedule[-1]] * (n - len(schedule)))
            return schedule[:n]

        # Random Markov chain
        all_regimes = list(MarketRegime)
        n_regimes = len(all_regimes)
        p_stay = 1.0 - config.regime_transition_prob
        p_switch = config.regime_transition_prob / (n_regimes - 1)
        trans_matrix = np.full((n_regimes, n_regimes), p_switch)
        np.fill_diagonal(trans_matrix, p_stay)

        schedule = []
        current_idx = rng.integers(0, n_regimes)
        for _ in range(n):
            schedule.append(all_regimes[current_idx])
            current_idx = int(rng.choice(n_regimes, p=trans_matrix[current_idx]))
        return schedule

    @staticmethod
    def _regime_params(
        regime: MarketRegime,
        sigma_annual: float,
        bh_event_bar: int = 0,
    ) -> tuple[float, float]:
        """Return (drift_annualised, vol_annualised) for a regime."""
        base_drift = REGIME_DRIFT[regime.value]
        vol_mult = REGIME_VOL_MULT[regime.value]

        if regime == MarketRegime.BLACK_HOLE_ACTIVE:
            # Drift accelerates as BH event matures then reverses
            if bh_event_bar < 20:
                # Accumulation phase: moderate positive drift
                drift = 0.08 + bh_event_bar * 0.01
            elif bh_event_bar < 35:
                # Breakout phase: explosive
                drift = 0.30
            else:
                # Reversal phase: strong mean reversion
                drift = -0.20
            return (drift, sigma_annual * vol_mult)

        return (base_drift, sigma_annual * vol_mult)

    @staticmethod
    def _intrabar_subpath(
        regime: MarketRegime,
        drift_ann: float,
        vol_ann: float,
        open_price: float,
        ou_theta: float,
        ou_kappa: float,
        ou_sigma: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Generate 6-point intrabar path (5 steps) starting from 1.0.

        Returns relative path (divide by open_price outside to scale).
        Returns array where path[0] = 1.0 (open), path[-1] = close.
        """
        n_sub = 5
        dt_sub = DT_15M / n_sub

        if regime == MarketRegime.MEAN_REVERTING:
            # OU sub-path scaled to price level
            path = np.empty(n_sub + 1)
            path[0] = open_price
            sqrt_dt_sub = math.sqrt(dt_sub)
            z = rng.standard_normal(n_sub)
            for j in range(n_sub):
                path[j + 1] = (
                    path[j]
                    + ou_kappa * (ou_theta - path[j]) * dt_sub
                    + ou_sigma * sqrt_dt_sub * z[j]
                )
            # Ensure no negative prices
            path = np.maximum(path, open_price * 0.01)
            return path / open_price

        # GBM sub-path (all other regimes)
        drift_sub = (drift_ann - 0.5 * vol_ann ** 2) * dt_sub
        diff_sub = vol_ann * math.sqrt(dt_sub)
        z = rng.standard_normal(n_sub)

        if regime == MarketRegime.VOLATILE:
            # Add small jump probability within bar
            jump_prob = 0.15
            for idx in range(n_sub):
                if rng.random() < jump_prob:
                    z[idx] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)

        log_returns = drift_sub + diff_sub * z
        log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
        return np.exp(log_path)


# ---------------------------------------------------------------------------
# Correlated multi-asset simulator
# ---------------------------------------------------------------------------

class CorrelatedAssetSimulator:
    """
    Generates correlated multi-asset price paths via Cholesky decomposition.

    Supports:
      - Regime-dependent correlation (contagion in VOLATILE)
      - BTC dominance effect: when BTC is in BH_ACTIVE, other crypto follows
        with a configurable lag in bars.

    Usage
    -----
    >>> sim = CorrelatedAssetSimulator(asset_names=["BTC", "ETH", "SOL"])
    >>> paths = sim.generate(n=1000, corr_matrix=corr, mus=mus, sigmas=sigmas)
    >>> paths["BTC"].shape
    (1001,)
    """

    def __init__(
        self,
        asset_names: Optional[list[str]] = None,
        btc_dominance_lag: int = 2,
        contagion_boost: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        asset_names:
            Names for each asset; first asset treated as BTC for dominance effect.
        btc_dominance_lag:
            Bars of lag before BTC BH_ACTIVE event propagates to other assets.
        contagion_boost:
            Added to all pairwise correlations when VOLATILE regime detected.
        seed:
            Optional random seed.
        """
        self.asset_names = asset_names or []
        self.btc_dominance_lag = btc_dominance_lag
        self.contagion_boost = contagion_boost
        self.seed = seed

    def generate(
        self,
        n: int,
        corr_matrix: NDArray[np.float64],
        mus: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        regime: Optional[MarketRegime] = None,
        btc_bh_active_bars: Optional[list[int]] = None,
        initial_prices: Optional[NDArray[np.float64]] = None,
        dt: float = DT_15M,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate correlated price paths.

        Parameters
        ----------
        n:
            Number of steps.
        corr_matrix:
            k x k correlation matrix (k = number of assets).
        mus:
            Annualised drift vector, shape (k,).
        sigmas:
            Annualised vol vector, shape (k,).
        regime:
            Current regime -- used to apply contagion boost in VOLATILE.
        btc_bh_active_bars:
            List of bar indices where BTC is in BH_ACTIVE; triggers lagged
            contagion for other crypto assets.
        initial_prices:
            Starting prices, shape (k,). Defaults to all 1.0.

        Returns
        -------
        dict mapping asset name -> np.ndarray of shape (n+1,)
        """
        rng = np.random.default_rng(self.seed)
        k = len(corr_matrix)
        if initial_prices is None:
            initial_prices = np.ones(k)

        # Apply contagion boost in VOLATILE or HIGH_VOL regimes
        effective_corr = corr_matrix.copy()
        if regime in (MarketRegime.VOLATILE, MarketRegime.BLACK_HOLE_ACTIVE):
            boost = np.full_like(effective_corr, self.contagion_boost)
            np.fill_diagonal(boost, 0.0)
            effective_corr = np.clip(effective_corr + boost, -0.99, 0.99)

        # Enforce positive-semidefinite
        effective_corr = self._nearest_psd(effective_corr)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(effective_corr)
        except np.linalg.LinAlgError:
            # Fallback: diagonal (uncorrelated)
            logger.warning("Cholesky decomposition failed -- using diagonal correlation.")
            L = np.diag(np.ones(k))

        # Covariance-scaled Cholesky (sigma_i * L_ij)
        cov_scale = sigmas[:, None] * L  # shape (k, k) broadcast

        # GBM log-return simulation
        drifts = (mus - 0.5 * sigmas ** 2) * dt
        z_raw = rng.standard_normal((n, k))   # (n, k) iid normals
        z_corr = (L @ z_raw.T).T              # (n, k) correlated normals

        log_returns = drifts[None, :] + sigmas[None, :] * math.sqrt(dt) * z_corr  # (n, k)

        # BTC dominance contagion: inject correlated shock with lag
        btc_set: set[int] = set(btc_bh_active_bars) if btc_bh_active_bars else set()
        if btc_set and k > 1:
            lag = self.btc_dominance_lag
            shock_intensity = 0.015  # per-bar extra return for lagged assets
            for bar_idx in btc_set:
                target_bar = bar_idx + lag
                if 0 <= target_bar < n:
                    # Other assets (idx 1..k-1) get a shock proportional to BTC move
                    btc_move = log_returns[bar_idx, 0]
                    for asset_idx in range(1, k):
                        log_returns[target_bar, asset_idx] += btc_move * 0.5

        # Build cumulative price paths
        log_paths = np.vstack([np.zeros((1, k)), np.cumsum(log_returns, axis=0)])  # (n+1, k)
        prices = initial_prices[None, :] * np.exp(log_paths)  # (n+1, k)

        names = self.asset_names if len(self.asset_names) == k else [f"asset_{i}" for i in range(k)]
        return {name: prices[:, i] for i, name in enumerate(names)}

    @staticmethod
    def _nearest_psd(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project a symmetric matrix to the nearest positive semidefinite matrix.

        Uses eigenvalue clipping (Higham 2002 simplified version).
        """
        # Symmetrize
        M = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-8)
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-scale to unit diagonal (correlation matrix)
        d = np.sqrt(np.diag(psd))
        d = np.where(d < 1e-12, 1.0, d)
        psd = psd / np.outer(d, d)
        np.fill_diagonal(psd, 1.0)
        return psd


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_trending_bull_config(
    n_bars: int = 1000,
    initial_price: float = 100.0,
    annual_vol: float = 0.20,
    seed: Optional[int] = 42,
) -> SimConfig:
    """Pre-built config: pure trending bull market."""
    return SimConfig(
        n_bars=n_bars,
        regime_sequence=[(MarketRegime.TRENDING_BULL, n_bars)],
        initial_price=initial_price,
        annual_vol=annual_vol,
        seed=seed,
    )


def make_mixed_regime_config(
    n_bars: int = 2000,
    initial_price: float = 100.0,
    annual_vol: float = 0.20,
    seed: Optional[int] = 42,
) -> SimConfig:
    """Pre-built config: alternating regimes with BH episode."""
    return SimConfig(
        n_bars=n_bars,
        regime_sequence=[
            (MarketRegime.TRENDING_BULL, 400),
            (MarketRegime.MEAN_REVERTING, 300),
            (MarketRegime.VOLATILE, 200),
            (MarketRegime.BLACK_HOLE_ACTIVE, 150),
            (MarketRegime.TRENDING_BEAR, 300),
            (MarketRegime.MEAN_REVERTING, 650),
        ],
        initial_price=initial_price,
        annual_vol=annual_vol,
        seed=seed,
    )


def simulate_default_crypto(
    symbol: str = "BTC",
    n_bars: int = 5000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Quick simulation of a crypto-like asset (high vol, BH events).

    Returns OHLCV DataFrame ready for LARSA backtesting.
    """
    vol_map = {"BTC": 0.75, "ETH": 0.90, "SOL": 1.20}
    vol = vol_map.get(symbol, 0.80)
    price_map = {"BTC": 40_000.0, "ETH": 2_500.0, "SOL": 80.0}
    p0 = price_map.get(symbol, 100.0)
    cfg = SimConfig(n_bars=n_bars, initial_price=p0, annual_vol=vol, seed=seed)
    return RegimeSwitchingMarket.generate(cfg)


# ---------------------------------------------------------------------------
# Module-level validation helpers (for tests / quick sanity checks)
# ---------------------------------------------------------------------------

def _check_ohlcv_invariants(df: pd.DataFrame) -> list[str]:
    """Return list of violated OHLCV invariants (empty list = all ok)."""
    violations = []
    if (df["high"] < df["open"]).any():
        violations.append("high < open on some bars")
    if (df["high"] < df["close"]).any():
        violations.append("high < close on some bars")
    if (df["low"] > df["open"]).any():
        violations.append("low > open on some bars")
    if (df["low"] > df["close"]).any():
        violations.append("low > close on some bars")
    if (df["volume"] <= 0).any():
        violations.append("non-positive volume on some bars")
    if (df["close"] <= 0).any():
        violations.append("non-positive close price on some bars")
    return violations
