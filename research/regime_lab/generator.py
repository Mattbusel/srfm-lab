"""
research/regime_lab/generator.py
=================================
Synthetic regime-aware price-series generators.

Classes
-------
MarkovRegimeGenerator       — Discrete-time Markov-chain + Gaussian returns
GARCHRegimeGenerator        — GARCH(1,1) with regime-switching variance
HestonRegimeGenerator       — Heston stochastic-vol with regime-dependent mean-reversion
JumpDiffusionGenerator      — Merton jump-diffusion dS = μS dt + σS dW + J dN
BootstrapScenarioGenerator  — Stationary block bootstrap from historical returns

Functions
---------
calibrate_to_history(historical_prices, model='markov') -> fitted generator
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime constants (mirrored from __init__ to avoid circular imports)
# ---------------------------------------------------------------------------
BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)

# ---------------------------------------------------------------------------
# Default regime parameters (annualised μ, daily σ)
# ---------------------------------------------------------------------------
DEFAULT_REGIME_PARAMS: Dict[str, Tuple[float, float]] = {
    BULL:     ( 0.25 / 252,  0.008),   # +25% annual, low daily vol
    BEAR:     (-0.20 / 252,  0.015),   # -20% annual, moderate vol
    SIDEWAYS: ( 0.03 / 252,  0.006),   # +3% annual, very low vol
    HIGH_VOL: ( 0.00 / 252,  0.030),   # flat return, high vol
}

DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array([
    #  BULL    BEAR   SIDE   HV
    [0.97,   0.01,  0.015,  0.005],   # from BULL
    [0.01,   0.96,  0.020,  0.010],   # from BEAR
    [0.02,   0.02,  0.950,  0.010],   # from SIDEWAYS
    [0.05,   0.05,  0.050,  0.850],   # from HIGH_VOL
], dtype=float)


# ===========================================================================
# 1. MarkovRegimeGenerator
# ===========================================================================

@dataclass
class MarkovGeneratorResult:
    prices:  np.ndarray   # (n_bars + 1,) including S0
    returns: np.ndarray   # (n_bars,)
    regimes: np.ndarray   # (n_bars,) string labels


class MarkovRegimeGenerator:
    """
    Discrete-time Markov-chain regime generator.

    At each bar the regime is drawn from the Markov chain and the return
    is drawn from N(mu_k, sigma_k**2) specific to that regime.

    Parameters
    ----------
    transition_matrix : (K, K) row-stochastic ndarray
        P[i, j] = probability of transitioning from regime i to regime j.
    regime_params : dict  regime → (mu, sigma) in per-bar units
        mu    : mean log-return per bar
        sigma : std-dev of log-return per bar
    initial_price : float (default 100.0)
    """

    def __init__(self,
                 transition_matrix: Optional[np.ndarray] = None,
                 regime_params: Optional[Dict[str, Tuple[float, float]]] = None,
                 initial_price: float = 100.0):
        self.transition_matrix = (DEFAULT_TRANSITION_MATRIX.copy()
                                  if transition_matrix is None else
                                  np.asarray(transition_matrix, dtype=float))
        self.regime_params  = regime_params or DEFAULT_REGIME_PARAMS.copy()
        self.initial_price  = initial_price

        # Validate
        K = len(REGIMES)
        if self.transition_matrix.shape != (K, K):
            # Attempt to use as-is if square
            if self.transition_matrix.ndim == 2 and self.transition_matrix.shape[0] == self.transition_matrix.shape[1]:
                K = self.transition_matrix.shape[0]
            else:
                raise ValueError(f"transition_matrix must be ({K},{K})")
        # Row-normalise
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self.transition_matrix / row_sums

        self._regime_list = list(self.regime_params.keys())
        self._K           = len(self._regime_list)
        self._regime_idx  = {r: i for i, r in enumerate(self._regime_list)}

    # ------------------------------------------------------------------ #

    def _stationary(self) -> np.ndarray:
        """Compute stationary distribution of the Markov chain."""
        A = self.transition_matrix
        eigvals, eigvecs = np.linalg.eig(A.T)
        # Eigenvalue closest to 1
        idx = np.argmin(np.abs(eigvals - 1.0))
        v   = eigvecs[:, idx].real
        v   = np.abs(v) / np.abs(v).sum()
        return v

    def generate(self, n_bars: int, seed: Optional[int] = None,
                 initial_regime: Optional[str] = None) -> MarkovGeneratorResult:
        """
        Generate a synthetic price path.

        Parameters
        ----------
        n_bars         : number of price bars to generate
        seed           : optional RNG seed
        initial_regime : starting regime; if None, drawn from stationary dist

        Returns
        -------
        MarkovGeneratorResult
        """
        rng = np.random.default_rng(seed)

        # Starting regime
        if initial_regime is not None:
            cur = self._regime_idx.get(initial_regime, 0)
        else:
            pi  = self._stationary()
            cur = int(rng.choice(self._K, p=pi))

        regimes = np.empty(n_bars, dtype=object)
        log_rets = np.zeros(n_bars)

        for t in range(n_bars):
            regime_name = self._regime_list[cur]
            regimes[t]  = regime_name
            mu, sigma   = self.regime_params[regime_name]
            log_rets[t] = rng.normal(mu, sigma)
            # Transition
            row = self.transition_matrix[cur]
            cur = int(rng.choice(self._K, p=row))

        prices = np.empty(n_bars + 1)
        prices[0] = self.initial_price
        for t in range(n_bars):
            prices[t + 1] = prices[t] * np.exp(log_rets[t])

        return MarkovGeneratorResult(
            prices=prices,
            returns=log_rets,
            regimes=regimes,
        )

    def generate_ohlc(self, n_bars: int, seed: Optional[int] = None,
                      intrabar_noise: float = 0.5) -> pd.DataFrame:
        """
        Generate a synthetic OHLC DataFrame with regime column.

        The high/low are synthesised via a random intrabar range proportional
        to the regime's sigma.
        """
        result = self.generate(n_bars, seed=seed)
        rng    = np.random.default_rng(seed)
        rows   = []
        for t in range(n_bars):
            close_prev = result.prices[t]
            close_curr = result.prices[t + 1]
            regime     = result.regimes[t]
            _, sigma   = self.regime_params[regime]
            intrabar   = abs(rng.normal(0, sigma * intrabar_noise))
            o = close_prev * np.exp(rng.normal(0, sigma * 0.1))
            h = max(o, close_curr) * np.exp(abs(rng.normal(0, intrabar)))
            l = min(o, close_curr) * np.exp(-abs(rng.normal(0, intrabar)))
            rows.append({
                "open":   o,
                "high":   h,
                "low":    l,
                "close":  close_curr,
                "regime": regime,
                "log_ret": result.returns[t],
            })
        df = pd.DataFrame(rows)
        df.index = pd.RangeIndex(n_bars)
        return df


# ===========================================================================
# 2. GARCHRegimeGenerator
# ===========================================================================

@dataclass
class GARCHGeneratorResult:
    returns:     np.ndarray   # (n_bars,) log-returns
    regimes:     np.ndarray   # (n_bars,) string labels
    volatilities: np.ndarray  # (n_bars,) conditional sigma (not variance)


class GARCHRegimeGenerator:
    """
    GARCH(1,1) with regime-switching variance.

    The conditional variance evolves as:
        h_t = omega_k + alpha_k * eps_{t-1}^2 + beta_k * h_{t-1}

    where k is the current regime and eps_{t-1} is the previous standardised
    residual scaled by sqrt(h_{t-1}).

    Parameters
    ----------
    regimes    : list of regime names (default all four)
    omega_dict : dict regime → omega (long-run variance component)
    alpha_dict : dict regime → alpha (ARCH coefficient)
    beta_dict  : dict regime → beta  (GARCH coefficient)
    transition_matrix : (K, K) row-stochastic Markov matrix for regime switching
    mu_dict    : dict regime → drift per bar (default 0)
    """

    # Sensible defaults for each regime
    _DEFAULTS: Dict[str, Dict[str, float]] = {
        BULL:     {"omega": 2e-6,  "alpha": 0.05, "beta": 0.93, "mu":  0.0003},
        BEAR:     {"omega": 1e-5,  "alpha": 0.10, "beta": 0.88, "mu": -0.0002},
        SIDEWAYS: {"omega": 1e-6,  "alpha": 0.03, "beta": 0.96, "mu":  0.0001},
        HIGH_VOL: {"omega": 5e-5,  "alpha": 0.15, "beta": 0.80, "mu":  0.0000},
    }

    def __init__(self,
                 regimes: Optional[List[str]] = None,
                 omega_dict: Optional[Dict[str, float]] = None,
                 alpha_dict: Optional[Dict[str, float]] = None,
                 beta_dict:  Optional[Dict[str, float]] = None,
                 mu_dict:    Optional[Dict[str, float]] = None,
                 transition_matrix: Optional[np.ndarray] = None):
        self._regimes = regimes or list(REGIMES)
        self._K       = len(self._regimes)
        self._ridx    = {r: i for i, r in enumerate(self._regimes)}

        self.omega_dict = omega_dict or {r: self._DEFAULTS[r]["omega"] for r in self._regimes}
        self.alpha_dict = alpha_dict or {r: self._DEFAULTS[r]["alpha"] for r in self._regimes}
        self.beta_dict  = beta_dict  or {r: self._DEFAULTS[r]["beta"]  for r in self._regimes}
        self.mu_dict    = mu_dict    or {r: self._DEFAULTS[r]["mu"]     for r in self._regimes}

        if transition_matrix is None:
            self.transition_matrix = DEFAULT_TRANSITION_MATRIX.copy()
        else:
            tm = np.asarray(transition_matrix, dtype=float)
            tm = tm / tm.sum(axis=1, keepdims=True)
            self.transition_matrix = tm

    # ------------------------------------------------------------------ #

    def generate(self, n_bars: int, seed: Optional[int] = None,
                 initial_regime: Optional[str] = None,
                 h0: Optional[float] = None) -> GARCHGeneratorResult:
        """
        Generate GARCH(1,1) return series with Markov regime switching.

        Parameters
        ----------
        n_bars         : number of bars
        seed           : RNG seed
        initial_regime : starting regime label
        h0             : starting conditional variance (default: omega / (1 - alpha - beta))

        Returns
        -------
        GARCHGeneratorResult
        """
        rng = np.random.default_rng(seed)

        # Starting regime
        cur: int
        if initial_regime is not None:
            cur = self._ridx.get(initial_regime, 0)
        else:
            cur = int(rng.integers(0, self._K))

        # Starting variance
        r0 = self._regimes[cur]
        if h0 is None:
            denom = max(1 - self.alpha_dict[r0] - self.beta_dict[r0], 1e-6)
            h0    = self.omega_dict[r0] / denom
        h_prev = h0

        returns      = np.zeros(n_bars)
        regimes      = np.empty(n_bars, dtype=object)
        volatilities = np.zeros(n_bars)

        eps_prev = 0.0  # previous residual

        for t in range(n_bars):
            regime_name = self._regimes[cur]
            regimes[t]  = regime_name

            omega = self.omega_dict[regime_name]
            alpha = self.alpha_dict[regime_name]
            beta  = self.beta_dict[regime_name]
            mu    = self.mu_dict[regime_name]

            # Update conditional variance
            h_t = omega + alpha * eps_prev ** 2 + beta * h_prev
            h_t = max(h_t, 1e-10)
            volatilities[t] = np.sqrt(h_t)

            # Draw innovation
            z          = float(rng.standard_normal())
            eps_t      = z * np.sqrt(h_t)
            returns[t] = mu + eps_t

            eps_prev = eps_t
            h_prev   = h_t

            # Regime transition
            row = self.transition_matrix[cur]
            cur = int(rng.choice(self._K, p=row))

        return GARCHGeneratorResult(
            returns=returns,
            regimes=regimes,
            volatilities=volatilities,
        )

    def generate_prices(self, n_bars: int, s0: float = 100.0,
                        seed: Optional[int] = None) -> pd.DataFrame:
        """Convenience: generate prices from GARCH returns."""
        result = self.generate(n_bars, seed=seed)
        prices = s0 * np.exp(np.cumsum(result.returns))
        prices = np.concatenate([[s0], prices])
        return pd.DataFrame({
            "price":      prices[1:],
            "log_ret":    result.returns,
            "sigma":      result.volatilities,
            "regime":     result.regimes,
        })


# ===========================================================================
# 3. HestonRegimeGenerator
# ===========================================================================

@dataclass
class HestonGeneratorResult:
    prices:   np.ndarray   # (n_bars + 1,)
    variances: np.ndarray  # (n_bars + 1,) instantaneous variance V_t
    regimes:  np.ndarray   # (n_bars,)


class HestonRegimeGenerator:
    """
    Heston stochastic-volatility model with regime-dependent mean-reversion.

    Model:
        dS = mu * S dt + sqrt(V) * S dW1
        dV = kappa_k * (theta_k - V) dt + xi * sqrt(V) dW2
        corr(dW1, dW2) = rho

    where k is the current Markov regime.

    Parameters
    ----------
    regimes : list of regime names
    v0      : initial variance (default 0.04 ≈ 20% annual vol)
    kappa   : base mean-reversion speed
    theta   : base long-run variance
    xi      : vol-of-vol
    rho     : correlation between asset and variance Brownians
    kappa_multipliers : dict regime → multiplier for kappa
    theta_multipliers : dict regime → multiplier for theta
    mu_dict           : dict regime → drift per unit time
    """

    _KAPPA_MULT: Dict[str, float] = {
        BULL:     1.5,    # faster mean reversion — low vol persists
        BEAR:     0.5,    # slow — high vol lingers
        SIDEWAYS: 2.0,    # very fast — variance snaps to theta
        HIGH_VOL: 0.3,    # sluggish — vol stays elevated
    }
    _THETA_MULT: Dict[str, float] = {
        BULL:     0.5,    # long-run vol below average
        BEAR:     2.0,    # long-run vol above average
        SIDEWAYS: 0.8,
        HIGH_VOL: 4.0,
    }
    _MU: Dict[str, float] = {
        BULL:      0.25 / 252,
        BEAR:     -0.20 / 252,
        SIDEWAYS:  0.03 / 252,
        HIGH_VOL:  0.00 / 252,
    }

    def __init__(self,
                 regimes: Optional[List[str]] = None,
                 v0:  float = 0.04,
                 kappa: float = 2.0,
                 theta: float = 0.04,
                 xi:    float = 0.5,
                 rho:   float = -0.7,
                 kappa_multipliers: Optional[Dict[str, float]] = None,
                 theta_multipliers: Optional[Dict[str, float]] = None,
                 mu_dict:           Optional[Dict[str, float]] = None,
                 transition_matrix: Optional[np.ndarray] = None):
        self._regimes = regimes or list(REGIMES)
        self._K       = len(self._regimes)
        self._ridx    = {r: i for i, r in enumerate(self._regimes)}

        self.v0    = v0
        self.kappa = kappa
        self.theta = theta
        self.xi    = xi
        self.rho   = rho

        self.kappa_mult = kappa_multipliers or self._KAPPA_MULT.copy()
        self.theta_mult = theta_multipliers or self._THETA_MULT.copy()
        self.mu_dict    = mu_dict           or self._MU.copy()

        if transition_matrix is None:
            self.transition_matrix = DEFAULT_TRANSITION_MATRIX.copy()
        else:
            tm = np.asarray(transition_matrix, dtype=float)
            self.transition_matrix = tm / tm.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #

    def generate(self, n_bars: int, dt: float = 1 / 252,
                 seed: Optional[int] = None,
                 initial_regime: Optional[str] = None) -> HestonGeneratorResult:
        """
        Simulate Heston paths with regime-switching using Euler-Maruyama.

        Parameters
        ----------
        n_bars         : number of simulation steps
        dt             : time step in years (default 1/252 = 1 trading day)
        seed           : RNG seed
        initial_regime : starting regime label

        Returns
        -------
        HestonGeneratorResult
        """
        rng = np.random.default_rng(seed)

        # Cholesky decomposition for correlated Brownians
        cov = np.array([[1.0, self.rho],
                        [self.rho, 1.0]])
        L   = np.linalg.cholesky(cov)

        # Starting regime
        if initial_regime is not None:
            cur = self._ridx.get(initial_regime, 0)
        else:
            cur = int(rng.integers(0, self._K))

        prices    = np.zeros(n_bars + 1)
        variances = np.zeros(n_bars + 1)
        regimes   = np.empty(n_bars, dtype=object)

        prices[0]    = 100.0
        variances[0] = self.v0
        sqrt_dt      = np.sqrt(dt)

        for t in range(n_bars):
            regime_name = self._regimes[cur]
            regimes[t]  = regime_name

            V_t  = max(variances[t], 1e-8)
            kappa_eff = self.kappa * self.kappa_mult.get(regime_name, 1.0)
            theta_eff = self.theta * self.theta_mult.get(regime_name, 1.0)
            mu_eff    = self.mu_dict.get(regime_name, 0.0)

            # Correlated Brownian increments
            z   = rng.standard_normal(2)
            w   = L @ z
            dW1 = w[0] * sqrt_dt
            dW2 = w[1] * sqrt_dt

            # Asset price (log-Euler)
            log_ret = (mu_eff - 0.5 * V_t) * dt + np.sqrt(V_t) * dW1
            prices[t + 1] = prices[t] * np.exp(log_ret)

            # Variance (full-truncation Euler)
            dV = kappa_eff * (theta_eff - V_t) * dt + self.xi * np.sqrt(V_t) * dW2
            variances[t + 1] = max(V_t + dV, 0.0)

            # Regime transition
            row = self.transition_matrix[cur]
            cur = int(rng.choice(self._K, p=row))

        return HestonGeneratorResult(
            prices=prices,
            variances=variances,
            regimes=regimes,
        )

    def generate_dataframe(self, n_bars: int, dt: float = 1 / 252,
                           seed: Optional[int] = None) -> pd.DataFrame:
        result = self.generate(n_bars, dt=dt, seed=seed)
        return pd.DataFrame({
            "price":    result.prices[1:],
            "variance": result.variances[1:],
            "vol":      np.sqrt(result.variances[1:]),
            "regime":   result.regimes,
        })


# ===========================================================================
# 4. JumpDiffusionGenerator — Merton (1976)
# ===========================================================================

@dataclass
class JumpDiffusionResult:
    prices:      np.ndarray   # (n_bars + 1,)
    returns:     np.ndarray   # (n_bars,) total log-return per bar
    jump_times:  np.ndarray   # boolean (n_bars,) — True when jump occurred
    jump_sizes:  np.ndarray   # log-jump size per bar (0 if no jump)
    diffusion:   np.ndarray   # diffusive component of return


class JumpDiffusionGenerator:
    """
    Merton (1976) jump-diffusion model.

    Dynamics:
        dS = (μ - λ*κ) S dt + σ S dW + J S dN

    where:
        N ~ Poisson(λ * dt) (number of jumps per bar)
        J ~ LogNormal(jump_mean, jump_std)  (jump size, already log-scale)
        κ  = E[J - 1]  (expected jump size correction)

    Parameters
    ----------
    mu             : annual drift
    sigma          : annual diffusion volatility
    jump_intensity : λ — expected number of jumps per year
    jump_mean      : mean log-jump size (negative → crash bias)
    jump_std       : std-dev of log-jump size
    dt             : bar length in years (default 1/252)
    """

    def __init__(self,
                 mu: float = 0.08,
                 sigma: float = 0.20,
                 jump_intensity: float = 5.0,
                 jump_mean: float = -0.05,
                 jump_std:  float = 0.08,
                 dt: float = 1 / 252,
                 initial_price: float = 100.0):
        self.mu             = mu
        self.sigma          = sigma
        self.jump_intensity = jump_intensity   # annual
        self.jump_mean      = jump_mean
        self.jump_std       = jump_std
        self.dt             = dt
        self.initial_price  = initial_price

        # Drift correction: κ = E[e^J - 1] = exp(jump_mean + 0.5*jump_std^2) - 1
        self._kappa = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1

    # ------------------------------------------------------------------ #

    def generate(self, n_bars: int, seed: Optional[int] = None
                 ) -> JumpDiffusionResult:
        """
        Simulate a Merton jump-diffusion path.

        Parameters
        ----------
        n_bars : number of bars
        seed   : RNG seed

        Returns
        -------
        JumpDiffusionResult
        """
        rng     = np.random.default_rng(seed)
        dt      = self.dt
        lam_dt  = self.jump_intensity * dt    # expected jumps per bar
        sqrt_dt = np.sqrt(dt)

        # Drift adjustment
        drift = (self.mu - 0.5 * self.sigma ** 2 - self.jump_intensity * self._kappa) * dt

        diffusion   = rng.standard_normal(n_bars) * self.sigma * sqrt_dt + drift
        n_jumps     = rng.poisson(lam_dt, size=n_bars)
        jump_times  = n_jumps > 0

        jump_sizes  = np.zeros(n_bars)
        for t in np.where(jump_times)[0]:
            # Sum of n_jumps[t] lognormal jumps
            nj = n_jumps[t]
            j_samples = rng.normal(self.jump_mean, self.jump_std, size=nj)
            jump_sizes[t] = j_samples.sum()

        log_rets = diffusion + jump_sizes
        prices   = np.empty(n_bars + 1)
        prices[0] = self.initial_price
        for t in range(n_bars):
            prices[t + 1] = prices[t] * np.exp(log_rets[t])

        return JumpDiffusionResult(
            prices=prices,
            returns=log_rets,
            jump_times=jump_times,
            jump_sizes=jump_sizes,
            diffusion=diffusion,
        )

    def generate_scenarios(self, n_bars: int, n_scenarios: int = 1000,
                            seed: Optional[int] = None) -> np.ndarray:
        """
        Generate multiple independent price paths.

        Returns
        -------
        np.ndarray of shape (n_scenarios, n_bars + 1)
        """
        paths = np.zeros((n_scenarios, n_bars + 1))
        rng   = np.random.default_rng(seed)
        for i in range(n_scenarios):
            result    = self.generate(n_bars, seed=rng.integers(0, 2**31))
            paths[i]  = result.prices
        return paths

    def tail_probability(self, threshold: float, horizon: int) -> float:
        """
        Estimate P(S_T / S_0 < threshold) analytically (CLT approximation).

        Parameters
        ----------
        threshold : price ratio (e.g. 0.80 for -20% loss)
        horizon   : number of bars

        Returns
        -------
        float — approximate probability
        """
        from scipy import stats as scipy_stats  # type: ignore
        dt     = self.dt
        lam_dt = self.jump_intensity * dt
        mu_total = ((self.mu - 0.5 * self.sigma ** 2
                     - self.jump_intensity * self._kappa) * dt * horizon
                    + lam_dt * horizon * self.jump_mean)
        var_total = ((self.sigma ** 2 * dt
                      + lam_dt * (self.jump_std ** 2 + self.jump_mean ** 2)) * horizon)
        log_threshold = np.log(threshold)
        z = (log_threshold - mu_total) / np.sqrt(var_total)
        return float(scipy_stats.norm.cdf(z))


# ===========================================================================
# 5. BootstrapScenarioGenerator — Stationary Block Bootstrap
# ===========================================================================

@dataclass
class BootstrapResult:
    scenarios:        np.ndarray   # (n_scenarios, n_bars)
    block_boundaries: Optional[List[List[int]]] = None  # block start indices per scenario


class BootstrapScenarioGenerator:
    """
    Stationary block bootstrap from historical returns.

    Block lengths are drawn from a Geometric distribution (mean = block_size)
    to ensure stationarity (Politis & Romano 1994).

    Parameters
    ----------
    historical_returns : 1-D array of observed log-returns
    block_size         : mean block length (default 20)
    preserve_vol_scaling : bool — scale blocks to match target vol (default False)
    """

    def __init__(self,
                 historical_returns: np.ndarray | pd.Series,
                 block_size: int = 20,
                 preserve_vol_scaling: bool = False):
        self.historical_returns  = np.asarray(historical_returns, dtype=float)
        self.block_size          = block_size
        self.preserve_vol_scaling = preserve_vol_scaling
        self._n_hist             = len(self.historical_returns)

        if self._n_hist < block_size:
            warnings.warn(
                f"Historical sample ({self._n_hist}) < block_size ({block_size}). "
                "Using block_size=max(5, n_hist//4).", stacklevel=2
            )
            self.block_size = max(5, self._n_hist // 4)

    # ------------------------------------------------------------------ #

    def generate(self, n_bars: int, n_scenarios: int = 1000,
                 seed: Optional[int] = None) -> BootstrapResult:
        """
        Generate bootstrap scenarios.

        Parameters
        ----------
        n_bars      : length of each synthetic path
        n_scenarios : number of independent bootstrap samples
        seed        : RNG seed

        Returns
        -------
        BootstrapResult with scenarios of shape (n_scenarios, n_bars)
        """
        rng           = np.random.default_rng(seed)
        scenarios     = np.zeros((n_scenarios, n_bars))
        all_blocks: List[List[int]] = []

        # Geometric block length probability
        p_geom = 1.0 / self.block_size

        for i in range(n_scenarios):
            path        = np.zeros(n_bars)
            t           = 0
            block_starts: List[int] = []

            while t < n_bars:
                # Block length ~ Geometric(p)
                block_len = int(rng.geometric(p_geom))
                block_len = min(block_len, n_bars - t)

                # Random starting position in history
                start = int(rng.integers(0, self._n_hist - 1))
                end   = min(start + block_len, self._n_hist)
                block = self.historical_returns[start:end]

                actual_len = len(block)
                path[t : t + actual_len] = block
                block_starts.append(start)
                t += actual_len

            if self.preserve_vol_scaling:
                target_vol = float(np.std(self.historical_returns, ddof=1))
                path_vol   = float(np.std(path, ddof=1)) or 1e-8
                path       = path * (target_vol / path_vol)

            scenarios[i] = path
            all_blocks.append(block_starts)

        return BootstrapResult(scenarios=scenarios, block_boundaries=all_blocks)

    def generate_prices(self, n_bars: int, n_scenarios: int = 1000,
                        s0: float = 100.0,
                        seed: Optional[int] = None) -> np.ndarray:
        """
        Return price paths of shape (n_scenarios, n_bars + 1).

        First column is always s0.
        """
        boot  = self.generate(n_bars, n_scenarios=n_scenarios, seed=seed)
        cumret = np.cumsum(boot.scenarios, axis=1)
        prices = s0 * np.exp(np.hstack([np.zeros((n_scenarios, 1)), cumret]))
        return prices

    def empirical_var(self, confidence: float = 0.95,
                      n_bars: int = 21,
                      n_scenarios: int = 10_000,
                      seed: Optional[int] = None) -> float:
        """
        Compute empirical VaR at *confidence* level over *n_bars* horizon.

        Returns
        -------
        float — positive number representing loss (e.g. 0.05 = 5% loss)
        """
        prices = self.generate_prices(n_bars, n_scenarios=n_scenarios, seed=seed)
        total_rets = (prices[:, -1] - prices[:, 0]) / prices[:, 0]
        return float(-np.percentile(total_rets, (1 - confidence) * 100))

    def empirical_cvar(self, confidence: float = 0.95,
                       n_bars: int = 21,
                       n_scenarios: int = 10_000,
                       seed: Optional[int] = None) -> float:
        """
        Compute empirical CVaR (Expected Shortfall) at *confidence* level.

        Returns
        -------
        float — positive number
        """
        prices = self.generate_prices(n_bars, n_scenarios=n_scenarios, seed=seed)
        total_rets = (prices[:, -1] - prices[:, 0]) / prices[:, 0]
        var_thresh = np.percentile(total_rets, (1 - confidence) * 100)
        tail_rets  = total_rets[total_rets <= var_thresh]
        if len(tail_rets) == 0:
            return 0.0
        return float(-np.mean(tail_rets))


# ===========================================================================
# 6. calibrate_to_history
# ===========================================================================

def calibrate_to_history(historical_prices: np.ndarray | pd.Series,
                         model: str = "markov",
                         regime_labels: Optional[np.ndarray] = None,
                         **kwargs: Any) -> Any:
    """
    Calibrate a synthetic generator to historical price data.

    Parameters
    ----------
    historical_prices : 1-D array of prices
    model             : 'markov', 'garch', 'heston', 'jump', 'bootstrap'
    regime_labels     : optional 1-D array of regime strings aligned to prices
                        (used for Markov/GARCH calibration)
    **kwargs          : passed to the generator constructor

    Returns
    -------
    Fitted generator instance.
    """
    prices = np.asarray(historical_prices, dtype=float)
    log_rets = np.diff(np.log(np.where(prices > 0, prices, 1e-10)))

    if model == "bootstrap":
        return BootstrapScenarioGenerator(log_rets, **kwargs)

    if model == "jump":
        mu        = float(np.mean(log_rets) * 252)
        sigma     = float(np.std(log_rets, ddof=1) * np.sqrt(252))
        # Heuristic jump detection: returns > 3 sigma
        z_scores  = np.abs(log_rets - np.mean(log_rets)) / (np.std(log_rets) + 1e-10)
        jumps     = log_rets[z_scores > 3]
        intensity = float(len(jumps) / (len(log_rets) / 252)) if len(jumps) > 0 else 5.0
        j_mean    = float(np.mean(jumps)) if len(jumps) > 0 else -0.03
        j_std     = float(np.std(jumps, ddof=1)) if len(jumps) > 1 else 0.05
        return JumpDiffusionGenerator(mu=mu, sigma=sigma,
                                      jump_intensity=intensity,
                                      jump_mean=j_mean,
                                      jump_std=j_std, **kwargs)

    if model in ("markov", "garch", "heston"):
        # Infer regime labels if not provided
        if regime_labels is None:
            from research.regime_lab.detector import RollingVolRegimeDetector
            det = RollingVolRegimeDetector()
            regime_labels = det.detect(prices[1:])   # align to returns

        # Ensure length matches returns
        rl = np.asarray(regime_labels, dtype=object)
        if len(rl) != len(log_rets):
            rl = rl[:len(log_rets)]

        # Estimate per-regime (mu, sigma)
        regime_params: Dict[str, Tuple[float, float]] = {}
        for r in REGIMES:
            mask = rl == r
            if mask.sum() > 1:
                mu_r = float(np.mean(log_rets[mask]))
                sg_r = float(np.std(log_rets[mask], ddof=1))
            else:
                mu_r, sg_r = DEFAULT_REGIME_PARAMS[r]
            regime_params[r] = (mu_r, sg_r)

        # Estimate transition matrix
        K    = len(REGIMES)
        r2i  = {r: i for i, r in enumerate(REGIMES)}
        trans = np.ones((K, K))  # Laplace smoothing
        for t in range(1, len(rl)):
            i = r2i.get(str(rl[t-1]), 2)
            j = r2i.get(str(rl[t]),   2)
            trans[i, j] += 1
        trans = trans / trans.sum(axis=1, keepdims=True)

        if model == "markov":
            return MarkovRegimeGenerator(
                transition_matrix=trans,
                regime_params=regime_params, **kwargs)

        if model == "garch":
            omega_d, alpha_d, beta_d, mu_d = {}, {}, {}, {}
            for r in REGIMES:
                mask = rl == r
                r_rets = log_rets[mask] if mask.sum() > 5 else log_rets
                mu_r   = float(np.mean(r_rets))
                var_r  = float(np.var(r_rets, ddof=1))
                # Rough GARCH(1,1) MoM estimates
                alpha_r = min(0.15, max(0.03, _estimate_garch_alpha(r_rets)))
                beta_r  = min(0.96, max(0.70, 1 - alpha_r - 0.05))
                omega_r = var_r * (1 - alpha_r - beta_r)
                omega_d[r] = max(omega_r, 1e-7)
                alpha_d[r] = alpha_r
                beta_d[r]  = beta_r
                mu_d[r]    = mu_r
            return GARCHRegimeGenerator(
                omega_dict=omega_d, alpha_dict=alpha_d,
                beta_dict=beta_d, mu_dict=mu_d,
                transition_matrix=trans, **kwargs)

        if model == "heston":
            var_overall = float(np.var(log_rets, ddof=1))
            return HestonRegimeGenerator(
                v0=var_overall,
                transition_matrix=trans, **kwargs)

    raise ValueError(f"Unknown model '{model}'. Choose: markov, garch, heston, jump, bootstrap")


def _estimate_garch_alpha(returns: np.ndarray) -> float:
    """Rough method-of-moments estimate for GARCH(1,1) alpha."""
    r2  = returns ** 2
    if len(r2) < 3:
        return 0.08
    # lag-1 autocorrelation of squared returns
    mean_r2 = float(np.mean(r2))
    cov     = float(np.mean((r2[1:] - mean_r2) * (r2[:-1] - mean_r2)))
    var     = float(np.var(r2))
    acf1    = cov / var if var > 0 else 0.0
    # alpha ≈ sqrt(acf1) (rough approximation)
    return max(0.03, min(0.20, float(np.sqrt(abs(acf1)))))


# ===========================================================================
# 7. Scenario utilities
# ===========================================================================

def combine_scenarios(*generators: Any, weights: Optional[List[float]] = None,
                      n_bars: int = 252,
                      n_scenarios: int = 1000,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Combine return paths from multiple generators via weighted mixing.

    Each scenario is drawn from one generator with probability proportional
    to its weight.

    Parameters
    ----------
    generators  : generator instances with generate() or generate_prices() methods
    weights     : list of weights (default equal)
    n_bars      : path length
    n_scenarios : number of output scenarios
    seed        : RNG seed

    Returns
    -------
    np.ndarray of shape (n_scenarios, n_bars) — log-return matrix
    """
    rng     = np.random.default_rng(seed)
    ng      = len(generators)
    if weights is None:
        weights = [1.0 / ng] * ng
    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()

    all_returns = np.zeros((n_scenarios, n_bars))
    choices     = rng.choice(ng, size=n_scenarios, p=weights_arr)

    for i, g_idx in enumerate(choices):
        g    = generators[g_idx]
        s    = int(rng.integers(0, 2**31))
        if isinstance(g, BootstrapScenarioGenerator):
            boot = g.generate(n_bars, n_scenarios=1, seed=s)
            all_returns[i] = boot.scenarios[0]
        elif isinstance(g, MarkovRegimeGenerator):
            res = g.generate(n_bars, seed=s)
            all_returns[i] = res.returns
        elif isinstance(g, GARCHRegimeGenerator):
            res = g.generate(n_bars, seed=s)
            all_returns[i] = res.returns
        elif isinstance(g, HestonRegimeGenerator):
            res = g.generate(n_bars, seed=s)
            # Compute log-returns from prices
            p = res.prices
            all_returns[i] = np.diff(np.log(np.where(p > 0, p, 1e-10)))[: n_bars]
        elif isinstance(g, JumpDiffusionGenerator):
            res = g.generate(n_bars, seed=s)
            all_returns[i] = res.returns
        else:
            # Generic fallback
            all_returns[i] = np.zeros(n_bars)

    return all_returns
