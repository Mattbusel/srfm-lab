"""
priors.py
=========
Prior distributions for each strategy parameter tracked by the Bayesian
updater.

Each prior is a dataclass wrapping a scipy.stats distribution.  All priors
expose a consistent interface:

    prior.sample(n)          -- draw *n* i.i.d. samples (numpy array)
    prior.log_prob(x)        -- log-density evaluated at scalar or array *x*
    prior.prob(x)            -- density (non-log) at *x*
    prior.mean               -- prior mean (property)
    prior.std                -- prior std-dev (property)
    prior.credible_interval(alpha) -- (lo, hi) symmetric CI

Design notes
------------
* Beta priors are parameterized with (alpha, beta) and an optional *scale*
  and *loc* to map the [0, 1] support to the physical parameter range.
* Normal priors are truncated to physically meaningful ranges so that SMC
  particles never escape valid parameter space.
* The Beta parameterization for ``min_hold_bars`` is conceptually a
  rescaled Beta: the *base* Beta draws a proportion in [0,1] which is then
  mapped to [1, 20] bars.  The stated (alpha=8, beta=2) places the bulk of
  mass near 8 on that scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Base helper — truncated Normal
# ---------------------------------------------------------------------------

def _truncnorm_params(mu: float, sigma: float, lo: float, hi: float):
    """Return (a, b) shape parameters for scipy.stats.truncnorm."""
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return a, b


# ---------------------------------------------------------------------------
# min_hold_bars  --  Beta(8, 2) rescaled to [1, 20]
# ---------------------------------------------------------------------------

@dataclass
class MinHoldBarsPrior:
    """
    Prior for ``min_hold_bars``.

    The IAE optimiser found 8 bars to be the sweet spot.  We encode this as
    a Beta(alpha, beta) distribution rescaled to [lo, hi] bars so that the
    prior mean sits at 8 and the distribution is right-skewed (we believe
    holding longer is safer than cutting too short).

    Parameters
    ----------
    alpha : float
        Beta shape -- controls left tail / peak location.
    beta  : float
        Beta shape -- controls right tail.
    lo    : float
        Lower bound of the rescaled support (default 1 bar).
    hi    : float
        Upper bound of the rescaled support (default 20 bars).
    """

    alpha: float = 8.0
    beta:  float = 2.0
    lo:    float = 1.0
    hi:    float = 20.0
    name:  str   = field(default="min_hold_bars", init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist(self):
        """Return the underlying scipy Beta distribution."""
        return stats.beta(self.alpha, self.beta)

    def _to_unit(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Map physical parameter value to [0, 1]."""
        return (np.asarray(x, dtype=float) - self.lo) / (self.hi - self.lo)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map [0, 1] back to physical parameter space."""
        return u * (self.hi - self.lo) + self.lo

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw *n* samples from the prior in physical space."""
        u = self._dist().rvs(size=n)
        return self._from_unit(u)

    def log_prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Log-density at physical value *x*.

        Accounts for the Jacobian of the affine rescaling so that the
        density integrates to 1 over [lo, hi].
        """
        u = self._to_unit(x)
        # Jacobian: dx = (hi - lo) * du  =>  log|J| = log(hi - lo)
        lp = self._dist().logpdf(u) - math.log(self.hi - self.lo)
        # Mask out-of-support values
        mask = (np.asarray(x) < self.lo) | (np.asarray(x) > self.hi)
        lp = np.where(mask, -np.inf, lp)
        return float(lp) if np.ndim(x) == 0 else lp

    def prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Density at *x*."""
        return np.exp(self.log_prob(x))

    @property
    def mean(self) -> float:
        """Prior mean in physical space."""
        return self._from_unit(self._dist().mean()).item()

    @property
    def std(self) -> float:
        """Prior std-dev in physical space."""
        return (self._dist().std() * (self.hi - self.lo)).item()

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Return (lo, hi) equal-tail credible interval at level *alpha*."""
        d = self._dist()
        lo_u = d.ppf(alpha / 2)
        hi_u = d.ppf(1 - alpha / 2)
        return self._from_unit(lo_u).item(), self._from_unit(hi_u).item()

    def __repr__(self) -> str:
        return (
            f"MinHoldBarsPrior(alpha={self.alpha}, beta={self.beta}, "
            f"lo={self.lo}, hi={self.hi}, mean={self.mean:.2f}, std={self.std:.2f})"
        )


# ---------------------------------------------------------------------------
# stale_15m_move  --  Beta(4, 2) * 0.02 (prior mean ≈ 0.0133, IAE result 0.008)
# ---------------------------------------------------------------------------

@dataclass
class Stale15mMovePrior:
    """
    Prior for ``stale_15m_move``.

    The parameter is a fractional price move threshold in [0, 0.02].
    Beta(4, 2) places the bulk of the mass in the upper half, reflecting
    our IAE finding that small values (0.008) sometimes lead to premature
    trade invalidation.

    Parameters
    ----------
    alpha : float
        Beta shape.
    beta  : float
        Beta shape.
    scale : float
        Multiplier to convert Beta [0,1] output to physical range [0, scale].
    """

    alpha: float = 4.0
    beta:  float = 2.0
    scale: float = 0.02
    name:  str   = field(default="stale_15m_move", init=False, repr=False)

    def _dist(self):
        return stats.beta(self.alpha, self.beta)

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist().rvs(size=n) * self.scale

    def log_prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        u = np.asarray(x, dtype=float) / self.scale
        lp = self._dist().logpdf(u) - math.log(self.scale)
        mask = (np.asarray(x) < 0) | (np.asarray(x) > self.scale)
        lp = np.where(mask, -np.inf, lp)
        return float(lp) if np.ndim(x) == 0 else lp

    def prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return np.exp(self.log_prob(x))

    @property
    def mean(self) -> float:
        return self._dist().mean() * self.scale

    @property
    def std(self) -> float:
        return self._dist().std() * self.scale

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        d = self._dist()
        return d.ppf(alpha / 2) * self.scale, d.ppf(1 - alpha / 2) * self.scale

    def __repr__(self) -> str:
        return (
            f"Stale15mMovePrior(alpha={self.alpha}, beta={self.beta}, "
            f"scale={self.scale}, mean={self.mean:.5f}, std={self.std:.5f})"
        )


# ---------------------------------------------------------------------------
# winner_protection_pct  --  Beta(3, 3) * 0.02 (symmetric prior, IAE 0.005)
# ---------------------------------------------------------------------------

@dataclass
class WinnerProtectionPctPrior:
    """
    Prior for ``winner_protection_pct``.

    A symmetric Beta(3, 3) * 0.02 prior reflects genuine uncertainty about
    the right protection level.  The IAE optimiser found 0.005 (25 % of the
    prior range), which is near the lower end -- suggesting winners should
    only be protected with a small buffer.

    Parameters
    ----------
    alpha : float
        Beta shape.
    beta  : float
        Beta shape.
    scale : float
        Multiplier to map [0,1] -> physical range.
    """

    alpha: float = 3.0
    beta:  float = 3.0
    scale: float = 0.02
    name:  str   = field(default="winner_protection_pct", init=False, repr=False)

    def _dist(self):
        return stats.beta(self.alpha, self.beta)

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist().rvs(size=n) * self.scale

    def log_prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        u = np.asarray(x, dtype=float) / self.scale
        lp = self._dist().logpdf(u) - math.log(self.scale)
        mask = (np.asarray(x) < 0) | (np.asarray(x) > self.scale)
        lp = np.where(mask, -np.inf, lp)
        return float(lp) if np.ndim(x) == 0 else lp

    def prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return np.exp(self.log_prob(x))

    @property
    def mean(self) -> float:
        return self._dist().mean() * self.scale

    @property
    def std(self) -> float:
        return self._dist().std() * self.scale

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        d = self._dist()
        return d.ppf(alpha / 2) * self.scale, d.ppf(1 - alpha / 2) * self.scale

    def __repr__(self) -> str:
        return (
            f"WinnerProtectionPctPrior(alpha={self.alpha}, beta={self.beta}, "
            f"scale={self.scale}, mean={self.mean:.5f}, std={self.std:.5f})"
        )


# ---------------------------------------------------------------------------
# garch_target_vol  --  TruncNormal(0.90, 0.10) on [0.5, 1.5]
# ---------------------------------------------------------------------------

@dataclass
class GarchTargetVolPrior:
    """
    Prior for ``garch_target_vol``.

    The GARCH target vol is the annualised volatility target used when
    sizing positions.  We use a truncated Normal centred at 0.90 with
    std 0.10, clipped to [0.5, 1.5] to avoid unphysical values.

    Parameters
    ----------
    mu    : float
        Prior mean (before truncation).
    sigma : float
        Prior std-dev (before truncation).
    lo    : float
        Lower truncation bound.
    hi    : float
        Upper truncation bound.
    """

    mu:    float = 0.90
    sigma: float = 0.10
    lo:    float = 0.50
    hi:    float = 1.50
    name:  str   = field(default="garch_target_vol", init=False, repr=False)

    def _dist(self):
        a, b = _truncnorm_params(self.mu, self.sigma, self.lo, self.hi)
        return stats.truncnorm(a, b, loc=self.mu, scale=self.sigma)

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist().rvs(size=n)

    def log_prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self._dist().logpdf(np.asarray(x, dtype=float))

    def prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return np.exp(self.log_prob(x))

    @property
    def mean(self) -> float:
        return self._dist().mean()

    @property
    def std(self) -> float:
        return self._dist().std()

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        d = self._dist()
        return d.ppf(alpha / 2), d.ppf(1 - alpha / 2)

    def __repr__(self) -> str:
        return (
            f"GarchTargetVolPrior(mu={self.mu}, sigma={self.sigma}, "
            f"lo={self.lo}, hi={self.hi}, mean={self.mean:.3f}, std={self.std:.3f})"
        )


# ---------------------------------------------------------------------------
# hour_boost_multiplier  --  TruncNormal(1.25, 0.15) on [1.0, 2.0]
# ---------------------------------------------------------------------------

@dataclass
class HourBoostMultiplierPrior:
    """
    Prior for ``hour_boost_multiplier``.

    A multiplier applied to signal strength during high-activity market
    hours.  Prior is TruncNormal(1.25, 0.15) on [1.0, 2.0], reflecting
    the belief that peak hours add a modest but meaningful edge.

    Parameters
    ----------
    mu    : float
        Prior mean (before truncation).
    sigma : float
        Prior std-dev.
    lo    : float
        Lower bound (must be >= 1.0 since a multiplier below 1 makes no sense).
    hi    : float
        Upper bound.
    """

    mu:    float = 1.25
    sigma: float = 0.15
    lo:    float = 1.00
    hi:    float = 2.00
    name:  str   = field(default="hour_boost_multiplier", init=False, repr=False)

    def _dist(self):
        a, b = _truncnorm_params(self.mu, self.sigma, self.lo, self.hi)
        return stats.truncnorm(a, b, loc=self.mu, scale=self.sigma)

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist().rvs(size=n)

    def log_prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self._dist().logpdf(np.asarray(x, dtype=float))

    def prob(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return np.exp(self.log_prob(x))

    @property
    def mean(self) -> float:
        return self._dist().mean()

    @property
    def std(self) -> float:
        return self._dist().std()

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        d = self._dist()
        return d.ppf(alpha / 2), d.ppf(1 - alpha / 2)

    def __repr__(self) -> str:
        return (
            f"HourBoostMultiplierPrior(mu={self.mu}, sigma={self.sigma}, "
            f"lo={self.lo}, hi={self.hi}, mean={self.mean:.3f}, std={self.std:.3f})"
        )


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

ALL_PRIORS = {
    "min_hold_bars":        MinHoldBarsPrior,
    "stale_15m_move":       Stale15mMovePrior,
    "winner_protection_pct": WinnerProtectionPctPrior,
    "garch_target_vol":     GarchTargetVolPrior,
    "hour_boost_multiplier": HourBoostMultiplierPrior,
}


def build_default_priors() -> dict:
    """Instantiate all priors with default hyperparameters."""
    return {name: cls() for name, cls in ALL_PRIORS.items()}
