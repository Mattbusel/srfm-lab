"""
posterior.py
============
Posterior computation for strategy parameter updating.

Two pathways:
1. **Conjugate update** -- for Beta-Binomial parameters
   (``min_hold_bars``, ``winner_protection_pct``, ``stale_15m_move``):
   the posterior is computed analytically by adding observed counts to
   the Beta prior shape parameters.

2. **Sequential Monte Carlo (SMC)** -- for non-conjugate parameters
   (``garch_target_vol``, ``hour_boost_multiplier``): we maintain a
   particle cloud that approximates the posterior.

SMC Algorithm
-------------
Given prior p(theta) and likelihood L(data | theta):

    1. Initialise:  particles[i] ~ prior,  weights[i] = 1/N
    2. Weighting:   w[i] *= L(data | particles[i])
    3. Normalise:   w[i] /= sum(w)
    4. Diagnose:    ESS = 1 / sum(w^2)   (effective sample size)
    5. Resample:    if ESS < N/2, perform systematic resampling
    6. Jitter:      particles += Normal(0, jitter_sigma) random walk
                    (keeps particle cloud diffuse; necessary for
                     non-stationary parameters)

Systematic resampling is preferred over multinomial resampling because it
has lower variance for the same computational cost (O(N) vs O(N log N)).

References
----------
* Del Moral, Doucet & Jasra (2006) -- Sequential Monte Carlo Samplers.
* Chopin (2002) -- A sequential particle filter method for static models.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .priors import (
    MinHoldBarsPrior,
    Stale15mMovePrior,
    WinnerProtectionPctPrior,
    GarchTargetVolPrior,
    HourBoostMultiplierPrior,
    build_default_priors,
)
from .likelihood import LikelihoodFactory

logger = logging.getLogger(__name__)

N_PARTICLES = 1000


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PosteriorEstimate:
    """
    Result of one parameter's posterior update.

    Attributes
    ----------
    param_name         : which parameter this estimate is for.
    mean               : posterior mean (point estimate for recommendation).
    std                : posterior std-dev (uncertainty measure).
    credible_interval_95: (lo, hi) 95 % equal-tail credible interval.
    particles          : raw particle array (None for conjugate updates).
    weights            : normalised particle weights (None for conjugate).
    method             : "conjugate" or "smc".
    n_effective        : effective sample size for SMC (None for conjugate).
    """

    param_name:            str
    mean:                  float
    std:                   float
    credible_interval_95:  Tuple[float, float]
    particles:             Optional[np.ndarray] = field(default=None, repr=False)
    weights:               Optional[np.ndarray] = field(default=None, repr=False)
    method:                str = "smc"
    n_effective:           Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "param_name": self.param_name,
            "mean": self.mean,
            "std": self.std,
            "ci95_lo": self.credible_interval_95[0],
            "ci95_hi": self.credible_interval_95[1],
            "method": self.method,
            "n_effective": self.n_effective,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PosteriorEstimate":
        return cls(
            param_name=d["param_name"],
            mean=d["mean"],
            std=d["std"],
            credible_interval_95=(d["ci95_lo"], d["ci95_hi"]),
            method=d.get("method", "smc"),
            n_effective=d.get("n_effective"),
        )


@dataclass
class ParameterPosteriors:
    """
    Collection of posterior estimates for all tracked parameters.

    Attributes
    ----------
    estimates : dict mapping param_name -> PosteriorEstimate.
    timestamp : ISO-8601 string of when this update was computed.
    n_trades  : number of trades that drove this update.
    """

    estimates: Dict[str, PosteriorEstimate]
    timestamp: str = ""
    n_trades:  int = 0

    def __getitem__(self, key: str) -> PosteriorEstimate:
        return self.estimates[key]

    def __contains__(self, key: str) -> bool:
        return key in self.estimates

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_trades":  self.n_trades,
            "estimates": {k: v.to_dict() for k, v in self.estimates.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParameterPosteriors":
        estimates = {
            k: PosteriorEstimate.from_dict(v)
            for k, v in d.get("estimates", {}).items()
        }
        return cls(
            estimates=estimates,
            timestamp=d.get("timestamp", ""),
            n_trades=d.get("n_trades", 0),
        )


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------

def systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    """
    Systematic resampling of *n* indices from normalised *weights*.

    More efficient than multinomial resampling: O(N) and lower variance.
    All N sub-intervals of [0,1] are sampled with a single uniform draw.

    Parameters
    ----------
    weights : 1-D normalised weight array.
    n       : number of indices to draw.

    Returns
    -------
    indices : integer array of length *n*.
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # numerical safety

    # Evenly spaced positions with a single random offset
    u = (np.arange(n, dtype=float) + np.random.uniform()) / n
    return np.searchsorted(cumsum, u)


# ---------------------------------------------------------------------------
# SMC posterior
# ---------------------------------------------------------------------------

class SMCPosterior:
    """
    Sequential Monte Carlo posterior for a single parameter.

    The particle cloud is initialised once from the prior and then
    updated sequentially as new trade batches arrive.  Particles are
    jittered after each resampling step to prevent degeneracy.

    Parameters
    ----------
    prior        : one of the prior objects from priors.py.
    n_particles  : number of particles (default 1000).
    jitter_sigma : std-dev of the Gaussian random walk applied after
                   resampling to maintain diversity.
    """

    def __init__(self, prior, n_particles: int = N_PARTICLES, jitter_sigma: float = 0.02):
        self.prior        = prior
        self.n_particles  = n_particles
        self.jitter_sigma = jitter_sigma

        # Initialise particles from prior
        self.particles: np.ndarray = prior.sample(n_particles)
        self.weights:   np.ndarray = np.ones(n_particles) / n_particles
        self.update_count: int     = 0

    # ------------------------------------------------------------------
    # Core SMC step
    # ------------------------------------------------------------------

    def update(self, trade_stats, likelihood_factory: LikelihoodFactory) -> "SMCPosterior":
        """
        Incorporate a new batch of trade evidence.

        Steps: weight -> normalise -> diagnose -> resample -> jitter.

        Parameters
        ----------
        trade_stats       : TradeStats instance describing the batch.
        likelihood_factory: LikelihoodFactory for computing log-likelihoods.

        Returns
        -------
        self (fluent interface).
        """
        param_name = self.prior.name

        # 1. Compute log-weights for current particles
        log_ll = likelihood_factory.log_likelihood(
            param_name, self.particles, trade_stats
        )

        # Guard against all-NaN / all-inf batches (no information)
        if not np.all(np.isfinite(log_ll)):
            n_bad = np.sum(~np.isfinite(log_ll))
            logger.warning(
                "%s: %d particles returned non-finite log-likelihood; clipping.",
                param_name, n_bad,
            )
            log_ll = np.where(np.isfinite(log_ll), log_ll, -1e6)

        # 2. Update log-weights
        log_w = np.log(self.weights + 1e-300) + log_ll

        # 3. Normalise (log-sum-exp for stability)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        self.weights = w

        # 4. Effective sample size
        ess = 1.0 / np.sum(w ** 2)
        logger.debug("%s ESS=%.1f / %d", param_name, ess, self.n_particles)

        # 5. Resample when ESS drops below half the particle count
        if ess < self.n_particles / 2:
            indices = systematic_resample(w, self.n_particles)
            self.particles = self.particles[indices].copy()
            self.weights   = np.ones(self.n_particles) / self.n_particles
            logger.debug("%s: resampled (ESS was %.1f)", param_name, ess)

            # 6. Jitter to maintain diversity
            self.particles += np.random.normal(
                0, self.jitter_sigma, size=self.n_particles
            )
            # Reflect out-of-bounds particles back into support
            self._enforce_bounds()

        self.update_count += 1
        return self

    def _enforce_bounds(self) -> None:
        """Clip or reflect particles that escaped the prior support."""
        prior = self.prior
        if hasattr(prior, "lo") and hasattr(prior, "hi"):
            lo, hi = prior.lo, prior.hi
        elif hasattr(prior, "scale"):
            lo, hi = 0.0, prior.scale
        else:
            return
        # Reflection: bounce off the bounds
        self.particles = np.clip(self.particles, lo, hi)

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def mean(self) -> float:
        """Weighted mean of particles."""
        return float(np.average(self.particles, weights=self.weights))

    def std(self) -> float:
        """Weighted std-dev of particles."""
        mu = self.mean()
        var = np.average((self.particles - mu) ** 2, weights=self.weights)
        return float(math.sqrt(max(var, 0.0)))

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Equal-tail credible interval by weighted empirical quantiles.

        Sorts particles by value, builds cumulative weight, and reads off
        the alpha/2 and (1-alpha/2) quantiles.
        """
        idx  = np.argsort(self.particles)
        p_s  = self.particles[idx]
        w_s  = self.weights[idx]
        cdf  = np.cumsum(w_s)
        lo   = float(p_s[np.searchsorted(cdf, alpha / 2)])
        hi   = float(p_s[np.searchsorted(cdf, 1 - alpha / 2)])
        return lo, hi

    def n_effective(self) -> float:
        """Effective sample size."""
        return float(1.0 / np.sum(self.weights ** 2))

    def to_estimate(self) -> PosteriorEstimate:
        """Convert to a PosteriorEstimate dataclass."""
        return PosteriorEstimate(
            param_name=self.prior.name,
            mean=self.mean(),
            std=self.std(),
            credible_interval_95=self.credible_interval(0.05),
            particles=self.particles.copy(),
            weights=self.weights.copy(),
            method="smc",
            n_effective=self.n_effective(),
        )


# ---------------------------------------------------------------------------
# Conjugate (Beta) posterior
# ---------------------------------------------------------------------------

class BetaConjugatePosterior:
    """
    Exact analytical posterior update for Beta-Binomial parameters.

    The prior is Beta(alpha_0, beta_0).  After observing k successes
    in n trials the posterior is Beta(alpha_0 + k, beta_0 + n - k).

    Parameters
    ----------
    prior : Beta-based prior (MinHoldBarsPrior, etc.).
    """

    def __init__(self, prior):
        self.prior   = prior
        self.alpha   = prior.alpha
        self.beta_   = prior.beta
        self.scale   = getattr(prior, "scale", None)
        self.lo      = getattr(prior, "lo", 0.0)
        self.hi      = getattr(prior, "hi", 1.0)

    def update(self, n_wins: int, n_trades: int) -> "BetaConjugatePosterior":
        """
        Perform analytical update.

        Parameters
        ----------
        n_wins   : number of winning trades observed.
        n_trades : total trades in the batch.
        """
        self.alpha  += n_wins
        self.beta_  += (n_trades - n_wins)
        return self

    def _dist(self):
        return __import__("scipy.stats", fromlist=["beta"]).beta(self.alpha, self.beta_)

    def mean_unit(self) -> float:
        """Posterior mean in [0, 1]."""
        return self.alpha / (self.alpha + self.beta_)

    def std_unit(self) -> float:
        """Posterior std in [0, 1]."""
        a, b = self.alpha, self.beta_
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def _to_physical(self, u: float) -> float:
        if self.scale is not None:
            return u * self.scale
        return u * (self.hi - self.lo) + self.lo

    def mean(self) -> float:
        return self._to_physical(self.mean_unit())

    def std(self) -> float:
        s = self.std_unit()
        if self.scale is not None:
            return s * self.scale
        return s * (self.hi - self.lo)

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        from scipy.stats import beta as beta_dist
        d = beta_dist(self.alpha, self.beta_)
        lo_u = d.ppf(alpha / 2)
        hi_u = d.ppf(1 - alpha / 2)
        return self._to_physical(lo_u), self._to_physical(hi_u)

    def to_estimate(self) -> PosteriorEstimate:
        return PosteriorEstimate(
            param_name=self.prior.name,
            mean=self.mean(),
            std=self.std(),
            credible_interval_95=self.credible_interval(0.05),
            particles=None,
            weights=None,
            method="conjugate",
            n_effective=None,
        )


# ---------------------------------------------------------------------------
# Posterior computer -- dispatches conjugate vs SMC
# ---------------------------------------------------------------------------

class PosteriorComputer:
    """
    Manages posterior state for all tracked parameters.

    Chooses conjugate updates for Beta-Binomial parameters and SMC for
    everything else.  State is mutable across calls so that each new
    trade batch incrementally refines the posterior.

    Parameters
    ----------
    priors       : dict of prior objects (from build_default_priors()).
    n_particles  : number of SMC particles.
    jitter_sigma : std-dev for SMC particle jitter after resampling.
    """

    # Parameters that have conjugate Beta-Binomial updates
    CONJUGATE_PARAMS = {"min_hold_bars", "stale_15m_move", "winner_protection_pct"}

    def __init__(
        self,
        priors: Optional[dict] = None,
        n_particles: int = N_PARTICLES,
        jitter_sigma: float = 0.02,
    ):
        self.priors       = priors or build_default_priors()
        self.n_particles  = n_particles
        self.jitter_sigma = jitter_sigma
        self._likelihood  = LikelihoodFactory()

        # Initialise posterior objects
        self._conjugate: Dict[str, BetaConjugatePosterior] = {}
        self._smc:       Dict[str, SMCPosterior]            = {}

        for name, prior in self.priors.items():
            if name in self.CONJUGATE_PARAMS:
                self._conjugate[name] = BetaConjugatePosterior(prior)
            else:
                self._smc[name] = SMCPosterior(
                    prior, n_particles=n_particles, jitter_sigma=jitter_sigma
                )

    def update(self, trade_stats) -> ParameterPosteriors:
        """
        Perform one round of posterior updates given new trade evidence.

        Parameters
        ----------
        trade_stats : TradeStats (from trade_batcher.py).

        Returns
        -------
        ParameterPosteriors with updated estimates for all parameters.
        """
        import datetime

        ts = trade_stats

        # Conjugate updates
        if "min_hold_bars" in self._conjugate:
            self._conjugate["min_hold_bars"].update(ts.n_wins, ts.n_trades)
        if "stale_15m_move" in self._conjugate:
            self._conjugate["stale_15m_move"].update(ts.n_entered, ts.n_total_signals)
        if "winner_protection_pct" in self._conjugate:
            self._conjugate["winner_protection_pct"].update(
                ts.n_protected_wins, ts.n_winners
            )

        # SMC updates
        for name, smc in self._smc.items():
            smc.update(ts, self._likelihood)

        # Collect estimates
        estimates: Dict[str, PosteriorEstimate] = {}
        for name, conj in self._conjugate.items():
            estimates[name] = conj.to_estimate()
        for name, smc in self._smc.items():
            estimates[name] = smc.to_estimate()

        return ParameterPosteriors(
            estimates=estimates,
            timestamp=datetime.datetime.utcnow().isoformat(),
            n_trades=ts.n_trades,
        )

    def get_current_estimates(self) -> Dict[str, PosteriorEstimate]:
        """Return current estimates without performing an update."""
        out = {}
        for name, conj in self._conjugate.items():
            out[name] = conj.to_estimate()
        for name, smc in self._smc.items():
            out[name] = smc.to_estimate()
        return out

    def reset_to_priors(self) -> None:
        """
        Reset all posteriors back to the prior state.

        Call this when a regime change is detected and historical
        evidence should be discarded.
        """
        self.__init__(
            priors=self.priors,
            n_particles=self.n_particles,
            jitter_sigma=self.jitter_sigma,
        )
        logger.info("Posteriors reset to priors.")
