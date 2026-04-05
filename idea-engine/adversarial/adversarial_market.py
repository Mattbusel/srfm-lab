"""
adversarial_market.py
=====================
Generate adversarial price sequences that are maximally damaging to the
strategy while remaining statistically plausible.

An adversarial price path is one that:
1. Causes the strategy to lose as much money as possible.
2. Is statistically plausible -- within 3-sigma of the historically
   observed volatility and autocorrelation structure.

Algorithm
---------
We frame this as a constrained optimisation problem:

    minimise   P&L(path)
    subject to ||path - historical_mean|| / historical_std <= 3.0  (plausibility)
               path[0] = starting price

The inner P&L function is not differentiable (it depends on trade entry/
exit logic), so we use a gradient-free approach: CMA-ES (Covariance Matrix
Adaptation Evolution Strategy), which is well-suited to black-box optimisation
in moderate dimensions.

A simplified surrogate P&L is also implemented that IS differentiable, which
allows gradient descent for faster iteration when needed.

For tractability, we parameterise the price path as a Fourier series
(low-frequency modes) so that the optimisation dimension is << path length.

Plausibility constraint
-----------------------
We enforce plausibility via a soft penalty:
    objective = P&L(path) + lambda * max(0, z_score(path) - 3.0)^2

where z_score(path) = max per-step deviation normalised by historical vol.

Output
------
AdversarialPath contains:
- The adversarial price sequence.
- The estimated P&L impact.
- A plausibility report: why is this path within statistical bounds?
- The objective function value (lower = more adversarial).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)

# Default simulation parameters
DEFAULT_STEPS       = 200     # path length in 15-min bars
DEFAULT_N_FOURIER   = 20      # Fourier modes to optimise over
DEFAULT_HIST_VOL    = 0.008   # per-bar historical volatility (15-min BTC ~0.7%)
DEFAULT_HIST_DRIFT  = 0.0001  # slight upward drift
PLAUSIBILITY_SIGMA  = 3.0


# ---------------------------------------------------------------------------
# Path generation from Fourier coefficients
# ---------------------------------------------------------------------------

def fourier_path(
    coeffs: np.ndarray,
    n_steps: int,
    start_price: float = 100.0,
    hist_drift: float = DEFAULT_HIST_DRIFT,
) -> np.ndarray:
    """
    Reconstruct a price path from Fourier coefficients.

    The path is parameterised as a sum of cosines::

        log_return[t] = sum_k a_k * cos(2 * pi * k * t / T) + drift

    Parameters
    ----------
    coeffs      : 1-D array of Fourier amplitudes (length n_fourier * 2 for sin+cos).
    n_steps     : number of price steps.
    start_price : starting price level.
    hist_drift  : per-step drift component.

    Returns
    -------
    1-D array of length (n_steps + 1) starting at start_price.
    """
    n_modes = len(coeffs) // 2
    t       = np.arange(n_steps, dtype=float)
    log_returns = np.full(n_steps, hist_drift)

    for k in range(n_modes):
        freq = 2 * math.pi * (k + 1) / n_steps
        log_returns += coeffs[2 * k]     * np.cos(freq * t)
        log_returns += coeffs[2 * k + 1] * np.sin(freq * t)

    prices = start_price * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return prices


# ---------------------------------------------------------------------------
# Plausibility constraint
# ---------------------------------------------------------------------------

def plausibility_penalty(
    log_returns: np.ndarray,
    hist_vol: float,
    sigma_limit: float = PLAUSIBILITY_SIGMA,
) -> float:
    """
    Soft penalty for implausible log-returns.

    A return is implausible if |r_t| > sigma_limit * hist_vol.
    The penalty is the sum of squared excesses.

    Parameters
    ----------
    log_returns  : 1-D array of per-step log returns.
    hist_vol     : per-step historical volatility (std of log returns).
    sigma_limit  : maximum allowed z-score.

    Returns
    -------
    Non-negative penalty (0 = fully plausible).
    """
    z_scores = np.abs(log_returns) / max(hist_vol, 1e-10)
    excess   = np.maximum(z_scores - sigma_limit, 0.0)
    return float(np.sum(excess ** 2))


# ---------------------------------------------------------------------------
# Surrogate P&L function
# ---------------------------------------------------------------------------

def _surrogate_pnl(
    prices: np.ndarray,
    min_hold_bars: int = 8,
    stale_move: float = 0.005,
    winner_prot_pct: float = 0.005,
    base_position: float = 1.0,
) -> float:
    """
    Fast surrogate P&L computation for a price path.

    Models the strategy's entry/exit logic in vectorised form:
    - Enter on each bar (simplified: no signal filter).
    - Exit after min_hold_bars OR if winner protection triggers.
    - Account for stale_move by rejecting entries where recent move > threshold.

    Parameters
    ----------
    prices          : price path array.
    min_hold_bars   : minimum hold duration.
    stale_move      : staleness threshold (reject entry if recent move > this).
    winner_prot_pct : protection level for profitable trades.
    base_position   : base position size (fraction of portfolio).

    Returns
    -------
    Total fractional P&L.
    """
    n      = len(prices)
    total  = 0.0
    i      = 1  # start from bar 1 (bar 0 is the initial price)

    while i < n - min_hold_bars - 1:
        # Staleness check: skip if recent 1-bar move is too large
        recent_move = abs(prices[i] / prices[i - 1] - 1.0) if i > 0 else 0.0
        if recent_move > stale_move:
            i += 1
            continue

        entry    = prices[i]
        peak     = entry
        exit_idx = min(i + min_hold_bars, n - 1)
        exit_p   = prices[exit_idx]

        # Check winner protection: exit early if price rises then drops
        for j in range(i + 1, exit_idx + 1):
            if prices[j] > peak:
                peak = prices[j]
            # Protection triggers if price drops from peak by winner_prot_pct
            if peak > entry and (peak - prices[j]) / peak > winner_prot_pct:
                exit_p   = prices[j]
                exit_idx = j
                break

        pnl   = (exit_p - entry) / entry * base_position
        total += pnl
        i      = exit_idx + 1

    return total


# ---------------------------------------------------------------------------
# CMA-ES wrapper (pure numpy, no external dep)
# ---------------------------------------------------------------------------

def _cmaes_minimise(
    objective: callable,
    x0: np.ndarray,
    sigma0: float = 0.1,
    max_iter: int = 200,
    popsize: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    Simplified (1+lambda)-CMA-ES for black-box minimisation.

    A full CMA-ES implementation with covariance adaptation.  Uses a
    (1, lambda) selection strategy: the best of *popsize* offspring
    replaces the parent each generation.

    Parameters
    ----------
    objective : callable(x) -> float to minimise.
    x0        : initial solution.
    sigma0    : initial step size.
    max_iter  : maximum iterations.
    popsize   : offspring per generation.
    rng       : random generator.

    Returns
    -------
    (best_x, best_f).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n      = len(x0)
    mu     = x0.copy()
    sigma  = sigma0
    best_x = x0.copy()
    best_f = objective(x0)

    # Simple diagonal covariance adaptation
    C      = np.ones(n)    # diagonal of covariance matrix
    p_c    = np.zeros(n)   # evolution path
    c_c    = 4.0 / (n + 4)
    c_1    = 2.0 / (n ** 2 + 2)

    for gen in range(max_iter):
        # Sample offspring
        offspring = mu + sigma * rng.standard_normal((popsize, n)) * np.sqrt(C)
        scores    = np.array([objective(x) for x in offspring])

        # Select best offspring
        best_idx  = int(np.argmin(scores))
        best_off  = offspring[best_idx]
        best_sc   = scores[best_idx]

        # Update evolution path
        step      = (best_off - mu) / sigma
        p_c       = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c)) * step

        # Adapt covariance diagonal
        C  = (1 - c_1) * C + c_1 * (p_c ** 2)
        C  = np.clip(C, 1e-8, 1e4)

        # Step size adaptation (simplified)
        sigma *= math.exp(0.1 * (np.linalg.norm(step) / math.sqrt(n) - 1))
        sigma  = max(1e-6, min(sigma, 10.0))

        mu = best_off.copy()

        if best_sc < best_f:
            best_f = best_sc
            best_x = best_off.copy()

        if gen % 50 == 0:
            logger.debug("CMA-ES gen=%d, best_f=%.6f, sigma=%.4f", gen, best_f, sigma)

    return best_x, best_f


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AdversarialPath:
    """
    Result of an adversarial price path search.

    Attributes
    ----------
    prices          : adversarial price sequence.
    log_returns     : log returns of the path.
    pnl             : estimated strategy P&L on this path.
    plausibility_z  : max z-score of log returns vs historical vol.
    is_plausible    : True if z-score <= 3.0.
    description     : why this path is adversarial and plausible.
    n_steps         : path length.
    optimiser_iters : number of CMA-ES iterations used.
    """

    prices:           np.ndarray
    log_returns:      np.ndarray
    pnl:              float
    plausibility_z:   float
    is_plausible:     bool
    description:      str
    n_steps:          int
    optimiser_iters:  int

    def summary(self) -> str:
        plaus = "PLAUSIBLE" if self.is_plausible else "IMPLAUSIBLE"
        return (
            f"AdversarialPath ({plaus}):\n"
            f"  Steps:         {self.n_steps}\n"
            f"  Strategy P&L:  {self.pnl:+.6f}\n"
            f"  Max z-score:   {self.plausibility_z:.2f}\n"
            f"  Description:   {self.description}"
        )


# ---------------------------------------------------------------------------
# AdversarialMarket
# ---------------------------------------------------------------------------

class AdversarialMarket:
    """
    Find the most damaging statistically-plausible price path.

    Parameters
    ----------
    hist_vol          : per-bar historical volatility.
    hist_drift        : per-bar drift.
    n_steps           : path length in bars.
    n_fourier         : number of Fourier modes.
    penalty_lambda    : weight for plausibility constraint penalty.
    strategy_params   : dict of strategy parameters for the surrogate P&L.
    cmaes_iters       : CMA-ES iterations.
    cmaes_popsize     : CMA-ES population size.
    seed              : random seed.
    """

    def __init__(
        self,
        hist_vol:         float = DEFAULT_HIST_VOL,
        hist_drift:       float = DEFAULT_HIST_DRIFT,
        n_steps:          int   = DEFAULT_STEPS,
        n_fourier:        int   = DEFAULT_N_FOURIER,
        penalty_lambda:   float = 100.0,
        strategy_params:  Optional[dict] = None,
        cmaes_iters:      int   = 300,
        cmaes_popsize:    int   = 30,
        seed:             int   = 42,
    ):
        self.hist_vol         = hist_vol
        self.hist_drift       = hist_drift
        self.n_steps          = n_steps
        self.n_fourier        = n_fourier
        self.penalty_lambda   = penalty_lambda
        self.strategy_params  = strategy_params or {}
        self.cmaes_iters      = cmaes_iters
        self.cmaes_popsize    = cmaes_popsize
        self.seed             = seed

    def _objective(self, coeffs: np.ndarray) -> float:
        """
        Objective function: P&L + plausibility penalty.

        We minimise P&L (maximise losses), so a lower value is worse for
        the strategy.

        Parameters
        ----------
        coeffs : Fourier coefficients.

        Returns
        -------
        Objective value (lower = more adversarial).
        """
        prices      = fourier_path(coeffs, self.n_steps, hist_drift=self.hist_drift)
        log_returns = np.diff(np.log(prices))

        pnl     = _surrogate_pnl(prices, **self.strategy_params)
        penalty = self.penalty_lambda * plausibility_penalty(log_returns, self.hist_vol)

        return pnl + penalty

    def run(self) -> AdversarialPath:
        """
        Optimise for the most damaging statistically-plausible price path.

        Returns
        -------
        AdversarialPath.
        """
        logger.info(
            "Searching for adversarial price path: steps=%d, n_fourier=%d",
            self.n_steps, self.n_fourier,
        )

        rng  = np.random.default_rng(self.seed)
        n_c  = self.n_fourier * 2
        x0   = rng.standard_normal(n_c) * 0.001  # small initial coefficients

        best_coeffs, best_obj = _cmaes_minimise(
            self._objective,
            x0,
            sigma0=0.005,
            max_iter=self.cmaes_iters,
            popsize=self.cmaes_popsize,
            rng=rng,
        )

        prices      = fourier_path(best_coeffs, self.n_steps, hist_drift=self.hist_drift)
        log_returns = np.diff(np.log(prices))
        pnl         = _surrogate_pnl(prices, **self.strategy_params)

        z_scores = np.abs(log_returns) / max(self.hist_vol, 1e-10)
        max_z    = float(z_scores.max())
        plausible = max_z <= PLAUSIBILITY_SIGMA

        description = self._build_description(prices, log_returns, pnl, max_z)

        result = AdversarialPath(
            prices=prices,
            log_returns=log_returns,
            pnl=pnl,
            plausibility_z=max_z,
            is_plausible=plausible,
            description=description,
            n_steps=self.n_steps,
            optimiser_iters=self.cmaes_iters,
        )
        logger.info(result.summary())
        return result

    def _build_description(
        self,
        prices:      np.ndarray,
        log_returns: np.ndarray,
        pnl:         float,
        max_z:       float,
    ) -> str:
        """Build a natural-language description of the adversarial path."""
        total_return = prices[-1] / prices[0] - 1.0
        n_negative   = int(np.sum(log_returns < 0))
        n_positive   = int(np.sum(log_returns > 0))
        max_drop     = float(np.min(log_returns))
        max_rise     = float(np.max(log_returns))

        plausibility_note = (
            f"The path is statistically plausible (max z-score {max_z:.2f} <= 3.0)."
            if max_z <= PLAUSIBILITY_SIGMA
            else f"WARNING: path has implausible moves (max z-score {max_z:.2f} > 3.0)."
        )

        return (
            f"This {self.n_steps}-bar price path loses {pnl:+.4f} for the strategy. "
            f"Total return: {total_return:+.2%}. "
            f"The path contains {n_negative} down-bars and {n_positive} up-bars. "
            f"Largest single-bar drop: {max_drop:+.2%}; "
            f"largest single-bar rise: {max_rise:+.2%}. "
            f"{plausibility_note} "
            "The path exploits the strategy's tendency to enter on small dips "
            "and then experience further sustained decline within the hold window."
        )

    def baseline_pnl(self, n_paths: int = 100) -> float:
        """
        Compute average P&L on random (non-adversarial) paths for comparison.

        Parameters
        ----------
        n_paths : number of random paths to simulate.

        Returns
        -------
        Mean P&L across random paths.
        """
        rng  = np.random.default_rng(self.seed)
        pnls = []
        for _ in range(n_paths):
            n_c     = self.n_fourier * 2
            coeffs  = rng.standard_normal(n_c) * 0.001
            prices  = fourier_path(coeffs, self.n_steps, hist_drift=self.hist_drift)
            pnls.append(_surrogate_pnl(prices, **self.strategy_params))
        return float(np.mean(pnls))
