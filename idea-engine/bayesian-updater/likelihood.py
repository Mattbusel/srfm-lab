"""
likelihood.py
=============
Likelihood functions for Bayesian parameter updating.

Given a parameter value *theta* and a batch of trades that were generated
under that parameter setting, these functions compute P(trades | theta).

Design overview
---------------
* For ``min_hold_bars`` and ``winner_protection_pct``: outcomes are
  binary (win / loss), so we model P(trade_outcome | param) with a
  Beta-Binomial likelihood.  The win probability p(theta) is a smooth
  function of theta; given p we use a Binomial likelihood for observed
  (n_wins, n_trades).

* For ``garch_target_vol``: each trade's P&L is approximately Normal
  with location and scale that depend on the vol target.  Higher vol
  targets mean larger expected moves but also larger realized P&L
  variance.  We model mu(v) = v * base_mu and sigma(v) = v * base_sigma.

* For ``stale_15m_move``: the threshold controls how often trades are
  cancelled before entry.  We model trade entry rate as a function of
  the threshold and then compute likelihood of observed entry/rejection
  counts.

* For ``hour_boost_multiplier``: multiplier affects both mean P&L and
  variance.  Normal likelihood with mean = multiplier * baseline_mu.

All log-likelihood functions return scalars.  They accept either scalar
theta or arrays (in which case they return arrays of the same shape,
useful for evaluating over particle clouds).

Numerical safety
----------------
We always return log-likelihoods (not raw likelihoods) to avoid underflow
when multiplying many small probabilities together.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    """Numerically stable log; clips to *floor* before taking log."""
    return np.log(np.maximum(x, floor))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Beta-Binomial log-likelihood helpers
# ---------------------------------------------------------------------------

def _log_beta_binomial(k: int, n: int, alpha: float, beta: float) -> float:
    """
    Log-likelihood of observing *k* successes in *n* trials under a
    Beta-Binomial model with shape parameters (alpha, beta).

    The marginal likelihood integrates out the success probability p::

        P(k | n, alpha, beta) = C(n,k) * B(k+alpha, n-k+beta) / B(alpha, beta)

    where B is the Beta function.

    Parameters
    ----------
    k     : int   -- number of wins observed.
    n     : int   -- total number of trades.
    alpha : float -- effective Beta shape for successes.
    beta  : float -- effective Beta shape for failures.
    """
    if n == 0:
        return 0.0
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    log_beta_num = betaln(k + alpha, n - k + beta)
    log_beta_den = betaln(alpha, beta)
    return log_comb + log_beta_num - log_beta_den


# ---------------------------------------------------------------------------
# min_hold_bars likelihood
# ---------------------------------------------------------------------------

def log_likelihood_min_hold(
    theta: Union[float, np.ndarray],
    n_wins: int,
    n_trades: int,
    sensitivity: float = 0.12,
    optimal_bars: float = 8.0,
) -> np.ndarray:
    """
    Log-likelihood of observed win/loss outcomes given ``min_hold_bars=theta``.

    We model the win probability as a function of hold length::

        p(theta) = sigmoid(sensitivity * (theta - optimal_bars))  + 0.50

    This means win rate is 0.50 at theta=0, rising toward 1.0 as theta
    increases, with the sharpest gain near `optimal_bars`.

    We then use a Beta-Binomial with effective concentration
    ``kappa = 20.0`` -- moderate confidence in the win-rate model.

    Parameters
    ----------
    theta         : parameter value(s) to evaluate at.
    n_wins        : observed wins in the batch.
    n_trades      : total trades in the batch.
    sensitivity   : steepness of win-rate sigmoid wrt bars.
    optimal_bars  : bars at which win-rate gradient is steepest.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    kappa = 20.0  # concentration; higher = tighter likelihood

    # p in (0.45, 0.95) to keep things realistic
    p = 0.45 + 0.50 * _sigmoid(sensitivity * (theta - optimal_bars))
    alpha_eff = p * kappa
    beta_eff  = (1 - p) * kappa

    # Vectorise over theta
    ll = np.array([
        _log_beta_binomial(n_wins, n_trades, float(a), float(b))
        for a, b in zip(alpha_eff, beta_eff)
    ])
    return ll.squeeze() if ll.size == 1 else ll


# ---------------------------------------------------------------------------
# garch_target_vol likelihood
# ---------------------------------------------------------------------------

def log_likelihood_garch_vol(
    theta: Union[float, np.ndarray],
    pnl_series: np.ndarray,
    base_mu_per_unit: float = 0.003,
    base_sigma_per_unit: float = 0.025,
) -> np.ndarray:
    """
    Log-likelihood of observed P&L series given ``garch_target_vol=theta``.

    Model:
        mu(theta)    = theta * base_mu_per_unit
        sigma(theta) = theta * base_sigma_per_unit

    Each trade's fractional P&L is assumed i.i.d. Normal(mu, sigma).
    The log-likelihood is the sum of Normal log-pdfs over the batch.

    Parameters
    ----------
    theta             : vol target(s) to evaluate at.
    pnl_series        : array of per-trade fractional P&L values.
    base_mu_per_unit  : expected P&L per unit of vol target.
    base_sigma_per_unit: P&L std per unit of vol target.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    pnl   = np.asarray(pnl_series, dtype=float)

    if len(pnl) == 0:
        return np.zeros_like(theta)

    # Clip to avoid degenerate distributions
    mu_arr    = theta * base_mu_per_unit
    sigma_arr = np.maximum(theta * base_sigma_per_unit, 1e-6)

    # For each theta value, sum Normal log-pdf over all trades
    ll = np.array([
        stats.norm.logpdf(pnl, loc=float(mu), scale=float(sig)).sum()
        for mu, sig in zip(mu_arr, sigma_arr)
    ])
    return ll.squeeze() if ll.size == 1 else ll


# ---------------------------------------------------------------------------
# stale_15m_move likelihood
# ---------------------------------------------------------------------------

def log_likelihood_stale_move(
    theta: Union[float, np.ndarray],
    n_entered: int,
    n_total_signals: int,
    move_dist_std: float = 0.008,
) -> np.ndarray:
    """
    Log-likelihood of observed entry rate given ``stale_15m_move=theta``.

    The threshold theta determines whether a signal is still "fresh": a
    signal is entered if the 15-minute move since signal time is < theta.
    Assuming 15m moves are half-Normal(0, move_dist_std), the entry
    probability is::

        p_entry(theta) = erf(theta / (sqrt(2) * move_dist_std))

    We then use a Binomial likelihood for (n_entered, n_total_signals).

    Parameters
    ----------
    theta            : threshold value(s) to evaluate at.
    n_entered        : signals that passed the staleness filter.
    n_total_signals  : all signals generated in the period.
    move_dist_std    : std-dev of the 15-minute move distribution.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if n_total_signals == 0:
        return np.zeros_like(theta)

    # p_entry via the half-Normal CDF
    p_entry = stats.norm.cdf(theta, loc=0, scale=move_dist_std)
    p_entry = np.clip(p_entry, 1e-9, 1 - 1e-9)

    ll = stats.binom.logpmf(n_entered, n_total_signals, p_entry)
    return ll.squeeze() if ll.size == 1 else ll


# ---------------------------------------------------------------------------
# winner_protection_pct likelihood
# ---------------------------------------------------------------------------

def log_likelihood_winner_prot(
    theta: Union[float, np.ndarray],
    n_protected_wins: int,
    n_winners: int,
    kappa: float = 15.0,
) -> np.ndarray:
    """
    Log-likelihood of observed winner-protection outcomes given
    ``winner_protection_pct=theta``.

    A "protected win" occurs when a trade that reached peak profit was
    exited above the protection threshold.  Higher theta means a tighter
    buffer, so more winners are protected -- but at the cost of earlier
    exits.  We model the protection rate as::

        p_protect(theta) = 1 - exp(-theta / tau)

    where tau = 0.005 (the IAE-derived value).  The likelihood for
    n_protected_wins out of n_winners is Beta-Binomial.

    Parameters
    ----------
    theta            : protection threshold(s) to evaluate at.
    n_protected_wins : winners that survived to exit above the threshold.
    n_winners        : total profitable trades in the batch.
    kappa            : concentration of the Beta-Binomial.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    tau = 0.005

    p_protect = 1.0 - np.exp(-theta / tau)
    p_protect = np.clip(p_protect, 1e-9, 1 - 1e-9)

    alpha_eff = p_protect * kappa
    beta_eff  = (1 - p_protect) * kappa

    ll = np.array([
        _log_beta_binomial(n_protected_wins, n_winners, float(a), float(b))
        for a, b in zip(alpha_eff, beta_eff)
    ])
    return ll.squeeze() if ll.size == 1 else ll


# ---------------------------------------------------------------------------
# hour_boost_multiplier likelihood
# ---------------------------------------------------------------------------

def log_likelihood_hour_boost(
    theta: Union[float, np.ndarray],
    boosted_pnl: np.ndarray,
    baseline_mu: float = 0.003,
    baseline_sigma: float = 0.025,
) -> np.ndarray:
    """
    Log-likelihood of P&L observed during boosted hours given
    ``hour_boost_multiplier=theta``.

    We assume boosted-hour trades have::

        mu_boosted    = theta * baseline_mu
        sigma_boosted = max(theta * 0.8, 1.0) * baseline_sigma

    (The sigma grows sublinearly with theta since the boost mainly
    increases expected return, not noise.)

    Parameters
    ----------
    theta          : multiplier value(s) to evaluate at.
    boosted_pnl    : per-trade P&L observed during boosted hours.
    baseline_mu    : baseline expected P&L per trade.
    baseline_sigma : baseline P&L std-dev per trade.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    pnl   = np.asarray(boosted_pnl, dtype=float)

    if len(pnl) == 0:
        return np.zeros_like(theta)

    ll = np.array([
        stats.norm.logpdf(
            pnl,
            loc=float(t) * baseline_mu,
            scale=max(float(t) * 0.80, 1.0) * baseline_sigma,
        ).sum()
        for t in theta
    ])
    return ll.squeeze() if ll.size == 1 else ll


# ---------------------------------------------------------------------------
# Dispatcher -- unified interface
# ---------------------------------------------------------------------------

class LikelihoodFactory:
    """
    Dispatch log-likelihood computation by parameter name.

    Usage::

        factory = LikelihoodFactory()
        ll = factory.log_likelihood("min_hold_bars", theta=8.0, trade_stats=stats)
    """

    def log_likelihood(
        self,
        param_name: str,
        theta: Union[float, np.ndarray],
        trade_stats,
    ) -> np.ndarray:
        """
        Compute log-likelihood for *param_name* at *theta* given *trade_stats*.

        Parameters
        ----------
        param_name  : one of the tracked parameter names.
        theta       : scalar or array of parameter values.
        trade_stats : TradeStats instance (from trade_batcher.py).
        """
        ts = trade_stats
        dispatch = {
            "min_hold_bars": lambda: log_likelihood_min_hold(
                theta, ts.n_wins, ts.n_trades
            ),
            "stale_15m_move": lambda: log_likelihood_stale_move(
                theta, ts.n_entered, ts.n_total_signals
            ),
            "winner_protection_pct": lambda: log_likelihood_winner_prot(
                theta, ts.n_protected_wins, ts.n_winners
            ),
            "garch_target_vol": lambda: log_likelihood_garch_vol(
                theta, np.asarray(ts.pnl_list, dtype=float)
            ),
            "hour_boost_multiplier": lambda: log_likelihood_hour_boost(
                theta, np.asarray(ts.boosted_hour_pnl, dtype=float)
            ),
        }
        if param_name not in dispatch:
            raise ValueError(f"Unknown parameter: {param_name!r}")
        return dispatch[param_name]()
