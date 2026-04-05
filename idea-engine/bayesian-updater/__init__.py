"""
bayesian-updater
================
Online Bayesian parameter updating for the IAE (Idea Automation Engine).

As live trades arrive, strategy parameters are updated toward what is working
NOW rather than relying purely on backtest-derived priors.  The module
implements conjugate-update shortcuts where available (Beta-Binomial) and
falls back to Sequential Monte Carlo (SMC) for non-conjugate cases.

Public surface
--------------
    BayesianUpdater   -- main entry point; call .update(trades)
    ParameterPosteriors
    PosteriorEstimate
    TradeStats
    DriftMonitor
    HypothesisEmitter
    Scheduler

Typical usage::

    from bayesian_updater import BayesianUpdater, load_live_trades

    updater = BayesianUpdater.from_state_file("state.json")
    trades  = load_live_trades()
    posts   = updater.update(trades)
    params  = updater.get_recommended_params()
    updater.save_state("state.json")
"""

from .priors import (
    MinHoldBarsPrior,
    Stale15mMovePrior,
    WinnerProtectionPctPrior,
    GarchTargetVolPrior,
    HourBoostMultiplierPrior,
)
from .posterior import PosteriorEstimate, ParameterPosteriors
from .trade_batcher import TradeStats, TradeBatcher, load_live_trades
from .updater import BayesianUpdater
from .drift_monitor import DriftMonitor, DriftAlert
from .hypothesis_emitter import HypothesisEmitter, IAEHypothesis
from .scheduler import UpdateScheduler

__all__ = [
    "MinHoldBarsPrior",
    "Stale15mMovePrior",
    "WinnerProtectionPctPrior",
    "GarchTargetVolPrior",
    "HourBoostMultiplierPrior",
    "PosteriorEstimate",
    "ParameterPosteriors",
    "TradeStats",
    "TradeBatcher",
    "load_live_trades",
    "BayesianUpdater",
    "DriftMonitor",
    "DriftAlert",
    "HypothesisEmitter",
    "IAEHypothesis",
    "UpdateScheduler",
]
