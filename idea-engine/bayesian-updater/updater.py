"""
updater.py
==========
BayesianUpdater -- the main entry point for online parameter updating.

Workflow
--------
1. New trades arrive (from live_trades.db or passed directly).
2. TradeBatcher aggregates them into TradeStats.
3. PosteriorComputer updates each parameter's posterior.
4. detect_drift() compares posterior means vs prior means.
5. get_recommended_params() returns posterior means as the new parameter
   suggestion dict, ready to be handed off to the IAE genome system.
6. State is serialised to JSON for persistence across restarts.

Drift detection
---------------
A parameter is flagged as drifted when::

    |posterior_mean - prior_mean| > drift_threshold * prior_std

The default threshold is 2.0 (two prior standard deviations).  When drift
is detected, a DriftFlag is emitted and the HypothesisEmitter is invoked
to convert the finding into a formal IAE hypothesis.

Persistence
-----------
State is stored in a JSON file with the following structure::

    {
        "last_update_timestamp": "...",
        "n_total_trades_seen": 1234,
        "last_trade_timestamp": "...",
        "posteriors": { ... PosteriorEstimate dicts ... }
    }

Note: particle clouds are NOT serialised (they are large and reproducible
from the prior).  Only the summary statistics (mean, std, CI) are saved.
On reload the SMC particles are re-initialised from the prior and
weighted to match the saved mean/std approximately -- good enough for a
warm start.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .priors import build_default_priors
from .posterior import PosteriorComputer, ParameterPosteriors, PosteriorEstimate
from .trade_batcher import TradeBatcher, TradeStats, Trade, load_live_trades

logger = logging.getLogger(__name__)

_HERE     = Path(__file__).resolve()
_REPO     = _HERE.parents[3]
_STATE_FILE = _REPO / "idea-engine" / "bayesian-updater" / "state.json"


# ---------------------------------------------------------------------------
# Drift flag
# ---------------------------------------------------------------------------

@dataclass
class DriftFlag:
    """
    Records that a parameter's posterior has drifted from its prior.

    Attributes
    ----------
    param_name     : name of the drifted parameter.
    prior_mean     : mean of the prior distribution.
    posterior_mean : current posterior mean.
    prior_std      : std of the prior.
    z_score        : (posterior_mean - prior_mean) / prior_std.
    direction      : "up" or "down".
    severity       : "moderate" (2-3 sigma) or "severe" (>3 sigma).
    """

    param_name:     str
    prior_mean:     float
    posterior_mean: float
    prior_std:      float
    z_score:        float
    direction:      str
    severity:       str


# ---------------------------------------------------------------------------
# BayesianUpdater
# ---------------------------------------------------------------------------

class BayesianUpdater:
    """
    Online Bayesian parameter updater for the IAE strategy genome.

    Parameters
    ----------
    priors          : dict of prior objects.  Defaults to build_default_priors().
    n_particles     : number of SMC particles (default 1000).
    drift_threshold : z-score threshold for flagging parameter drift.
    state_file      : path for JSON state persistence.
    db_path         : path to live_trades.db.
    """

    def __init__(
        self,
        priors: Optional[dict] = None,
        n_particles: int = 1000,
        drift_threshold: float = 2.0,
        state_file: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.priors           = priors or build_default_priors()
        self.n_particles      = n_particles
        self.drift_threshold  = drift_threshold
        self.state_file       = state_file or _STATE_FILE
        self.db_path          = db_path

        self._computer        = PosteriorComputer(
            priors=self.priors,
            n_particles=n_particles,
        )
        self._batcher         = TradeBatcher(db_path=db_path)

        self._last_update_ts: Optional[str]  = None
        self._n_total_trades: int             = 0
        self._last_trade_ts:  Optional[str]   = None
        self._current_posteriors: Optional[ParameterPosteriors] = None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update(self, new_trades: Optional[List[Trade]] = None) -> ParameterPosteriors:
        """
        Incorporate new trade evidence and return updated posteriors.

        Parameters
        ----------
        new_trades : optional pre-loaded list of Trade objects.  If None,
                     the updater loads from the database using
                     ``last_trade_timestamp`` as the cutoff so we never
                     re-process the same trades.

        Returns
        -------
        ParameterPosteriors -- updated estimates for all tracked parameters.
        """
        import datetime

        # Load trades if not provided
        if new_trades is None:
            new_trades = self._batcher.load_all(
                since_timestamp=self._last_trade_ts
            )

        if not new_trades:
            logger.info("No new trades to process.")
            return self._current_posteriors or self._empty_posteriors()

        trade_stats = TradeStats.from_trades(new_trades, tag="all")
        logger.info(
            "Updating posteriors with %d trades (win_rate=%.3f, avg_pnl=%.5f)",
            trade_stats.n_trades, trade_stats.win_rate, trade_stats.avg_pnl,
        )

        self._current_posteriors = self._computer.update(trade_stats)
        self._n_total_trades    += trade_stats.n_trades
        self._last_update_ts     = datetime.datetime.utcnow().isoformat()

        # Record the timestamp of the latest trade to avoid re-processing
        timestamps = [t.timestamp for t in new_trades if t.timestamp]
        if timestamps:
            self._last_trade_ts = max(timestamps)

        return self._current_posteriors

    def update_from_db(self) -> ParameterPosteriors:
        """
        Convenience method: load all new trades from DB and update.

        Uses the stored ``last_trade_ts`` cutoff so that only genuinely
        new trades are incorporated.
        """
        return self.update(new_trades=None)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self) -> List[DriftFlag]:
        """
        Compare current posterior means vs prior means.

        A parameter is flagged as drifted when::

            |posterior_mean - prior_mean| > drift_threshold * prior_std

        Returns
        -------
        List of DriftFlag objects (may be empty if no drift detected).
        """
        if self._current_posteriors is None:
            return []

        flags: List[DriftFlag] = []

        for name, prior in self.priors.items():
            if name not in self._current_posteriors:
                continue

            post    = self._current_posteriors[name]
            p_mean  = prior.mean
            p_std   = prior.std

            if p_std < 1e-10:
                continue

            z = (post.mean - p_mean) / p_std
            if abs(z) > self.drift_threshold:
                direction = "up" if z > 0 else "down"
                severity  = "severe" if abs(z) > 3.0 else "moderate"
                flag = DriftFlag(
                    param_name=name,
                    prior_mean=p_mean,
                    posterior_mean=post.mean,
                    prior_std=p_std,
                    z_score=z,
                    direction=direction,
                    severity=severity,
                )
                flags.append(flag)
                logger.warning(
                    "DRIFT [%s] %s z=%.2f: prior_mean=%.4f -> posterior_mean=%.4f",
                    severity.upper(), name, z, p_mean, post.mean,
                )

        return flags

    # ------------------------------------------------------------------
    # Parameter recommendations
    # ------------------------------------------------------------------

    def get_recommended_params(self) -> Dict[str, float]:
        """
        Return the posterior means as a parameter suggestion dict.

        The returned dict can be passed directly to ParamManager or used
        as a delta on top of BASELINE_PARAMS.

        Returns
        -------
        dict of {param_name: posterior_mean}.  Returns prior means if no
        update has been performed yet.
        """
        if self._current_posteriors is None:
            logger.info("No posterior update performed yet; returning prior means.")
            return {name: prior.mean for name, prior in self.priors.items()}

        return {
            name: est.mean
            for name, est in self._current_posteriors.estimates.items()
        }

    def get_uncertainty(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the 95% credible intervals for each tracked parameter.

        Returns
        -------
        dict of {param_name: (lo, hi)}.
        """
        if self._current_posteriors is None:
            return {name: prior.credible_interval() for name, prior in self.priors.items()}
        return {
            name: est.credible_interval_95
            for name, est in self._current_posteriors.estimates.items()
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[Path] = None) -> None:
        """
        Serialise updater state to a JSON file.

        Particles are NOT saved (too large; will be re-initialised).
        Only summary statistics and metadata are persisted.

        Parameters
        ----------
        path : override the default state file path.
        """
        p = path or self.state_file
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "last_update_timestamp": self._last_update_ts,
            "last_trade_timestamp":  self._last_trade_ts,
            "n_total_trades_seen":   self._n_total_trades,
            "drift_threshold":       self.drift_threshold,
            "posteriors":            {},
        }

        if self._current_posteriors:
            state["posteriors"] = {
                name: est.to_dict()
                for name, est in self._current_posteriors.estimates.items()
            }

        with open(p, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("Updater state saved to %s", p)

    def load_state(self, path: Optional[Path] = None) -> bool:
        """
        Restore updater state from a previously saved JSON file.

        Parameters
        ----------
        path : override the default state file path.

        Returns
        -------
        True if state was loaded successfully, False otherwise.
        """
        p = path or self.state_file
        if not p.exists():
            logger.info("No saved state at %s; starting from priors.", p)
            return False

        try:
            with open(p) as f:
                state = json.load(f)

            self._last_update_ts = state.get("last_update_timestamp")
            self._last_trade_ts  = state.get("last_trade_timestamp")
            self._n_total_trades = state.get("n_total_trades_seen", 0)

            if state.get("posteriors"):
                from .posterior import PosteriorEstimate, ParameterPosteriors
                estimates = {
                    k: PosteriorEstimate.from_dict(v)
                    for k, v in state["posteriors"].items()
                }
                self._current_posteriors = ParameterPosteriors(
                    estimates=estimates,
                    timestamp=state.get("last_update_timestamp", ""),
                    n_trades=self._n_total_trades,
                )

            logger.info(
                "Loaded updater state from %s (n_trades=%d, last_update=%s)",
                p, self._n_total_trades, self._last_update_ts,
            )
            return True

        except Exception as exc:
            logger.error("Failed to load state from %s: %s", p, exc)
            return False

    @classmethod
    def from_state_file(
        cls,
        path: Optional[Path] = None,
        **kwargs,
    ) -> "BayesianUpdater":
        """
        Create a BayesianUpdater and attempt to restore its state.

        Parameters
        ----------
        path   : path to the state JSON file.
        kwargs : passed through to __init__.

        Returns
        -------
        BayesianUpdater with state restored if available.
        """
        updater = cls(state_file=path, **kwargs)
        updater.load_state(path)
        return updater

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_posteriors(self) -> ParameterPosteriors:
        """Return a ParameterPosteriors containing prior estimates."""
        import datetime
        from .posterior import PosteriorEstimate, ParameterPosteriors

        estimates = {}
        for name, prior in self.priors.items():
            estimates[name] = PosteriorEstimate(
                param_name=name,
                mean=prior.mean,
                std=prior.std,
                credible_interval_95=prior.credible_interval(),
                method="prior",
            )
        return ParameterPosteriors(
            estimates=estimates,
            timestamp=datetime.datetime.utcnow().isoformat(),
            n_trades=0,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianUpdater("
            f"n_params={len(self.priors)}, "
            f"n_trades={self._n_total_trades}, "
            f"last_update={self._last_update_ts!r})"
        )
