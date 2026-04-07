"""
ml/training/hyperband.py

Hyperband early stopping algorithm for hyperparameter optimisation.

Hyperband is a principled early-stopping strategy that extends successive
halving (SHA) to multiple brackets, each with a different trade-off between
number of configurations and resources per configuration.

Reference:
  Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization", JMLR 2018.

Notation (matching the paper):
  R     -- max resource (iterations) per configuration
  eta   -- halving factor
  s_max = floor(log_eta(R))
  B     = (s_max + 1) * R   -- total budget

Usage example::

    def get_params(rng):
        return {
            "learning_rate": 10 ** rng.uniform(-4, -1),
            "n_units": int(rng.integers(32, 512)),
        }

    def train_and_eval(params, resource):
        model = build_model(**params)
        model.train(steps=resource)
        return model.val_loss()   # lower is better

    hb = Hyperband(get_params=get_params, train_and_eval=train_and_eval,
                   max_iter=81, eta=3, maximize=False)
    result = hb.run()
    print(result.best_config, result.best_val)
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """
    Result for a single (config, resource) evaluation.

    Attributes
    ----------
    trial_id  : unique identifier across all brackets and rounds
    bracket   : Hyperband bracket index
    round     : successive halving round within the bracket
    config    : hyperparameter dictionary
    resource  : number of units of resource used
    val_score : validation score (higher is better after sign flip if minimising)
    elapsed   : wall-clock seconds for this evaluation
    """

    trial_id: int
    bracket: int
    round: int
    config: Dict[str, Any]
    resource: int
    val_score: float
    elapsed: float = 0.0


@dataclass
class BracketResult:
    """
    Summary of one Hyperband bracket (fixed s value).

    Attributes
    ----------
    bracket          : s value
    configs_tried    : total configurations evaluated in this bracket
    best_config      : configuration with highest val_score in bracket
    best_val         : best val_score seen in this bracket
    trials           : list of all TrialResult objects in bracket
    """

    bracket: int
    configs_tried: int
    best_config: Dict[str, Any]
    best_val: float
    trials: List[TrialResult] = field(default_factory=list)


@dataclass
class BestResult:
    """
    Overall best result across all Hyperband brackets.

    Attributes
    ----------
    config           : best hyperparameter configuration found
    val_score        : best validation score
    total_iters_used : total resource units consumed
    n_configs        : total configurations evaluated
    n_brackets       : number of brackets run
    bracket_results  : per-bracket BracketResult objects
    """

    config: Dict[str, Any]
    val_score: float
    total_iters_used: int
    n_configs: int
    n_brackets: int
    bracket_results: List[BracketResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ResourceSchedule  # compute allocation table
# ---------------------------------------------------------------------------

@dataclass
class ResourceSchedule:
    """
    Pre-computes the resource allocation table for all Hyperband brackets.

    Parameters
    ----------
    max_iter : int
        Maximum resource units per configuration (R in the paper).
    eta : int
        Halving factor (default 3).
    """

    max_iter: int
    eta: int

    def __post_init__(self):
        self.s_max = math.floor(math.log(self.max_iter) / math.log(self.eta))
        self.total_budget = (self.s_max + 1) * self.max_iter

    def bracket_schedule(self, s: int) -> List[Dict[str, int]]:
        """
        Return the successive halving schedule for bracket s.

        Each entry is {"round": i, "n_configs": n_i, "resource": r_i}.
        """
        # n = ceil((s_max+1)/(s+1)) * eta^s
        n = math.ceil((self.s_max + 1) / (s + 1)) * (self.eta ** s)
        schedule = []
        for i in range(s + 1):
            n_i = math.floor(n * (self.eta ** (-i)))
            r_i = self.max_iter * (self.eta ** (i - s))
            r_i = max(1, int(r_i))
            schedule.append(
                {"round": i, "n_configs": max(1, n_i), "resource": r_i}
            )
        return schedule

    def full_schedule(self) -> List[Dict[str, Any]]:
        """Return full schedule across all brackets."""
        rows = []
        for s in range(self.s_max, -1, -1):
            for entry in self.bracket_schedule(s):
                rows.append({"bracket": s, **entry})
        return rows

    def total_resource_usage(self) -> int:
        """Estimate total resource units consumed across all brackets."""
        total = 0
        for s in range(self.s_max, -1, -1):
            for entry in self.bracket_schedule(s):
                total += entry["n_configs"] * entry["resource"]
        return total

    def __repr__(self) -> str:
        return (
            f"ResourceSchedule(max_iter={self.max_iter}, eta={self.eta}, "
            f"s_max={self.s_max}, est_total_resource={self.total_resource_usage()})"
        )


# ---------------------------------------------------------------------------
# HyperbandLogger
# ---------------------------------------------------------------------------

class HyperbandLogger:
    """
    Tracks all trial results for post-run analysis.

    Parameters
    ----------
    maximize : bool
        True if higher val_score is better (e.g. accuracy, Sharpe).
        False if lower is better (e.g. loss).
    """

    def __init__(self, maximize: bool = True):
        self.maximize = maximize
        self._trials: List[TrialResult] = []
        self._trial_counter = 0

    def log(self, trial: TrialResult) -> None:
        self._trials.append(trial)
        self._trial_counter += 1
        logger.debug(
            "Trial %d | bracket=%d round=%d resource=%d score=%.6f",
            trial.trial_id,
            trial.bracket,
            trial.round,
            trial.resource,
            trial.val_score,
        )

    @property
    def all_trials(self) -> List[TrialResult]:
        return list(self._trials)

    @property
    def best_trial(self) -> Optional[TrialResult]:
        if not self._trials:
            return None
        return max(self._trials, key=lambda t: t.val_score)

    def trials_dataframe(self):
        """Return all trials as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "trial_id": t.trial_id,
                    "bracket": t.bracket,
                    "round": t.round,
                    "resource": t.resource,
                    "val_score": t.val_score,
                    "elapsed": t.elapsed,
                    **{f"param_{k}": v for k, v in t.config.items()},
                }
                for t in self._trials
            ]
        )

    def bracket_summary(self) -> "pd.DataFrame":
        """Per-bracket statistics."""
        import pandas as pd

        df = self.trials_dataframe()
        if df.empty:
            return df

        return (
            df.groupby("bracket")
            .agg(
                n_trials=("trial_id", "count"),
                best_score=("val_score", "max"),
                mean_score=("val_score", "mean"),
                total_resource=("resource", "sum"),
            )
            .reset_index()
        )

    def next_trial_id(self) -> int:
        return self._trial_counter


# ---------------------------------------------------------------------------
# SuccessiveHalving  # inner loop
# ---------------------------------------------------------------------------

class SuccessiveHalving:
    """
    Successive halving: given n_configs configurations and r_min starting
    resource, repeatedly evaluate and keep the top 1/eta.

    Parameters
    ----------
    get_params : Callable[[], dict]
        Samples a hyperparameter configuration.
    train_and_eval : Callable[[dict, int], float]
        Trains a model with given config for `resource` units,
        returns a validation score.
    eta : int
        Halving rate.
    maximize : bool
        True if higher score is better.
    logger : HyperbandLogger
    bracket : int
        Current Hyperband bracket index (for logging).
    """

    def __init__(
        self,
        get_params: Callable,
        train_and_eval: Callable,
        eta: int,
        maximize: bool,
        logger: HyperbandLogger,
        bracket: int,
        rng: np.random.Generator,
    ):
        self.get_params = get_params
        self.train_and_eval = train_and_eval
        self.eta = eta
        self.maximize = maximize
        self.logger = logger
        self.bracket = bracket
        self.rng = rng

    def run(
        self, n: int, r: int, n_rounds: int
    ) -> Tuple[List[Dict[str, Any]], BracketResult]:
        """
        Run successive halving starting with n configs at resource r.

        Parameters
        ----------
        n : int
            Number of starting configurations.
        r : int
            Starting resource per configuration.
        n_rounds : int
            Total number of SHA rounds (= s+1 for bracket s).

        Returns
        -------
        (surviving_configs, BracketResult)
        """
        # sample initial configurations
        configs = [self.get_params(self.rng) for _ in range(n)]
        bracket_trials: List[TrialResult] = []

        for round_idx in range(n_rounds):
            resource = int(r * (self.eta ** round_idx))
            scored: List[Tuple[float, Dict]] = []

            for cfg in configs:
                t0 = time.perf_counter()
                score = self.train_and_eval(cfg, resource)
                elapsed = time.perf_counter() - t0

                # invert if minimising so we can always keep max
                stored_score = score if self.maximize else -score

                trial = TrialResult(
                    trial_id=self.logger.next_trial_id(),
                    bracket=self.bracket,
                    round=round_idx,
                    config=cfg,
                    resource=resource,
                    val_score=stored_score,
                    elapsed=elapsed,
                )
                self.logger.log(trial)
                bracket_trials.append(trial)
                scored.append((stored_score, cfg))

            # keep top 1/eta configurations
            scored.sort(key=lambda x: x[0], reverse=True)
            n_keep = max(1, math.floor(len(scored) / self.eta))
            configs = [cfg for _, cfg in scored[:n_keep]]

        best_score, best_cfg = max(scored, key=lambda x: x[0])

        result = BracketResult(
            bracket=self.bracket,
            configs_tried=len(bracket_trials),
            best_config=best_cfg,
            best_val=best_score if self.maximize else -best_score,
            trials=bracket_trials,
        )
        return configs, result


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------

class Hyperband:
    """
    Hyperband: successive halving / Hyperband algorithm for hyperparameter
    optimisation.

    Efficiently allocates compute budget by early-stopping bad configurations
    across multiple brackets.

        R = 81    -- max resource per configuration (default)
        eta = 3   -- halving factor (default)

    Parameters
    ----------
    get_params : Callable[[np.random.Generator], dict]
        Function that takes an RNG and returns a hyperparameter dict.
    train_and_eval : Callable[[dict, int], float]
        Function that trains a model with the given params for `resource`
        units (e.g. epochs) and returns a validation score.
    max_iter : int
        Maximum resource per configuration (R).  Should be a power of eta.
    eta : int
        Halving factor; recommended value 3.
    maximize : bool
        True if higher val_score is better.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        get_params: Callable,
        train_and_eval: Callable,
        max_iter: int = 81,
        eta: int = 3,
        maximize: bool = True,
        seed: Optional[int] = None,
    ):
        self.get_params = get_params
        self.train_and_eval = train_and_eval
        self.max_iter = max_iter
        self.eta = eta
        self.maximize = maximize
        self.seed = seed

        self._schedule = ResourceSchedule(max_iter=max_iter, eta=eta)
        self._logger = HyperbandLogger(maximize=maximize)
        self._rng = np.random.default_rng(seed)

    @property
    def schedule(self) -> ResourceSchedule:
        return self._schedule

    @property
    def logger(self) -> HyperbandLogger:
        return self._logger

    def run(self) -> BestResult:
        """
        Execute the Hyperband algorithm.

        Outer loop: iterate brackets from s_max down to 0.
        Each bracket runs successive halving with a different
        n_configs / r_min trade-off.

        Returns
        -------
        BestResult with best config, score, and run statistics.
        """
        s_max = self._schedule.s_max
        bracket_results: List[BracketResult] = []
        total_iters = 0
        total_configs = 0

        # iterate brackets from most-configurations (s_max) to fewest
        for s in range(s_max, -1, -1):
            # number of initial configurations for this bracket
            n = math.ceil(
                (s_max + 1) / (s + 1) * (self.eta ** s)
            )
            # minimum resource for this bracket
            r = self.max_iter * (self.eta ** (-s))
            r = max(1, int(r))

            logger.info(
                "Hyperband bracket s=%d: n=%d configs, r_min=%d, rounds=%d",
                s, n, r, s + 1,
            )

            sha = SuccessiveHalving(
                get_params=self.get_params,
                train_and_eval=self.train_and_eval,
                eta=self.eta,
                maximize=self.maximize,
                logger=self._logger,
                bracket=s,
                rng=self._rng,
            )

            _, bracket_result = sha.run(n=n, r=r, n_rounds=s + 1)
            bracket_results.append(bracket_result)

            # accumulate statistics
            for trial in bracket_result.trials:
                total_iters += trial.resource
            total_configs += bracket_result.configs_tried

        # find overall best across all brackets
        best_trial = self._logger.best_trial
        if best_trial is None:
            raise RuntimeError("No trials were run  # check get_params and train_and_eval.")

        return BestResult(
            config=best_trial.config,
            val_score=best_trial.val_score if self.maximize else -best_trial.val_score,
            total_iters_used=total_iters,
            n_configs=total_configs,
            n_brackets=len(bracket_results),
            bracket_results=bracket_results,
        )

    def __repr__(self) -> str:
        return (
            f"Hyperband(max_iter={self.max_iter}, eta={self.eta}, "
            f"maximize={self.maximize}, "
            f"s_max={self._schedule.s_max}, "
            f"est_budget={self._schedule.total_resource_usage()})"
        )
