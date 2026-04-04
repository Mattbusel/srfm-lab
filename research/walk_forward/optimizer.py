"""
research/walk_forward/optimizer.py
────────────────────────────────────
Multi-backend parameter optimizer for walk-forward analysis.

Supports:
  • Grid Search        — exhaustive combinatorial scan
  • Random Search      — Monte Carlo sampling from param space
  • Sobol Search       — quasi-random low-discrepancy sequences
  • Bayesian Optimization — Gaussian Process + Expected Improvement

All backends share the same interface: strategy_fn × param_space × splitter → OptResult.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .splits import WFSplit, CPCVSplitter
from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    profit_factor,
    win_rate,
    max_drawdown,
    PerformanceStats,
    compute_performance_stats,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trial:
    """Single parameter evaluation trial."""
    trial_id:    int
    params:      Dict[str, Any]
    score:       float
    is_score:    float = 0.0
    oos_score:   float = 0.0
    n_trades:    int   = 0
    elapsed_sec: float = 0.0
    error:       Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and np.isfinite(self.score)


@dataclass
class OptResult:
    """
    Result from parameter optimization.

    Attributes
    ----------
    best_params        : parameter dict with highest OOS score.
    best_score         : best score achieved.
    all_trials         : all evaluated trials (sorted by score descending).
    param_importance   : dict of param_name → importance score in [0, 1].
    convergence_curve  : list of best-so-far scores by trial number.
    method             : optimization method name.
    metric             : scoring metric name.
    elapsed_sec        : total optimization time.
    n_trials           : total number of trials evaluated.
    """
    best_params:       Dict[str, Any]
    best_score:        float
    all_trials:        List[Trial]
    param_importance:  Dict[str, float]
    convergence_curve: List[float]
    method:            str
    metric:            str
    elapsed_sec:       float = 0.0
    n_trials:          int   = 0

    def top_k(self, k: int = 10) -> List[Trial]:
        """Return top-k trials by score."""
        return sorted(self.all_trials, key=lambda t: t.score, reverse=True)[:k]

    def param_values_for(self, param: str) -> Tuple[List, List[float]]:
        """Return (param_values, scores) for a given parameter across all trials."""
        vals   = [t.params.get(param) for t in self.all_trials if t.success]
        scores = [t.score             for t in self.all_trials if t.success]
        return (vals, scores)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trials to DataFrame."""
        rows = []
        for t in self.all_trials:
            row = dict(t.params)
            row["score"]       = t.score
            row["trial_id"]    = t.trial_id
            row["n_trades"]    = t.n_trades
            row["elapsed_sec"] = t.elapsed_sec
            row["error"]       = t.error
            rows.append(row)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ParamSpace definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContinuousParam:
    """A continuous parameter with uniform distribution in [low, high]."""
    name: str
    low:  float
    high: float
    log:  bool = False      # if True, sample in log-space

    def sample(self, u: float) -> float:
        """Map uniform [0,1] → parameter value."""
        if self.log:
            log_low  = math.log(self.low)
            log_high = math.log(self.high)
            return math.exp(log_low + u * (log_high - log_low))
        return self.low + u * (self.high - self.low)


@dataclass
class DiscreteParam:
    """A discrete parameter choosing from a list of values."""
    name:   str
    values: List[Any]

    def sample(self, u: float) -> Any:
        """Map uniform [0,1] → one of the discrete values."""
        idx = int(u * len(self.values))
        idx = min(idx, len(self.values) - 1)
        return self.values[idx]


ParamSpec = Union[ContinuousParam, DiscreteParam]


def build_param_space(spec: Dict[str, Any]) -> List[ParamSpec]:
    """
    Build a list of ParamSpec from a dict specification.

    Dict format:
      {'cf': (0.001, 0.003),          # continuous uniform
       'cf_log': (0.001, 0.003, True), # continuous log-space
       'bh_form': [1.2, 1.5, 2.0],    # discrete
      }
    """
    specs: List[ParamSpec] = []
    for name, val in spec.items():
        if isinstance(val, list):
            specs.append(DiscreteParam(name=name, values=val))
        elif isinstance(val, tuple):
            if len(val) == 2:
                specs.append(ContinuousParam(name=name, low=val[0], high=val[1]))
            elif len(val) == 3:
                specs.append(ContinuousParam(name=name, low=val[0], high=val[1], log=bool(val[2])))
            else:
                raise ValueError(f"Tuple for '{name}' must be (low, high) or (low, high, log_bool)")
        else:
            raise ValueError(f"Unsupported param spec for '{name}': {type(val)}")
    return specs


def sample_params(param_specs: List[ParamSpec], u_vector: np.ndarray) -> Dict[str, Any]:
    """Map a unit hypercube point u_vector → parameter dict."""
    if len(u_vector) != len(param_specs):
        raise ValueError(f"u_vector length {len(u_vector)} ≠ n_params {len(param_specs)}")
    return {spec.name: spec.sample(float(u)) for spec, u in zip(param_specs, u_vector)}


# ─────────────────────────────────────────────────────────────────────────────
# Shared scorer
# ─────────────────────────────────────────────────────────────────────────────

def _score_result(
    trades_list: List[Dict],
    metric:      str,
    starting_equity: float = 100_000.0,
) -> float:
    """Compute a scalar score from a list of trade dicts."""
    if not trades_list:
        return -np.inf

    pnl_arr = np.array([t.get("pnl", 0.0) for t in trades_list], dtype=np.float64)
    pnl_arr = pnl_arr[np.isfinite(pnl_arr)]

    if len(pnl_arr) == 0:
        return -np.inf

    pos_arr = np.array([t.get("dollar_pos", 1.0) for t in trades_list], dtype=np.float64)
    pos_arr = np.where(np.abs(pos_arr) < 1e-6, 1.0, pos_arr)
    ret_arr = pnl_arr / pos_arr
    ret_arr = np.where(np.isfinite(ret_arr), ret_arr, 0.0)

    if metric == "sharpe":
        return sharpe_ratio(ret_arr)
    elif metric == "sortino":
        return sortino_ratio(ret_arr)
    elif metric == "calmar":
        eq  = starting_equity + np.cumsum(pnl_arr)
        eq  = np.concatenate([[starting_equity], eq])
        mdd = max_drawdown(eq)
        return calmar_ratio(ret_arr, max_dd=mdd)
    elif metric == "profit_factor":
        return profit_factor(pnl_arr)
    elif metric == "win_rate":
        return win_rate(pnl_arr)
    elif metric == "total_pnl":
        return float(np.sum(pnl_arr))
    elif metric == "expectancy":
        return float(np.mean(pnl_arr))
    else:
        raise ValueError(f"Unknown metric: '{metric}'")


def _call_and_score(
    strategy_fn:    Callable,
    params:         Dict[str, Any],
    splits:         List[WFSplit],
    trades:         pd.DataFrame,
    metric:         str,
    starting_equity: float = 100_000.0,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate strategy on IS splits and return mean IS score + IS trade list.
    """
    all_is_trades: List[Dict] = []
    scores: List[float] = []

    for split in splits:
        train = trades.iloc[split.train_idx].copy().reset_index(drop=True)
        if len(train) < 5:
            continue
        try:
            raw = strategy_fn(train, params)
            if isinstance(raw, pd.DataFrame):
                t_list = raw.to_dict(orient="records")
            elif isinstance(raw, (list, tuple)):
                t_list = [t if isinstance(t, dict) else vars(t) for t in raw]
            else:
                t_list = []

            sc = _score_result(t_list, metric, starting_equity)
            if np.isfinite(sc):
                scores.append(sc)
                all_is_trades.extend(t_list)
        except Exception as e:
            logger.debug("_call_and_score error: %s", e)
            continue

    mean_score = float(np.mean(scores)) if scores else -np.inf
    return mean_score, all_is_trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    strategy_fn:     Callable,
    param_grid:      Dict[str, List[Any]],
    trades:          pd.DataFrame,
    splitter:        List[WFSplit],
    metric:          str   = "sharpe",
    starting_equity: float = 100_000.0,
    verbose:         bool  = True,
) -> OptResult:
    """
    Exhaustive grid search over all parameter combinations.

    Evaluates every combination in the Cartesian product of param_grid on the
    IS portions of the provided splits. Selects the combination with highest
    mean IS score.

    Parameters
    ----------
    strategy_fn     : callable(train_trades, params) → List[dict].
    param_grid      : dict of param_name → list of values.
    trades          : full trades DataFrame.
    splitter        : list of WFSplit objects for IS/OOS partitioning.
    metric          : scoring metric ('sharpe', 'sortino', 'calmar', 'profit_factor').
    starting_equity : initial equity for performance calculations.
    verbose         : log progress.

    Returns
    -------
    OptResult

    Examples
    --------
    >>> result = grid_search(my_fn, {'cf': [0.001, 0.002], 'bh_form': [1.2, 1.5]},
    ...                      trades, splits)
    >>> print(result.best_params, result.best_score)
    """
    t0   = time.perf_counter()
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_total = len(combos)

    if verbose:
        logger.info("Grid search: %d combinations, metric='%s'", n_total, metric)

    trials: List[Trial] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    convergence: List[float] = []

    for trial_id, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        t_start = time.perf_counter()
        try:
            score, _ = _call_and_score(strategy_fn, params, splitter, trades, metric, starting_equity)
            err = None
        except Exception as e:
            score = -np.inf
            err   = str(e)

        trial = Trial(
            trial_id    = trial_id,
            params      = params,
            score       = score,
            elapsed_sec = time.perf_counter() - t_start,
            error       = err,
        )
        trials.append(trial)

        if score > best_score:
            best_score  = score
            best_params = params

        convergence.append(best_score)

        if verbose and (trial_id + 1) % max(1, n_total // 10) == 0:
            logger.info(
                "Grid search: %d/%d | best=%.4f | current=%.4f | params=%s",
                trial_id + 1, n_total, best_score, score, params,
            )

    elapsed = time.perf_counter() - t0
    importance = param_importance_from_trials(trials, param_grid)

    return OptResult(
        best_params       = best_params,
        best_score        = best_score,
        all_trials        = sorted(trials, key=lambda t: t.score, reverse=True),
        param_importance  = importance,
        convergence_curve = convergence,
        method            = "grid_search",
        metric            = metric,
        elapsed_sec       = elapsed,
        n_trials          = len(trials),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Search
# ─────────────────────────────────────────────────────────────────────────────

def random_search(
    strategy_fn:     Callable,
    param_space:     Union[Dict[str, Any], List[ParamSpec]],
    trades:          pd.DataFrame,
    splitter:        List[WFSplit],
    n_iter:          int   = 100,
    metric:          str   = "sharpe",
    starting_equity: float = 100_000.0,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> OptResult:
    """
    Random search over the parameter space.

    Samples `n_iter` parameter configurations uniformly at random from the
    specified parameter space. More efficient than grid search when the number
    of parameters is large (Bergstra & Bengio, 2012).

    Parameters
    ----------
    strategy_fn  : callable(train_trades, params) → List[dict].
    param_space  : either a dict spec (see build_param_space) or List[ParamSpec].
    trades       : full trades DataFrame.
    splitter     : list of WFSplit objects.
    n_iter       : number of random samples.
    metric       : scoring metric.
    seed         : random seed for reproducibility.

    Returns
    -------
    OptResult

    References
    ----------
    Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter
    Optimization. JMLR.
    """
    if isinstance(param_space, dict):
        param_specs = build_param_space(param_space)
    else:
        param_specs = param_space

    rng  = np.random.default_rng(seed)
    n_dims = len(param_specs)

    t0 = time.perf_counter()
    if verbose:
        logger.info("Random search: %d iterations, metric='%s'", n_iter, metric)

    trials:     List[Trial] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    convergence: List[float] = []

    for trial_id in range(n_iter):
        u      = rng.uniform(0.0, 1.0, size=n_dims)
        params = sample_params(param_specs, u)
        t_start = time.perf_counter()

        try:
            score, _ = _call_and_score(strategy_fn, params, splitter, trades, metric, starting_equity)
            err = None
        except Exception as e:
            score = -np.inf
            err   = str(e)

        trial = Trial(
            trial_id    = trial_id,
            params      = params,
            score       = score,
            elapsed_sec = time.perf_counter() - t_start,
            error       = err,
        )
        trials.append(trial)

        if score > best_score:
            best_score  = score
            best_params = params

        convergence.append(best_score)

        if verbose and (trial_id + 1) % max(1, n_iter // 10) == 0:
            logger.info(
                "Random search: %d/%d | best=%.4f | current=%.4f",
                trial_id + 1, n_iter, best_score, score,
            )

    elapsed = time.perf_counter() - t0
    importance = param_importance_from_specs(trials, param_specs)

    return OptResult(
        best_params       = best_params,
        best_score        = best_score,
        all_trials        = sorted(trials, key=lambda t: t.score, reverse=True),
        param_importance  = importance,
        convergence_curve = convergence,
        method            = "random_search",
        metric            = metric,
        elapsed_sec       = elapsed,
        n_trials          = len(trials),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sobol Search
# ─────────────────────────────────────────────────────────────────────────────

def sobol_search(
    strategy_fn:     Callable,
    param_space:     Union[Dict[str, Any], List[ParamSpec]],
    trades:          pd.DataFrame,
    splitter:        List[WFSplit],
    n_iter:          int   = 200,
    metric:          str   = "sharpe",
    starting_equity: float = 100_000.0,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> OptResult:
    """
    Quasi-random search using Sobol sequences.

    Sobol sequences provide better coverage of the parameter space than pure
    random sampling by using low-discrepancy quasi-random numbers. This
    typically requires fewer evaluations to find a good solution.

    The Sobol sequence is generated via scipy.stats.qmc.Sobol and mapped to
    the parameter space via quantile transformation.

    Parameters
    ----------
    strategy_fn  : callable(train_trades, params) → List[dict].
    param_space  : dict spec or List[ParamSpec].
    trades       : full trades DataFrame.
    splitter     : list of WFSplit objects.
    n_iter       : number of Sobol samples (should be a power of 2 for best coverage).
    metric       : scoring metric.
    seed         : random seed.

    Returns
    -------
    OptResult

    References
    ----------
    Sobol, I.M. (1967). On the distribution of points in a cube and the
    approximate evaluation of integrals. USSR Computational Mathematics and
    Mathematical Physics.
    """
    if isinstance(param_space, dict):
        param_specs = build_param_space(param_space)
    else:
        param_specs = param_space

    n_dims = len(param_specs)

    # Generate Sobol sequence
    try:
        from scipy.stats.qmc import Sobol
        sampler   = Sobol(d=n_dims, scramble=True, seed=seed)
        # Round up n_iter to next power of 2 for Sobol efficiency
        n_sobol   = 2 ** math.ceil(math.log2(max(n_iter, 2)))
        u_matrix  = sampler.random(n_sobol)[:n_iter]
    except ImportError:
        warnings.warn(
            "scipy.stats.qmc not available — falling back to random search",
            ImportWarning, stacklevel=2,
        )
        return random_search(
            strategy_fn, param_specs, trades, splitter,
            n_iter=n_iter, metric=metric, starting_equity=starting_equity,
            seed=seed, verbose=verbose,
        )

    t0 = time.perf_counter()
    if verbose:
        logger.info("Sobol search: %d iterations, metric='%s'", n_iter, metric)

    trials:     List[Trial] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    convergence: List[float] = []

    for trial_id, u in enumerate(u_matrix):
        params  = sample_params(param_specs, u)
        t_start = time.perf_counter()

        try:
            score, _ = _call_and_score(strategy_fn, params, splitter, trades, metric, starting_equity)
            err = None
        except Exception as e:
            score = -np.inf
            err   = str(e)

        trial = Trial(
            trial_id    = trial_id,
            params      = params,
            score       = score,
            elapsed_sec = time.perf_counter() - t_start,
            error       = err,
        )
        trials.append(trial)

        if score > best_score:
            best_score  = score
            best_params = params

        convergence.append(best_score)

        if verbose and (trial_id + 1) % max(1, n_iter // 10) == 0:
            logger.info(
                "Sobol search: %d/%d | best=%.4f | current=%.4f",
                trial_id + 1, n_iter, best_score, score,
            )

    elapsed = time.perf_counter() - t0
    importance = param_importance_from_specs(trials, param_specs)

    return OptResult(
        best_params       = best_params,
        best_score        = best_score,
        all_trials        = sorted(trials, key=lambda t: t.score, reverse=True),
        param_importance  = importance,
        convergence_curve = convergence,
        method            = "sobol_search",
        metric            = metric,
        elapsed_sec       = elapsed,
        n_trials          = len(trials),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bayesian Optimization
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_opt(
    strategy_fn:     Callable,
    param_space:     Union[Dict[str, Any], List[ParamSpec]],
    trades:          pd.DataFrame,
    splitter:        List[WFSplit],
    n_iter:          int   = 50,
    n_init:          int   = 10,
    metric:          str   = "sharpe",
    acquisition:     str   = "ei",   # 'ei' or 'ucb'
    xi:              float = 0.01,   # EI exploration parameter
    kappa:           float = 2.576,  # UCB exploration parameter
    starting_equity: float = 100_000.0,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> OptResult:
    """
    Bayesian optimization with Gaussian Process surrogate model.

    Uses a GP surrogate model trained on observed (params, score) pairs to
    predict where in the parameter space the score is likely to be highest.
    The acquisition function balances exploration vs exploitation.

    Algorithm:
    1. Evaluate n_init random points (warm-up).
    2. Fit GP on observed points.
    3. Maximize acquisition function to select next point.
    4. Evaluate strategy at next point.
    5. Update GP and repeat for n_iter - n_init steps.

    Parameters
    ----------
    strategy_fn  : callable(train_trades, params) → List[dict].
    param_space  : dict spec or List[ParamSpec].
    trades       : full trades DataFrame.
    splitter     : list of WFSplit objects.
    n_iter       : total evaluations (including warm-up).
    n_init       : number of warm-up random evaluations.
    metric       : scoring metric.
    acquisition  : 'ei' (Expected Improvement) or 'ucb' (Upper Confidence Bound).
    xi           : EI exploration-exploitation trade-off (higher = more explore).
    kappa        : UCB exploration parameter.
    seed         : random seed.

    Returns
    -------
    OptResult

    References
    ----------
    Snoek, J., Larochelle, H., & Adams, R.P. (2012). Practical Bayesian
    Optimization of Machine Learning Algorithms. NIPS.
    """
    if isinstance(param_space, dict):
        param_specs = build_param_space(param_space)
    else:
        param_specs = param_space

    n_dims = len(param_specs)
    rng    = np.random.default_rng(seed)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        from scipy.stats import norm as scipy_norm
        from scipy.optimize import minimize as scipy_minimize
    except ImportError as e:
        warnings.warn(
            f"Bayesian optimization requires scikit-learn and scipy: {e}. "
            "Falling back to Sobol search.",
            ImportWarning, stacklevel=2,
        )
        return sobol_search(
            strategy_fn, param_specs, trades, splitter,
            n_iter=n_iter, metric=metric, starting_equity=starting_equity,
            seed=seed, verbose=verbose,
        )

    t0 = time.perf_counter()
    if verbose:
        logger.info(
            "Bayesian opt: %d iters (%d warm-up), acquisition='%s', metric='%s'",
            n_iter, n_init, acquisition, metric,
        )

    # GP kernel: Matern 5/2 + white noise
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-3, 10.0), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=int(seed),
    )

    trials:     List[Trial] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    convergence: List[float] = []

    # Observed points in unit hypercube
    X_obs: List[np.ndarray] = []
    y_obs: List[float]      = []

    def _evaluate(u: np.ndarray, trial_id: int) -> Trial:
        params  = sample_params(param_specs, np.clip(u, 0.0, 1.0))
        t_start = time.perf_counter()
        try:
            score, _ = _call_and_score(strategy_fn, params, splitter, trades, metric, starting_equity)
            err = None
        except Exception as e:
            score = -np.inf
            err   = str(e)
        return Trial(
            trial_id    = trial_id,
            params      = params,
            score       = score,
            elapsed_sec = time.perf_counter() - t_start,
            error       = err,
        )

    # ── Warm-up phase: random sampling ──────────────────────────────────────
    n_init_actual = min(n_init, n_iter)
    for trial_id in range(n_init_actual):
        u     = rng.uniform(0.0, 1.0, size=n_dims)
        trial = _evaluate(u, trial_id)
        trials.append(trial)

        if trial.success:
            X_obs.append(u)
            y_obs.append(trial.score)

        if trial.score > best_score:
            best_score  = trial.score
            best_params = trial.params

        convergence.append(best_score)

        if verbose:
            logger.info(
                "Bayes warm-up %d/%d: score=%.4f | best=%.4f",
                trial_id + 1, n_init_actual, trial.score, best_score,
            )

    # ── Bayesian phase ──────────────────────────────────────────────────────
    for trial_id in range(n_init_actual, n_iter):
        if len(X_obs) < 3:
            # Not enough observations — fall back to random
            u = rng.uniform(0.0, 1.0, size=n_dims)
        else:
            # Fit GP
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_arr, y_arr)

            # Maximize acquisition function
            u = _maximize_acquisition(
                gp=gp, n_dims=n_dims, acquisition=acquisition,
                y_best=best_score, xi=xi, kappa=kappa, rng=rng,
            )

        trial = _evaluate(u, trial_id)
        trials.append(trial)

        if trial.success:
            X_obs.append(u)
            y_obs.append(trial.score)

        if trial.score > best_score:
            best_score  = trial.score
            best_params = trial.params

        convergence.append(best_score)

        if verbose:
            logger.info(
                "Bayes iter %d/%d: score=%.4f | best=%.4f | params=%s",
                trial_id + 1, n_iter, trial.score, best_score,
                {k: round(v, 4) if isinstance(v, float) else v for k, v in trial.params.items()},
            )

    elapsed    = time.perf_counter() - t0
    importance = param_importance_from_specs(trials, param_specs)

    return OptResult(
        best_params       = best_params,
        best_score        = best_score,
        all_trials        = sorted(trials, key=lambda t: t.score, reverse=True),
        param_importance  = importance,
        convergence_curve = convergence,
        method            = "bayesian_opt",
        metric            = metric,
        elapsed_sec       = elapsed,
        n_trials          = len(trials),
    )


def _maximize_acquisition(
    gp:          object,
    n_dims:      int,
    acquisition: str,
    y_best:      float,
    xi:          float,
    kappa:       float,
    rng:         np.random.Generator,
    n_restarts:  int = 10,
) -> np.ndarray:
    """
    Find the point in [0,1]^n_dims that maximises the acquisition function.

    Uses multi-start L-BFGS-B optimization.

    Parameters
    ----------
    gp          : fitted GaussianProcessRegressor.
    n_dims      : dimensionality.
    acquisition : 'ei' or 'ucb'.
    y_best      : current best observed score.
    xi          : EI exploration parameter.
    kappa       : UCB exploration parameter.
    rng         : numpy random generator for starting points.
    n_restarts  : number of random restarts for optimizer.

    Returns
    -------
    numpy array of shape (n_dims,) in [0, 1]^n_dims.
    """
    from scipy.optimize import minimize as scipy_minimize
    from scipy.stats import norm as scipy_norm

    def neg_acq(u: np.ndarray) -> float:
        u_2d = u.reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(u_2d, return_std=True)
        mu    = float(mu[0])
        sigma = float(sigma[0])

        if sigma < 1e-10:
            return 0.0

        if acquisition == "ei":
            z    = (mu - y_best - xi) / sigma
            acq  = (mu - y_best - xi) * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
            return -float(acq)
        elif acquisition == "ucb":
            return -(mu + kappa * sigma)
        else:
            raise ValueError(f"Unknown acquisition: '{acquisition}'")

    best_u    = rng.uniform(0.0, 1.0, size=n_dims)
    best_val  = neg_acq(best_u)
    bounds    = [(0.0, 1.0)] * n_dims

    for _ in range(n_restarts):
        x0 = rng.uniform(0.0, 1.0, size=n_dims)
        try:
            res = scipy_minimize(neg_acq, x0, method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_u   = res.x
        except Exception:
            pass

    return np.clip(best_u, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Importance
# ─────────────────────────────────────────────────────────────────────────────

def param_importance(opt_result: OptResult) -> Dict[str, float]:
    """
    Return parameter importance scores from an OptResult.

    Wrapper that re-computes importance if needed or returns the stored value.

    Parameters
    ----------
    opt_result : OptResult from any optimizer.

    Returns
    -------
    Dict of param_name → importance in [0, 1].
    """
    return opt_result.param_importance


def param_importance_from_trials(
    trials:     List[Trial],
    param_grid: Dict[str, List[Any]],
) -> Dict[str, float]:
    """
    Compute variance-based parameter importance from discrete grid trials.

    Importance(param) = std(mean_score_per_param_value) / total_score_std.

    Parameters
    ----------
    trials     : list of Trial objects.
    param_grid : original parameter grid.

    Returns
    -------
    Dict of param_name → importance in [0, 1].
    """
    successful = [t for t in trials if t.success]
    if len(successful) < 3:
        return {k: 1.0 / len(param_grid) for k in param_grid}

    importances: Dict[str, float] = {}

    for param, values in param_grid.items():
        # Group trials by this param's value
        value_scores: Dict[Any, List[float]] = {}
        for t in successful:
            v = t.params.get(param)
            if v is not None:
                value_scores.setdefault(v, []).append(t.score)

        # Variance of group means
        group_means = [np.mean(scores) for scores in value_scores.values()]
        if len(group_means) >= 2:
            importances[param] = float(np.std(group_means))
        else:
            importances[param] = 0.0

    # Normalise to [0, 1]
    total = sum(importances.values())
    if total > 1e-12:
        importances = {k: v / total for k, v in importances.items()}
    else:
        importances = {k: 1.0 / len(importances) for k in importances}

    return importances


def param_importance_from_specs(
    trials:      List[Trial],
    param_specs: List[ParamSpec],
) -> Dict[str, float]:
    """
    Compute variance-based parameter importance from continuous param specs.

    Uses correlation between param value and score as a proxy for importance.

    Parameters
    ----------
    trials      : list of Trial objects.
    param_specs : list of ParamSpec objects.

    Returns
    -------
    Dict of param_name → importance in [0, 1].
    """
    successful = [t for t in trials if t.success]
    if len(successful) < 5:
        n = len(param_specs)
        return {spec.name: 1.0 / n for spec in param_specs}

    scores = np.array([t.score for t in successful])
    importances: Dict[str, float] = {}

    for spec in param_specs:
        values = np.array([t.params.get(spec.name, np.nan) for t in successful], dtype=np.float64)
        valid  = np.isfinite(values) & np.isfinite(scores)

        if valid.sum() < 5:
            importances[spec.name] = 0.0
            continue

        # Use absolute Pearson correlation as importance proxy
        try:
            corr = float(np.abs(np.corrcoef(values[valid], scores[valid])[0, 1]))
            importances[spec.name] = corr if np.isfinite(corr) else 0.0
        except Exception:
            importances[spec.name] = 0.0

    # Normalise
    total = sum(importances.values())
    if total > 1e-12:
        importances = {k: v / total for k, v in importances.items()}
    else:
        n = len(param_specs)
        importances = {spec.name: 1.0 / n for spec in param_specs}

    return importances


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    opt_result: OptResult,
    save_path:  Optional[str] = None,
    show:       bool = True,
) -> None:
    """
    Plot the convergence curve (best score vs trial number).

    Parameters
    ----------
    opt_result : OptResult from any optimizer.
    save_path  : if provided, save the figure to this path.
    show       : if True, display the figure interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping convergence plot")
        return

    curve = opt_result.convergence_curve
    if not curve:
        logger.warning("No convergence data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(curve) + 1), curve, "b-", linewidth=2, label="Best score so far")

    # Also plot individual trial scores
    trial_scores = [t.score for t in opt_result.all_trials]
    ax.scatter(
        [t.trial_id + 1 for t in opt_result.all_trials],
        trial_scores,
        alpha=0.3, s=20, c="gray", label="Individual trials",
    )

    ax.axhline(opt_result.best_score, color="red", linestyle="--", alpha=0.7,
               label=f"Best: {opt_result.best_score:.4f}")

    ax.set_xlabel("Trial Number")
    ax.set_ylabel(f"Score ({opt_result.metric})")
    ax.set_title(f"Optimization Convergence — {opt_result.method}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved convergence plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_param_importance(
    opt_result: OptResult,
    save_path:  Optional[str] = None,
    show:       bool = True,
) -> None:
    """
    Plot parameter importance as a horizontal bar chart.

    Parameters
    ----------
    opt_result : OptResult.
    save_path  : optional save path.
    show       : display figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping importance plot")
        return

    importance = opt_result.param_importance
    if not importance:
        logger.warning("No importance data to plot")
        return

    params  = list(importance.keys())
    values  = [importance[p] for p in params]
    indices = np.argsort(values)[::-1]  # descending

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.5)))
    bars = ax.barh(
        range(len(params)),
        [values[i] for i in indices],
        color="steelblue", alpha=0.8,
    )
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([params[i] for i in indices])
    ax.set_xlabel("Importance Score (normalised)")
    ax.set_title(f"Parameter Importance — {opt_result.method}")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved importance plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_param_heatmap(
    opt_result: OptResult,
    p1:         str,
    p2:         str,
    save_path:  Optional[str] = None,
    show:       bool = True,
    n_bins:     int  = 15,
) -> None:
    """
    Plot a 2-D heatmap of score vs two parameter dimensions.

    Parameters
    ----------
    opt_result : OptResult.
    p1         : x-axis parameter name.
    p2         : y-axis parameter name.
    save_path  : optional save path.
    show       : display figure.
    n_bins     : number of bins per axis for the heatmap.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap plot")
        return

    successful = [t for t in opt_result.all_trials if t.success]
    if len(successful) < 4:
        logger.warning("Insufficient data for heatmap (< 4 successful trials)")
        return

    x_vals = np.array([t.params.get(p1, np.nan) for t in successful], dtype=np.float64)
    y_vals = np.array([t.params.get(p2, np.nan) for t in successful], dtype=np.float64)
    scores = np.array([t.score                                         for t in successful], dtype=np.float64)

    valid = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(scores)
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    scores = scores[valid]

    if len(x_vals) < 4:
        logger.warning("Not enough valid data for heatmap")
        return

    # Bin data into 2D grid
    x_bins = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    y_bins = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
    grid   = np.full((n_bins, n_bins), np.nan)

    for xi in range(n_bins):
        for yi in range(n_bins):
            mask = (
                (x_vals >= x_bins[xi]) & (x_vals < x_bins[xi + 1]) &
                (y_vals >= y_bins[yi]) & (y_vals < y_bins[yi + 1])
            )
            if mask.sum() > 0:
                grid[yi, xi] = float(np.mean(scores[mask]))

    fig, ax = plt.subplots(figsize=(9, 7))
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

    im = ax.pcolormesh(
        x_centers, y_centers, grid,
        cmap="RdYlGn", shading="auto",
    )
    plt.colorbar(im, ax=ax, label=f"Mean {opt_result.metric}")

    # Mark best point
    best_x = opt_result.best_params.get(p1)
    best_y = opt_result.best_params.get(p2)
    if best_x is not None and best_y is not None:
        ax.scatter([best_x], [best_y], marker="*", s=200, c="black",
                   zorder=5, label="Best")
        ax.legend()

    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(f"Score Heatmap: {p1} vs {p2} — {opt_result.method}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ParamOptimizer — unified interface
# ─────────────────────────────────────────────────────────────────────────────

class ParamOptimizer:
    """
    Unified interface for all optimization backends.

    Wraps grid_search, random_search, sobol_search, and bayesian_opt
    with a consistent API and shared configuration.

    Parameters
    ----------
    strategy_fn     : callable(train_trades, params) → List[dict].
    metric          : scoring metric for IS optimization.
    starting_equity : initial equity for performance metrics.
    verbose         : log optimization progress.
    seed            : global random seed.

    Examples
    --------
    >>> opt = ParamOptimizer(my_strategy, metric='sharpe')
    >>> result = opt.optimize(
    ...     method='bayesian', trades=trades, splitter=splits,
    ...     param_space={'cf': (0.001, 0.003), 'bh_form': [1.2, 1.5, 2.0]},
    ...     n_iter=50,
    ... )
    >>> print(result.best_params)
    """

    def __init__(
        self,
        strategy_fn:     Callable,
        metric:          str   = "sharpe",
        starting_equity: float = 100_000.0,
        verbose:         bool  = True,
        seed:            int   = 42,
    ) -> None:
        self.strategy_fn     = strategy_fn
        self.metric          = metric
        self.starting_equity = starting_equity
        self.verbose         = verbose
        self.seed            = seed

    def optimize(
        self,
        method:      str,
        trades:      pd.DataFrame,
        splitter:    List[WFSplit],
        param_space: Union[Dict[str, Any], List[ParamSpec]],
        n_iter:      int = 100,
        n_init:      int = 10,
        **kwargs,
    ) -> OptResult:
        """
        Run optimization with the specified backend.

        Parameters
        ----------
        method      : 'grid', 'random', 'sobol', 'bayesian'.
        trades      : full trades DataFrame.
        splitter    : list of WFSplit for IS evaluation.
        param_space : dict spec or List[ParamSpec].
                      For 'grid', use dict with list values.
        n_iter      : total iterations (not used for grid search).
        n_init      : Bayesian warm-up evaluations.
        **kwargs    : passed to the underlying optimizer.

        Returns
        -------
        OptResult
        """
        method = method.lower().strip()

        common = dict(
            strategy_fn     = self.strategy_fn,
            trades          = trades,
            splitter        = splitter,
            metric          = self.metric,
            starting_equity = self.starting_equity,
            verbose         = self.verbose,
        )

        if method in ("grid", "grid_search"):
            if not isinstance(param_space, dict):
                raise ValueError("grid search requires dict param_space with list values")
            return grid_search(param_grid=param_space, **common)

        elif method in ("random", "random_search"):
            return random_search(
                param_space = param_space,
                n_iter      = n_iter,
                seed        = self.seed,
                **common,
                **kwargs,
            )

        elif method in ("sobol", "sobol_search", "quasi_random"):
            return sobol_search(
                param_space = param_space,
                n_iter      = n_iter,
                seed        = self.seed,
                **common,
                **kwargs,
            )

        elif method in ("bayesian", "bayesian_opt", "bo", "gp"):
            return bayesian_opt(
                param_space = param_space,
                n_iter      = n_iter,
                n_init      = n_init,
                seed        = self.seed,
                **common,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unknown optimization method: '{method}'. "
                "Choose from: grid, random, sobol, bayesian"
            )
