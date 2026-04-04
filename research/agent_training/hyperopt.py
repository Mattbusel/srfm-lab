"""
research/agent_training/hyperopt.py

Hyperparameter optimisation for RL agents.

Implements random search with cross-validation. No external libraries —
pure Python + numpy.

Classes:
    AgentHyperSearch  — random search over a parameter space
    HyperSearchResult — best params and full trial history
    CVResult          — k-fold cross-validation result
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np

from research.agent_training.trainer import AgentTrainer, TrainingConfig, EvalResult
from research.agent_training.environment import TradingEnvironment


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------


def loguniform(low: float, high: float, rng: np.random.Generator) -> float:
    """Sample from log-uniform distribution in [low, high]."""
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


def randint(low: int, high: int, rng: np.random.Generator) -> int:
    """Sample integer in [low, high]."""
    return int(rng.integers(low, high + 1))


def choice(options: list, rng: np.random.Generator) -> Any:
    """Sample one item uniformly from a list."""
    return options[int(rng.integers(0, len(options)))]


def uniform(low: float, high: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(low, high))


class ParamSpace:
    """
    Define a hyperparameter search space.

    Each parameter is defined with a type and range:
        ps.add_loguniform('lr', 1e-5, 1e-2)
        ps.add_int('hidden_dim', 64, 512)
        ps.add_choice('activation', ['relu', 'tanh', 'gelu'])
        ps.add_uniform('dropout', 0.0, 0.3)
        ps.add_fixed('gamma', 0.99)
    """

    def __init__(self) -> None:
        self._params: dict[str, tuple] = {}

    def add_loguniform(self, name: str, low: float, high: float) -> "ParamSpace":
        self._params[name] = ("loguniform", low, high)
        return self

    def add_uniform(self, name: str, low: float, high: float) -> "ParamSpace":
        self._params[name] = ("uniform", low, high)
        return self

    def add_int(self, name: str, low: int, high: int) -> "ParamSpace":
        self._params[name] = ("int", low, high)
        return self

    def add_choice(self, name: str, options: list) -> "ParamSpace":
        self._params[name] = ("choice", options)
        return self

    def add_fixed(self, name: str, value: Any) -> "ParamSpace":
        self._params[name] = ("fixed", value)
        return self

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample one configuration."""
        out = {}
        for name, spec in self._params.items():
            kind = spec[0]
            if kind == "loguniform":
                out[name] = loguniform(spec[1], spec[2], rng)
            elif kind == "uniform":
                out[name] = uniform(spec[1], spec[2], rng)
            elif kind == "int":
                out[name] = randint(spec[1], spec[2], rng)
            elif kind == "choice":
                out[name] = choice(spec[1], rng)
            elif kind == "fixed":
                out[name] = spec[1]
        return out

    def __repr__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self._params.items()]
        return "ParamSpace(\n" + "\n".join(lines) + "\n)"


# ---------------------------------------------------------------------------
# Default parameter space for RL agents
# ---------------------------------------------------------------------------


def default_agent_param_space() -> ParamSpace:
    """Standard parameter space covering the most impactful hyperparameters."""
    ps = ParamSpace()
    ps.add_loguniform("lr", 1e-5, 1e-2)
    ps.add_uniform("gamma", 0.90, 0.999)
    ps.add_int("batch_size", 32, 256)
    ps.add_int("hidden_dim", 64, 512)
    ps.add_int("network_depth", 1, 4)
    ps.add_loguniform("epsilon_decay", 0.990, 0.9999)
    ps.add_uniform("tau", 0.001, 0.05)
    ps.add_uniform("dropout", 0.0, 0.3)
    ps.add_choice("activation", ["relu", "tanh", "gelu"])
    ps.add_fixed("n_actions", 21)
    return ps


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    trial_id: int
    params: dict[str, Any]
    score: float
    eval_result: Optional[EvalResult]
    training_time_s: float
    converged: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# HyperSearchResult
# ---------------------------------------------------------------------------


@dataclass
class HyperSearchResult:
    """Result of a full hyperparameter search run."""

    best_params: dict[str, Any]
    best_score: float
    best_trial_id: int
    all_trials: list[TrialResult]
    n_trials: int
    total_time_s: float
    agent_class_name: str

    def top_k_trials(self, k: int = 5) -> list[TrialResult]:
        """Return the k best trials sorted by score (descending)."""
        return sorted(
            [t for t in self.all_trials if t.error is None],
            key=lambda t: t.score,
            reverse=True,
        )[:k]

    def param_importance(self) -> dict[str, float]:
        """
        Estimate parameter importance as correlation with trial score.

        Only works for numeric parameters.
        """
        numeric_params = {
            k for k in self.best_params
            if isinstance(self.best_params[k], (int, float))
        }
        importances = {}
        scores = np.array([t.score for t in self.all_trials if t.error is None])
        for param in numeric_params:
            values = np.array(
                [t.params.get(param, 0.0) for t in self.all_trials if t.error is None],
                dtype=np.float64,
            )
            if values.std() > 1e-10 and scores.std() > 1e-10:
                corr = float(np.corrcoef(values, scores)[0, 1])
            else:
                corr = 0.0
            importances[param] = abs(corr)
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# CVResult
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Result of k-fold cross-validation on an agent configuration."""

    params: dict[str, Any]
    fold_scores: list[float]
    mean_score: float
    std_score: float
    fold_eval_results: list[EvalResult]
    n_folds: int


# ---------------------------------------------------------------------------
# AgentHyperSearch
# ---------------------------------------------------------------------------


class AgentHyperSearch:
    """
    Hyperparameter optimisation for RL agents via random search.

    The scoring function defaults to mean OOS sharpe ratio.
    All search is done by evaluating agents after short training runs.

    Args:
        n_trials_per_worker : Trials per search invocation.
        score_fn            : Optional custom score(EvalResult) -> float.
        verbose             : Print trial progress.
        seed                : RNG seed.
    """

    def __init__(
        self,
        score_fn: Optional[Callable[[EvalResult], float]] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.score_fn = score_fn or self._default_score
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _default_score(eval_res: EvalResult) -> float:
        """Default scoring: Sharpe - 0.3 * MaxDrawdown."""
        return eval_res.mean_sharpe - 0.3 * eval_res.mean_max_drawdown

    def random_search(
        self,
        env_fn: Callable[[], TradingEnvironment],
        agent_class: type,
        param_space: Optional[ParamSpace] = None,
        n_trials: int = 50,
        train_episodes: int = 100,
        eval_episodes: int = 10,
    ) -> HyperSearchResult:
        """
        Random search over the hyperparameter space.

        Args:
            env_fn         : Callable that returns a fresh TradingEnvironment.
            agent_class    : Class of agent to instantiate (DQNAgent, etc.).
            param_space    : ParamSpace defining the search space.
                             Defaults to default_agent_param_space().
            n_trials       : Number of random trials.
            train_episodes : Training episodes per trial.
            eval_episodes  : Evaluation episodes per trial.

        Returns:
            HyperSearchResult.
        """
        if param_space is None:
            param_space = default_agent_param_space()

        trials: list[TrialResult] = []
        best_score = -math.inf
        best_params: dict = {}
        best_trial_id = 0
        start_time = time.time()

        for trial_id in range(n_trials):
            params = param_space.sample(self._rng)
            t0 = time.time()

            try:
                env = env_fn()
                agent = self._build_agent(agent_class, env.obs_dim, params)
                cfg = self._build_training_config(params, train_episodes)

                trainer = AgentTrainer(agent, env, cfg)
                trainer.train(n_episodes=train_episodes, eval_every=train_episodes + 1)
                eval_res = trainer.evaluate(eval_episodes)

                score = self.score_fn(eval_res)
                trial = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    score=score,
                    eval_result=eval_res,
                    training_time_s=time.time() - t0,
                    converged=True,
                )

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_trial_id = trial_id

            except Exception as e:
                trial = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    score=-math.inf,
                    eval_result=None,
                    training_time_s=time.time() - t0,
                    converged=False,
                    error=str(e),
                )

            trials.append(trial)

            if self.verbose:
                status = f"score={trial.score:.4f}" if trial.error is None else f"ERROR: {trial.error[:50]}"
                print(f"Trial {trial_id+1:3d}/{n_trials} | {status} | best={best_score:.4f}")

        return HyperSearchResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_id=best_trial_id,
            all_trials=trials,
            n_trials=n_trials,
            total_time_s=time.time() - start_time,
            agent_class_name=agent_class.__name__,
        )

    def cross_validate_agent(
        self,
        agent_class: type,
        params: dict[str, Any],
        env_fn: Callable[[], TradingEnvironment],
        n_folds: int = 5,
        train_episodes: int = 100,
        eval_episodes: int = 10,
    ) -> CVResult:
        """
        K-fold cross-validation for a fixed hyperparameter configuration.

        Each fold uses a different random seed to ensure independence.

        Args:
            agent_class    : Agent class.
            params         : Hyperparameter dict.
            env_fn         : Environment factory.
            n_folds        : Number of folds.
            train_episodes : Training episodes per fold.
            eval_episodes  : Eval episodes per fold.

        Returns:
            CVResult.
        """
        fold_scores = []
        fold_eval_results = []

        for fold in range(n_folds):
            seed = int(self._rng.integers(0, 100_000))
            env = env_fn()
            env.seed(seed)

            fold_params = {**params, "seed": seed}
            agent = self._build_agent(agent_class, env.obs_dim, fold_params)
            cfg = self._build_training_config(fold_params, train_episodes, seed=seed)

            trainer = AgentTrainer(agent, env, cfg)
            trainer.train(n_episodes=train_episodes, eval_every=train_episodes + 1)
            eval_res = trainer.evaluate(eval_episodes)

            score = self.score_fn(eval_res)
            fold_scores.append(score)
            fold_eval_results.append(eval_res)

            if self.verbose:
                print(f"  Fold {fold+1}/{n_folds}: score={score:.4f}")

        return CVResult(
            params=params,
            fold_scores=fold_scores,
            mean_score=float(np.mean(fold_scores)),
            std_score=float(np.std(fold_scores)),
            fold_eval_results=fold_eval_results,
            n_folds=n_folds,
        )

    # ------------------------------------------------------------------
    # Agent and config construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_agent(agent_class: type, obs_dim: int, params: dict) -> Any:
        """
        Instantiate an agent from a parameter dict.

        Handles common parameter names used across agent types.
        """
        from research.agent_training.agents import (
            DQNAgent, DDQNAgent, D3QNAgent, TD3Agent, PPOAgent, EnsembleAgent
        )

        lr = float(params.get("lr", 1e-3))
        gamma = float(params.get("gamma", 0.99))
        hidden_dim = int(params.get("hidden_dim", 256))
        depth = int(params.get("network_depth", 2))
        hidden_dims = [hidden_dim] * depth
        epsilon_decay = float(params.get("epsilon_decay", 0.995))
        n_actions = int(params.get("n_actions", 21))
        seed = params.get("seed", None)

        if agent_class in (DQNAgent, DDQNAgent):
            return agent_class(
                obs_dim=obs_dim,
                n_actions=n_actions,
                lr=lr,
                gamma=gamma,
                epsilon_decay=epsilon_decay,
                hidden_dims=hidden_dims,
                seed=seed,
            )
        elif agent_class is D3QNAgent:
            return D3QNAgent(
                obs_dim=obs_dim,
                n_actions=n_actions,
                hidden_dim=hidden_dim,
                lr=lr,
                gamma=gamma,
                epsilon_decay=epsilon_decay,
                seed=seed,
            )
        elif agent_class is TD3Agent:
            return TD3Agent(
                obs_dim=obs_dim,
                action_dim=1,
                hidden_dims=hidden_dims,
                lr_actor=lr / 3,
                lr_critic=lr,
                gamma=gamma,
                seed=seed,
            )
        elif agent_class is PPOAgent:
            return PPOAgent(
                obs_dim=obs_dim,
                action_dim=1,
                lr_actor=lr / 3,
                lr_critic=lr,
                gamma=gamma,
                hidden_dims=hidden_dims,
                seed=seed,
            )
        elif agent_class is EnsembleAgent:
            return EnsembleAgent(
                obs_dim=obs_dim,
                n_actions=n_actions,
                hidden_dim=hidden_dim,
                lr=lr,
                seed=seed,
            )
        else:
            # Generic: try constructor with obs_dim and common args
            return agent_class(obs_dim=obs_dim, lr=lr, gamma=gamma)

    @staticmethod
    def _build_training_config(
        params: dict,
        n_episodes: int,
        seed: Optional[int] = None,
    ) -> TrainingConfig:
        batch_size = int(params.get("batch_size", 64))
        tau = float(params.get("tau", 0.005))
        return TrainingConfig(
            n_episodes=n_episodes,
            eval_every=n_episodes + 1,  # no mid-training eval
            batch_size=batch_size,
            tau=tau,
            verbose=False,
            seed=seed,
            early_stop_patience=n_episodes,  # disable early stopping
            use_curriculum=False,
        )

    def plot_search_results(
        self,
        result: HyperSearchResult,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot trial scores over search and parameter correlation."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        valid = [t for t in result.all_trials if t.error is None]
        scores = [t.score for t in valid]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Hyperparameter Search: {result.agent_class_name}", fontsize=13)

        ax = axes[0]
        ax.plot(scores, marker="o", alpha=0.5, color="steelblue")
        ax.axhline(result.best_score, color="green", linestyle="--", label=f"Best={result.best_score:.4f}")
        ax.set_title("Trial Scores")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.legend()

        ax = axes[1]
        imp = result.param_importance()
        if imp:
            params_sorted = list(imp.keys())[:10]
            values_sorted = [imp[p] for p in params_sorted]
            ax.barh(params_sorted, values_sorted, color="darkorange")
            ax.set_title("Parameter Importance (|corr|)")
            ax.set_xlabel("|Correlation with Score|")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
