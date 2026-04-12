"""
lumina/experiment_tracker.py

MLflow/W&B experiment tracking integration for Lumina.

Covers:
  - Experiment logging (metrics, parameters, artifacts)
  - Hyperparameter tracking and comparison
  - Artifact storage (models, plots, configs)
  - Metric comparison across runs
  - Model registry integration
  - A/B testing framework for model comparison
  - Automated hyperparameter search (Bayesian, random, grid)
  - Run metadata and tagging
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    experiment_name: str = "lumina_experiment"
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    # Tracker backends to use
    use_mlflow: bool = False
    use_wandb: bool = False
    use_local: bool = True
    # Local logging
    log_dir: str = "runs"
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_registry_uri: Optional[str] = None
    # W&B
    wandb_project: str = "lumina"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"   # "online" | "offline" | "disabled"


@dataclass
class RunSummary:
    """Summary of a completed experiment run."""
    run_id: str
    experiment_name: str
    run_name: str
    start_time: float
    end_time: float
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    tags: Dict[str, str]


# ---------------------------------------------------------------------------
# Local run logger (no dependencies)
# ---------------------------------------------------------------------------

class LocalRunLogger:
    """
    Simple local experiment logger.
    Stores runs as JSON files in a directory.
    """

    def __init__(self, log_dir: Union[str, pathlib.Path]):
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id: Optional[str] = None
        self._run_dir: Optional[pathlib.Path] = None
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._params: Dict[str, Any] = {}
        self._tags: Dict[str, str] = {}
        self._start_time: float = 0.0

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        self._run_id = str(uuid.uuid4())[:8]
        run_name = run_name or f"run_{self._run_id}"
        self._run_dir = self.log_dir / experiment_name / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = {}
        self._params = {}
        self._tags = tags or {}
        self._start_time = time.time()
        logger.info(f"[LocalLogger] Started run {run_name} ({self._run_id})")
        return self._run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        (self._run_dir / "params.json").write_text(json.dumps(self._params, indent=2, default=str))

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append({"value": value, "step": step, "time": time.time()})
        # Append to JSONL for streaming
        with open(self._run_dir / f"metric_{key}.jsonl", "a") as f:
            f.write(json.dumps({"value": value, "step": step}) + "\n")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, path: Union[str, pathlib.Path], artifact_name: Optional[str] = None) -> None:
        path = pathlib.Path(path)
        if path.exists():
            dest_name = artifact_name or path.name
            import shutil
            shutil.copy2(path, self._run_dir / dest_name)

    def log_dict(self, data: Dict[str, Any], filename: str = "data.json") -> None:
        (self._run_dir / filename).write_text(json.dumps(data, indent=2, default=str))

    def end_run(self, status: str = "FINISHED") -> None:
        summary = {
            "run_id": self._run_id,
            "status": status,
            "start_time": self._start_time,
            "end_time": time.time(),
            "duration_sec": time.time() - self._start_time,
            "params": self._params,
            "final_metrics": {k: v[-1]["value"] for k, v in self._metrics.items()},
            "tags": self._tags,
        }
        (self._run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"[LocalLogger] Ended run {self._run_id} ({status})")

    def get_metric_history(self, key: str) -> List[float]:
        return [entry["value"] for entry in self._metrics.get(key, [])]


# ---------------------------------------------------------------------------
# MLflow tracker
# ---------------------------------------------------------------------------

class MLflowTracker:
    """MLflow experiment tracking integration."""

    def __init__(self, config: RunConfig):
        if not _MLFLOW_AVAILABLE:
            raise ImportError("Install mlflow: pip install mlflow")
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        self._active_run = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        self._active_run = mlflow.start_run(
            run_name=run_name or self.config.run_name,
            tags=tags or self.config.tags,
        )
        return self._active_run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._active_run:
            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active_run:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._active_run:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, artifact_path: str = "model") -> None:
        if self._active_run:
            mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, path: str) -> None:
        if self._active_run:
            mlflow.log_artifact(path)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run:
            mlflow.end_run(status=status)
            self._active_run = None

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> "pd.DataFrame":
        import pandas as pd
        return mlflow.search_runs(
            experiment_names=[self.config.experiment_name],
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results,
        )

    def get_best_run(self, metric: str = "val_loss", maximize: bool = False) -> Optional[Dict[str, Any]]:
        try:
            runs = self.search_runs(
                order_by=[f"metrics.{metric} {'DESC' if maximize else 'ASC'}"],
                max_results=1,
            )
            if len(runs) > 0:
                return runs.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to get best run: {e}")
        return None


# ---------------------------------------------------------------------------
# Weights & Biases tracker
# ---------------------------------------------------------------------------

class WandBTracker:
    """Weights & Biases experiment tracking integration."""

    def __init__(self, config: RunConfig):
        if not _WANDB_AVAILABLE:
            raise ImportError("Install wandb: pip install wandb")
        self.config = config
        self._run = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        self._run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name or self.config.run_name,
            config=config_dict or {},
            tags=tags or list(self.config.tags.values()),
            mode=self.config.wandb_mode,
            reinit=True,
        )
        return self._run.id

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run:
            wandb.config.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._run:
            wandb.log({key: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run:
            wandb.log(metrics, step=step)

    def log_model(self, model: nn.Module, name: str = "model") -> None:
        if self._run:
            artifact = wandb.Artifact(name, type="model")
            with artifact.new_file("model.pt", mode="wb") as f:
                torch.save(model.state_dict(), f)
            self._run.log_artifact(artifact)

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        if self._run:
            artifact = wandb.Artifact(pathlib.Path(path).stem, type=artifact_type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    def log_table(self, key: str, data: Dict[str, List]) -> None:
        if self._run:
            wandb.log({key: wandb.Table(data=list(zip(*data.values())), columns=list(data.keys()))})

    def end_run(self) -> None:
        if self._run:
            self._run.finish()
            self._run = None

    def get_run_url(self) -> Optional[str]:
        if self._run:
            return self._run.url
        return None


# ---------------------------------------------------------------------------
# Unified experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Unified experiment tracking that can log to multiple backends simultaneously.
    Supports MLflow, W&B, and local file logging.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self._trackers: List[Any] = []
        self._local: Optional[LocalRunLogger] = None
        self._run_id: Optional[str] = None
        self._active: bool = False

        # Initialize backends
        if config.use_local:
            self._local = LocalRunLogger(config.log_dir)
            self._trackers.append(self._local)
        if config.use_mlflow and _MLFLOW_AVAILABLE:
            try:
                self._mlflow = MLflowTracker(config)
                self._trackers.append(self._mlflow)
            except Exception as e:
                logger.warning(f"MLflow init failed: {e}")
        if config.use_wandb and _WANDB_AVAILABLE:
            try:
                self._wandb = WandBTracker(config)
                self._trackers.append(self._wandb)
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")

    @contextlib.contextmanager
    def run(
        self,
        run_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for a training run."""
        self.start_run(run_name, tags)
        if params:
            self.log_params(params)
        try:
            yield self
            self.end_run("FINISHED")
        except Exception as e:
            self.end_run("FAILED")
            raise

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        self._run_id = str(uuid.uuid4())[:8]
        run_name = run_name or f"run_{self._run_id}"

        for tracker in self._trackers:
            try:
                if isinstance(tracker, LocalRunLogger):
                    tracker.start_run(self.config.experiment_name, run_name, tags)
                elif isinstance(tracker, MLflowTracker):
                    tracker.start_run(run_name, tags)
                elif isinstance(tracker, WandBTracker):
                    tracker.start_run(run_name)
            except Exception as e:
                logger.warning(f"Tracker {type(tracker).__name__} start failed: {e}")

        self._active = True
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        for tracker in self._trackers:
            try:
                tracker.end_run(status if hasattr(tracker, 'end_run') else None)
            except Exception as e:
                logger.warning(f"Tracker end_run failed: {e}")
        self._active = False

    def log_params(self, params: Dict[str, Any]) -> None:
        for tracker in self._trackers:
            try:
                tracker.log_params(params)
            except Exception as e:
                logger.debug(f"log_params failed for {type(tracker).__name__}: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        for tracker in self._trackers:
            try:
                tracker.log_metric(key, value, step)
            except Exception as e:
                logger.debug(f"log_metric failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for tracker in self._trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.debug(f"log_metrics failed: {e}")

    def log_model(self, model: nn.Module, name: str = "model") -> None:
        for tracker in self._trackers:
            try:
                tracker.log_model(model, name)
            except Exception as e:
                logger.debug(f"log_model failed: {e}")

    def log_artifact(self, path: Union[str, pathlib.Path]) -> None:
        for tracker in self._trackers:
            try:
                tracker.log_artifact(str(path))
            except Exception as e:
                logger.debug(f"log_artifact failed: {e}")

    def log_dict(self, data: Dict[str, Any], filename: str = "data.json") -> None:
        if self._local:
            self._local.log_dict(data, filename)

    def log_hyperparams(self, model: nn.Module, optimizer: Any) -> None:
        """Auto-log model and optimizer hyperparameters."""
        params = {
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if hasattr(optimizer, "param_groups"):
            pg = optimizer.param_groups[0]
            params["lr"] = pg.get("lr", 0)
            params["weight_decay"] = pg.get("weight_decay", 0)
            params["optimizer"] = type(optimizer).__name__
        self.log_params(params)

    def get_metric_history(self, key: str) -> List[float]:
        if self._local:
            return self._local.get_metric_history(key)
        return []


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

@dataclass
class HPSearchConfig:
    """Hyperparameter search configuration."""
    method: str = "bayesian"    # "random" | "grid" | "bayesian"
    n_trials: int = 20
    metric: str = "val_loss"
    direction: str = "minimize"  # "minimize" | "maximize"
    timeout_sec: Optional[float] = None
    n_jobs: int = 1


class HyperparameterSearch:
    """
    Automated hyperparameter search with optional Optuna backend.
    """

    def __init__(
        self,
        config: HPSearchConfig,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.config = config
        self.tracker = tracker
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_metric: float = float("inf") if config.direction == "minimize" else float("-inf")
        self._trial_results: List[Dict[str, Any]] = []

    def random_search(
        self,
        param_space: Dict[str, Any],
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> Dict[str, Any]:
        """
        Random hyperparameter search.

        param_space: Dict of param_name -> (low, high) for floats,
                     or [values] for categorical.
        """
        best_params = {}
        best_metric = float("inf") if self.config.direction == "minimize" else float("-inf")

        for trial in range(self.config.n_trials):
            params = {}
            for name, space in param_space.items():
                if isinstance(space, (list, tuple)) and len(space) == 2 and isinstance(space[0], float):
                    low, high = space
                    params[name] = low + (high - low) * np.random.random()
                elif isinstance(space, list):
                    params[name] = np.random.choice(space)
                elif isinstance(space, tuple) and len(space) == 3 and isinstance(space[0], int):
                    low, high, step = space
                    choices = list(range(low, high + 1, step))
                    params[name] = np.random.choice(choices)
                else:
                    params[name] = space

            try:
                metric = objective_fn(params)
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                metric = float("inf") if self.config.direction == "minimize" else float("-inf")

            improved = (
                (metric < best_metric) if self.config.direction == "minimize"
                else (metric > best_metric)
            )
            if improved:
                best_metric = metric
                best_params = params.copy()

            self._trial_results.append({
                "trial": trial,
                "params": params,
                "metric": metric,
                "is_best": improved,
            })

            if self.tracker:
                self.tracker.log_metrics({
                    f"hpsearch/{self.config.metric}": metric,
                    "hpsearch/best_metric": best_metric,
                }, step=trial)

            logger.info(f"Trial {trial}: metric={metric:.4f}, best={best_metric:.4f}")

        self._best_params = best_params
        self._best_metric = best_metric
        return best_params

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> Dict[str, Any]:
        """Exhaustive grid search."""
        import itertools

        keys = list(param_grid.keys())
        values_list = list(param_grid.values())
        all_combinations = list(itertools.product(*values_list))

        if len(all_combinations) > self.config.n_trials:
            # Randomly sample from grid
            indices = np.random.choice(len(all_combinations), self.config.n_trials, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        best_params = {}
        best_metric = float("inf") if self.config.direction == "minimize" else float("-inf")

        for trial, combo in enumerate(all_combinations):
            params = dict(zip(keys, combo))
            metric = objective_fn(params)

            improved = (
                (metric < best_metric) if self.config.direction == "minimize"
                else (metric > best_metric)
            )
            if improved:
                best_metric = metric
                best_params = params.copy()

            self._trial_results.append({"trial": trial, "params": params, "metric": metric})

        self._best_params = best_params
        self._best_metric = best_metric
        return best_params

    def bayesian_search(
        self,
        param_space: Dict[str, Any],
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna if available."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def optuna_objective(trial: optuna.Trial) -> float:
                params = {}
                for name, space in param_space.items():
                    if isinstance(space, (list, tuple)) and len(space) == 2 and all(isinstance(s, float) for s in space):
                        params[name] = trial.suggest_float(name, space[0], space[1])
                    elif isinstance(space, list):
                        params[name] = trial.suggest_categorical(name, space)
                    elif isinstance(space, tuple) and len(space) == 3 and isinstance(space[0], int):
                        params[name] = trial.suggest_int(name, space[0], space[1], step=space[2])
                    else:
                        params[name] = space
                return objective_fn(params)

            direction = "minimize" if self.config.direction == "minimize" else "maximize"
            study = optuna.create_study(direction=direction)
            study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_sec,
                n_jobs=self.config.n_jobs,
            )
            self._best_params = study.best_params
            self._best_metric = study.best_value
            self._trial_results = [
                {"trial": t.number, "params": t.params, "metric": t.value}
                for t in study.trials
            ]
            return study.best_params

        except ImportError:
            logger.warning("Optuna not available; falling back to random search.")
            return self.random_search(param_space, objective_fn)

    def run(
        self,
        param_space: Dict[str, Any],
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> Dict[str, Any]:
        """Run hyperparameter search with configured method."""
        if self.config.method == "random":
            return self.random_search(param_space, objective_fn)
        elif self.config.method == "grid":
            return self.grid_search(param_space, objective_fn)
        elif self.config.method == "bayesian":
            return self.bayesian_search(param_space, objective_fn)
        else:
            raise ValueError(f"Unknown search method: {self.config.method}")

    def best_params(self) -> Optional[Dict[str, Any]]:
        return self._best_params

    def best_metric(self) -> float:
        return self._best_metric

    def results_dataframe(self):
        try:
            import pandas as pd
            return pd.DataFrame(self._trial_results)
        except ImportError:
            return self._trial_results


# ---------------------------------------------------------------------------
# A/B testing framework
# ---------------------------------------------------------------------------

@dataclass
class ABTestConfig:
    model_a_name: str
    model_b_name: str
    test_name: str
    metric: str = "sharpe_ratio"
    significance_level: float = 0.05
    min_samples: int = 100
    max_samples: int = 10_000


class ABTestingFramework:
    """
    A/B testing framework for comparing financial model versions.
    Uses sequential testing to stop early when significance is achieved.
    """

    def __init__(self, config: ABTestConfig, tracker: Optional[ExperimentTracker] = None):
        self.config = config
        self.tracker = tracker
        self._samples_a: List[float] = []
        self._samples_b: List[float] = []
        self._result: Optional[Dict[str, Any]] = None

    def add_sample(self, model: str, metric_value: float) -> None:
        """Add a metric observation for model A or B."""
        if model == self.config.model_a_name:
            self._samples_a.append(metric_value)
        elif model == self.config.model_b_name:
            self._samples_b.append(metric_value)

        n = min(len(self._samples_a), len(self._samples_b))
        if n >= self.config.min_samples:
            self._check_significance()

    def _check_significance(self) -> Optional[Dict[str, Any]]:
        """Run statistical test and check for significance."""
        try:
            from scipy import stats as sp_stats
            a = np.array(self._samples_a)
            b = np.array(self._samples_b)
            t_stat, p_value = sp_stats.ttest_ind(a, b)
            significant = p_value < self.config.significance_level
            winner = None
            if significant:
                winner = self.config.model_a_name if np.mean(a) > np.mean(b) else self.config.model_b_name

            result = {
                "n_samples_a": len(a),
                "n_samples_b": len(b),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "std_a": float(np.std(a, ddof=1)),
                "std_b": float(np.std(b, ddof=1)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
                "winner": winner,
                "lift": float((np.mean(a) - np.mean(b)) / (np.mean(b) + 1e-10)),
            }
            self._result = result
            if self.tracker and significant:
                self.tracker.log_dict(result, f"ab_test_{self.config.test_name}.json")
            return result
        except ImportError:
            return None

    def summary(self) -> Optional[Dict[str, Any]]:
        return self._result or self._check_significance()


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

class RunComparer:
    """
    Compare multiple experiment runs on a set of metrics.
    """

    def __init__(self, log_dir: Union[str, pathlib.Path]):
        self.log_dir = pathlib.Path(log_dir)

    def load_run(self, experiment: str, run_name: str) -> Dict[str, Any]:
        run_dir = self.log_dir / experiment / run_name
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        return {}

    def compare(
        self,
        experiment: str,
        metric_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compare all runs in an experiment."""
        experiment_dir = self.log_dir / experiment
        if not experiment_dir.exists():
            return []

        results = []
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text())
                row = {
                    "run_name": run_dir.name,
                    "status": summary.get("status"),
                    "duration_sec": summary.get("duration_sec", 0),
                }
                if metric_keys:
                    for k in metric_keys:
                        row[k] = summary.get("final_metrics", {}).get(k, float("nan"))
                else:
                    row.update(summary.get("final_metrics", {}))
                results.append(row)

        return sorted(results, key=lambda x: x.get(metric_keys[0] if metric_keys else "duration_sec", 0))

    def best_run(self, experiment: str, metric: str, maximize: bool = False) -> Optional[Dict[str, Any]]:
        """Find the best run for a given metric."""
        runs = self.compare(experiment, metric_keys=[metric])
        if not runs:
            return None
        return max(runs, key=lambda x: x.get(metric, float("-inf")) if maximize
                   else -x.get(metric, float("inf")))


# ---------------------------------------------------------------------------
# Callback-based integration
# ---------------------------------------------------------------------------

class TrainingCallback:
    """
    Callback interface for hooking into training loops.
    """

    def on_train_begin(self, logs: Optional[Dict] = None) -> None: pass
    def on_train_end(self, logs: Optional[Dict] = None) -> None: pass
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None: pass
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None: pass
    def on_step_begin(self, step: int, logs: Optional[Dict] = None) -> None: pass
    def on_step_end(self, step: int, logs: Optional[Dict] = None) -> None: pass


class ExperimentTrackerCallback(TrainingCallback):
    """Callback that logs to ExperimentTracker during training."""

    def __init__(self, tracker: ExperimentTracker, log_every_n_steps: int = 10):
        self.tracker = tracker
        self.log_every_n_steps = log_every_n_steps

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        if logs:
            self.tracker.log_params(logs)

    def on_step_end(self, step: int, logs: Optional[Dict] = None) -> None:
        if logs and step % self.log_every_n_steps == 0:
            self.tracker.log_metrics(
                {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                step=step,
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if logs:
            self.tracker.log_metrics(
                {f"epoch/{k}": v for k, v in logs.items() if isinstance(v, (int, float))},
                step=epoch,
            )

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        self.tracker.end_run("FINISHED")


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation metric."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._wait: int = 0
        self._best_weights: Optional[Dict] = None
        self.stopped_epoch: int = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> bool:
        if logs is None or self.monitor not in logs:
            return False

        current = logs[self.monitor]
        improved = (
            (current < self._best_value - self.min_delta) if self.mode == "min"
            else (current > self._best_value + self.min_delta)
        )

        if improved:
            self._best_value = current
            self._wait = 0
            if self.restore_best_weights and "model" in logs:
                self._best_weights = copy.deepcopy(logs["model"].state_dict())
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping at epoch {epoch}. Best {self.monitor}: {self._best_value:.4f}")
                if self.restore_best_weights and self._best_weights is not None and "model" in logs:
                    logs["model"].load_state_dict(self._best_weights)
                return True   # Stop signal

        return False


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "RunConfig",
    "RunSummary",
    # Trackers
    "LocalRunLogger",
    "MLflowTracker",
    "WandBTracker",
    "ExperimentTracker",
    # HP Search
    "HPSearchConfig",
    "HyperparameterSearch",
    # A/B Testing
    "ABTestConfig",
    "ABTestingFramework",
    # Comparison
    "RunComparer",
    # Callbacks
    "TrainingCallback",
    "ExperimentTrackerCallback",
    "EarlyStoppingCallback",
]
