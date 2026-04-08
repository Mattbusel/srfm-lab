"""
experiment-tracker/experiment_manager.py

Experiment management layer on top of the experiment tracker.

Provides high-level operations:
  - Create, run, compare, and archive experiments
  - Config versioning with full parameter change history
  - Side-by-side metric comparison across experiments
  - Statistical comparison: paired t-test on OOS returns
  - Hyperparameter search: grid search and random search
  - Early stopping: halt if OOS Sharpe drops below threshold
  - Experiment tagging and filtering
  - Leaderboard: rank experiments by configurable metrics
  - Export: full experiment lineage to JSON

Usage:
    from experiment_tracker.experiment_manager import ExperimentManager

    mgr = ExperimentManager()
    exp_id = mgr.create("test_momentum", hypothesis_id=42,
                         config={"lookback": 20, "threshold": 0.5})
    mgr.run(exp_id, run_fn=my_backtest)
    mgr.compare([exp_id, other_id])
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    CREATED   = "created"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    STOPPED   = "early_stopped"
    ARCHIVED  = "archived"


class SearchStrategy(str, Enum):
    GRID   = "grid"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """Full in-memory representation of one experiment."""
    id: str
    name: str
    hypothesis_id: str | int | None
    config: dict[str, Any]
    config_version: int
    status: ExperimentStatus
    tags: list[str]
    results: dict[str, float]        # metric_name -> value
    oos_returns: list[float]         # for statistical comparison
    config_history: list[dict[str, Any]]  # version history
    parent_id: str | None            # experiment this was derived from
    created_at: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    error_message: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis_id": self.hypothesis_id,
            "config": self.config,
            "config_version": self.config_version,
            "status": self.status.value,
            "tags": self.tags,
            "results": {k: round(v, 6) for k, v in self.results.items()},
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }

    @property
    def sharpe(self) -> float | None:
        return self.results.get("sharpe")

    @property
    def oos_sharpe(self) -> float | None:
        return self.results.get("oos_sharpe")

    @property
    def total_return(self) -> float | None:
        return self.results.get("total_return")


@dataclass
class ConfigChange:
    """Record of a single config parameter change."""
    experiment_id: str
    version: int
    param_name: str
    old_value: Any
    new_value: Any
    changed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple experiments."""
    experiment_ids: list[str]
    experiment_names: list[str]
    metrics_table: dict[str, list[float | None]]  # metric -> [val_per_exp]
    config_diffs: dict[str, list[Any]]             # param -> [val_per_exp]
    statistical_tests: dict[str, dict[str, float]] # pair_key -> {t_stat, p_value}
    best_by_metric: dict[str, str]                 # metric -> best experiment_id
    summary: str

    def to_dataframe(self) -> pd.DataFrame:
        """Return a wide-format DataFrame: rows=metrics, cols=experiments."""
        data: dict[str, list[float | None]] = {}
        for metric, values in self.metrics_table.items():
            data[metric] = values
        return pd.DataFrame(data, index=self.experiment_names).T


@dataclass
class LeaderboardEntry:
    """One row of the leaderboard."""
    rank: int
    experiment_id: str
    experiment_name: str
    primary_metric: float
    metrics: dict[str, float]
    tags: list[str]
    status: str


@dataclass
class SearchResult:
    """Result of a hyperparameter search."""
    search_id: str
    strategy: SearchStrategy
    n_experiments: int
    param_space: dict[str, list[Any]]
    best_experiment_id: str
    best_config: dict[str, Any]
    best_metric: float
    all_results: list[dict[str, Any]]  # sorted by primary metric
    duration_seconds: float


@dataclass
class EarlyStopConfig:
    """Configuration for early stopping."""
    metric: str = "oos_sharpe"
    threshold: float = 0.0
    patience: int = 3              # stop after N consecutive checks below threshold
    check_interval_seconds: float = 60.0


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """
    High-level experiment management: create, run, compare, search, archive.

    All state is held in-memory (dict of Experiment objects).  For persistent
    storage, integrate with the ExperimentTracker from tracker.py.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._config_changes: list[ConfigChange] = []
        self._counter = 0

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        hypothesis_id: str | int | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new experiment and return its ID.
        """
        self._counter += 1
        exp_id = self._generate_id(name)
        now = datetime.now(timezone.utc).isoformat()

        exp = Experiment(
            id=exp_id,
            name=name,
            hypothesis_id=hypothesis_id,
            config=dict(config or {}),
            config_version=1,
            status=ExperimentStatus.CREATED,
            tags=list(tags or []),
            results={},
            oos_returns=[],
            config_history=[{"version": 1, "config": dict(config or {}), "timestamp": now}],
            parent_id=parent_id,
            created_at=now,
            started_at=None,
            finished_at=None,
            duration_seconds=None,
            error_message=None,
            metadata=dict(metadata or {}),
        )
        self._experiments[exp_id] = exp
        logger.info("Created experiment %s: %s", exp_id, name)
        return exp_id

    # ------------------------------------------------------------------
    # Update config (versioned)
    # ------------------------------------------------------------------

    def update_config(
        self,
        experiment_id: str,
        updates: dict[str, Any],
    ) -> int:
        """
        Update experiment config parameters.  Creates a new version.
        Returns the new config version number.
        """
        exp = self._get(experiment_id)
        if exp.status not in (ExperimentStatus.CREATED, ExperimentStatus.FAILED):
            raise ValueError(
                f"Cannot update config of experiment in status {exp.status.value}"
            )

        old_config = dict(exp.config)
        for key, new_val in updates.items():
            old_val = exp.config.get(key)
            exp.config[key] = new_val
            self._config_changes.append(ConfigChange(
                experiment_id=experiment_id,
                version=exp.config_version + 1,
                param_name=key,
                old_value=old_val,
                new_value=new_val,
            ))

        exp.config_version += 1
        exp.config_history.append({
            "version": exp.config_version,
            "config": dict(exp.config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": dict(updates),
        })

        logger.info(
            "Updated experiment %s config to v%d: %s",
            experiment_id, exp.config_version, updates,
        )
        return exp.config_version

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        experiment_id: str,
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        early_stop: EarlyStopConfig | None = None,
    ) -> dict[str, float]:
        """
        Execute an experiment by calling run_fn with the experiment config.

        run_fn signature: (config: dict) -> {"sharpe": float, "oos_returns": [...], ...}
        """
        exp = self._get(experiment_id)
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()

        try:
            raw_result = run_fn(dict(exp.config))

            # Extract results
            oos_returns = raw_result.pop("oos_returns", [])
            if isinstance(oos_returns, np.ndarray):
                oos_returns = oos_returns.tolist()
            exp.oos_returns = oos_returns

            # Store metrics
            for key, val in raw_result.items():
                if isinstance(val, (int, float)):
                    exp.results[key] = float(val)

            # Early stopping check (post-hoc for single-shot runs)
            if early_stop and early_stop.metric in exp.results:
                metric_val = exp.results[early_stop.metric]
                if metric_val < early_stop.threshold:
                    exp.status = ExperimentStatus.STOPPED
                    logger.info(
                        "Experiment %s early-stopped: %s=%.4f < %.4f",
                        experiment_id, early_stop.metric,
                        metric_val, early_stop.threshold,
                    )
                else:
                    exp.status = ExperimentStatus.COMPLETED
            else:
                exp.status = ExperimentStatus.COMPLETED

        except Exception as exc:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = str(exc)
            logger.error("Experiment %s failed: %s", experiment_id, exc)
            raise
        finally:
            exp.duration_seconds = time.monotonic() - t0
            exp.finished_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Experiment %s completed in %.1fs: %s",
            experiment_id, exp.duration_seconds, exp.results,
        )
        return dict(exp.results)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare(
        self,
        experiment_ids: list[str],
        metrics: list[str] | None = None,
    ) -> ComparisonResult:
        """
        Side-by-side comparison of multiple experiments.
        Includes statistical tests on OOS returns where available.
        """
        experiments = [self._get(eid) for eid in experiment_ids]

        # Gather all metric names
        all_metrics: set[str] = set()
        for exp in experiments:
            all_metrics.update(exp.results.keys())
        if metrics:
            all_metrics = all_metrics & set(metrics)
        metric_list = sorted(all_metrics)

        # Build metrics table
        metrics_table: dict[str, list[float | None]] = {}
        for m in metric_list:
            metrics_table[m] = [
                exp.results.get(m) for exp in experiments
            ]

        # Config diffs
        all_params: set[str] = set()
        for exp in experiments:
            all_params.update(exp.config.keys())
        config_diffs: dict[str, list[Any]] = {}
        for param in sorted(all_params):
            values = [exp.config.get(param) for exp in experiments]
            if len(set(str(v) for v in values)) > 1:
                config_diffs[param] = values

        # Statistical tests (paired t-test on OOS returns)
        stat_tests = self._pairwise_ttests(experiments)

        # Best by metric
        best_by: dict[str, str] = {}
        for m in metric_list:
            vals = metrics_table[m]
            valid = [(v, experiments[i].id) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_val, best_id = max(valid, key=lambda x: x[0])
                best_by[m] = best_id

        # Summary
        summary_parts = [f"Comparing {len(experiments)} experiments:"]
        for exp in experiments:
            sharpe = exp.results.get("sharpe", exp.results.get("oos_sharpe"))
            sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
            summary_parts.append(f"  {exp.name} ({exp.id[:8]}): Sharpe={sharpe_str}")

        return ComparisonResult(
            experiment_ids=experiment_ids,
            experiment_names=[e.name for e in experiments],
            metrics_table=metrics_table,
            config_diffs=config_diffs,
            statistical_tests=stat_tests,
            best_by_metric=best_by,
            summary="\n".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def _pairwise_ttests(
        self, experiments: list[Experiment],
    ) -> dict[str, dict[str, float]]:
        """Paired t-test on OOS returns between all experiment pairs."""
        from scipy.stats import ttest_rel

        results: dict[str, dict[str, float]] = {}
        for i, exp_a in enumerate(experiments):
            for j, exp_b in enumerate(experiments):
                if j <= i:
                    continue
                if not exp_a.oos_returns or not exp_b.oos_returns:
                    continue
                # Align to same length
                min_len = min(len(exp_a.oos_returns), len(exp_b.oos_returns))
                if min_len < 5:
                    continue
                a = np.array(exp_a.oos_returns[:min_len])
                b = np.array(exp_b.oos_returns[:min_len])
                try:
                    t_stat, p_value = ttest_rel(a, b)
                    pair_key = f"{exp_a.id[:8]}_vs_{exp_b.id[:8]}"
                    results[pair_key] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "mean_diff": float(np.mean(a - b)),
                        "n_samples": min_len,
                        "significant_5pct": p_value < 0.05,
                    }
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        base_name: str,
        hypothesis_id: str | int | None,
        param_space: dict[str, list[Any]],
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        primary_metric: str = "oos_sharpe",
        fixed_params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        early_stop: EarlyStopConfig | None = None,
    ) -> SearchResult:
        """
        Exhaustive grid search over parameter space.
        """
        return self._search(
            strategy=SearchStrategy.GRID,
            base_name=base_name,
            hypothesis_id=hypothesis_id,
            param_space=param_space,
            run_fn=run_fn,
            primary_metric=primary_metric,
            fixed_params=fixed_params,
            tags=tags,
            early_stop=early_stop,
            n_random=None,
        )

    def random_search(
        self,
        base_name: str,
        hypothesis_id: str | int | None,
        param_space: dict[str, list[Any]],
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        n_trials: int = 50,
        primary_metric: str = "oos_sharpe",
        fixed_params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        early_stop: EarlyStopConfig | None = None,
        seed: int | None = None,
    ) -> SearchResult:
        """
        Random search over parameter space.
        """
        return self._search(
            strategy=SearchStrategy.RANDOM,
            base_name=base_name,
            hypothesis_id=hypothesis_id,
            param_space=param_space,
            run_fn=run_fn,
            primary_metric=primary_metric,
            fixed_params=fixed_params,
            tags=tags,
            early_stop=early_stop,
            n_random=n_trials,
            seed=seed,
        )

    def _search(
        self,
        strategy: SearchStrategy,
        base_name: str,
        hypothesis_id: str | int | None,
        param_space: dict[str, list[Any]],
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        primary_metric: str,
        fixed_params: dict[str, Any] | None,
        tags: list[str] | None,
        early_stop: EarlyStopConfig | None,
        n_random: int | None,
        seed: int | None = None,
    ) -> SearchResult:
        """Internal search implementation for grid and random."""
        t0 = time.monotonic()
        search_id = hashlib.sha256(
            f"{base_name}_{time.time()}".encode()
        ).hexdigest()[:10]

        # Generate parameter combinations
        if strategy == SearchStrategy.GRID:
            combos = list(self._grid_combos(param_space))
        else:
            rng = random.Random(seed)
            combos = [
                self._random_combo(param_space, rng)
                for _ in range(n_random or 50)
            ]

        logger.info(
            "Starting %s search '%s': %d combinations",
            strategy.value, base_name, len(combos),
        )

        all_results: list[dict[str, Any]] = []
        search_tags = list(tags or []) + [f"search:{search_id}"]

        for idx, combo in enumerate(combos):
            # Merge fixed params
            config = dict(fixed_params or {})
            config.update(combo)

            exp_name = f"{base_name}_s{search_id}_{idx:04d}"
            exp_id = self.create(
                name=exp_name,
                hypothesis_id=hypothesis_id,
                config=config,
                tags=search_tags,
            )

            try:
                results = self.run(exp_id, run_fn, early_stop=early_stop)
                metric_val = results.get(primary_metric, float("-inf"))
                all_results.append({
                    "experiment_id": exp_id,
                    "config": config,
                    "metric": metric_val,
                    "results": results,
                })
            except Exception as exc:
                logger.warning("Search trial %d failed: %s", idx, exc)
                all_results.append({
                    "experiment_id": exp_id,
                    "config": config,
                    "metric": float("-inf"),
                    "results": {},
                    "error": str(exc),
                })

        # Sort by primary metric descending
        all_results.sort(key=lambda r: r["metric"], reverse=True)

        best = all_results[0] if all_results else {
            "experiment_id": "", "config": {}, "metric": float("-inf"),
        }

        elapsed = time.monotonic() - t0
        return SearchResult(
            search_id=search_id,
            strategy=strategy,
            n_experiments=len(combos),
            param_space=param_space,
            best_experiment_id=best["experiment_id"],
            best_config=best["config"],
            best_metric=best["metric"],
            all_results=all_results,
            duration_seconds=elapsed,
        )

    @staticmethod
    def _grid_combos(
        param_space: dict[str, list[Any]],
    ) -> Iterator[dict[str, Any]]:
        """Generate all combinations from the parameter grid."""
        keys = sorted(param_space.keys())
        for values in itertools.product(*(param_space[k] for k in keys)):
            yield dict(zip(keys, values))

    @staticmethod
    def _random_combo(
        param_space: dict[str, list[Any]],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Pick one random combination from the parameter space."""
        return {
            k: rng.choice(v) for k, v in param_space.items()
        }

    # ------------------------------------------------------------------
    # Tagging & filtering
    # ------------------------------------------------------------------

    def add_tag(self, experiment_id: str, tag: str) -> None:
        exp = self._get(experiment_id)
        if tag not in exp.tags:
            exp.tags.append(tag)

    def remove_tag(self, experiment_id: str, tag: str) -> None:
        exp = self._get(experiment_id)
        if tag in exp.tags:
            exp.tags.remove(tag)

    def filter_by_tags(
        self, tags: list[str], match_all: bool = True,
    ) -> list[Experiment]:
        """Return experiments matching the given tags."""
        results = []
        for exp in self._experiments.values():
            tag_set = set(exp.tags)
            if match_all:
                if set(tags).issubset(tag_set):
                    results.append(exp)
            else:
                if set(tags) & tag_set:
                    results.append(exp)
        return results

    def filter_by_status(self, status: ExperimentStatus) -> list[Experiment]:
        return [e for e in self._experiments.values() if e.status == status]

    def filter_by_hypothesis(self, hypothesis_id: str | int) -> list[Experiment]:
        return [
            e for e in self._experiments.values()
            if e.hypothesis_id == hypothesis_id
        ]

    def search_experiments(
        self,
        name_contains: str | None = None,
        min_sharpe: float | None = None,
        max_sharpe: float | None = None,
        tags: list[str] | None = None,
        status: ExperimentStatus | None = None,
    ) -> list[Experiment]:
        """Flexible search across experiments."""
        results = list(self._experiments.values())

        if name_contains:
            results = [e for e in results if name_contains.lower() in e.name.lower()]
        if min_sharpe is not None:
            results = [
                e for e in results
                if (e.sharpe or e.oos_sharpe or float("-inf")) >= min_sharpe
            ]
        if max_sharpe is not None:
            results = [
                e for e in results
                if (e.sharpe or e.oos_sharpe or float("inf")) <= max_sharpe
            ]
        if tags:
            results = [e for e in results if set(tags).issubset(set(e.tags))]
        if status:
            results = [e for e in results if e.status == status]

        return results

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(
        self,
        primary_metric: str = "oos_sharpe",
        top_n: int = 20,
        tags: list[str] | None = None,
        include_statuses: list[ExperimentStatus] | None = None,
    ) -> list[LeaderboardEntry]:
        """
        Rank experiments by a primary metric.
        """
        include = include_statuses or [ExperimentStatus.COMPLETED, ExperimentStatus.STOPPED]
        candidates = [
            e for e in self._experiments.values()
            if e.status in include
        ]
        if tags:
            candidates = [e for e in candidates if set(tags).issubset(set(e.tags))]

        # Sort by metric
        def _sort_key(e: Experiment) -> float:
            return e.results.get(primary_metric, float("-inf"))

        candidates.sort(key=_sort_key, reverse=True)
        candidates = candidates[:top_n]

        entries = []
        for rank, exp in enumerate(candidates, 1):
            entries.append(LeaderboardEntry(
                rank=rank,
                experiment_id=exp.id,
                experiment_name=exp.name,
                primary_metric=exp.results.get(primary_metric, float("nan")),
                metrics=dict(exp.results),
                tags=list(exp.tags),
                status=exp.status.value,
            ))
        return entries

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def archive(self, experiment_id: str) -> None:
        """Mark an experiment as archived."""
        exp = self._get(experiment_id)
        exp.status = ExperimentStatus.ARCHIVED
        logger.info("Archived experiment %s", experiment_id)

    def archive_below_metric(
        self, metric: str, threshold: float,
    ) -> list[str]:
        """Archive all completed experiments below a metric threshold."""
        archived = []
        for exp in list(self._experiments.values()):
            if exp.status in (ExperimentStatus.COMPLETED, ExperimentStatus.STOPPED):
                val = exp.results.get(metric)
                if val is not None and val < threshold:
                    exp.status = ExperimentStatus.ARCHIVED
                    archived.append(exp.id)
        logger.info("Archived %d experiments below %s < %.4f", len(archived), metric, threshold)
        return archived

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_lineage(self, experiment_id: str) -> dict[str, Any]:
        """
        Export full experiment lineage: config history, parent chain,
        results, and metadata.
        """
        exp = self._get(experiment_id)

        # Walk parent chain
        chain: list[dict[str, Any]] = []
        current = exp
        visited: set[str] = set()
        while current is not None:
            if current.id in visited:
                break
            visited.add(current.id)
            chain.append({
                "id": current.id,
                "name": current.name,
                "config": current.config,
                "config_version": current.config_version,
                "results": current.results,
                "status": current.status.value,
            })
            if current.parent_id and current.parent_id in self._experiments:
                current = self._experiments[current.parent_id]
            else:
                current = None

        # Config changes for this experiment
        changes = [
            asdict(c) for c in self._config_changes
            if c.experiment_id == experiment_id
        ]

        return {
            "experiment": exp.to_dict(),
            "config_history": exp.config_history,
            "config_changes": changes,
            "parent_chain": list(reversed(chain)),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_all(self) -> list[dict[str, Any]]:
        """Export all experiments as a list of dicts."""
        return [exp.to_dict() for exp in self._experiments.values()]

    def export_to_json(self, path: str | Path) -> None:
        """Write all experiments to a JSON file."""
        data = {
            "experiments": self.export_all(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "n_experiments": len(self._experiments),
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info("Exported %d experiments to %s", len(self._experiments), path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, experiment_id: str) -> Experiment:
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return exp

    def _generate_id(self, name: str) -> str:
        self._counter += 1
        raw = f"{name}_{self._counter}_{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, experiment_id: str) -> Experiment:
        """Public accessor."""
        return self._get(experiment_id)

    @property
    def n_experiments(self) -> int:
        return len(self._experiments)

    def list_experiments(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Experiment]:
        """List experiments with pagination."""
        all_exps = sorted(
            self._experiments.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return all_exps[offset: offset + limit]
