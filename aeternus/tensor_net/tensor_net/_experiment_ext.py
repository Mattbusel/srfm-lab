"""Extension for experiment_runner.py — appended programmatically."""


# ---------------------------------------------------------------------------
# Section: Experiment logging and analysis utilities
# ---------------------------------------------------------------------------

import os
import json
import time
import hashlib
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    name: str
    elapsed_s: float
    n_iterations: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    throughput: float   # iterations per second
    metadata: dict = field(default_factory=dict)


def benchmark_function(
    fn,
    args: tuple = (),
    kwargs: dict | None = None,
    n_warmup: int = 3,
    n_iters: int = 10,
    name: str = "unknown",
) -> BenchmarkResult:
    """
    Benchmark a function with warmup and multiple iterations.

    Parameters
    ----------
    fn : callable
    args : tuple
    kwargs : dict, optional
    n_warmup : int
    n_iters : int
    name : str

    Returns
    -------
    BenchmarkResult
    """
    if kwargs is None:
        kwargs = {}

    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return BenchmarkResult(
        name=name,
        elapsed_s=times_arr.sum(),
        n_iterations=n_iters,
        mean_s=float(times_arr.mean()),
        std_s=float(times_arr.std()),
        min_s=float(times_arr.min()),
        max_s=float(times_arr.max()),
        throughput=float(n_iters / times_arr.sum()),
    )


def compare_benchmarks(results: list) -> dict:
    """
    Compare multiple BenchmarkResult objects.

    Returns dict with speedup ratios relative to the slowest.
    """
    if not results:
        return {}
    sorted_results = sorted(results, key=lambda r: r.mean_s)
    baseline_mean = results[0].mean_s
    comparison = {}
    for r in sorted_results:
        comparison[r.name] = {
            "mean_s": r.mean_s,
            "std_s": r.std_s,
            "speedup_vs_baseline": baseline_mean / (r.mean_s + 1e-12),
            "throughput": r.throughput,
        }
    return comparison


class ExperimentLogger:
    """
    Simple structured logger for experiment results.

    Appends JSON records to a log file and maintains an in-memory buffer.

    Parameters
    ----------
    log_path : str
        Path to the log file.
    buffer_size : int
        Flush to disk every N records.
    """

    def __init__(self, log_path: str, buffer_size: int = 10) -> None:
        self.log_path = log_path
        self.buffer_size = buffer_size
        self._buffer: list = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None

    def log(self, record: dict) -> None:
        """Append a record (dict) to the log."""
        record["_timestamp"] = time.time()
        self._buffer.append(record)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to disk."""
        if not self._buffer:
            return
        with open(self.log_path, "a") as fh:
            for rec in self._buffer:
                fh.write(json.dumps(rec) + "\n")
        self._buffer.clear()

    def read_all(self) -> list:
        """Read all records from the log file."""
        self.flush()
        records = []
        if not os.path.exists(self.log_path):
            return records
        with open(self.log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def close(self) -> None:
        self.flush()


class HyperparameterGrid:
    """
    Grid search over hyperparameters.

    Parameters
    ----------
    param_grid : dict mapping param_name -> list of values
    """

    def __init__(self, param_grid: dict) -> None:
        self.param_grid = param_grid
        self._combinations: list = []
        self._build_combinations()

    def _build_combinations(self) -> None:
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        self._combinations = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

    def __len__(self) -> int:
        return len(self._combinations)

    def __iter__(self):
        return iter(self._combinations)

    def __getitem__(self, idx: int) -> dict:
        return self._combinations[idx]

    def combination_hash(self, combo: dict) -> str:
        """Return a short hash of a hyperparameter combination."""
        key = json.dumps(combo, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]


class GridSearchRunner:
    """
    Runs a function over a hyperparameter grid and records results.

    Parameters
    ----------
    grid : HyperparameterGrid
    objective_fn : callable(params: dict) -> dict
        Function that accepts hyperparameters and returns a result dict
        with at least a ``"score"`` key.
    logger : ExperimentLogger, optional
    verbose : bool
    """

    def __init__(
        self,
        grid: HyperparameterGrid,
        objective_fn,
        logger=None,
        verbose: bool = False,
    ) -> None:
        self.grid = grid
        self.objective_fn = objective_fn
        self.logger = logger
        self.verbose = verbose
        self.results: list = []
        self.best_params: dict | None = None
        self.best_score: float = float("-inf")

    def run(self) -> list:
        """
        Execute grid search.

        Returns
        -------
        list of (params, result) tuples.
        """
        for i, params in enumerate(self.grid):
            if self.verbose:
                print(f"  [{i+1}/{len(self.grid)}] params={params}")
            try:
                result = self.objective_fn(params)
                score = float(result.get("score", float("-inf")))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = dict(params)
                entry = {"params": params, "result": result, "score": score}
                self.results.append(entry)
                if self.logger is not None:
                    self.logger.log(entry)
            except Exception as exc:
                warnings.warn(f"GridSearch error for params={params}: {exc}")
                self.results.append({"params": params, "result": {"error": str(exc)}, "score": float("-inf")})

        return self.results

    def top_k(self, k: int = 5) -> list:
        """Return top-k results by score."""
        return sorted(self.results, key=lambda x: x["score"], reverse=True)[:k]

    def summary(self) -> dict:
        if not self.results:
            return {}
        scores = [r["score"] for r in self.results if np.isfinite(r["score"])]
        return {
            "n_evaluated": len(self.results),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "mean_score": float(np.mean(scores)) if scores else None,
            "std_score": float(np.std(scores)) if scores else None,
        }


class RandomSearchRunner:
    """
    Random hyperparameter search.

    Parameters
    ----------
    param_distributions : dict
        Mapping param_name -> list of values or callable() -> value.
    objective_fn : callable(params: dict) -> dict
    n_trials : int
    random_seed : int
    logger : ExperimentLogger, optional
    verbose : bool
    """

    def __init__(
        self,
        param_distributions: dict,
        objective_fn,
        n_trials: int = 20,
        random_seed: int = 42,
        logger=None,
        verbose: bool = False,
    ) -> None:
        self.param_distributions = param_distributions
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.rng = np.random.default_rng(random_seed)
        self.logger = logger
        self.verbose = verbose
        self.results: list = []
        self.best_params: dict | None = None
        self.best_score: float = float("-inf")

    def _sample(self) -> dict:
        params = {}
        for name, dist in self.param_distributions.items():
            if callable(dist):
                params[name] = dist()
            elif isinstance(dist, list):
                params[name] = dist[int(self.rng.integers(0, len(dist)))]
            else:
                params[name] = dist
        return params

    def run(self) -> list:
        """Execute random search."""
        for i in range(self.n_trials):
            params = self._sample()
            if self.verbose:
                print(f"  [{i+1}/{self.n_trials}] params={params}")
            try:
                result = self.objective_fn(params)
                score = float(result.get("score", float("-inf")))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = dict(params)
                entry = {"params": params, "result": result, "score": score}
                self.results.append(entry)
                if self.logger is not None:
                    self.logger.log(entry)
            except Exception as exc:
                warnings.warn(f"RandomSearch error for params={params}: {exc}")
                self.results.append({"params": params, "result": {"error": str(exc)}, "score": float("-inf")})

        return self.results

    def top_k(self, k: int = 5) -> list:
        return sorted(self.results, key=lambda x: x["score"], reverse=True)[:k]

    def summary(self) -> dict:
        if not self.results:
            return {}
        scores = [r["score"] for r in self.results if np.isfinite(r["score"])]
        return {
            "n_trials": self.n_trials,
            "n_evaluated": len(self.results),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "mean_score": float(np.mean(scores)) if scores else None,
            "std_score": float(np.std(scores)) if scores else None,
        }


class ExperimentCheckpointer:
    """
    Checkpoint manager for long-running experiments.

    Saves periodic checkpoints and allows resuming from the latest.

    Parameters
    ----------
    checkpoint_dir : str
    experiment_name : str
    save_every_n : int
        Save a checkpoint every N steps.
    keep_last_k : int
        Keep only the last K checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str = "/tmp/tt_checkpoints",
        experiment_name: str = "exp",
        save_every_n: int = 10,
        keep_last_k: int = 3,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.save_every_n = save_every_n
        self.keep_last_k = keep_last_k
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._step = 0
        self._saved_checkpoints: list = []

    def step(self, state: dict) -> bool:
        """
        Increment step counter and optionally save a checkpoint.

        Parameters
        ----------
        state : dict
            Serialisable state dict.

        Returns
        -------
        bool : True if a checkpoint was saved.
        """
        self._step += 1
        if self._step % self.save_every_n == 0:
            self.save(state, self._step)
            return True
        return False

    def save(self, state: dict, step: int) -> str:
        """Save a checkpoint. Returns the path."""
        fname = f"{self.experiment_name}_step{step:06d}.json"
        path = os.path.join(self.checkpoint_dir, fname)
        serialisable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                serialisable[k] = {"_type": "ndarray", "data": v.tolist(), "shape": list(v.shape)}
            else:
                try:
                    json.dumps(v)
                    serialisable[k] = v
                except TypeError:
                    serialisable[k] = str(v)
        serialisable["_step"] = step
        serialisable["_timestamp"] = time.time()
        with open(path, "w") as fh:
            json.dump(serialisable, fh)
        self._saved_checkpoints.append(path)
        # Prune old checkpoints
        while len(self._saved_checkpoints) > self.keep_last_k:
            old = self._saved_checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)
        return path

    def load_latest(self) -> dict | None:
        """Load the most recent checkpoint."""
        if not self._saved_checkpoints:
            # Try to find existing checkpoints on disk
            import glob
            pattern = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_step*.json")
            files = sorted(glob.glob(pattern))
            if not files:
                return None
            path = files[-1]
        else:
            path = self._saved_checkpoints[-1]

        if not os.path.exists(path):
            return None

        with open(path) as fh:
            data = json.load(fh)
        # Reconstruct numpy arrays
        for k, v in data.items():
            if isinstance(v, dict) and v.get("_type") == "ndarray":
                data[k] = np.array(v["data"], dtype=np.float32).reshape(v["shape"])
        return data

    @property
    def current_step(self) -> int:
        return self._step
