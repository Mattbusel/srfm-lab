"""
experiment_runner.py — CLI experiment runner for TensorNet (Project AETERNUS).

Provides:
  - argparse-based CLI for running TT/Tucker/CP/MPS experiments
  - YAML config file loading and merging with CLI arguments
  - Experiment logging to JSON (results, metrics, configs)
  - Comparison plots: TT vs Tucker vs CP vs Full
  - Reproducible seeding (JAX, NumPy, Python random)
  - Result caching with SHA-256 config hashing
  - Experiment registry and lookup
  - Multi-run averaging with standard deviation
  - HTML report generation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================================
# Config loading
# ============================================================================

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Config dict.
    """
    try:
        import yaml
    except ImportError:
        warnings.warn("PyYAML not installed; returning empty config.")
        return {}

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-merge two config dicts. Override takes precedence.

    Args:
        base: Base configuration.
        override: Overrides to apply.

    Returns:
        Merged config dict.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a config dict.

    Args:
        config: Config to hash.

    Returns:
        Hex string hash.
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ============================================================================
# Reproducible seeding
# ============================================================================

def set_global_seed(seed: int) -> jax.random.PRNGKey:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed.

    Returns:
        JAX PRNG key.
    """
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


# ============================================================================
# Experiment result structures
# ============================================================================

@dataclass
class ExperimentResult:
    """Result container for a single experiment run."""
    experiment_name: str
    config: Dict[str, Any]
    config_hash: str
    timestamp: float
    elapsed_seconds: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)
        )
        return d


@dataclass
class ComparisonResult:
    """Result container for method comparison experiments."""
    dataset: str
    methods: List[str]
    metrics_per_method: Dict[str, Dict[str, float]]
    rank_per_method: Dict[str, int]
    n_params_per_method: Dict[str, int]
    elapsed_per_method: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Result caching
# ============================================================================

class ResultCache:
    """Cache experiment results by config hash.

    Args:
        cache_dir: Directory to store cached results.
    """

    def __init__(self, cache_dir: str = ".experiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cfg_hash: str, name: str) -> Path:
        return self.cache_dir / f"{name}_{cfg_hash}.json"

    def get(self, cfg_hash: str, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result.

        Args:
            cfg_hash: Config hash.
            name: Experiment name.

        Returns:
            Cached result dict or None.
        """
        p = self._cache_path(cfg_hash, name)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def store(self, result: ExperimentResult) -> None:
        """Store an experiment result.

        Args:
            result: ExperimentResult to cache.
        """
        p = self._cache_path(result.config_hash, result.experiment_name)
        with open(p, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def list_cached(self) -> List[str]:
        """List all cached experiment hashes."""
        return [f.stem for f in self.cache_dir.glob("*.json")]


# ============================================================================
# Experiment runner
# ============================================================================

class ExperimentRunner:
    """Run and compare tensor decomposition experiments.

    Supports running multiple methods on the same data and comparing
    reconstruction error, compression ratio, and runtime.

    Args:
        output_dir: Directory for saving results.
        use_cache: Whether to cache and reuse results.
        verbose: Print progress.
    """

    SUPPORTED_METHODS = ["tt", "tucker", "cp", "full", "mps"]

    def __init__(
        self,
        output_dir: str = "./experiments",
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.cache = ResultCache(str(self.output_dir / "cache")) if use_cache else None
        self._results: List[ExperimentResult] = []

    def run(
        self,
        name: str,
        data: np.ndarray,
        config: Dict[str, Any],
        experiment_fn: Callable[[np.ndarray, Dict[str, Any]], Dict[str, float]],
        force_rerun: bool = False,
    ) -> ExperimentResult:
        """Run a single named experiment.

        Args:
            name: Experiment name.
            data: Input data.
            config: Experiment config.
            experiment_fn: Function (data, config) -> metrics_dict.
            force_rerun: Ignore cache and rerun.

        Returns:
            ExperimentResult.
        """
        cfg_hash = config_hash(config)

        if not force_rerun and self.cache is not None:
            cached = self.cache.get(cfg_hash, name)
            if cached is not None:
                if self.verbose:
                    print(f"[cache] Loaded result for '{name}' (hash={cfg_hash})")
                result = ExperimentResult(
                    experiment_name=cached["experiment_name"],
                    config=cached["config"],
                    config_hash=cached["config_hash"],
                    timestamp=cached["timestamp"],
                    elapsed_seconds=cached["elapsed_seconds"],
                    metrics=cached["metrics"],
                    metadata=cached.get("metadata", {}),
                )
                self._results.append(result)
                return result

        seed = config.get("seed", 42)
        set_global_seed(seed)

        if self.verbose:
            print(f"Running '{name}' (hash={cfg_hash}) ...", flush=True)

        t0 = time.time()
        error = None
        metrics: Dict[str, float] = {}

        try:
            metrics = experiment_fn(data, config)
        except Exception as e:
            error = str(e)
            warnings.warn(f"Experiment '{name}' failed: {e}")

        elapsed = time.time() - t0

        result = ExperimentResult(
            experiment_name=name,
            config=config,
            config_hash=cfg_hash,
            timestamp=time.time(),
            elapsed_seconds=elapsed,
            metrics=metrics,
            error=error,
        )

        if self.cache is not None and error is None:
            self.cache.store(result)

        self._results.append(result)

        if self.verbose:
            print(f"  Elapsed: {elapsed:.2f}s")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6g}")

        return result

    def run_multi(
        self,
        name: str,
        data: np.ndarray,
        config: Dict[str, Any],
        experiment_fn: Callable[[np.ndarray, Dict[str, Any]], Dict[str, float]],
        n_runs: int = 5,
    ) -> Dict[str, Any]:
        """Run an experiment multiple times and average results.

        Args:
            name: Experiment name.
            data: Input data.
            config: Base config.
            experiment_fn: Experiment function.
            n_runs: Number of repetitions.

        Returns:
            Dict with mean and std of each metric.
        """
        all_metrics: Dict[str, List[float]] = {}

        for run in range(n_runs):
            cfg = dict(config)
            cfg["seed"] = config.get("seed", 42) + run
            result = self.run(f"{name}_run{run}", data, cfg, experiment_fn, force_rerun=True)
            for k, v in result.metrics.items():
                all_metrics.setdefault(k, []).append(v)

        summary: Dict[str, Any] = {}
        for k, vals in all_metrics.items():
            arr = np.array(vals)
            summary[f"{k}_mean"] = float(arr.mean())
            summary[f"{k}_std"] = float(arr.std())
            summary[f"{k}_values"] = vals

        return summary

    def compare_methods(
        self,
        data: np.ndarray,
        rank: int = 8,
        methods: Optional[List[str]] = None,
        dataset_name: str = "dataset",
    ) -> ComparisonResult:
        """Compare TT, Tucker, CP, and Full decomposition methods.

        Args:
            data: Input data array.
            rank: Bond dimension / Tucker rank / CP rank.
            methods: List of methods to compare. Defaults to all.
            dataset_name: Name for the dataset.

        Returns:
            ComparisonResult with per-method metrics.
        """
        if methods is None:
            methods = ["tt", "tucker", "cp", "full"]

        metrics_per_method: Dict[str, Dict[str, float]] = {}
        rank_per_method: Dict[str, int] = {}
        n_params_per_method: Dict[str, int] = {}
        elapsed_per_method: Dict[str, float] = {}

        flat = data.reshape(data.shape[0], -1)
        orig_norm = float(np.linalg.norm(flat))

        for method in methods:
            t0 = time.time()
            m_result = _run_decomposition(flat, rank, method)
            elapsed = time.time() - t0

            recon_err = float(np.linalg.norm(flat - m_result["recon"])) / (orig_norm + 1e-15)

            metrics_per_method[method] = {
                "reconstruction_error": recon_err,
                "compression_ratio": m_result["compression_ratio"],
                "variance_explained": m_result.get("variance_explained", 0.0),
            }
            rank_per_method[method] = rank
            n_params_per_method[method] = m_result["n_params"]
            elapsed_per_method[method] = elapsed

            if self.verbose:
                print(
                    f"  {method:8s}: error={recon_err:.6f}, "
                    f"ratio={m_result['compression_ratio']:.2f}x, "
                    f"time={elapsed:.2f}s"
                )

        return ComparisonResult(
            dataset=dataset_name,
            methods=methods,
            metrics_per_method=metrics_per_method,
            rank_per_method=rank_per_method,
            n_params_per_method=n_params_per_method,
            elapsed_per_method=elapsed_per_method,
        )

    def save_results_json(self, path: Optional[str] = None) -> str:
        """Save all results to a JSON file.

        Args:
            path: Output path. Defaults to output_dir/results.json.

        Returns:
            Path where results were saved.
        """
        if path is None:
            path = str(self.output_dir / "results.json")

        all_dicts = [r.to_dict() for r in self._results]
        with open(path, "w") as f:
            json.dump(all_dicts, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved {len(all_dicts)} results to {path}")

        return path

    def plot_comparison(
        self,
        comparison: ComparisonResult,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Plot method comparison bar charts.

        Args:
            comparison: ComparisonResult from compare_methods.
            save_path: Optional path to save figure.
            show: Whether to call plt.show().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available.")
            return

        methods = comparison.methods
        n_methods = len(methods)
        x = np.arange(n_methods)

        errors = [comparison.metrics_per_method[m]["reconstruction_error"] for m in methods]
        ratios = [comparison.metrics_per_method[m]["compression_ratio"] for m in methods]
        elapsed = [comparison.elapsed_per_method[m] for m in methods]
        n_params = [comparison.n_params_per_method[m] for m in methods]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Decomposition Method Comparison — {comparison.dataset}", fontsize=13)

        def bar_plot(ax, values, ylabel, title, log=False):
            bars = ax.bar(x, values, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:n_methods])
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if log and all(v > 0 for v in values):
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3g}",
                    ha="center", va="bottom", fontsize=8,
                )

        bar_plot(axes[0, 0], errors, "Relative Error", "Reconstruction Error", log=True)
        bar_plot(axes[0, 1], ratios, "Compression Ratio", "Compression Ratio")
        bar_plot(axes[1, 0], elapsed, "Time (s)", "Runtime", log=True)
        bar_plot(axes[1, 1], n_params, "# Parameters", "Parameter Count", log=True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def _run_decomposition(
    flat_data: np.ndarray,
    rank: int,
    method: str,
) -> Dict[str, Any]:
    """Run a specific decomposition method on flattened data.

    Args:
        flat_data: Data matrix (n, d).
        rank: Decomposition rank.
        method: "tt", "tucker", "cp", or "full".

    Returns:
        Dict with recon, n_params, compression_ratio, variance_explained.
    """
    n, d = flat_data.shape
    orig_size = n * d

    U, s, Vt = np.linalg.svd(flat_data, full_matrices=False)
    r = min(rank, len(s))

    if method in ("tt", "mps"):
        recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
        n_params = n * r + r + r * d
    elif method == "tucker":
        # Tucker as 2-mode: U * s (left factor) and Vt (right factor)
        recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
        n_params = n * r + r * r + r * d  # core + two factors
    elif method == "cp":
        # CP as rank-1 sum (proxy)
        recon = sum(
            s[i] * np.outer(U[:, i], Vt[i, :])
            for i in range(r)
        )
        n_params = r * (n + d)
    elif method == "full":
        recon = flat_data.copy()
        n_params = orig_size
    else:
        raise ValueError(f"Unknown method: {method}")

    s_sq = s ** 2
    var_explained = float(s_sq[:r].sum() / (s_sq.sum() + 1e-15))

    return {
        "recon": recon,
        "n_params": n_params,
        "compression_ratio": orig_size / max(1, n_params),
        "variance_explained": var_explained,
    }


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="TensorNet Experiment Runner — AETERNUS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML config file path.")
    parser.add_argument("--experiment", type=str, default="comparison",
                        choices=["comparison", "rank_sweep", "stability", "custom"],
                        help="Experiment type.")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (CSV/npy).")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples.")
    parser.add_argument("--n-features", type=int, default=64, help="Feature dimension.")
    parser.add_argument("--rank", type=int, default=8, help="TT/Tucker/CP rank.")
    parser.add_argument("--max-rank", type=int, default=32, help="Maximum rank for sweeps.")
    parser.add_argument("--methods", nargs="+", default=["tt", "tucker", "cp", "full"],
                        help="Decomposition methods to compare.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                        help="Output directory.")
    parser.add_argument("--no-cache", action="store_true", help="Disable result caching.")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots.")
    parser.add_argument("--save-plots", type=str, default=None, help="Directory to save plots.")

    return parser


def load_or_generate_data(
    args: argparse.Namespace,
) -> np.ndarray:
    """Load data from file or generate synthetic data.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Data array (n_samples, n_features).
    """
    if args.data is not None:
        path = args.data
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".csv"):
            try:
                import pandas as pd
                data = pd.read_csv(path).values
            except ImportError:
                data = np.genfromtxt(path, delimiter=",", skip_header=1)
        else:
            raise ValueError(f"Unsupported data format: {path}")
        return data.astype(np.float32)

    # Generate synthetic data
    rng = np.random.default_rng(args.seed)
    # Low-rank structure for interesting compression comparison
    U = rng.normal(0, 1, (args.n_samples, args.rank))
    V = rng.normal(0, 1, (args.rank, args.n_features))
    data = (U @ V + rng.normal(0, 0.1, (args.n_samples, args.n_features))).astype(np.float32)
    return data


def run_cli(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        Exit code (0 = success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load YAML config
    config = {}
    if args.config:
        config = load_yaml_config(args.config)

    # Merge with CLI args
    cli_config = {
        "seed": args.seed,
        "rank": args.rank,
        "max_rank": args.max_rank,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "methods": args.methods,
    }
    config = merge_configs(config, cli_config)

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        verbose=args.verbose,
    )

    data = load_or_generate_data(args)
    print(f"Data shape: {data.shape}")

    if args.experiment == "comparison":
        print("\n--- Method Comparison ---")
        comparison = runner.compare_methods(
            data,
            rank=config.get("rank", 8),
            methods=config.get("methods", ["tt", "tucker", "cp", "full"]),
            dataset_name=args.data or "synthetic",
        )

        # Save JSON
        result_path = runner.save_results_json()
        print(f"\nResults saved to {result_path}")

        if args.plot or args.save_plots:
            save_path = None
            if args.save_plots:
                os.makedirs(args.save_plots, exist_ok=True)
                save_path = os.path.join(args.save_plots, "comparison.png")
            runner.plot_comparison(comparison, save_path=save_path, show=args.plot)

    elif args.experiment == "rank_sweep":
        print("\n--- Rank Sweep ---")
        from tensor_net.rank_selection import rank_sweep_cv

        ranks = [1, 2, 4, 8, 16, 32][:sum(1 for r in [1, 2, 4, 8, 16, 32] if r <= args.max_rank)]
        sweep_result = rank_sweep_cv(data, ranks=ranks, verbose=args.verbose)
        print(sweep_result.summary())

        if args.plot or args.save_plots:
            from tensor_net.rank_selection import plot_rank_sweep
            save_path = None
            if args.save_plots:
                os.makedirs(args.save_plots, exist_ok=True)
                save_path = os.path.join(args.save_plots, "rank_sweep.png")
            plot_rank_sweep(sweep_result, save_path=save_path, show=args.plot)

    return 0


def main():
    """Entry point for the experiment runner CLI."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()


# ============================================================================
# HTML report generation
# ============================================================================

def generate_html_report(
    results: List[ExperimentResult],
    comparison: Optional["ComparisonResult"] = None,
    output_path: str = "./experiment_report.html",
    title: str = "TensorNet Experiment Report",
) -> str:
    """Generate an HTML report from experiment results.

    Args:
        results: List of ExperimentResult objects.
        comparison: Optional ComparisonResult for method comparison.
        output_path: Path to save the HTML report.
        title: Report title.

    Returns:
        Path to saved HTML file.
    """
    rows = []
    for r in results:
        metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in r.metrics.items())
        status = "OK" if r.error is None else f"ERROR: {r.error}"
        rows.append(
            f"<tr>"
            f"<td>{r.experiment_name}</td>"
            f"<td>{r.config_hash}</td>"
            f"<td>{r.elapsed_seconds:.2f}s</td>"
            f"<td>{metrics_str}</td>"
            f"<td>{status}</td>"
            f"</tr>"
        )

    comparison_section = ""
    if comparison is not None:
        method_rows = []
        for m in comparison.methods:
            metrics = comparison.metrics_per_method[m]
            method_rows.append(
                f"<tr>"
                f"<td>{m}</td>"
                f"<td>{metrics['reconstruction_error']:.6f}</td>"
                f"<td>{metrics['compression_ratio']:.2f}x</td>"
                f"<td>{comparison.elapsed_per_method[m]:.3f}s</td>"
                f"<td>{comparison.n_params_per_method[m]}</td>"
                f"</tr>"
            )
        comparison_section = f"""
        <h2>Method Comparison — {comparison.dataset}</h2>
        <table border="1" cellpadding="4">
        <tr><th>Method</th><th>Error</th><th>Compression</th><th>Time</th><th>#Params</th></tr>
        {"".join(method_rows)}
        </table>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        h1, h2 {{ color: #4fc3f7; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th {{ background: #0d47a1; color: white; padding: 8px; text-align: left; }}
        td {{ padding: 6px; border: 1px solid #444; }}
        tr:nth-child(even) {{ background: #263238; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Total experiments: {len(results)}</p>
    {comparison_section}
    <h2>Experiment Results</h2>
    <table>
    <tr><th>Name</th><th>Hash</th><th>Time</th><th>Metrics</th><th>Status</th></tr>
    {"".join(rows)}
    </table>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


# ============================================================================
# Experiment registry
# ============================================================================

class ExperimentRegistry:
    """Registry for experiment definitions.

    Allows registering named experiment functions and listing/running them.

    Usage::

        registry = ExperimentRegistry()

        @registry.register("tt_compression")
        def run_tt(data, config):
            ...
            return {"mse": 0.01}

        result = registry.run("tt_compression", data, config)
    """

    def __init__(self):
        self._experiments: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register an experiment function.

        Args:
            name: Experiment name.
            description: Human-readable description.
            tags: Optional list of tags.

        Returns:
            Decorator function.
        """
        def decorator(fn: Callable) -> Callable:
            self._experiments[name] = fn
            self._metadata[name] = {
                "description": description,
                "tags": tags or [],
                "function_name": fn.__name__,
            }
            return fn
        return decorator

    def run(
        self,
        name: str,
        data: np.ndarray,
        config: Dict[str, Any],
    ) -> ExperimentResult:
        """Run a registered experiment.

        Args:
            name: Experiment name.
            data: Input data.
            config: Experiment config.

        Returns:
            ExperimentResult.
        """
        if name not in self._experiments:
            raise KeyError(f"Experiment '{name}' not registered.")

        runner = ExperimentRunner(verbose=False)
        return runner.run(name, data, config, self._experiments[name])

    def list_experiments(
        self,
        tag_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all registered experiments.

        Args:
            tag_filter: If provided, only return experiments with this tag.

        Returns:
            List of experiment metadata dicts.
        """
        experiments = []
        for name, meta in self._metadata.items():
            if tag_filter is None or tag_filter in meta.get("tags", []):
                experiments.append({"name": name, **meta})
        return experiments

    def run_all(
        self,
        data: np.ndarray,
        config: Dict[str, Any],
        tag_filter: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """Run all registered (optionally tag-filtered) experiments.

        Args:
            data: Input data.
            config: Common config.
            tag_filter: Optional tag filter.

        Returns:
            List of ExperimentResult objects.
        """
        to_run = self.list_experiments(tag_filter)
        results = []
        for exp in to_run:
            try:
                r = self.run(exp["name"], data, config)
                results.append(r)
            except Exception as e:
                warnings.warn(f"Experiment '{exp['name']}' failed: {e}")
        return results


# ============================================================================
# Multi-metric experiment comparison
# ============================================================================

def compare_experiments_table(
    results: List[ExperimentResult],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format experiment results as a comparison table string.

    Args:
        results: List of ExperimentResult objects.
        metrics: Metrics to include. If None, uses all.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to compare."

    if metrics is None:
        metrics = sorted(set(k for r in results for k in r.metrics))

    # Header
    col_width = 16
    name_width = 30
    header = f"{'Experiment':{name_width}}"
    for m in metrics:
        header += f" {m[:col_width-1]:{col_width}}"
    header += f" {'Time(s)':{col_width}}"

    sep = "-" * len(header)
    rows = [sep, header, sep]

    for r in results:
        row = f"{r.experiment_name[:name_width-1]:{name_width}}"
        for m in metrics:
            val = r.metrics.get(m)
            if val is not None:
                row += f" {val:{col_width}.4g}"
            else:
                row += f" {'N/A':{col_width}}"
        row += f" {r.elapsed_seconds:{col_width}.2f}"
        rows.append(row)

    rows.append(sep)
    return "\n".join(rows)


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

