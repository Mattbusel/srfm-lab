"""
walk_forward_engine.py — Full Walk-Forward Validation Engine

Finds best params in-sample, applies out-of-sample.
Tracks: param stability, OOS performance, degradation ratio (IS Sharpe / OOS Sharpe).

Usage:
    from tools.walk_forward_engine import WalkForwardEngine, WalkForwardResult
    engine = WalkForwardEngine(data, strategy_fn, param_grid, train_bars=252, test_bars=63)
    result = engine.run()
    WalkForwardEngine.plot_results(result)
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_idx:       int
    train_start:      int
    train_end:        int
    test_start:       int
    test_end:         int
    best_params:      Dict[str, Any]
    is_sharpe:        float     # in-sample Sharpe with best params
    oos_sharpe:       float     # out-of-sample Sharpe
    is_return:        float
    oos_return:       float
    is_equity_curve:  List[float]
    oos_equity_curve: List[float]
    all_param_sharpes: List[Tuple[Dict[str, Any], float]]  # (params, IS Sharpe)


@dataclass
class WalkForwardResult:
    windows:            List[WindowResult]
    combined_oos_curve: List[float]          # spliced OOS equity curve
    is_sharpes:         List[float]
    oos_sharpes:        List[float]
    degradation_ratio:  float                # mean(IS) / mean(OOS)
    param_stability:    Dict[str, float]     # fraction of windows each param was chosen
    summary_stats:      Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy wrapper: default simple BH trend-following
# ─────────────────────────────────────────────────────────────────────────────

def default_bh_strategy(
    closes: np.ndarray,
    params: Dict[str, Any],
    starting_equity: float = 1.0,
) -> np.ndarray:
    """
    Default strategy: long when BH active, flat otherwise.
    Returns equity curve array.
    """
    from srfm_core import MinkowskiClassifier, BlackHoleDetector

    cf          = params.get("cf", 0.001)
    bh_form     = params.get("bh_form", 1.5)
    bh_collapse = params.get("bh_collapse", 1.0)
    bh_decay    = params.get("bh_decay", 0.95)
    pos_size    = params.get("pos_size", 0.25)

    mc  = MinkowskiClassifier(cf=float(cf))
    bh  = BlackHoleDetector(float(bh_form), float(bh_collapse), float(bh_decay))
    mc.update(float(closes[0]))

    equity = starting_equity
    curve  = [equity]

    for i in range(1, len(closes)):
        bit = mc.update(float(closes[i]))
        bh.update(bit, float(closes[i]), float(closes[i-1]))
        bar_ret = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)

        if bh.bh_active and bh.bh_dir > 0:
            equity *= (1.0 + pos_size * bar_ret)
        elif bh.bh_active and bh.bh_dir < 0:
            equity *= (1.0 - pos_size * bar_ret)

        equity = max(0.0, equity)
        curve.append(equity)

    return np.array(curve)


def compute_sharpe(equity_curve: np.ndarray, periods_per_year: float = 252) -> float:
    """Compute annualized Sharpe ratio from equity curve."""
    if len(equity_curve) < 3:
        return 0.0
    rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    std  = rets.std()
    if std < 1e-10:
        return 0.0
    return float(rets.mean() / std * math.sqrt(periods_per_year))


def compute_cagr(equity_curve: np.ndarray, periods_per_year: float = 252) -> float:
    """Compute CAGR from equity curve."""
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    n_years = max(0.001, len(equity_curve) / periods_per_year)
    return float((equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1.0)


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown as a fraction (negative value)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd   = (equity_curve - peak) / (peak + 1e-10)
    return float(dd.min())


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Full walk-forward validation engine.

    Parameters
    ----------
    data          : price DataFrame with 'close' column, or 1D close array
    strategy_fn   : callable(closes, params, starting_equity) → equity_curve array
                    Defaults to default_bh_strategy if None
    param_grid    : dict of {param_name: [values_to_try]}
    train_bars    : number of bars in in-sample training window
    test_bars     : number of bars in out-of-sample test window
    step_bars     : how many bars to advance window each iteration
                    (default = test_bars for non-overlapping OOS)
    metric        : optimization metric: 'sharpe' | 'cagr' | 'calmar'
    starting_equity: starting equity for simulation
    verbose       : print progress
    """

    def __init__(
        self,
        data: Any,
        strategy_fn: Optional[Callable] = None,
        param_grid:  Optional[Dict[str, List[Any]]] = None,
        train_bars:  int = 252,
        test_bars:   int = 63,
        step_bars:   Optional[int] = None,
        metric:      str = "sharpe",
        starting_equity: float = 1.0,
        verbose: bool = True,
    ):
        self.strategy_fn    = strategy_fn or default_bh_strategy
        self.param_grid     = param_grid  or {
            "cf":       [0.0008, 0.001, 0.0012, 0.0015],
            "bh_form":  [1.0, 1.5, 2.0],
            "pos_size": [0.15, 0.25, 0.35],
        }
        self.train_bars     = train_bars
        self.test_bars      = test_bars
        self.step_bars      = step_bars or test_bars
        self.metric         = metric
        self.starting_equity = starting_equity
        self.verbose        = verbose

        # Extract close array
        if isinstance(data, pd.DataFrame):
            for col in ("close", "Close", "CLOSE"):
                if col in data.columns:
                    self.closes = data[col].values.astype(float)
                    break
            else:
                self.closes = data.iloc[:, 0].values.astype(float)
        else:
            self.closes = np.array(data, dtype=float)

        self._param_combinations = self._build_param_combinations()

    def _build_param_combinations(self) -> List[Dict[str, Any]]:
        """Expand param_grid into list of all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    def _evaluate_params(
        self, closes: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        """Run strategy and return (metric_value, equity_curve)."""
        try:
            curve = self.strategy_fn(closes, params, self.starting_equity)
        except Exception:
            return -1e9, np.ones(len(closes))

        if self.metric == "sharpe":
            m = compute_sharpe(curve)
        elif self.metric == "cagr":
            m = compute_cagr(curve)
        elif self.metric == "calmar":
            dd = abs(compute_max_drawdown(curve))
            m  = compute_cagr(curve) / (dd + 1e-3)
        else:
            m = compute_sharpe(curve)

        return float(m) if math.isfinite(float(m)) else -1e9, curve

    def run(self) -> WalkForwardResult:
        """
        Run walk-forward validation over all available data.

        Returns WalkForwardResult with per-window stats and spliced OOS curve.
        """
        n = len(self.closes)
        n_windows = max(0, (n - self.train_bars - self.test_bars) // self.step_bars + 1)

        if n_windows == 0:
            raise ValueError(
                f"Insufficient data: {n} bars < {self.train_bars + self.test_bars} required "
                f"(train={self.train_bars} + test={self.test_bars})"
            )

        if self.verbose:
            print(f"Walk-forward: {n} bars, {len(self._param_combinations)} param combos, "
                  f"{n_windows} windows")

        windows: List[WindowResult] = []
        combined_oos: List[float] = [self.starting_equity]

        for w in range(n_windows):
            train_start = w * self.step_bars
            train_end   = train_start + self.train_bars
            test_start  = train_end
            test_end    = min(test_start + self.test_bars, n)

            if test_end <= test_start:
                break

            train_closes = self.closes[train_start:train_end]
            test_closes  = self.closes[test_start:test_end]

            if self.verbose:
                print(f"  Window {w+1}/{n_windows}: train[{train_start}:{train_end}] "
                      f"test[{test_start}:{test_end}]")

            # In-sample optimization
            best_params = self._param_combinations[0]
            best_is     = -1e9
            all_param_sharpes = []

            for params in self._param_combinations:
                is_metric, is_curve = self._evaluate_params(train_closes, params)
                all_param_sharpes.append((dict(params), float(is_metric)))
                if is_metric > best_is:
                    best_is     = is_metric
                    best_params = params
                    best_is_curve = is_curve

            # Out-of-sample evaluation with best params
            oos_metric, oos_curve = self._evaluate_params(test_closes, best_params)
            is_sharpe  = compute_sharpe(best_is_curve)
            oos_sharpe = compute_sharpe(oos_curve)

            # Splice OOS curve
            oos_scale = combined_oos[-1] / (oos_curve[0] + 1e-10)
            combined_oos.extend([v * oos_scale for v in oos_curve[1:]])

            windows.append(WindowResult(
                window_idx=w,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=dict(best_params),
                is_sharpe=float(is_sharpe),
                oos_sharpe=float(oos_sharpe),
                is_return=float(compute_cagr(best_is_curve)),
                oos_return=float(compute_cagr(oos_curve)),
                is_equity_curve=best_is_curve.tolist(),
                oos_equity_curve=oos_curve.tolist(),
                all_param_sharpes=all_param_sharpes,
            ))

            if self.verbose:
                print(f"    Best: {best_params}  IS Sharpe={is_sharpe:.2f}  OOS Sharpe={oos_sharpe:.2f}")

        # Aggregate stats
        is_sharpes  = [w.is_sharpe  for w in windows]
        oos_sharpes = [w.oos_sharpe for w in windows]
        mean_is  = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        degradation = mean_is / (mean_oos + 1e-10) if mean_oos != 0 else float("inf")

        # Parameter stability: which param value was chosen most often?
        param_stability = self._compute_param_stability(windows)

        # OOS combined curve stats
        oos_arr    = np.array(combined_oos)
        oos_sharpe_combined = compute_sharpe(oos_arr)
        oos_cagr_combined   = compute_cagr(oos_arr)
        oos_dd_combined     = compute_max_drawdown(oos_arr)

        summary_stats = {
            "n_windows":         len(windows),
            "n_param_combos":    len(self._param_combinations),
            "mean_is_sharpe":    round(mean_is, 3),
            "mean_oos_sharpe":   round(mean_oos, 3),
            "degradation_ratio": round(float(degradation), 3) if math.isfinite(float(degradation)) else 99.0,
            "oos_sharpe_combined": round(float(oos_sharpe_combined), 3),
            "oos_cagr_combined":   round(float(oos_cagr_combined), 4),
            "oos_maxdd_combined":  round(float(oos_dd_combined), 4),
            "pct_windows_oos_positive": round(float(np.mean([v > 0 for v in oos_sharpes])), 3),
        }

        return WalkForwardResult(
            windows=windows,
            combined_oos_curve=combined_oos,
            is_sharpes=is_sharpes,
            oos_sharpes=oos_sharpes,
            degradation_ratio=float(degradation),
            param_stability=param_stability,
            summary_stats=summary_stats,
        )

    def _compute_param_stability(
        self, windows: List[WindowResult]
    ) -> Dict[str, float]:
        """
        For each parameter, compute the fraction of windows that chose each value.
        Returns: {f"{param}={value}": fraction} for dominant choice per param.
        """
        if not windows:
            return {}

        stability = {}
        keys = list(self.param_grid.keys())

        for key in keys:
            value_counts: Dict[str, int] = {}
            for w in windows:
                val = str(w.best_params.get(key, "N/A"))
                value_counts[val] = value_counts.get(val, 0) + 1
            total = len(windows)
            dominant = max(value_counts, key=lambda k: value_counts[k])
            fraction = value_counts[dominant] / total
            stability[f"{key}={dominant}"] = round(fraction, 3)

        return stability

    @staticmethod
    def plot_results(result: "WalkForwardResult") -> None:
        """
        Plot walk-forward results:
          1. IS vs OOS Sharpe per window
          2. Combined OOS equity curve
          3. Parameter stability bar chart
          4. Degradation ratio visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Walk-Forward Validation Results", fontsize=12, fontweight="bold")

        # 1. IS vs OOS Sharpe
        ax = axes[0, 0]
        x = np.arange(len(result.is_sharpes))
        ax.plot(x, result.is_sharpes,  "b-o", markersize=4, linewidth=1.0, label="IS Sharpe")
        ax.plot(x, result.oos_sharpes, "r-s", markersize=4, linewidth=1.0, label="OOS Sharpe")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.fill_between(x, result.oos_sharpes, 0,
                       where=[v < 0 for v in result.oos_sharpes],
                       alpha=0.2, color="red", label="Negative OOS")
        ax.set_title("IS vs OOS Sharpe per Window", fontsize=9)
        ax.set_xlabel("Window #"); ax.set_ylabel("Sharpe Ratio")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # 2. Combined OOS equity curve
        ax = axes[0, 1]
        eq = np.array(result.combined_oos_curve)
        ax.plot(eq, color="steelblue", linewidth=1.2)
        ax.fill_between(range(len(eq)), 1.0, eq, where=eq >= 1.0, alpha=0.15, color="green")
        ax.fill_between(range(len(eq)), 1.0, eq, where=eq < 1.0, alpha=0.15, color="red")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"Combined OOS Equity Curve\n"
                     f"Sharpe={result.summary_stats['oos_sharpe_combined']:.2f}  "
                     f"CAGR={result.summary_stats['oos_cagr_combined']:.1%}", fontsize=9)
        ax.set_xlabel("Bar"); ax.set_ylabel("Portfolio (×Start)")
        ax.grid(alpha=0.3)

        # 3. Parameter stability
        ax = axes[1, 0]
        if result.param_stability:
            labels = list(result.param_stability.keys())
            values = list(result.param_stability.values())
            ax.barh(labels, values, color="teal", alpha=0.7)
            ax.axvline(0.5, color="orange", linestyle="--", linewidth=0.8, label="50%")
            ax.set_title("Parameter Stability (dominant choice %)", fontsize=9)
            ax.set_xlabel("Fraction of Windows"); ax.legend(fontsize=7)
            ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No param data", ha="center", va="center")

        # 4. Degradation analysis
        ax = axes[1, 1]
        ratios = [w.is_sharpe / (abs(w.oos_sharpe) + 1e-3) for w in result.windows if w.is_sharpe > 0]
        if ratios:
            ax.hist(ratios, bins=min(20, len(ratios)), color="orange", alpha=0.7)
            ax.axvline(1.0, color="green", linestyle="--", linewidth=0.8, label="No degradation")
            ax.axvline(result.degradation_ratio, color="red", linestyle="-",
                      linewidth=1.5, label=f"Mean={result.degradation_ratio:.2f}")
            ax.set_title("IS/OOS Sharpe Ratio Distribution", fontsize=9)
            ax.set_xlabel("Degradation Ratio"); ax.set_ylabel("Count")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        plt.tight_layout()
        try:
            out = Path(__file__).parent.parent / "results" / "walk_forward_results.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=120, bbox_inches="tight")
            print(f"Plot saved → {out}")
        except Exception as e:
            print(f"Could not save plot: {e}")
        plt.show()

    @staticmethod
    def compute_degradation_ratio(result: "WalkForwardResult") -> float:
        """
        IS Sharpe / OOS Sharpe.
        Close to 1.0 = no overfitting.
        > 2.0 = significant overfitting.
        """
        return result.degradation_ratio

    def print_summary(self, result: "WalkForwardResult") -> None:
        """Print a formatted summary of walk-forward results."""
        stats = result.summary_stats
        print("\n" + "=" * 55)
        print("  Walk-Forward Validation Summary")
        print("=" * 55)
        print(f"  Windows:             {stats['n_windows']}")
        print(f"  Param combos:        {stats['n_param_combos']}")
        print(f"  Mean IS Sharpe:      {stats['mean_is_sharpe']:.3f}")
        print(f"  Mean OOS Sharpe:     {stats['mean_oos_sharpe']:.3f}")
        print(f"  Degradation ratio:   {stats['degradation_ratio']:.3f}")
        print(f"  OOS Sharpe (combined): {stats['oos_sharpe_combined']:.3f}")
        print(f"  OOS CAGR (combined):   {stats['oos_cagr_combined']:.1%}")
        print(f"  OOS Max DD:            {stats['oos_maxdd_combined']:.1%}")
        print(f"  Windows with OOS>0:   {stats['pct_windows_oos_positive']:.0%}")
        print("\n  Parameter Stability:")
        for k, v in result.param_stability.items():
            print(f"    {k}: {v:.0%} of windows")
        print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    closes: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    train_bars: int = 252,
    test_bars:  int = 63,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Convenience function: run walk-forward with default BH strategy.

    Parameters
    ----------
    closes     : array of close prices
    param_grid : parameter grid for sweep (uses default if None)
    train_bars : in-sample window length
    test_bars  : out-of-sample window length
    verbose    : print progress

    Returns
    -------
    WalkForwardResult
    """
    engine = WalkForwardEngine(
        data=closes,
        param_grid=param_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        verbose=verbose,
    )
    result = engine.run()
    if verbose:
        engine.print_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--csv",    default=None, help="Path to OHLCV CSV")
    parser.add_argument("--n-bars", type=int, default=2000)
    parser.add_argument("--train",  type=int, default=252)
    parser.add_argument("--test",   type=int, default=63)
    parser.add_argument("--plot",   action="store_true")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        closes = df["close"].dropna().values
    else:
        rng = np.random.default_rng(42)
        closes = np.empty(args.n_bars)
        closes[0] = 4500.0
        for i in range(1, args.n_bars):
            closes[i] = closes[i-1] * (1.0 + 0.0001 + 0.0008 * rng.standard_normal())

    result = run_walk_forward(closes, train_bars=args.train, test_bars=args.test)

    if args.plot:
        WalkForwardEngine.plot_results(result)
