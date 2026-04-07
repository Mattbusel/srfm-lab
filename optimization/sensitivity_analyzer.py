"""
optimization/sensitivity_analyzer.py
======================================
Parameter sensitivity analysis for the LARSA trading system.

Identifies which parameters have the greatest impact on strategy
performance (Sharpe ratio), using three complementary methods:

  1. One-At-a-Time (OAT) -- vary each param +/-10%, +/-25%, record Sharpe change
  2. Sobol indices        -- variance-based global sensitivity (SALib)
  3. Morris method        -- trajectory elementary effects, efficient screening

Also generates a Plotly tornado chart HTML report.

Classes:
  SensitivityAnalyzer  -- main analysis class
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(_REPO_ROOT))

from config.param_schema import ParamSchema  # noqa: E402
from config.param_manager import LiveParams, ParamManager  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from SALib.sample import saltelli, morris as morris_sample
    from SALib.analyze import sobol, morris as morris_analyze
    _SALIB_AVAILABLE = True
except ImportError:
    _SALIB_AVAILABLE = False
    logger.debug("SALib not available -- Sobol/Morris methods disabled")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.debug("plotly not available -- HTML report disabled")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OATResult:
    """Result of a single one-at-a-time sensitivity evaluation."""
    param_name: str
    base_sharpe: float
    perturbation_pct: float
    perturbed_value: float
    perturbed_sharpe: float
    sharpe_change: float
    sharpe_change_pct: float
    direction: str  # "up" or "down"


@dataclass
class SensitivityReport:
    """Aggregated sensitivity analysis results."""
    oat_results: list[OATResult] = field(default_factory=list)
    sobol_indices: dict[str, float] = field(default_factory=dict)
    morris_mu_star: dict[str, float] = field(default_factory=dict)
    morris_sigma: dict[str, float] = field(default_factory=dict)
    critical_params: list[str] = field(default_factory=list)
    param_importance_rank: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "oat_results": [
                {
                    "param_name": r.param_name,
                    "perturbation_pct": r.perturbation_pct,
                    "sharpe_change_pct": r.sharpe_change_pct,
                    "direction": r.direction,
                }
                for r in self.oat_results
            ],
            "sobol_indices": self.sobol_indices,
            "morris_mu_star": self.morris_mu_star,
            "morris_sigma": self.morris_sigma,
            "critical_params": self.critical_params,
            "param_importance_rank": self.param_importance_rank,
        }


# ---------------------------------------------------------------------------
# BacktestEvaluator -- thin wrapper for sensitivity analysis
# ---------------------------------------------------------------------------

class _BacktestEvaluator:
    """
    Evaluates Sharpe ratio for a given parameter dict and price data.

    This class is deliberately minimal. The full BacktestRunner is
    used when available; otherwise a synthetic proxy is used for
    testing the sensitivity infrastructure without requiring price data.
    """

    def __init__(self, bars: dict[str, pd.DataFrame], initial_capital: float = 100_000.0) -> None:
        self._bars = bars
        self._initial_capital = initial_capital
        self._engine_available = False
        try:
            from backtest.engine import BacktestEngine  # noqa: F401
            from backtest.strategy_adapter import StrategyAdapter  # noqa: F401
            self._engine_available = True
        except ImportError:
            pass

    def evaluate(self, params: dict[str, Any]) -> float:
        """Run backtest with params and return annualized Sharpe ratio."""
        if not self._engine_available or not self._bars:
            return self._synthetic_sharpe(params)
        try:
            from backtest.engine import BacktestEngine
            from backtest.strategy_adapter import StrategyAdapter
            adapter = StrategyAdapter(params=params)
            symbols = list(self._bars.keys())
            engine = BacktestEngine(symbols=symbols, initial_capital=self._initial_capital)
            engine.register_handler(adapter)
            results = engine.run_simple(bars=self._bars, signal_fn=adapter.on_bar)
            equity = results.get("equity_curve")
            if equity is None or len(equity) < 20:
                return float("-inf")
            if not isinstance(equity, pd.Series):
                equity = pd.Series(equity)
            rets = np.log(equity / equity.shift(1)).dropna()
            if len(rets) < 10 or rets.std(ddof=1) < 1e-12:
                return 0.0
            return float(rets.mean() / rets.std(ddof=1) * math.sqrt(252 * 26))
        except Exception as exc:
            logger.debug("Backtest evaluation error: %s", exc)
            return float("-inf")

    @staticmethod
    def _synthetic_sharpe(params: dict[str, Any]) -> float:
        """
        Synthetic Sharpe proxy for testing without real price data.
        This is a deterministic function of params that mimics realistic
        sensitivity patterns. NOT used in production.
        """
        sharpe = 1.0
        sharpe += 0.3 * (1.5 - params.get("CF_BULL_THRESH", 1.2))
        sharpe -= 0.2 * max(0, params.get("CF_BULL_THRESH", 1.2) - 2.0)
        sharpe += 0.15 * (2.0 - params.get("BH_MASS_THRESH", 1.92))
        sharpe += 0.25 * (1.0 - params.get("NAV_OMEGA_SCALE_K", 0.5))
        sharpe += 0.10 * params.get("ML_SIGNAL_BOOST", 1.2)
        sharpe -= 0.05 * params.get("GARCH_ALPHA", 0.09) * 10
        sharpe += 0.12 * (0.5 - abs(params.get("NAV_EMA_ALPHA", 0.05) - 0.05))
        # Add small noise for realism
        seed = int(sum(abs(hash(str(v))) for v in params.values()) % 10000)
        rng = np.random.default_rng(seed)
        sharpe += float(rng.normal(0, 0.05))
        return float(sharpe)


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """
    Parameter sensitivity analysis for LARSA.

    Supports three methods:
      - OAT (one-at-a-time): fast, interpretable, local
      - Sobol: global variance-based, requires SALib
      - Morris: efficient screening, requires SALib

    Usage::

        analyzer = SensitivityAnalyzer(bars=price_data)
        report = analyzer.run_full_analysis()
        critical = analyzer.get_critical_params(threshold=0.05)
        analyzer.plot_tornado(output="sensitivity_report.html")
    """

    _NUMERIC_TYPES = {"float", "int"}

    def __init__(
        self,
        bars: Optional[dict[str, pd.DataFrame]] = None,
        schema: Optional[ParamSchema] = None,
        base_params: Optional[dict[str, Any]] = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        """
        Args:
            bars: {symbol: OHLCV DataFrame} for backtesting
            schema: ParamSchema instance (defaults if None)
            base_params: parameter values to use as the baseline
            initial_capital: starting equity
        """
        self._bars = bars or {}
        self._schema = schema or ParamSchema()
        self._base_params = base_params or self._schema.defaults()
        self._evaluator = _BacktestEvaluator(self._bars, initial_capital)
        self._report: Optional[SensitivityReport] = None

        # Cache for evaluated param combinations
        self._eval_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, params: dict[str, Any]) -> str:
        """Create a deterministic cache key for a parameter dict."""
        sorted_items = sorted(params.items())
        return json.dumps(sorted_items, sort_keys=True, default=str)

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate with caching."""
        key = self._cache_key(params)
        if key not in self._eval_cache:
            self._eval_cache[key] = self._evaluator.evaluate(params)
        return self._eval_cache[key]

    def _get_numeric_params(self) -> list[str]:
        """Return names of all numeric (float/int) parameters."""
        names = []
        for name in self._schema.parameter_names:
            spec = self._schema.get_spec(name)
            if spec["type"] in self._NUMERIC_TYPES:
                names.append(name)
        return names

    def _perturb_value(
        self,
        name: str,
        base_value: Any,
        pct: float,
    ) -> Any:
        """
        Perturb a parameter value by pct fraction of its range.
        Clamps to schema bounds. Returns the perturbed value.
        """
        spec = self._schema.get_spec(name)
        ptype = spec["type"]
        lo = spec.get("min", -1e9)
        hi = spec.get("max", 1e9)

        if ptype == "float":
            step = hi - lo
            delta = step * pct
            new_val = float(np.clip(base_value + delta, lo, hi))
            return new_val
        elif ptype == "int":
            step = hi - lo
            delta = max(1, int(round(step * pct)))
            new_val = int(np.clip(base_value + delta, lo, hi))
            return new_val
        else:
            return base_value

    # ------------------------------------------------------------------
    # One-At-a-Time (OAT) sensitivity
    # ------------------------------------------------------------------

    def oat_sensitivity(
        self,
        perturbations: tuple[float, ...] = (-0.25, -0.10, 0.10, 0.25),
    ) -> list[OATResult]:
        """
        One-at-a-time sensitivity analysis.

        For each numeric parameter, varies it by each perturbation fraction
        of its total schema range while holding all others at base values.
        Records the resulting Sharpe ratio change.

        Args:
            perturbations: fractions of parameter range to shift (+/-)

        Returns list of OATResult, one per (param, perturbation) combination.
        """
        numeric_params = self._get_numeric_params()
        base_sharpe = self._evaluate(self._base_params)
        logger.info(
            "OAT sensitivity: base Sharpe=%.4f, %d params, %d perturbations each",
            base_sharpe, len(numeric_params), len(perturbations),
        )

        results: list[OATResult] = []

        for name in numeric_params:
            base_val = self._base_params.get(name, self._schema.get_spec(name)["default"])

            for pct in perturbations:
                perturbed_val = self._perturb_value(name, base_val, pct)
                if perturbed_val == base_val:
                    continue  # no change possible (at boundary)

                perturbed_params = dict(self._base_params)
                perturbed_params[name] = perturbed_val

                # Repair constraints that may be violated by the perturbation
                perturbed_params = self._repair_constraints(perturbed_params)

                sharpe = self._evaluate(perturbed_params)
                if not math.isfinite(sharpe) or not math.isfinite(base_sharpe):
                    continue

                delta = sharpe - base_sharpe
                if base_sharpe != 0:
                    delta_pct = delta / abs(base_sharpe) * 100.0
                else:
                    delta_pct = 0.0

                results.append(OATResult(
                    param_name=name,
                    base_sharpe=base_sharpe,
                    perturbation_pct=pct * 100.0,
                    perturbed_value=perturbed_val,
                    perturbed_sharpe=sharpe,
                    sharpe_change=delta,
                    sharpe_change_pct=delta_pct,
                    direction="up" if pct > 0 else "down",
                ))

        logger.info("OAT complete: %d evaluations", len(results))
        return results

    def _repair_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """Minimal constraint repair for OAT perturbations."""
        p = dict(params)
        if p.get("CF_BEAR_THRESH", 999) < p.get("CF_BULL_THRESH", 0):
            p["CF_BEAR_THRESH"] = p["CF_BULL_THRESH"]
        if p.get("BH_MASS_EXTREME", 999) <= p.get("BH_MASS_THRESH", 0):
            p["BH_MASS_EXTREME"] = p["BH_MASS_THRESH"] + 0.5
        if p.get("MAX_HOLD_BARS", 999) <= p.get("MIN_HOLD_BARS", 0):
            p["MAX_HOLD_BARS"] = p["MIN_HOLD_BARS"] + 4
        if p.get("GARCH_ALPHA", 0) + p.get("GARCH_BETA", 0) >= 1.0:
            total = p["GARCH_ALPHA"] + p["GARCH_BETA"]
            scale = 0.97 / total
            p["GARCH_ALPHA"] = round(p["GARCH_ALPHA"] * scale, 6)
            p["GARCH_BETA"] = round(p["GARCH_BETA"] * scale, 6)
        if p.get("OU_KAPPA_MIN", 0) >= p.get("OU_KAPPA_MAX", 1):
            p["OU_KAPPA_MAX"] = p["OU_KAPPA_MIN"] + 0.1
        return p

    # ------------------------------------------------------------------
    # Sobol sensitivity indices
    # ------------------------------------------------------------------

    def sobol_indices(self, n_samples: int = 1000) -> dict[str, float]:
        """
        Compute Sobol first-order sensitivity indices for numeric parameters.

        Uses the Saltelli sampling scheme and variance-based decomposition.
        Requires SALib (pip install SALib).

        Args:
            n_samples: base number of samples (total = n_samples * (2*N+2))

        Returns dict of {param_name: S1_index}.
        """
        if not _SALIB_AVAILABLE:
            logger.warning("SALib not available -- Sobol indices cannot be computed")
            return {}

        numeric_params = self._get_numeric_params()
        n_params = len(numeric_params)
        if n_params == 0:
            return {}

        bounds = []
        for name in numeric_params:
            spec = self._schema.get_spec(name)
            lo = float(spec.get("min", 0.0))
            hi = float(spec.get("max", 1.0))
            bounds.append([lo, hi])

        problem = {
            "num_vars": n_params,
            "names": numeric_params,
            "bounds": bounds,
        }

        logger.info("Sobol: sampling %d * (2*%d+2) = %d points", n_samples, n_params, n_samples * (2 * n_params + 2))

        try:
            X = saltelli.sample(problem, n_samples, calc_second_order=False)
        except Exception as exc:
            logger.error("Saltelli sampling failed: %s", exc)
            return {}

        Y = np.zeros(len(X))
        for i, x in enumerate(X):
            params = dict(self._base_params)
            for j, name in enumerate(numeric_params):
                spec = self._schema.get_spec(name)
                if spec["type"] == "int":
                    params[name] = int(round(x[j]))
                else:
                    params[name] = float(x[j])
            params = self._repair_constraints(params)
            Y[i] = self._evaluate(params)
            if i % 500 == 0 and i > 0:
                logger.debug("Sobol: %d/%d evaluations complete", i, len(X))

        # Replace inf/nan with mean
        finite_mask = np.isfinite(Y)
        if finite_mask.sum() > 0:
            Y[~finite_mask] = Y[finite_mask].mean()
        else:
            logger.warning("All Sobol evaluations returned non-finite Sharpe")
            return {}

        try:
            Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            result = {
                name: float(np.clip(s1, 0.0, 1.0))
                for name, s1 in zip(numeric_params, Si["S1"])
                if math.isfinite(s1)
            }
            logger.info("Sobol complete: top 5 = %s", sorted(result.items(), key=lambda x: -x[1])[:5])
            return result
        except Exception as exc:
            logger.error("Sobol analysis failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Morris method
    # ------------------------------------------------------------------

    def morris_indices(
        self,
        n_trajectories: int = 20,
        n_levels: int = 4,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute Morris elementary effects for numeric parameters.

        Morris method is computationally cheaper than Sobol while still
        identifying non-influential parameters (mu_star ~ 0) and
        non-linear/interaction-prone parameters (sigma/mu_star > 0.5).

        Args:
            n_trajectories: number of Morris trajectories
            n_levels: number of grid levels for each parameter

        Returns (mu_star, sigma) dicts keyed by parameter name.
        mu_star: mean absolute elementary effect (importance)
        sigma: std of elementary effects (non-linearity indicator)
        """
        if not _SALIB_AVAILABLE:
            logger.warning("SALib not available -- Morris method disabled")
            return {}, {}

        numeric_params = self._get_numeric_params()
        n_params = len(numeric_params)
        if n_params == 0:
            return {}, {}

        bounds = []
        for name in numeric_params:
            spec = self._schema.get_spec(name)
            lo = float(spec.get("min", 0.0))
            hi = float(spec.get("max", 1.0))
            bounds.append([lo, hi])

        problem = {
            "num_vars": n_params,
            "names": numeric_params,
            "bounds": bounds,
        }

        try:
            X = morris_sample.sample(
                problem,
                N=n_trajectories,
                num_levels=n_levels,
                optimal_trajectories=None,
            )
        except Exception as exc:
            logger.error("Morris sampling failed: %s", exc)
            return {}, {}

        logger.info("Morris: evaluating %d trajectories x %d+1 = %d points", n_trajectories, n_params, len(X))

        Y = np.zeros(len(X))
        for i, x in enumerate(X):
            params = dict(self._base_params)
            for j, name in enumerate(numeric_params):
                spec = self._schema.get_spec(name)
                if spec["type"] == "int":
                    params[name] = int(round(x[j]))
                else:
                    params[name] = float(x[j])
            params = self._repair_constraints(params)
            Y[i] = self._evaluate(params)

        finite_mask = np.isfinite(Y)
        if finite_mask.sum() > 0:
            Y[~finite_mask] = Y[finite_mask].mean()

        try:
            Si = morris_analyze.analyze(problem, X, Y, print_to_console=False)
            mu_star = {
                name: float(v)
                for name, v in zip(numeric_params, Si["mu_star"])
                if math.isfinite(v)
            }
            sigma = {
                name: float(v)
                for name, v in zip(numeric_params, Si["sigma"])
                if math.isfinite(v)
            }
            logger.info(
                "Morris complete: top mu_star = %s",
                sorted(mu_star.items(), key=lambda x: -x[1])[:5],
            )
            return mu_star, sigma
        except Exception as exc:
            logger.error("Morris analysis failed: %s", exc)
            return {}, {}

    # ------------------------------------------------------------------
    # OAT summary statistics
    # ------------------------------------------------------------------

    def _oat_importance(self, oat_results: list[OATResult]) -> dict[str, float]:
        """
        Compute per-parameter importance from OAT results.

        Importance = mean(|sharpe_change_pct|) across all perturbations.
        """
        from collections import defaultdict
        by_param: dict[str, list[float]] = defaultdict(list)
        for r in oat_results:
            by_param[r.param_name].append(abs(r.sharpe_change_pct))
        return {name: float(np.mean(vals)) for name, vals in by_param.items()}

    # ------------------------------------------------------------------
    # Critical parameters
    # ------------------------------------------------------------------

    def get_critical_params(self, threshold: float = 0.05) -> list[str]:
        """
        Return parameters where a 10% perturbation causes >threshold Sharpe change.

        threshold is expressed as a fraction (0.05 = 5% Sharpe change).
        Uses OAT results with +/-10% perturbations.

        Returns sorted list of critical parameter names (most sensitive first).
        """
        oat_10pct = self.oat_sensitivity(perturbations=(-0.10, 0.10))
        importance = self._oat_importance(oat_10pct)
        critical = {
            name: imp
            for name, imp in importance.items()
            if imp > threshold * 100.0  # threshold is fraction, imp is pct
        }
        return sorted(critical.keys(), key=lambda n: -critical[n])

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        n_sobol_samples: int = 1000,
        n_morris_trajectories: int = 20,
        run_sobol: bool = True,
        run_morris: bool = True,
    ) -> SensitivityReport:
        """
        Run all three sensitivity methods and compile a SensitivityReport.

        Args:
            n_sobol_samples: Saltelli sample count for Sobol
            n_morris_trajectories: trajectory count for Morris
            run_sobol: whether to run Sobol (requires SALib)
            run_morris: whether to run Morris (requires SALib)

        Returns a SensitivityReport with all results populated.
        """
        report = SensitivityReport()

        # OAT
        logger.info("Running OAT sensitivity analysis...")
        report.oat_results = self.oat_sensitivity()

        # Sobol
        if run_sobol:
            logger.info("Running Sobol sensitivity analysis...")
            report.sobol_indices = self.sobol_indices(n_samples=n_sobol_samples)

        # Morris
        if run_morris:
            logger.info("Running Morris sensitivity analysis...")
            mu_star, sigma = self.morris_indices(n_trajectories=n_morris_trajectories)
            report.morris_mu_star = mu_star
            report.morris_sigma = sigma

        # Critical params (10% perturbation, >5% Sharpe change)
        report.critical_params = self.get_critical_params(threshold=0.05)

        # Combined importance ranking
        report.param_importance_rank = self._rank_params(report)

        self._report = report
        return report

    def _rank_params(self, report: SensitivityReport) -> list[str]:
        """
        Combine OAT, Sobol, and Morris scores into a unified ranking.

        Normalizes each method's scores to [0,1] then computes a
        weighted average: OAT 40%, Sobol 40%, Morris mu_star 20%.
        """
        numeric_params = self._get_numeric_params()
        if not numeric_params:
            return []

        scores: dict[str, float] = {name: 0.0 for name in numeric_params}

        # OAT contribution (40%)
        oat_imp = self._oat_importance(report.oat_results)
        if oat_imp:
            max_oat = max(oat_imp.values()) or 1.0
            for name in numeric_params:
                scores[name] += 0.4 * oat_imp.get(name, 0.0) / max_oat

        # Sobol contribution (40%)
        if report.sobol_indices:
            max_s1 = max(report.sobol_indices.values()) or 1.0
            for name in numeric_params:
                scores[name] += 0.4 * report.sobol_indices.get(name, 0.0) / max_s1

        # Morris mu_star contribution (20%)
        if report.morris_mu_star:
            max_mu = max(report.morris_mu_star.values()) or 1.0
            for name in numeric_params:
                scores[name] += 0.2 * report.morris_mu_star.get(name, 0.0) / max_mu

        return sorted(numeric_params, key=lambda n: -scores[n])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_tornado(
        self,
        output: str = "sensitivity_report.html",
        n_top: int = 15,
    ) -> Optional[str]:
        """
        Generate a Plotly tornado chart showing parameter sensitivity.

        The chart shows the Sharpe change for +/-10% and +/-25%
        perturbations of the top n_top most sensitive parameters.
        Saves to output path and returns the path string.

        Requires plotly (pip install plotly).
        """
        if not _PLOTLY_AVAILABLE:
            logger.warning("plotly not available -- cannot generate HTML report")
            return None

        oat_results = self.oat_sensitivity(perturbations=(-0.25, -0.10, 0.10, 0.25))
        if not oat_results:
            logger.warning("No OAT results available for tornado chart")
            return None

        # Build per-param rows for plotting
        imp = self._oat_importance(oat_results)
        ranked = sorted(imp.keys(), key=lambda n: -imp[n])[:n_top]

        # Group results by param and perturbation
        data: dict[str, dict[float, float]] = {}
        for r in oat_results:
            if r.param_name not in ranked:
                continue
            if r.param_name not in data:
                data[r.param_name] = {}
            data[r.param_name][r.perturbation_pct] = r.sharpe_change_pct

        # Build traces
        colors_pos = ["#2196F3", "#42A5F5"]  # blue shades for +
        colors_neg = ["#F44336", "#EF9A9A"]  # red shades for -
        pert_labels = {-25.0: "-25%", -10.0: "-10%", 10.0: "+10%", 25.0: "+25%"}

        fig = go.Figure()

        for i, pct in enumerate([25.0, 10.0]):
            x_pos = [data[p].get(pct, 0.0) for p in ranked]
            x_neg = [data[p].get(-pct, 0.0) for p in ranked]
            fig.add_trace(go.Bar(
                y=ranked,
                x=x_pos,
                name=f"+{pct:.0f}%",
                orientation="h",
                marker_color=colors_pos[i],
            ))
            fig.add_trace(go.Bar(
                y=ranked,
                x=x_neg,
                name=f"-{pct:.0f}%",
                orientation="h",
                marker_color=colors_neg[i],
            ))

        fig.update_layout(
            title="LARSA Parameter Sensitivity -- Sharpe Change vs Perturbation",
            xaxis_title="Sharpe Ratio Change (%)",
            yaxis_title="Parameter",
            barmode="overlay",
            height=max(400, n_top * 35),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

        out_path = Path(output)
        try:
            pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn")
            logger.info("Tornado chart saved to %s", out_path)
            return str(out_path)
        except Exception as exc:
            logger.error("Failed to write tornado chart: %s", exc)
            return None

    def export_json(self, output: str = "sensitivity_results.json") -> str:
        """Export full sensitivity report to JSON."""
        report = self._report or self.run_full_analysis(run_sobol=False, run_morris=False)
        out_path = Path(output)
        with open(out_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        logger.info("Sensitivity report exported to %s", out_path)
        return str(out_path)

    def print_summary(self) -> None:
        """Print a text summary of sensitivity results to stdout."""
        report = self._report
        if report is None:
            print("No analysis run yet. Call run_full_analysis() first.")
            return

        print("\n=== LARSA Parameter Sensitivity Summary ===")
        print(f"Critical parameters (>5% Sharpe change at 10% perturbation):")
        for i, name in enumerate(report.critical_params, 1):
            print(f"  {i:2d}. {name}")

        print(f"\nTop 10 by combined importance ranking:")
        for i, name in enumerate(report.param_importance_rank[:10], 1):
            sobol = report.sobol_indices.get(name, float("nan"))
            mu = report.morris_mu_star.get(name, float("nan"))
            print(f"  {i:2d}. {name:<35}  Sobol S1={sobol:.4f}  Morris mu*={mu:.4f}")

        if report.sobol_indices:
            total_s1 = sum(report.sobol_indices.values())
            print(f"\nTotal first-order Sobol variance explained: {total_s1:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SensitivityAnalyzer()
    report = analyzer.run_full_analysis(run_sobol=False, run_morris=False)
    analyzer.print_summary()
    critical = analyzer.get_critical_params(threshold=0.05)
    print(f"\nCritical params: {critical}")
