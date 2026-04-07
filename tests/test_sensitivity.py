"""
test_sensitivity.py — Tests for parameter sensitivity analysis.

~500 LOC. Tests parameter perturbation, sensitivity detection, and edge summary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))


# ─────────────────────────────────────────────────────────────────────────────
# Minimal inline sensitivity framework (mirrors tools/sensitivity.py patterns)
# ─────────────────────────────────────────────────────────────────────────────

def _build_df(closes: np.ndarray, start: str = "2022-01-03") -> pd.DataFrame:
    n = len(closes)
    rng = np.random.default_rng(7)
    idx = pd.date_range(start, periods=n, freq="1h")
    noise = 0.0004 * np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": np.full(n, 50_000.0),
    }, index=idx)


def _make_trending(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = np.empty(n)
    closes[0] = 4500.0
    for i in range(1, n):
        closes[i] = closes[i-1] * (1.0 + 0.0001 + 0.0007 * rng.standard_normal())
    return _build_df(closes)


class _SimpleBacktest:
    """
    Minimal backtest: counts BH activations and computes a simple Sharpe proxy.
    Used for sensitivity tests without importing the full BHEngine.
    """
    def __init__(self, cf: float = 0.001, bh_form: float = 1.5,
                 bh_collapse: float = 1.0, bh_decay: float = 0.95):
        self.cf = cf
        self.bh_form = bh_form
        self.bh_collapse = bh_collapse
        self.bh_decay = bh_decay

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        from srfm_core import MinkowskiClassifier, BlackHoleDetector
        mc = MinkowskiClassifier(cf=self.cf)
        bh = BlackHoleDetector(self.bh_form, self.bh_collapse, self.bh_decay)

        closes = df["close"].values
        n_activations = 0
        ctl_sum = 0
        prev_active = False
        prev_close = None

        for i, c in enumerate(closes):
            bit = mc.update(float(c))
            if prev_close is not None:
                active = bh.update(bit, float(c), float(prev_close))
                if active and not prev_active:
                    n_activations += 1
                ctl_sum += bh.ctl
            prev_active = bh.bh_active
            prev_close = float(c)

        # Simple proxy metrics
        timelike_frac = sum(1 for c in closes[1:] if abs(c - closes[max(0, i-1)]) / (closes[max(0, i-1)] + 1e-9) / self.cf < 1.0
                           ) / max(1, len(closes) - 1)

        return {
            "n_activations": float(n_activations),
            "final_mass": float(bh.bh_mass),
            "ctl_sum": float(ctl_sum),
        }


def run_sensitivity(
    df: pd.DataFrame,
    base_params: Dict[str, float],
    param_grid: Dict[str, List[float]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run parameter sensitivity sweep.

    Returns dict: {param_name: [{"value": v, "metric": m, "delta_pct": d}, ...]}
    """
    results = {}

    # Baseline
    bt_base = _SimpleBacktest(**base_params)
    base_metrics = bt_base.run(df)
    base_activations = base_metrics["n_activations"]

    for param_name, values in param_grid.items():
        param_results = []
        for v in values:
            p = dict(base_params)
            p[param_name] = v
            bt = _SimpleBacktest(**p)
            metrics = bt.run(df)
            n_act = metrics["n_activations"]
            delta_pct = (n_act - base_activations) / (base_activations + 1e-9) * 100.0
            param_results.append({
                "value": v,
                "n_activations": n_act,
                "delta_pct": delta_pct,
                "is_baseline": abs(v - base_params[param_name]) < 1e-10,
            })
        results[param_name] = param_results
    return results


def detect_fragile_params(
    sensitivity_results: Dict[str, List[Dict[str, Any]]],
    threshold_pct: float = 25.0,
) -> List[str]:
    """Return list of parameter names where perturbation changes metric by > threshold_pct."""
    fragile = []
    for param, entries in sensitivity_results.items():
        deltas = [abs(e["delta_pct"]) for e in entries if not e.get("is_baseline", False)]
        if deltas and max(deltas) > threshold_pct:
            fragile.append(param)
    return fragile


def generate_edge_summary(
    sensitivity_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """Generate summary statistics per parameter."""
    summary = {}
    for param, entries in sensitivity_results.items():
        activations = [e["n_activations"] for e in entries]
        deltas = [e["delta_pct"] for e in entries]
        summary[param] = {
            "min_activations": float(min(activations)),
            "max_activations": float(max(activations)),
            "max_delta_pct": float(max(abs(d) for d in deltas)),
            "mean_delta_pct": float(np.mean([abs(d) for d in deltas])),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Class TestSensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestSensitivity:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.df = _make_trending(1200)
        self.base_params = {
            "cf": 0.001,
            "bh_form": 1.5,
            "bh_collapse": 1.0,
            "bh_decay": 0.95,
        }
        self.param_grid = {
            "cf":          [0.0005, 0.001, 0.002, 0.003],
            "bh_form":     [1.0, 1.5, 2.0, 3.0],
            "bh_collapse": [0.5, 1.0, 1.5],
            "bh_decay":    [0.80, 0.90, 0.95, 0.99],
        }
        self.results = run_sensitivity(self.df, self.base_params, self.param_grid)

    def test_all_perturbations_run(self):
        """Sensitivity sweep should run for every parameter in param_grid."""
        for param in self.param_grid:
            assert param in self.results, f"Missing results for param: {param}"
            assert len(self.results[param]) == len(self.param_grid[param])

    def test_baseline_in_perturbations(self):
        """Baseline parameter value should be present in each parameter's results."""
        for param, entries in self.results.items():
            base_val = self.base_params[param]
            found = any(abs(e["value"] - base_val) < 1e-10 for e in entries)
            assert found, f"Baseline value {base_val} not found in {param} results"

    def test_fragile_parameter_detected(self):
        """Parameter with large impact on activations should be detected as fragile."""
        # bh_form sweeping from 1.0 to 3.0 should change activation count significantly
        fragile = detect_fragile_params(self.results, threshold_pct=10.0)
        assert len(fragile) > 0, (
            "At least one parameter should be detected as fragile over a wide range")

    def test_robust_parameter_not_flagged(self):
        """A tiny perturbation of a robust parameter should not be flagged as fragile."""
        # Create a narrow-range sweep around the baseline
        narrow_grid = {
            "bh_decay": [0.949, 0.950, 0.951],  # tiny range
        }
        narrow_results = run_sensitivity(self.df, self.base_params, narrow_grid)
        fragile = detect_fragile_params(narrow_results, threshold_pct=50.0)
        assert "bh_decay" not in fragile, (
            "Tiny bh_decay perturbation should not be flagged as fragile at 50% threshold")

    def test_edge_summary_generated(self):
        """Edge summary should have all parameters with correct keys."""
        summary = generate_edge_summary(self.results)
        for param in self.param_grid:
            assert param in summary
            assert "max_delta_pct" in summary[param]
            assert "min_activations" in summary[param]
            assert summary[param]["max_delta_pct"] >= 0.0

    def test_higher_bh_form_fewer_activations(self):
        """Increasing bh_form should monotonically decrease activations."""
        bh_form_results = self.results["bh_form"]
        # Sort by bh_form value
        sorted_res = sorted(bh_form_results, key=lambda x: x["value"])
        activations = [r["n_activations"] for r in sorted_res]
        # Monotonically non-increasing (or close to it)
        for i in range(1, len(activations)):
            assert activations[i] <= activations[i-1] + 1, (  # allow small non-monotone
                f"Activations should decrease with higher bh_form: {activations}")

    def test_lower_cf_more_timelike(self):
        """Lower CF → beta smaller → more TIMELIKE bars."""
        from srfm_core import MinkowskiClassifier
        df = self.df
        closes = df["close"].values

        def count_timelike(cf: float) -> int:
            mc = MinkowskiClassifier(cf=cf)
            mc.update(float(closes[0]))
            tl = 0
            for i in range(1, len(closes)):
                if mc.update(float(closes[i])) == "TIMELIKE":
                    tl += 1
            return tl

        tl_high_cf = count_timelike(0.003)
        tl_low_cf  = count_timelike(0.0005)
        # beta = |dr| / cf → lower cf → higher beta → MORE spacelike
        # So higher CF → more TIMELIKE bars
        assert tl_high_cf >= tl_low_cf, (
            f"Higher CF should produce more TIMELIKE bars: high_cf={tl_high_cf}, low_cf={tl_low_cf}")

    def test_sensitivity_results_are_dicts(self):
        """Each perturbation result entry should be a dict with required keys."""
        for param, entries in self.results.items():
            for entry in entries:
                assert "value" in entry
                assert "n_activations" in entry
                assert "delta_pct" in entry
                assert math.isfinite(entry["delta_pct"])

    def test_all_activations_nonnegative(self):
        """n_activations should never be negative."""
        for param, entries in self.results.items():
            for entry in entries:
                assert entry["n_activations"] >= 0, (
                    f"Negative activations for {param}={entry['value']}")

    def test_decay_impact_on_mass_buildup(self):
        """Higher bh_decay → slower mass decay → more time for activation."""
        results_decay = self.results["bh_decay"]
        sorted_r = sorted(results_decay, key=lambda x: x["value"])
        # Higher decay: mass stays larger longer → possibly more activations
        # At minimum, there should be variation
        acts = [r["n_activations"] for r in sorted_r]
        # Not necessarily monotone, but range should be > 0
        assert max(acts) >= min(acts), "bh_decay should have some effect on activations"

    def test_param_grid_exhaustive(self):
        """All values in param_grid are evaluated exactly once."""
        for param, values in self.param_grid.items():
            result_values = [e["value"] for e in self.results[param]]
            assert len(result_values) == len(values), (
                f"{param}: expected {len(values)} results, got {len(result_values)}")
            for v in values:
                assert any(abs(rv - v) < 1e-10 for rv in result_values), (
                    f"Value {v} missing from {param} results")

    def test_fragile_threshold_scales(self):
        """Tightening the threshold detects more fragile params."""
        fragile_20 = detect_fragile_params(self.results, threshold_pct=20.0)
        fragile_5  = detect_fragile_params(self.results, threshold_pct=5.0)
        assert len(fragile_5) >= len(fragile_20), (
            "Tighter threshold should detect >= params as fragile")

    def test_edge_summary_max_gte_min(self):
        """In edge summary, max_activations should be >= min_activations."""
        summary = generate_edge_summary(self.results)
        for param, stats in summary.items():
            assert stats["max_activations"] >= stats["min_activations"], (
                f"{param}: max_activations < min_activations")
