"""
adversarial_tester.py — Adversarial testing framework for trading hypotheses.

Tries to break hypotheses via:
  - Data snooping bias check
  - Overfitting detection (in-sample vs out-of-sample degradation)
  - Regime robustness (performance across regimes)
  - Transaction cost sensitivity
  - Parameter sensitivity / perturbation analysis
  - Look-ahead bias scanner
  - Survivorship bias check
  - Stress testing via crash / tail scenarios
  - Devil's advocate report
  - Confidence deflation based on test failures
"""

from __future__ import annotations

import copy
import inspect
import itertools
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Test result structures
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class TestResult:
    test_name: str
    status: TestStatus
    score: float           # 0 = perfect, 1 = catastrophic failure
    description: str
    details: Dict = field(default_factory=dict)
    confidence_deflation: float = 0.0   # how much to reduce hypothesis confidence

    def to_dict(self) -> Dict:
        return {
            "test": self.test_name,
            "status": self.status.value,
            "score": round(self.score, 4),
            "description": self.description,
            "confidence_deflation": round(self.confidence_deflation, 4),
            "details": self.details,
        }


@dataclass
class AdversarialReport:
    hypothesis_id: str
    hypothesis_text: str
    original_confidence: float
    test_results: List[TestResult]
    final_confidence: float
    generated_at: float = field(default_factory=time.time)

    def summary(self) -> Dict:
        by_status = {s.value: 0 for s in TestStatus}
        for r in self.test_results:
            by_status[r.status.value] += 1
        total_deflation = sum(r.confidence_deflation for r in self.test_results)
        return {
            "hypothesis_id": self.hypothesis_id,
            "original_confidence": round(self.original_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "total_deflation": round(total_deflation, 4),
            "status_counts": by_status,
            "critical_failures": [
                r.test_name for r in self.test_results
                if r.status == TestStatus.FAIL
            ],
            "warnings": [
                r.test_name for r in self.test_results
                if r.status == TestStatus.WARN
            ],
        }

    def devils_advocate(self) -> str:
        """Generate a devil's advocate summary of hypothesis weaknesses."""
        failures = [r for r in self.test_results if r.status == TestStatus.FAIL]
        warnings = [r for r in self.test_results if r.status == TestStatus.WARN]

        lines = [
            f"DEVIL'S ADVOCATE REPORT",
            f"Hypothesis: {self.hypothesis_text[:120]}",
            f"Confidence: {self.original_confidence:.2%} → {self.final_confidence:.2%}",
            "",
        ]

        if not failures and not warnings:
            lines.append("No significant weaknesses found. Proceed with caution — absence of evidence is not evidence of absence.")
        else:
            if failures:
                lines.append("CRITICAL WEAKNESSES:")
                for r in failures:
                    lines.append(f"  ✗ {r.test_name}: {r.description}")
                    if r.details:
                        for k, v in list(r.details.items())[:3]:
                            lines.append(f"      {k}: {v}")
            if warnings:
                lines.append("MODERATE CONCERNS:")
                for r in warnings:
                    lines.append(f"  △ {r.test_name}: {r.description}")

        lines.append("")
        lines.append(f"Recommendation: {'DO NOT TRADE — too many critical failures' if len(failures) >= 3 else 'Proceed with reduced sizing' if failures or warnings else 'Cleared for trading'}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            **self.summary(),
            "test_results": [r.to_dict() for r in self.test_results],
            "devils_advocate": self.devils_advocate(),
        }


# ---------------------------------------------------------------------------
# Backtest result type (minimal interface)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Minimal backtest result expected by the adversarial tester."""
    returns: np.ndarray           # per-bar strategy returns
    dates: Optional[np.ndarray] = None
    regime_labels: Optional[np.ndarray] = None  # one per bar
    params: Dict = field(default_factory=dict)   # parameter values used
    in_sample_mask: Optional[np.ndarray] = None  # bool array, True=in-sample

    def sharpe(self, mask: Optional[np.ndarray] = None) -> float:
        r = self.returns[mask] if mask is not None else self.returns
        if len(r) < 2 or r.std() < 1e-9:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    def cagr(self, mask: Optional[np.ndarray] = None) -> float:
        r = self.returns[mask] if mask is not None else self.returns
        if len(r) == 0:
            return 0.0
        cum = float(np.prod(1 + r))
        years = len(r) / 252.0
        return float(cum ** (1 / max(years, 0.1)) - 1)

    def max_drawdown(self, mask: Optional[np.ndarray] = None) -> float:
        r = self.returns[mask] if mask is not None else self.returns
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-9)
        return float(dd.min())

    def annual_turnover(self, positions: Optional[np.ndarray] = None) -> float:
        if positions is None or len(positions) < 2:
            return 0.0
        return float(np.abs(np.diff(positions)).sum() / len(positions) * 252)


# ---------------------------------------------------------------------------
# Scenario library (crash / stress scenarios)
# ---------------------------------------------------------------------------

CRASH_SCENARIOS = {
    "2008_gfc": {
        "description": "2008 Global Financial Crisis",
        "shock_returns": np.array([-0.05, -0.08, -0.10, -0.12, -0.07, 0.04, 0.02, -0.06, -0.04, -0.09]),
        "vol_multiplier": 4.0,
        "correlation_shift": 0.8,
    },
    "2020_covid": {
        "description": "2020 COVID crash",
        "shock_returns": np.array([-0.04, -0.10, -0.12, -0.15, -0.09, 0.08, 0.06, 0.04, 0.03, 0.05]),
        "vol_multiplier": 5.0,
        "correlation_shift": 0.9,
    },
    "1987_black_monday": {
        "description": "1987 Black Monday crash",
        "shock_returns": np.array([-0.22, -0.08, 0.05, 0.03, 0.02, 0.04, 0.01, -0.02, 0.03, 0.02]),
        "vol_multiplier": 8.0,
        "correlation_shift": 0.95,
    },
    "2000_dot_com": {
        "description": "2000-2002 Dot-com bust",
        "shock_returns": np.array([-0.03, -0.04, -0.05, -0.02, -0.04, -0.03, -0.01, -0.04, -0.03, -0.02]) * 10,
        "vol_multiplier": 2.5,
        "correlation_shift": 0.6,
    },
    "rate_shock": {
        "description": "Rapid interest rate spike (+300bps in 6 months)",
        "shock_returns": np.array([-0.02, -0.03, -0.01, -0.04, -0.02, -0.01, -0.03, -0.02, -0.01, -0.02]),
        "vol_multiplier": 2.0,
        "correlation_shift": 0.5,
    },
    "liquidity_crisis": {
        "description": "Sudden liquidity freeze — bid-ask spreads 10x",
        "shock_returns": np.array([-0.01, -0.03, -0.05, -0.02, -0.01, -0.02, 0.01, 0.02, -0.01, 0.01]),
        "vol_multiplier": 3.0,
        "correlation_shift": 0.85,
        "spread_multiplier": 10.0,
    },
}


# ---------------------------------------------------------------------------
# Individual test implementations
# ---------------------------------------------------------------------------

class _Tests:

    # --- Data snooping bias -------------------------------------------------

    @staticmethod
    def data_snooping_bias(
        baseline_sharpe: float,
        n_strategies_tested: int,
        backtest_length_years: float,
    ) -> TestResult:
        """
        Estimate Bonferroni-adjusted p-value inflation from multiple testing.
        Uses the Haircut Sharpe Ratio from Harvey, Liu & Zhu (2016).
        """
        if n_strategies_tested <= 1:
            return TestResult("data_snooping_bias", TestStatus.PASS, 0.0,
                              "Only 1 strategy tested — no data snooping concern.", {})

        # Expected max Sharpe from N iid normal trials
        # E[max(Z_1,...,Z_N)] ≈ sqrt(2 * ln(N))
        expected_max = float(np.sqrt(2 * np.log(n_strategies_tested)))
        # Haircut = expected_max / (sqrt(T) * original_sharpe)
        t = max(backtest_length_years * 252, 1.0)
        t_stat = baseline_sharpe * np.sqrt(t / 252)
        haircut = expected_max / (max(t_stat, 0.01))
        adjusted_sharpe = baseline_sharpe * max(0.0, 1.0 - haircut)

        deflation = min(0.30, haircut * 0.3)
        score = min(1.0, haircut)

        if haircut > 0.5:
            status = TestStatus.FAIL
            desc = f"Severe data snooping risk. Adjusted Sharpe={adjusted_sharpe:.2f} (haircut={haircut:.0%})"
        elif haircut > 0.25:
            status = TestStatus.WARN
            desc = f"Moderate snooping risk. Adjusted Sharpe={adjusted_sharpe:.2f} (haircut={haircut:.0%})"
        else:
            status = TestStatus.PASS
            desc = f"Low snooping risk. Adjusted Sharpe={adjusted_sharpe:.2f}"

        return TestResult("data_snooping_bias", status, score, desc,
                          {"n_strategies": n_strategies_tested, "haircut": round(haircut, 4),
                           "original_sharpe": round(baseline_sharpe, 4),
                           "adjusted_sharpe": round(adjusted_sharpe, 4)},
                          confidence_deflation=deflation)

    # --- Overfitting detection -----------------------------------------------

    @staticmethod
    def overfitting_detection(result: BacktestResult) -> TestResult:
        if result.in_sample_mask is None:
            return TestResult("overfitting_detection", TestStatus.SKIP, 0.0,
                              "No in/out-of-sample split provided.", {})

        is_mask = result.in_sample_mask
        oos_mask = ~is_mask

        if oos_mask.sum() < 20:
            return TestResult("overfitting_detection", TestStatus.SKIP, 0.0,
                              "Insufficient OOS data.", {})

        is_sharpe = result.sharpe(is_mask)
        oos_sharpe = result.sharpe(oos_mask)
        is_cagr = result.cagr(is_mask)
        oos_cagr = result.cagr(oos_mask)

        if is_sharpe <= 0:
            degradation = 1.0
        else:
            degradation = max(0.0, (is_sharpe - oos_sharpe) / (abs(is_sharpe) + 1e-9))

        score = float(np.clip(degradation, 0.0, 1.0))
        deflation = min(0.40, degradation * 0.5)

        if degradation > 0.60:
            status = TestStatus.FAIL
            desc = f"Severe overfitting. IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f} ({degradation:.0%} degradation)"
        elif degradation > 0.30:
            status = TestStatus.WARN
            desc = f"Moderate overfitting. IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f}"
        else:
            status = TestStatus.PASS
            desc = f"Good IS/OOS consistency. IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f}"

        return TestResult("overfitting_detection", status, score, desc,
                          {"is_sharpe": round(is_sharpe, 4), "oos_sharpe": round(oos_sharpe, 4),
                           "is_cagr": round(is_cagr, 4), "oos_cagr": round(oos_cagr, 4),
                           "degradation": round(degradation, 4)},
                          confidence_deflation=deflation)

    # --- Regime robustness ---------------------------------------------------

    @staticmethod
    def regime_robustness(result: BacktestResult) -> TestResult:
        if result.regime_labels is None or len(set(result.regime_labels)) < 2:
            return TestResult("regime_robustness", TestStatus.SKIP, 0.0,
                              "No regime labels provided.", {})

        regimes = sorted(set(result.regime_labels))
        regime_sharpes: Dict[str, float] = {}

        for reg in regimes:
            mask = result.regime_labels == reg
            if mask.sum() >= 20:
                regime_sharpes[str(reg)] = round(result.sharpe(mask), 3)

        if len(regime_sharpes) < 2:
            return TestResult("regime_robustness", TestStatus.SKIP, 0.0,
                              "Insufficient data in multiple regimes.", {})

        sharpe_values = list(regime_sharpes.values())
        n_negative = sum(1 for s in sharpe_values if s < 0)
        worst = min(sharpe_values)
        best = max(sharpe_values)
        spread = best - worst

        score = float(np.clip(n_negative / len(sharpe_values) + spread / 5.0, 0.0, 1.0))
        deflation = min(0.25, n_negative / len(sharpe_values) * 0.3 + max(0, spread - 2.0) * 0.05)

        if n_negative > len(sharpe_values) / 2:
            status = TestStatus.FAIL
            desc = f"Strategy fails in {n_negative}/{len(sharpe_values)} regimes."
        elif n_negative > 0 or spread > 3.0:
            status = TestStatus.WARN
            desc = f"Regime-dependent: works in some but not all regimes. Spread={spread:.2f}"
        else:
            status = TestStatus.PASS
            desc = f"Robust across {len(regime_sharpes)} regimes. Spread={spread:.2f}"

        return TestResult("regime_robustness", status, score, desc,
                          {"regime_sharpes": regime_sharpes, "worst": round(worst, 3),
                           "best": round(best, 3), "n_negative": n_negative},
                          confidence_deflation=deflation)

    # --- Transaction cost sensitivity ----------------------------------------

    @staticmethod
    def transaction_cost_sensitivity(
        result: BacktestResult,
        positions: np.ndarray,
        base_bps: float = 5.0,
        stress_bps_levels: Optional[List[float]] = None,
    ) -> TestResult:
        if stress_bps_levels is None:
            stress_bps_levels = [5.0, 10.0, 20.0, 50.0]

        turnover_per_bar = float(np.abs(np.diff(positions)).mean()) if len(positions) > 1 else 0.01
        base_sharpe = result.sharpe()

        sharpe_at_cost: Dict[str, float] = {}
        for bps in stress_bps_levels:
            cost_per_bar = turnover_per_bar * bps / 10_000
            adj_returns = result.returns - cost_per_bar
            if adj_returns.std() > 1e-9:
                s = float(adj_returns.mean() / adj_returns.std() * np.sqrt(252))
            else:
                s = 0.0
            sharpe_at_cost[f"{bps}bps"] = round(s, 3)

        # At what cost level does Sharpe turn negative?
        breakeven_bps: Optional[float] = None
        for bps in np.arange(1.0, 200.0, 1.0):
            cost_per_bar = turnover_per_bar * bps / 10_000
            adj_returns = result.returns - cost_per_bar
            if adj_returns.mean() <= 0:
                breakeven_bps = float(bps)
                break

        if breakeven_bps is not None and breakeven_bps < 10.0:
            status = TestStatus.FAIL
            desc = f"Strategy breaks even at {breakeven_bps:.0f}bps — too sensitive to costs."
            score = 0.8
            deflation = 0.30
        elif breakeven_bps is not None and breakeven_bps < 25.0:
            status = TestStatus.WARN
            desc = f"Moderate cost sensitivity: breakeven at {breakeven_bps:.0f}bps."
            score = 0.4
            deflation = 0.15
        else:
            status = TestStatus.PASS
            desc = f"Robust to costs. Breakeven at {breakeven_bps:.0f}bps." if breakeven_bps else "Very robust to costs."
            score = 0.1
            deflation = 0.0

        return TestResult("transaction_cost_sensitivity", status, score, desc,
                          {"base_sharpe": round(base_sharpe, 3), "sharpe_at_cost": sharpe_at_cost,
                           "breakeven_bps": breakeven_bps, "turnover_daily": round(turnover_per_bar, 4)},
                          confidence_deflation=deflation)

    # --- Parameter sensitivity -----------------------------------------------

    @staticmethod
    def parameter_sensitivity(
        backtest_fn: Callable[[Dict], BacktestResult],
        base_params: Dict,
        perturbation_pct: float = 0.20,
        n_samples: int = 30,
    ) -> TestResult:
        """
        Perturb each numeric parameter by ±perturbation_pct and measure Sharpe stability.
        """
        base_result = backtest_fn(base_params)
        base_sharpe = base_result.sharpe()

        sharpe_samples = []
        param_sensitivity: Dict[str, float] = {}

        numeric_params = {k: v for k, v in base_params.items() if isinstance(v, (int, float))}

        if not numeric_params:
            return TestResult("parameter_sensitivity", TestStatus.SKIP, 0.0,
                              "No numeric parameters to perturb.", {})

        for param_name, base_val in numeric_params.items():
            param_sharpes = []
            perturb_range = np.linspace(1 - perturbation_pct, 1 + perturbation_pct,
                                        max(3, n_samples // len(numeric_params)))
            for mult in perturb_range:
                perturbed = {**base_params, param_name: base_val * mult}
                try:
                    r = backtest_fn(perturbed)
                    param_sharpes.append(r.sharpe())
                except Exception:
                    pass
            if param_sharpes:
                sensitivity = float(np.std(param_sharpes) / (abs(base_sharpe) + 1e-9))
                param_sensitivity[param_name] = round(sensitivity, 4)
                sharpe_samples.extend(param_sharpes)

        overall_stability = float(np.std(sharpe_samples) / (abs(base_sharpe) + 1e-9)) if sharpe_samples else 0.0
        n_negative_params = sum(1 for s in sharpe_samples if s < 0)

        if overall_stability > 1.0 or n_negative_params / max(len(sharpe_samples), 1) > 0.3:
            status = TestStatus.FAIL
            score = min(1.0, overall_stability)
            desc = f"Highly parameter-sensitive. Stability={overall_stability:.2f}"
            deflation = 0.25
        elif overall_stability > 0.5:
            status = TestStatus.WARN
            score = overall_stability * 0.7
            desc = f"Moderately parameter-sensitive. Stability={overall_stability:.2f}"
            deflation = 0.10
        else:
            status = TestStatus.PASS
            score = overall_stability * 0.3
            desc = f"Parameter-stable. Stability={overall_stability:.2f}"
            deflation = 0.0

        return TestResult("parameter_sensitivity", status, score, desc,
                          {"overall_stability": round(overall_stability, 4),
                           "base_sharpe": round(base_sharpe, 3),
                           "param_sensitivity": param_sensitivity},
                          confidence_deflation=deflation)

    # --- Look-ahead bias scanner ---------------------------------------------

    @staticmethod
    def look_ahead_bias(
        source_code: Optional[str] = None,
        lag_features: Optional[List[str]] = None,
        forward_returns_used_in_fit: bool = False,
    ) -> TestResult:
        """Heuristic look-ahead bias check."""
        issues = []
        score = 0.0

        if forward_returns_used_in_fit:
            issues.append("Forward returns appear to be used in fitting/training.")
            score += 0.6

        if source_code:
            # Scan for common look-ahead patterns
            suspicious_patterns = [
                ("shift(-", "Negative shift (future data)"),
                (".shift(-", "Negative shift on series"),
                ("future_", "Variable named with 'future_' prefix"),
                ("next_day", "Variable referencing next day"),
                ("label_", "Label variable — may embed future info"),
                ("_forward", "Forward-looking variable naming"),
            ]
            for pattern, desc in suspicious_patterns:
                if pattern.lower() in source_code.lower():
                    issues.append(f"Suspicious pattern: '{pattern}' — {desc}")
                    score += 0.2

        if lag_features:
            non_lagged = [f for f in lag_features if not any(
                k in f.lower() for k in ("lag", "prev", "t-", "shift", "delay", "past")
            )]
            if non_lagged:
                issues.append(f"Features without explicit lag indicator: {non_lagged[:5]}")
                score += 0.1 * len(non_lagged)

        score = min(1.0, score)
        deflation = min(0.50, score * 0.7)

        if score >= 0.5:
            status = TestStatus.FAIL
            desc = f"Look-ahead bias suspected. {len(issues)} issue(s) found."
        elif score > 0.1:
            status = TestStatus.WARN
            desc = f"Possible look-ahead bias. {len(issues)} concern(s)."
        else:
            status = TestStatus.PASS
            desc = "No obvious look-ahead bias detected."

        return TestResult("look_ahead_bias", status, score, desc,
                          {"issues": issues}, confidence_deflation=deflation)

    # --- Survivorship bias ---------------------------------------------------

    @staticmethod
    def survivorship_bias(
        universe_size_at_start: int,
        universe_size_at_end: int,
        delisted_included: bool = False,
        backtest_universe_is_current_index: bool = False,
    ) -> TestResult:
        """Check if backtest suffers from survivorship bias."""
        issues = []
        score = 0.0

        if backtest_universe_is_current_index:
            issues.append("Universe appears to be today's index constituents — classic survivorship bias.")
            score += 0.7

        if not delisted_included and universe_size_at_end < universe_size_at_start:
            survival_rate = universe_size_at_end / max(universe_size_at_start, 1)
            attrition_bias = 1.0 - survival_rate
            issues.append(f"Universe shrank by {1-survival_rate:.0%} without delisted stocks included.")
            score += attrition_bias * 0.5

        score = min(1.0, score)
        deflation = min(0.35, score * 0.4)

        if score >= 0.6:
            status = TestStatus.FAIL
            desc = "High survivorship bias risk."
        elif score > 0.2:
            status = TestStatus.WARN
            desc = "Moderate survivorship bias risk."
        else:
            status = TestStatus.PASS
            desc = "Survivorship bias risk appears low."

        return TestResult("survivorship_bias", status, score, desc,
                          {"issues": issues, "survival_rate": round(universe_size_at_end / max(universe_size_at_start, 1), 3),
                           "delisted_included": delisted_included},
                          confidence_deflation=deflation)

    # --- Stress test ---------------------------------------------------------

    @staticmethod
    def stress_test(
        returns: np.ndarray,
        positions: Optional[np.ndarray] = None,
        scenarios: Optional[Dict] = None,
    ) -> TestResult:
        """Apply crash scenarios and measure strategy response."""
        if scenarios is None:
            scenarios = CRASH_SCENARIOS

        scenario_results: Dict[str, Dict] = {}
        n_bad = 0

        for name, scenario in scenarios.items():
            shock = scenario["shock_returns"]
            vol_mult = scenario.get("vol_multiplier", 2.0)

            # Scale strategy returns by shock: assumes some correlation with market
            correlation = 0.6  # assumed market correlation
            shocked_returns = returns * (1 + (vol_mult - 1) * correlation) + shock.mean() * correlation

            if len(shocked_returns) > len(shock):
                shocked_window = shocked_returns[:len(shock)]
            else:
                shocked_window = shocked_returns

            cum = float(np.prod(1 + shocked_window) - 1)
            worst_day = float(shocked_window.min()) if len(shocked_window) > 0 else 0.0
            dd = float(min(np.minimum.accumulate(np.cumprod(1 + shocked_window)) - 1.0)) if len(shocked_window) > 0 else 0.0

            scenario_results[name] = {
                "description": scenario["description"],
                "scenario_cum_return": round(cum, 4),
                "worst_day": round(worst_day, 4),
                "max_drawdown": round(dd, 4),
            }

            if cum < -0.15 or worst_day < -0.10:
                n_bad += 1

        fraction_bad = n_bad / max(len(scenarios), 1)
        score = float(np.clip(fraction_bad, 0.0, 1.0))
        deflation = min(0.20, fraction_bad * 0.25)

        if fraction_bad > 0.5:
            status = TestStatus.FAIL
            desc = f"Strategy catastrophically fails in {n_bad}/{len(scenarios)} crash scenarios."
        elif fraction_bad > 0.25:
            status = TestStatus.WARN
            desc = f"Strategy struggles in {n_bad}/{len(scenarios)} crash scenarios."
        else:
            status = TestStatus.PASS
            desc = f"Strategy survives most stress scenarios ({n_bad}/{len(scenarios)} failures)."

        return TestResult("stress_test", status, score, desc,
                          {"scenario_results": scenario_results, "n_bad": n_bad},
                          confidence_deflation=deflation)


# ---------------------------------------------------------------------------
# AdversarialTester
# ---------------------------------------------------------------------------

class AdversarialTester:
    """
    Orchestrates the full adversarial test suite for a trading hypothesis.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._reports: List[AdversarialReport] = []

    def run(
        self,
        hypothesis_id: str,
        hypothesis_text: str,
        original_confidence: float,
        backtest_result: BacktestResult,
        positions: Optional[np.ndarray] = None,
        # Data snooping inputs
        n_strategies_tested: int = 1,
        backtest_length_years: float = 5.0,
        # Overfitting: uses backtest_result.in_sample_mask
        # Regime: uses backtest_result.regime_labels
        # Transaction costs
        base_cost_bps: float = 5.0,
        # Parameter sensitivity
        backtest_fn: Optional[Callable[[Dict], BacktestResult]] = None,
        base_params: Optional[Dict] = None,
        # Look-ahead
        source_code: Optional[str] = None,
        lag_features: Optional[List[str]] = None,
        forward_returns_in_fit: bool = False,
        # Survivorship
        universe_size_start: int = 500,
        universe_size_end: int = 500,
        delisted_included: bool = False,
        universe_is_current_index: bool = False,
        # Stress
        stress_scenarios: Optional[Dict] = None,
    ) -> AdversarialReport:
        """Run the full adversarial test suite and return a report."""

        if positions is None:
            positions = np.sign(backtest_result.returns)

        results: List[TestResult] = []

        if self.verbose:
            print(f"[AdversarialTester] Testing hypothesis: {hypothesis_text[:60]}...")

        # 1. Data snooping
        r = _Tests.data_snooping_bias(
            backtest_result.sharpe(), n_strategies_tested, backtest_length_years
        )
        results.append(r)
        self._log(r)

        # 2. Overfitting
        r = _Tests.overfitting_detection(backtest_result)
        results.append(r)
        self._log(r)

        # 3. Regime robustness
        r = _Tests.regime_robustness(backtest_result)
        results.append(r)
        self._log(r)

        # 4. Transaction costs
        r = _Tests.transaction_cost_sensitivity(backtest_result, positions, base_cost_bps)
        results.append(r)
        self._log(r)

        # 5. Parameter sensitivity
        if backtest_fn is not None and base_params is not None:
            r = _Tests.parameter_sensitivity(backtest_fn, base_params)
        else:
            r = TestResult("parameter_sensitivity", TestStatus.SKIP, 0.0,
                           "No backtest function provided for perturbation.", {})
        results.append(r)
        self._log(r)

        # 6. Look-ahead bias
        r = _Tests.look_ahead_bias(source_code, lag_features, forward_returns_in_fit)
        results.append(r)
        self._log(r)

        # 7. Survivorship bias
        r = _Tests.survivorship_bias(
            universe_size_start, universe_size_end, delisted_included, universe_is_current_index
        )
        results.append(r)
        self._log(r)

        # 8. Stress test
        r = _Tests.stress_test(backtest_result.returns, positions, stress_scenarios)
        results.append(r)
        self._log(r)

        # Compute final confidence
        total_deflation = sum(r.confidence_deflation for r in results)
        final_confidence = float(np.clip(original_confidence * (1.0 - total_deflation), 0.0, 1.0))

        report = AdversarialReport(
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis_text,
            original_confidence=original_confidence,
            test_results=results,
            final_confidence=final_confidence,
        )
        self._reports.append(report)

        if self.verbose:
            print(f"[AdversarialTester] Done. Confidence: {original_confidence:.2%} → {final_confidence:.2%}")
            print(report.devils_advocate())

        return report

    def _log(self, r: TestResult) -> None:
        if self.verbose:
            status_icons = {TestStatus.PASS: "✓", TestStatus.WARN: "△",
                            TestStatus.FAIL: "✗", TestStatus.SKIP: "—"}
            icon = status_icons.get(r.status, "?")
            print(f"  {icon} {r.test_name}: {r.description}")

    def all_reports(self) -> List[AdversarialReport]:
        return list(self._reports)

    def export_reports(self, path: str) -> None:
        data = [r.to_dict() for r in self._reports]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    T = 1000

    # Simulate a strategy that works in-sample but not out-of-sample
    is_returns = np.random.normal(0.0008, 0.012, T // 2)   # good IS
    oos_returns = np.random.normal(-0.0001, 0.015, T // 2)  # bad OOS
    all_returns = np.concatenate([is_returns, oos_returns])

    in_sample_mask = np.zeros(T, dtype=bool)
    in_sample_mask[:T//2] = True

    regime_labels = np.array(
        ["bull"] * (T // 3) + ["bear"] * (T // 3) + ["neutral"] * (T - 2 * T // 3)
    )

    result = BacktestResult(
        returns=all_returns,
        regime_labels=regime_labels,
        in_sample_mask=in_sample_mask,
        params={"lookback": 20, "threshold": 0.5, "holding": 5},
    )

    positions = np.cumsum(np.sign(all_returns)) / T  # toy positions

    tester = AdversarialTester(verbose=True)
    report = tester.run(
        hypothesis_id="hyp_001",
        hypothesis_text="Buy when 20-day momentum positive, sell when negative. Rebalance weekly.",
        original_confidence=0.72,
        backtest_result=result,
        positions=positions,
        n_strategies_tested=15,
        backtest_length_years=4.0,
        base_cost_bps=8.0,
        lag_features=["momentum_20d", "vol_10d", "price_roc"],
        universe_is_current_index=True,
    )

    print("\n--- REPORT SUMMARY ---")
    print(json.dumps(report.summary(), indent=2))
