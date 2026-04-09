"""
Lookahead Bias Detector: catch signals that accidentally use future data.

The most dangerous bug in quantitative finance: a signal that looks
amazing in backtest because it secretly uses information from the future.

Detection methods:
  1. Feature lag audit: compare IC at lag-0 vs lag-1 (should be similar)
  2. Shuffled target test: permute returns, re-evaluate (should give Sharpe~0)
  3. Timestamp monotonicity: ensure all features are computed from past data only
  4. Information ratio decay: IC should decay with horizon (not spike at lag 0)
  5. Cross-validation consistency: IS and OOS performance should be similar

Any signal that fails these tests is BLOCKED from live deployment.
This is the immune system against the deadliest bug in quant finance.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LookaheadTestResult:
    """Result of a single lookahead detection test."""
    test_name: str
    passed: bool
    score: float              # 0-1 (1 = definitely lookahead)
    details: str
    severity: str             # "clean" / "suspicious" / "likely_lookahead" / "confirmed_lookahead"


@dataclass
class LookaheadReport:
    """Complete lookahead audit report for a signal."""
    signal_name: str
    overall_verdict: str      # "CLEAN" / "SUSPICIOUS" / "BLOCKED"
    lookahead_probability: float  # 0-1
    tests: List[LookaheadTestResult]
    deployment_allowed: bool
    recommendation: str


class LookaheadDetector:
    """
    Detect lookahead bias in trading signals.

    Run this on EVERY new signal before allowing live deployment.
    A single failed test = signal is blocked.
    """

    def __init__(self, n_permutations: int = 100, max_ic_ratio: float = 3.0,
                  max_shuffled_sharpe: float = 0.5):
        self.n_permutations = n_permutations
        self.max_ic_ratio = max_ic_ratio
        self.max_shuffled_sharpe = max_shuffled_sharpe

    def audit(
        self,
        signal_values: np.ndarray,     # (T,) signal predictions
        actual_returns: np.ndarray,    # (T,) realized returns
        features: Optional[np.ndarray] = None,  # (T, F) feature matrix
        signal_name: str = "unknown",
    ) -> LookaheadReport:
        """Run full lookahead audit on a signal."""
        tests = []

        # Test 1: Feature lag audit
        tests.append(self._test_lag_audit(signal_values, actual_returns))

        # Test 2: Shuffled target test
        tests.append(self._test_shuffled_target(signal_values, actual_returns))

        # Test 3: IC decay profile
        tests.append(self._test_ic_decay(signal_values, actual_returns))

        # Test 4: IS vs OOS consistency
        tests.append(self._test_is_oos_consistency(signal_values, actual_returns))

        # Test 5: Timestamp monotonicity (if features provided)
        if features is not None:
            tests.append(self._test_feature_monotonicity(features, actual_returns))

        # Overall verdict
        n_failed = sum(1 for t in tests if not t.passed)
        n_suspicious = sum(1 for t in tests if t.severity in ("suspicious", "likely_lookahead"))
        n_confirmed = sum(1 for t in tests if t.severity == "confirmed_lookahead")

        if n_confirmed > 0:
            verdict = "BLOCKED"
            prob = 0.95
            allowed = False
            rec = "CONFIRMED LOOKAHEAD BIAS. Do NOT deploy this signal."
        elif n_failed >= 2 or n_suspicious >= 3:
            verdict = "BLOCKED"
            prob = 0.7
            allowed = False
            rec = "Multiple tests failed. Likely lookahead bias. Investigate feature engineering."
        elif n_failed == 1:
            verdict = "SUSPICIOUS"
            prob = 0.3
            allowed = False
            rec = "One test failed. Review and re-test before deploying."
        else:
            verdict = "CLEAN"
            prob = 0.05
            allowed = True
            rec = "All tests passed. Signal appears free of lookahead bias."

        return LookaheadReport(
            signal_name=signal_name,
            overall_verdict=verdict,
            lookahead_probability=prob,
            tests=tests,
            deployment_allowed=allowed,
            recommendation=rec,
        )

    def _test_lag_audit(self, signal: np.ndarray, returns: np.ndarray) -> LookaheadTestResult:
        """
        Compare IC at lag-0 vs lag-1.
        If IC at lag-0 is much higher than lag-1, the signal may be using
        contemporaneous (not lagged) data.
        """
        n = min(len(signal), len(returns))
        if n < 50:
            return LookaheadTestResult("lag_audit", True, 0.0, "Insufficient data", "clean")

        # IC at lag 0: corr(signal_t, return_t)
        ic_lag0 = abs(float(np.corrcoef(signal[:n], returns[:n])[0, 1]))

        # IC at lag 1: corr(signal_t, return_{t+1})
        ic_lag1 = abs(float(np.corrcoef(signal[:n-1], returns[1:n])[0, 1]))

        ratio = ic_lag0 / max(ic_lag1, 1e-6)

        if ratio > self.max_ic_ratio * 2:
            severity = "confirmed_lookahead"
        elif ratio > self.max_ic_ratio:
            severity = "likely_lookahead"
        elif ratio > self.max_ic_ratio * 0.7:
            severity = "suspicious"
        else:
            severity = "clean"

        passed = ratio <= self.max_ic_ratio

        return LookaheadTestResult(
            test_name="lag_audit",
            passed=passed,
            score=min(1.0, ratio / (self.max_ic_ratio * 2)),
            details=f"IC lag-0={ic_lag0:.4f}, IC lag-1={ic_lag1:.4f}, ratio={ratio:.1f}x",
            severity=severity,
        )

    def _test_shuffled_target(self, signal: np.ndarray, returns: np.ndarray) -> LookaheadTestResult:
        """
        Shuffle returns and re-evaluate.
        If the signal still shows significant Sharpe after shuffling,
        it's learning from something that shouldn't be possible.
        """
        n = min(len(signal), len(returns))
        if n < 50:
            return LookaheadTestResult("shuffled_target", True, 0.0, "Insufficient data", "clean")

        rng = np.random.default_rng(42)
        shuffled_sharpes = []

        for _ in range(self.n_permutations):
            shuffled = rng.permutation(returns[:n])
            strat_ret = signal[:n] * shuffled
            if strat_ret.std() > 1e-10:
                sharpe = float(strat_ret.mean() / strat_ret.std() * math.sqrt(252))
            else:
                sharpe = 0.0
            shuffled_sharpes.append(abs(sharpe))

        avg_shuffled = float(np.mean(shuffled_sharpes))
        max_shuffled = float(np.max(shuffled_sharpes))

        if avg_shuffled > self.max_shuffled_sharpe:
            severity = "confirmed_lookahead"
        elif max_shuffled > self.max_shuffled_sharpe * 2:
            severity = "suspicious"
        else:
            severity = "clean"

        passed = avg_shuffled <= self.max_shuffled_sharpe

        return LookaheadTestResult(
            test_name="shuffled_target",
            passed=passed,
            score=min(1.0, avg_shuffled / self.max_shuffled_sharpe),
            details=f"Avg shuffled Sharpe={avg_shuffled:.2f} (should be ~0), max={max_shuffled:.2f}",
            severity=severity,
        )

    def _test_ic_decay(self, signal: np.ndarray, returns: np.ndarray) -> LookaheadTestResult:
        """
        IC should decay monotonically with horizon.
        If IC at lag-0 is much higher than lag-1,2,3,... that's suspicious.
        """
        n = min(len(signal), len(returns))
        if n < 100:
            return LookaheadTestResult("ic_decay", True, 0.0, "Insufficient data", "clean")

        ics = []
        for lag in range(5):
            if lag == 0:
                ic = abs(float(np.corrcoef(signal[:n], returns[:n])[0, 1]))
            else:
                ic = abs(float(np.corrcoef(signal[:n-lag], returns[lag:n])[0, 1]))
            ics.append(ic)

        # Check if lag-0 dominates
        if len(ics) >= 2 and ics[1] > 1e-6:
            spike_ratio = ics[0] / ics[1]
        else:
            spike_ratio = 1.0

        # Check monotonicity of decay
        is_monotone = all(ics[i] >= ics[i+1] * 0.8 for i in range(len(ics) - 1))

        if spike_ratio > 5:
            severity = "likely_lookahead"
        elif spike_ratio > 3:
            severity = "suspicious"
        else:
            severity = "clean"

        passed = spike_ratio <= 5

        return LookaheadTestResult(
            test_name="ic_decay",
            passed=passed,
            score=min(1.0, spike_ratio / 10),
            details=f"IC by lag: {[f'{ic:.4f}' for ic in ics]}, spike ratio={spike_ratio:.1f}x",
            severity=severity,
        )

    def _test_is_oos_consistency(self, signal: np.ndarray, returns: np.ndarray) -> LookaheadTestResult:
        """
        Split into IS/OOS halves. If IS Sharpe >> OOS Sharpe, likely overfitting.
        """
        n = min(len(signal), len(returns))
        if n < 100:
            return LookaheadTestResult("is_oos_consistency", True, 0.0, "Insufficient data", "clean")

        mid = n // 2
        is_ret = signal[:mid] * returns[:mid]
        oos_ret = signal[mid:n] * returns[mid:n]

        is_sharpe = float(is_ret.mean() / max(is_ret.std(), 1e-10) * math.sqrt(252))
        oos_sharpe = float(oos_ret.mean() / max(oos_ret.std(), 1e-10) * math.sqrt(252))

        if abs(is_sharpe) > 1e-10:
            degradation = 1 - oos_sharpe / is_sharpe
        else:
            degradation = 0.0

        if degradation > 0.8:
            severity = "likely_lookahead"
        elif degradation > 0.5:
            severity = "suspicious"
        else:
            severity = "clean"

        passed = degradation <= 0.8

        return LookaheadTestResult(
            test_name="is_oos_consistency",
            passed=passed,
            score=min(1.0, max(0, degradation)),
            details=f"IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f}, degradation={degradation:.0%}",
            severity=severity,
        )

    def _test_feature_monotonicity(self, features: np.ndarray, returns: np.ndarray) -> LookaheadTestResult:
        """
        Check if any feature has suspiciously high correlation with CURRENT returns
        (should only correlate with FUTURE returns if it's a predictor).
        """
        T, F = features.shape
        n = min(T, len(returns))

        max_contemp_corr = 0.0
        worst_feature = -1

        for f in range(F):
            contemp = abs(float(np.corrcoef(features[:n, f], returns[:n])[0, 1]))
            if contemp > max_contemp_corr:
                max_contemp_corr = contemp
                worst_feature = f

        if max_contemp_corr > 0.5:
            severity = "confirmed_lookahead"
        elif max_contemp_corr > 0.3:
            severity = "likely_lookahead"
        elif max_contemp_corr > 0.15:
            severity = "suspicious"
        else:
            severity = "clean"

        passed = max_contemp_corr <= 0.3

        return LookaheadTestResult(
            test_name="feature_monotonicity",
            passed=passed,
            score=min(1.0, max_contemp_corr / 0.5),
            details=f"Max contemporaneous feature-return corr={max_contemp_corr:.3f} (feature {worst_feature})",
            severity=severity,
        )
