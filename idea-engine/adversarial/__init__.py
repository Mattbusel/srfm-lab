"""
adversarial
===========
Adversarial backtester for the IAE strategy genome.

This module tries to BREAK the strategy to find hidden risks before they
surface in live trading.  It does NOT try to make the strategy look good;
it tries to find conditions under which it fails catastrophically.

Modules
-------
fuzzer              -- Latin-hypercube parameter space fuzzer (10 000 combos).
worst_case_finder   -- Grid search for single-parameter damage analysis.
regime_stress       -- Historical stress tests: COVID, Luna, FTX.
correlation_attack  -- Correlation-spike and assumption-error testing.
overfitting_detector-- Walk-forward and CPCV overfitting detection.
adversarial_market  -- Gradient-based adversarial price path generation.
report_generator    -- Composite risk report with probability-weighted scenarios.

Usage::

    from adversarial import AdversarialRunner

    runner = AdversarialRunner(trade_data=trades_df, params=current_params)
    report = runner.full_report()
    print(report.summary())
"""

from .fuzzer             import ParameterFuzzer, FuzzerResult
from .worst_case_finder  import WorstCaseFinder, SensitivityResult
from .regime_stress      import RegimeStressor, StressResult
from .correlation_attack import CorrelationAttacker, CorrelationResult
from .overfitting_detector import OverfittingDetector, OverfittingReport
from .adversarial_market import AdversarialMarket, AdversarialPath
from .report_generator   import ReportGenerator, AdversarialReport

__all__ = [
    "ParameterFuzzer",
    "FuzzerResult",
    "WorstCaseFinder",
    "SensitivityResult",
    "RegimeStressor",
    "StressResult",
    "CorrelationAttacker",
    "CorrelationResult",
    "OverfittingDetector",
    "OverfittingReport",
    "AdversarialMarket",
    "AdversarialPath",
    "ReportGenerator",
    "AdversarialReport",
]
