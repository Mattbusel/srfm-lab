"""
research/reconciliation/__init__.py
====================================
Live vs Backtest Reconciliation Pipeline for srfm-lab.

This package provides end-to-end tooling to compare live trading results
against backtest simulations, decompose performance differences, detect
signal drift, measure slippage/market impact, attribute PnL, and audit for
data-leakage / lookahead bias.

Quick-start
-----------
>>> from research.reconciliation import (
...     LiveTradeLoader, BacktestTradeLoader, merge_live_backtest,
...     SlippageAnalyzer, SignalDriftDetector, PnLAttributionEngine,
...     DataLeakageAuditor, generate_full_report, ReconciliationReport,
... )
>>> live   = LiveTradeLoader("tools/backtest_output/live_trades.db").load()
>>> bt     = BacktestTradeLoader("tools/backtest_output/crypto_trades.csv").load()
>>> merged = merge_live_backtest(live, bt)
>>> report = generate_full_report(live, bt, output_dir="research/reconciliation/output")

Module overview
---------------
loader        – data ingestion & normalisation (LiveTradeLoader, BacktestTradeLoader, TradeRecord)
slippage      – fill-quality analysis and market-impact modelling (SlippageAnalyzer)
drift         – BH-activation overlap, regime-drift, signal autocorrelation (SignalDriftDetector)
attribution   – Brinson-Hood-Beebower-style PnL decomposition (PnLAttributionEngine)
leakage       – lookahead-bias / over-fitting detection (DataLeakageAuditor)
report        – HTML + console reconciliation report (ReconciliationReport, generate_full_report)
cli           – Click command-line interface (`recon` command group)
"""

from __future__ import annotations

# ── loader ──────────────────────────────────────────────────────────────────
from research.reconciliation.loader import (
    TradeRecord,
    LiveTradeLoader,
    BacktestTradeLoader,
    merge_live_backtest,
)

# ── slippage ─────────────────────────────────────────────────────────────────
from research.reconciliation.slippage import (
    SlippageAnalyzer,
    FillReport,
    SlippageStats,
)

# ── drift ────────────────────────────────────────────────────────────────────
from research.reconciliation.drift import (
    SignalDriftDetector,
    LjungBoxResult,
    ParameterStabilityResult,
)

# ── attribution ───────────────────────────────────────────────────────────────
from research.reconciliation.attribution import (
    PnLAttributionEngine,
    AttributionReport,
)

# ── leakage ───────────────────────────────────────────────────────────────────
from research.reconciliation.leakage import (
    DataLeakageAuditor,
    LeakageReport,
)

# ── report ────────────────────────────────────────────────────────────────────
from research.reconciliation.report import (
    ReconciliationReport,
    generate_full_report,
)

__all__ = [
    # loader
    "TradeRecord",
    "LiveTradeLoader",
    "BacktestTradeLoader",
    "merge_live_backtest",
    # slippage
    "SlippageAnalyzer",
    "FillReport",
    "SlippageStats",
    # drift
    "SignalDriftDetector",
    "LjungBoxResult",
    "ParameterStabilityResult",
    # attribution
    "PnLAttributionEngine",
    "AttributionReport",
    # leakage
    "DataLeakageAuditor",
    "LeakageReport",
    # report
    "ReconciliationReport",
    "generate_full_report",
]

__version__ = "1.0.0"
__author__ = "srfm-lab"
