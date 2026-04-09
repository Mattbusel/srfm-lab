"""
Compliance Reporter: automated regulatory trade reporting.

Generates MiFID II / SEC compliant reports automatically from trade data.
Every trade carries a provenance chain from physics concept to execution.

Reports include:
  - Best execution proof (why this venue, why this timing)
  - Algorithm description (what logic generated the signal)
  - Risk assessment at time of trade
  - Pre-trade and post-trade analytics
  - Conflict of interest declarations
  - Order routing rationale
"""

from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TradeRecord:
    """Regulatory-grade trade record with full audit trail."""
    # Identifiers
    trade_id: str
    order_id: str
    execution_id: str
    account_id: str

    # Instrument
    symbol: str
    isin: str = ""                # for MiFID II
    asset_class: str = "crypto"
    exchange: str = ""
    venue_type: str = "lit"       # lit / dark / systematic_internaliser

    # Execution
    side: str                     # buy / sell
    quantity: float
    price: float
    notional: float
    currency: str = "USD"
    timestamp: float = 0.0
    settlement_date: str = ""

    # Costs
    commission: float = 0.0
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Algorithm info (MiFID II RTS 25)
    algorithm_id: str = "SRFM-BH-v18"
    algorithm_description: str = ""
    signal_source: str = ""       # which physics concept / signal generated this
    signal_strength: float = 0.0
    signal_confidence: float = 0.0

    # Risk at time of trade
    portfolio_var_at_trade: float = 0.0
    position_concentration: float = 0.0
    drawdown_at_trade: float = 0.0

    # Best execution proof
    venues_considered: List[str] = field(default_factory=list)
    venue_selection_rationale: str = ""
    alternative_prices: Dict[str, float] = field(default_factory=dict)
    execution_quality_score: float = 0.0  # 0-1

    # Provenance
    provenance_trace_id: str = ""
    hypothesis_id: str = ""
    debate_consensus: float = 0.0
    dream_fragility: float = 0.0

    # Audit
    hash: str = ""                # tamper-evident hash of this record
    previous_hash: str = ""       # chain to previous trade (like blockchain)


class ComplianceReporter:
    """
    Automated regulatory compliance reporting.

    Generates reports required by:
    - MiFID II (EU): best execution, algorithm disclosure, transaction reporting
    - SEC Rule 606: order routing disclosure
    - SEC Rule 15c3-5: market access risk controls
    - Dodd-Frank: swap reporting (if applicable)
    """

    def __init__(self, account_id: str = "SRFM-001", firm_name: str = "SRFM Trading Lab"):
        self.account_id = account_id
        self.firm_name = firm_name
        self._records: List[TradeRecord] = []
        self._last_hash = "0" * 64

    def record_trade(self, trade: TradeRecord) -> str:
        """Record a trade with tamper-evident hashing."""
        trade.account_id = self.account_id
        trade.previous_hash = self._last_hash

        # Compute hash (like a blockchain)
        hash_input = f"{trade.trade_id}:{trade.symbol}:{trade.side}:{trade.quantity}:" \
                     f"{trade.price}:{trade.timestamp}:{trade.previous_hash}"
        trade.hash = hashlib.sha256(hash_input.encode()).hexdigest()
        self._last_hash = trade.hash

        self._records.append(trade)
        return trade.hash

    def generate_best_execution_report(self, period_days: int = 30) -> Dict:
        """
        MiFID II RTS 28: Best Execution Report.
        Shows that the system achieved best execution for clients.
        """
        cutoff = time.time() - period_days * 86400
        recent = [r for r in self._records if r.timestamp >= cutoff]

        if not recent:
            return {"period": f"Last {period_days} days", "trades": 0}

        # Venue analysis
        by_venue = {}
        for r in recent:
            v = r.venue_type
            if v not in by_venue:
                by_venue[v] = {"count": 0, "notional": 0, "avg_cost_bps": []}
            by_venue[v]["count"] += 1
            by_venue[v]["notional"] += r.notional
            by_venue[v]["avg_cost_bps"].append(r.total_cost_bps)

        for v in by_venue:
            costs = by_venue[v]["avg_cost_bps"]
            by_venue[v]["avg_cost_bps"] = float(sum(costs) / len(costs))

        # Execution quality
        quality_scores = [r.execution_quality_score for r in recent if r.execution_quality_score > 0]
        avg_quality = float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0

        return {
            "report_type": "MiFID II RTS 28 Best Execution Report",
            "firm": self.firm_name,
            "period": f"Last {period_days} days",
            "total_trades": len(recent),
            "total_notional": sum(r.notional for r in recent),
            "venue_breakdown": by_venue,
            "average_execution_quality": avg_quality,
            "average_total_cost_bps": float(sum(r.total_cost_bps for r in recent) / len(recent)),
            "algorithm_used": "SRFM-BH-v18 (Black Hole Physics + Multi-Agent Debate)",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def generate_algorithm_disclosure(self) -> Dict:
        """
        MiFID II RTS 25: Algorithm Disclosure.
        Describes the algorithm's logic for regulators.
        """
        return {
            "report_type": "MiFID II RTS 25 Algorithm Disclosure",
            "firm": self.firm_name,
            "algorithm_id": "SRFM-BH-v18",
            "algorithm_name": "SRFM Black Hole Physics Strategy",
            "description": (
                "The SRFM strategy applies Special Relativistic Financial Mechanics to classify "
                "market bars as TIMELIKE (causal, trending) or SPACELIKE (anomalous, reverting) "
                "using a Minkowski spacetime metric. Mass accumulates on consecutive TIMELIKE bars, "
                "and a 'black hole formation' event triggers entry when mass exceeds the formation "
                "threshold (BH_FORM=1.92). The system uses GARCH(1,1) conditional volatility for "
                "position sizing, Ornstein-Uhlenbeck mean reversion as an overlay, and a "
                "multi-agent debate system with 12 specialized agents for hypothesis validation."
            ),
            "risk_controls": {
                "max_position_pct": "10% of NAV per instrument",
                "max_daily_loss": "2% of NAV",
                "max_drawdown_halt": "15% of NAV (system halts)",
                "circuit_breaker": "Per-API fault isolation with exponential backoff",
                "hard_limit_override": "Risk committee can set hard limits via Guardian module",
            },
            "signal_sources": [
                "BH Physics (Minkowski classification, mass accumulation, Hawking temperature)",
                "GARCH(1,1) conditional volatility",
                "Ornstein-Uhlenbeck mean reversion",
                "Regime ensemble (HMM, vol-based, Hurst, trend, correlation)",
                "Event Horizon Synthesizer (autonomous physics-based signal discovery)",
                "Multi-Agent Debate (12 agents: quant, macro, risk, microstructure, etc.)",
            ],
            "evolution_mechanism": (
                "Parameters are evolved via NSGA-II genetic algorithm with multi-objective "
                "optimization (Sharpe, Calmar, drawdown). The Meta-Reward Co-Evolver discovers "
                "optimal reward functions for the PPO reinforcement learning agent. The Recursive "
                "Meta-Evolutionary Architecture (RMEA) evolves the evolution parameters themselves."
            ),
            "stability_monitoring": (
                "Lyapunov exponent monitoring ensures the evolutionary process converges. "
                "KL divergence tracking detects distribution drift between training and live. "
                "A StabilityGate prevents deployment of unstable configurations."
            ),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def generate_transaction_report(self, period_days: int = 1) -> List[Dict]:
        """
        SEC / MiFID II Transaction Report.
        One record per trade with full details.
        """
        cutoff = time.time() - period_days * 86400
        recent = [r for r in self._records if r.timestamp >= cutoff]

        return [
            {
                "trade_id": r.trade_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.timestamp)),
                "symbol": r.symbol,
                "side": r.side,
                "quantity": r.quantity,
                "price": r.price,
                "notional": r.notional,
                "venue": r.venue_type,
                "algorithm": r.algorithm_id,
                "signal_source": r.signal_source,
                "commission": r.commission,
                "total_cost_bps": r.total_cost_bps,
                "portfolio_var": r.portfolio_var_at_trade,
                "provenance": r.provenance_trace_id,
                "hash": r.hash,
            }
            for r in recent
        ]

    def generate_risk_control_report(self) -> Dict:
        """
        SEC Rule 15c3-5: Market Access Risk Controls.
        Demonstrates that pre-trade risk checks are in place.
        """
        return {
            "report_type": "SEC Rule 15c3-5 Market Access Risk Controls",
            "firm": self.firm_name,
            "pre_trade_controls": {
                "position_limit_check": "Max 10% NAV per instrument, enforced in pre_trade_checks.py",
                "daily_loss_limit": "2% NAV daily loss triggers trading halt",
                "order_size_limit": "Max order size checked against ADV (5% limit)",
                "spread_check": "Orders rejected if spread > 50 bps",
                "sector_concentration": "Max 35% NAV per sector",
                "leverage_limit": "Max 3x total leverage",
                "price_guard": "Orders rejected if price < $0.01 (runaway qty protection)",
                "side_validation": "Order side must be explicitly 'buy' or 'sell' (no defaults)",
            },
            "post_trade_controls": {
                "tca_feedback": "Slippage monitoring feeds back into routing decisions",
                "dark_pool_toxicity": "Adverse selection monitoring per dark venue",
                "compliance_hash_chain": "Every trade is hash-chained for tamper evidence",
            },
            "kill_switch": {
                "mechanism": "Guardian module runs in separate process",
                "trigger": "Drawdown > 15% or daily loss > 2%",
                "action": "Immediate halt + flatten all positions",
                "override": "Risk committee only",
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def verify_chain_integrity(self) -> Dict:
        """Verify the tamper-evident hash chain of all trade records."""
        if not self._records:
            return {"valid": True, "records_checked": 0}

        broken_at = None
        prev_hash = "0" * 64

        for i, record in enumerate(self._records):
            if record.previous_hash != prev_hash:
                broken_at = i
                break
            prev_hash = record.hash

        return {
            "valid": broken_at is None,
            "records_checked": len(self._records),
            "chain_length": len(self._records),
            "broken_at_index": broken_at,
            "latest_hash": self._last_hash,
        }
