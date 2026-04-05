"""
macro-factor/pipeline.py
─────────────────────────
Macro Factor Pipeline — Orchestrate daily factor computation.

This is the top-level entry point for the macro-factor module.
It:
  1. Fetches all six factor signals (DXY, Rates, VIX, Gold, Equity, Liquidity).
  2. Classifies the macro regime.
  3. Persists factor signals and regime to SQLite.
  4. Emits IAE hypotheses for extreme factor readings.
  5. Returns a MacroReport with all results.

Cadence: designed to run once daily (after US market close).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .factor_store import FactorStore
from .regime_classifier import RegimeClassifier, RegimeClassification, MacroRegime
from .signal_adapter import SignalAdapter, REGIME_MULTIPLIERS
from .factors.dxy import compute_dxy, DXYResult, dxy_summary
from .factors.rates import compute_rates, RatesResult, rates_summary
from .factors.vix import compute_vix, VIXResult, vix_summary
from .factors.gold import compute_gold, GoldResult, gold_summary
from .factors.equity_momentum import compute_equity_momentum, EquityMomentumResult, equity_summary
from .factors.liquidity import compute_liquidity, LiquidityResult, liquidity_summary

sys.path.insert(0, str(Path(__file__).parent.parent))
from hypothesis.types import Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class MacroReport:
    """Full output of a single MacroFactorPipeline run."""
    regime:              MacroRegime
    composite_score:     float
    position_multiplier: float
    classification:      Optional[RegimeClassification]
    dxy:                 Optional[DXYResult]        = None
    rates:               Optional[RatesResult]       = None
    vix:                 Optional[VIXResult]         = None
    gold:                Optional[GoldResult]        = None
    equity:              Optional[EquityMomentumResult] = None
    liquidity:           Optional[LiquidityResult]   = None
    hypotheses:          List[Hypothesis]            = field(default_factory=list)
    errors:              Dict[str, str]              = field(default_factory=dict)
    run_at:              str                         = ""

    def summary(self) -> str:
        lines = [
            f"=== MACRO REPORT {self.run_at[:10]} ===",
            f"Regime: {self.regime.value}  composite={self.composite_score:+.3f}  "
            f"position_mult={self.position_multiplier:.2f}x",
            "",
        ]
        if self.dxy:       lines.append("  " + dxy_summary(self.dxy))
        if self.rates:     lines.append("  " + rates_summary(self.rates))
        if self.vix:       lines.append("  " + vix_summary(self.vix))
        if self.gold:      lines.append("  " + gold_summary(self.gold))
        if self.equity:    lines.append("  " + equity_summary(self.equity))
        if self.liquidity: lines.append("  " + liquidity_summary(self.liquidity))
        if self.errors:
            lines.append(f"\n  Errors: {self.errors}")
        lines.append(f"\n  Hypotheses generated: {len(self.hypotheses)}")
        for h in self.hypotheses:
            lines.append(f"    [{h.type.value}] {h.description[:100]}")
        return "\n".join(lines)


class MacroFactorPipeline:
    """Orchestrate macro factor computation, regime classification, and hypothesis generation.

    Parameters
    ----------
    db_path:
        Path to the shared IAE SQLite database.
    """

    def __init__(
        self,
        db_path: Path | str = "C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db",
    ) -> None:
        self.store     = FactorStore(db_path=db_path)
        self.classifier = RegimeClassifier()
        self.adapter   = SignalAdapter()
        logger.info("MacroFactorPipeline initialised (db=%s)", db_path)

    def run(self) -> MacroReport:
        """Execute the full macro factor pipeline.

        Downloads all factor data, classifies regime, persists results,
        and generates IAE hypotheses.

        Returns
        -------
        MacroReport with all factor results, regime, hypotheses, and errors.
        """
        logger.info("MacroFactorPipeline.run() started")
        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        errors: Dict[str, str] = {}
        factor_payloads: Dict[str, dict] = {}
        factor_signals:  Dict[str, float] = {}

        # --- Compute each factor ---
        dxy_result: Optional[DXYResult] = None
        try:
            dxy_result = compute_dxy()
            factor_signals["dxy"] = dxy_result.signal
            factor_payloads["dxy"] = _result_to_dict(dxy_result)
            logger.info("DXY: %s", dxy_summary(dxy_result))
        except Exception as exc:
            logger.warning("DXY factor failed: %s", exc)
            errors["dxy"] = str(exc)

        rates_result: Optional[RatesResult] = None
        try:
            rates_result = compute_rates()
            factor_signals["rates"] = rates_result.signal
            factor_payloads["rates"] = _result_to_dict(rates_result)
            logger.info("Rates: %s", rates_summary(rates_result))
        except Exception as exc:
            logger.warning("Rates factor failed: %s", exc)
            errors["rates"] = str(exc)

        vix_result: Optional[VIXResult] = None
        try:
            vix_result = compute_vix()
            factor_signals["vix"] = vix_result.signal
            factor_payloads["vix"] = _result_to_dict(vix_result)
            logger.info("VIX: %s", vix_summary(vix_result))
        except Exception as exc:
            logger.warning("VIX factor failed: %s", exc)
            errors["vix"] = str(exc)

        gold_result: Optional[GoldResult] = None
        try:
            gold_result = compute_gold()
            factor_signals["gold"] = gold_result.signal
            factor_payloads["gold"] = _result_to_dict(gold_result)
            logger.info("Gold: %s", gold_summary(gold_result))
        except Exception as exc:
            logger.warning("Gold factor failed: %s", exc)
            errors["gold"] = str(exc)

        equity_result: Optional[EquityMomentumResult] = None
        try:
            equity_result = compute_equity_momentum()
            factor_signals["equity_momentum"] = equity_result.signal
            factor_payloads["equity_momentum"] = _result_to_dict(equity_result)
            logger.info("Equity: %s", equity_summary(equity_result))
        except Exception as exc:
            logger.warning("Equity factor failed: %s", exc)
            errors["equity_momentum"] = str(exc)

        liquidity_result: Optional[LiquidityResult] = None
        try:
            liquidity_result = compute_liquidity()
            factor_signals["liquidity"] = liquidity_result.signal
            factor_payloads["liquidity"] = _result_to_dict(liquidity_result)
            logger.info("Liquidity: %s", liquidity_summary(liquidity_result))
        except Exception as exc:
            logger.warning("Liquidity factor failed: %s", exc)
            errors["liquidity"] = str(exc)

        # --- Persist factor signals ---
        self.store.upsert_all_factors(today, factor_signals, factor_payloads)

        # --- Classify regime ---
        try:
            classification = self.classifier.classify(
                dxy=dxy_result,
                rates=rates_result,
                vix=vix_result,
                gold=gold_result,
                equity=equity_result,
                liquidity=liquidity_result,
            )
        except Exception as exc:
            logger.error("Regime classification failed: %s", exc)
            errors["regime_classifier"] = str(exc)
            classification = None

        regime     = classification.regime            if classification else MacroRegime.RISK_NEUTRAL
        composite  = classification.composite_score   if classification else 0.0
        multiplier = classification.position_multiplier if classification else 1.0

        # --- Persist regime ---
        if classification:
            self.store.upsert_regime(
                date=today,
                regime=regime.value,
                composite_score=composite,
                position_multiplier=multiplier,
                crisis_override=classification.crisis_override,
                crisis_reason=classification.crisis_reason,
                confidence=classification.confidence,
                payload={
                    "component_signals": classification.component_signals,
                    "errors": errors,
                },
            )

        # --- Generate hypotheses ---
        hypotheses: List[Hypothesis] = []
        if classification:
            try:
                hypotheses = self.adapter.adapt(
                    classification=classification,
                    dxy=dxy_result,
                    vix=vix_result,
                    equity=equity_result,
                    liquidity=liquidity_result,
                )
            except Exception as exc:
                logger.warning("Hypothesis generation failed: %s", exc)
                errors["hypothesis_gen"] = str(exc)

        report = MacroReport(
            regime=regime,
            composite_score=round(composite, 4),
            position_multiplier=multiplier,
            classification=classification,
            dxy=dxy_result,
            rates=rates_result,
            vix=vix_result,
            gold=gold_result,
            equity=equity_result,
            liquidity=liquidity_result,
            hypotheses=hypotheses,
            errors=errors,
            run_at=datetime.now(timezone.utc).isoformat(),
        )

        # Log regime transition if any
        transition = self.store.get_regime_transition()
        if transition:
            logger.warning(
                "MACRO REGIME TRANSITION: %s → %s", transition[0], transition[1]
            )

        logger.info(
            "MacroFactorPipeline.run() complete — regime=%s composite=%.3f mult=%.2fx "
            "hypotheses=%d",
            regime.value, composite, multiplier, len(hypotheses),
        )
        return report


def _result_to_dict(obj: object) -> dict:
    """Convert a dataclass result to a plain dict for JSON serialisation."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return {"value": str(obj)}
