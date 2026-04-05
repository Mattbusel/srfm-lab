"""
report_generator.py
===================
Generate a comprehensive adversarial risk report by aggregating results
from all adversarial sub-modules.

Report contents
---------------
1. **Top 5 worst-case scenarios** with estimated probabilities.
   Scenarios are drawn from fuzzer (worst parameter combos), regime
   stress tests, and adversarial market paths.

2. **Parameter sensitivity ranking**.
   From WorstCaseFinder: which parameters to watch most closely.

3. **Overfitting risk score per hypothesis**.
   From OverfittingDetector: walk-forward and CPCV results.

4. **Recommended guardrails**.
   Hard stops and circuit breakers derived from the analysis.

Output formats
--------------
- Plain text summary (suitable for Slack / email).
- Structured dict (suitable for JSON export or DB insertion).
- Markdown report (suitable for the idea-dashboard).

Usage::

    runner  = ReportGenerator()
    report  = runner.run_full(trade_pnl=pnl_arr, current_params=params)
    print(report.text_summary())
    with open("adversarial_report.md", "w") as f:
        f.write(report.markdown())
"""

from __future__ import annotations

import logging
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fuzzer             import ParameterFuzzer, FuzzerResult
from .worst_case_finder  import WorstCaseFinder, SensitivityResult
from .regime_stress      import RegimeStressor, StressResult
from .correlation_attack import CorrelationAttacker, CorrelationResult
from .overfitting_detector import OverfittingDetector, OverfittingReport
from .adversarial_market import AdversarialMarket, AdversarialPath

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail recommendations
# ---------------------------------------------------------------------------

@dataclass
class Guardrail:
    """A single recommended hard stop or circuit breaker."""

    name:         str
    description:  str
    trigger:      str       # e.g. "drawdown > 15%"
    action:       str       # e.g. "halt trading for 4 hours"
    source:       str       # which analysis generated this
    priority:     str = "HIGH"   # HIGH / MEDIUM / LOW


# ---------------------------------------------------------------------------
# Worst-case scenario
# ---------------------------------------------------------------------------

@dataclass
class WorstCaseScenario:
    """One of the top-N worst-case scenarios."""

    rank:               int
    scenario_type:      str    # "REGIME", "PARAMETER", "ADVERSARIAL_PATH", "CORRELATION"
    name:               str
    estimated_pnl:      float
    estimated_probability: float
    description:        str
    source_module:      str


# ---------------------------------------------------------------------------
# Full report container
# ---------------------------------------------------------------------------

@dataclass
class AdversarialReport:
    """
    Aggregated adversarial risk report.

    Attributes
    ----------
    timestamp              : report generation time.
    top_scenarios          : top 5 worst-case scenarios.
    sensitivity_ranking    : ordered list of (param, sensitivity) tuples.
    overfitting_summary    : per-hypothesis overfit status.
    guardrails             : recommended guardrails.
    critical_corr          : correlation level at which portfolio blows up.
    regime_worst_case      : name of the most dangerous historical regime.
    adversarial_path_pnl   : P&L of the adversarial price path.
    overall_risk_score     : composite risk score in [0, 10].
    """

    timestamp:              str
    top_scenarios:          List[WorstCaseScenario]
    sensitivity_ranking:    List[Tuple[str, float]]
    overfitting_summary:    Dict[str, bool]
    guardrails:             List[Guardrail]
    critical_corr:          float
    regime_worst_case:      str
    adversarial_path_pnl:   float
    overall_risk_score:     float
    raw_results:            dict = field(default_factory=dict, repr=False)

    def text_summary(self) -> str:
        lines = [
            "=" * 60,
            "ADVERSARIAL RISK REPORT",
            f"Generated: {self.timestamp}",
            f"Overall risk score: {self.overall_risk_score:.1f} / 10",
            "=" * 60,
            "",
            "TOP 5 WORST-CASE SCENARIOS:",
        ]
        for sc in self.top_scenarios:
            lines.append(
                f"  {sc.rank}. [{sc.scenario_type}] {sc.name}\n"
                f"     P&L: {sc.estimated_pnl:+.4f}  "
                f"Probability: {sc.estimated_probability:.1%}\n"
                f"     {sc.description}"
            )
        lines += [
            "",
            "PARAMETER SENSITIVITY (most dangerous first):",
        ]
        for rank, (param, sens) in enumerate(self.sensitivity_ranking[:6], 1):
            lines.append(f"  {rank}. {param:<25s}  dP&L/dparam = {sens:+.4f}")
        lines += [
            "",
            f"OVERFITTING STATUS:",
            f"  Overfit hypotheses: {sum(self.overfitting_summary.values())} / {len(self.overfitting_summary)}",
            "",
            "GUARDRAILS:",
        ]
        for g in self.guardrails:
            lines.append(
                f"  [{g.priority}] {g.name}\n"
                f"    Trigger: {g.trigger}\n"
                f"    Action:  {g.action}"
            )
        lines += [
            "",
            f"Critical correlation level: {self.critical_corr:.3f}",
            f"Most dangerous regime: {self.regime_worst_case}",
            f"Adversarial path P&L: {self.adversarial_path_pnl:+.6f}",
        ]
        return "\n".join(lines)

    def markdown(self) -> str:
        """Generate a Markdown version of the report."""
        lines = [
            "# Adversarial Risk Report",
            f"**Generated:** {self.timestamp}  ",
            f"**Risk Score:** {self.overall_risk_score:.1f} / 10",
            "",
            "## Top 5 Worst-Case Scenarios",
            "",
            "| Rank | Type | Name | P&L | Probability |",
            "|------|------|------|-----|-------------|",
        ]
        for sc in self.top_scenarios:
            lines.append(
                f"| {sc.rank} | {sc.scenario_type} | {sc.name} | "
                f"{sc.estimated_pnl:+.4f} | {sc.estimated_probability:.1%} |"
            )
        lines += [
            "",
            "## Parameter Sensitivity",
            "",
            "| Rank | Parameter | dP&L/dparam |",
            "|------|-----------|-------------|",
        ]
        for rank, (param, sens) in enumerate(self.sensitivity_ranking, 1):
            lines.append(f"| {rank} | `{param}` | {sens:+.4f} |")
        lines += [
            "",
            "## Guardrails",
            "",
        ]
        for g in self.guardrails:
            lines.append(f"### {g.name} `[{g.priority}]`")
            lines.append(f"- **Trigger:** {g.trigger}")
            lines.append(f"- **Action:** {g.action}")
            lines.append(f"- **Source:** {g.source}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_risk_score": self.overall_risk_score,
            "top_scenarios": [
                {
                    "rank": sc.rank,
                    "type": sc.scenario_type,
                    "name": sc.name,
                    "pnl": sc.estimated_pnl,
                    "probability": sc.estimated_probability,
                    "description": sc.description,
                }
                for sc in self.top_scenarios
            ],
            "sensitivity_ranking": [
                {"param": p, "sensitivity": s}
                for p, s in self.sensitivity_ranking
            ],
            "overfitting": self.overfitting_summary,
            "guardrails": [
                {
                    "name": g.name,
                    "trigger": g.trigger,
                    "action": g.action,
                    "priority": g.priority,
                }
                for g in self.guardrails
            ],
            "critical_corr": self.critical_corr,
            "regime_worst_case": self.regime_worst_case,
            "adversarial_path_pnl": self.adversarial_path_pnl,
        }


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Orchestrate all adversarial analyses and compile a single risk report.

    Parameters
    ----------
    current_params : dict of current strategy parameters.
    n_fuzz_samples : number of fuzzer parameter combinations.
    n_stress_paths : Monte Carlo paths per stress regime.
    n_corr_paths   : Monte Carlo paths for correlation attack.
    run_adversarial_market : whether to run the (slow) adversarial path search.
    seed           : base random seed.
    """

    def __init__(
        self,
        current_params: Optional[Dict] = None,
        n_fuzz_samples: int   = 1000,    # reduced from 10k for fast default run
        n_stress_paths: int   = 500,
        n_corr_paths:   int   = 500,
        run_adversarial_market: bool = True,
        seed:           int   = 42,
    ):
        from .worst_case_finder import CURRENT_PARAMS
        self.current_params    = current_params or CURRENT_PARAMS.copy()
        self.n_fuzz_samples    = n_fuzz_samples
        self.n_stress_paths    = n_stress_paths
        self.n_corr_paths      = n_corr_paths
        self.run_adv_market    = run_adversarial_market
        self.seed              = seed

    def run_full(
        self,
        trade_pnl: Optional[np.ndarray] = None,
        hypothesis_pnl: Optional[dict]  = None,
    ) -> AdversarialReport:
        """
        Run all adversarial analyses and produce a comprehensive report.

        Parameters
        ----------
        trade_pnl      : historical per-trade P&L array (for overfitting detection).
        hypothesis_pnl : per-hypothesis P&L for walk-forward validation.

        Returns
        -------
        AdversarialReport.
        """
        logger.info("Starting full adversarial analysis suite.")

        # Use synthetic P&L if none provided
        if trade_pnl is None:
            rng = np.random.default_rng(self.seed)
            trade_pnl = rng.normal(0.003, 0.018, 500)

        trade_pnl = np.asarray(trade_pnl, dtype=float)
        raw = {}

        # 1. Fuzzer
        logger.info("Running parameter fuzzer (%d samples)...", self.n_fuzz_samples)
        fuzzer = ParameterFuzzer(n_samples=self.n_fuzz_samples, seed=self.seed)
        raw["fuzzer"] = fuzzer.run()

        # 2. Worst-case finder
        logger.info("Running worst-case sensitivity analysis...")
        wcf = WorstCaseFinder(current_params=self.current_params, seed=self.seed)
        raw["worst_case"] = wcf.run()

        # 3. Regime stress
        logger.info("Running regime stress tests...")
        stressor  = RegimeStressor(n_paths=self.n_stress_paths, seed=self.seed)
        raw["regime"] = stressor.run_all()

        # 4. Correlation attack
        logger.info("Running correlation attack...")
        corr_att = CorrelationAttacker(
            current_corr_factor=self.current_params.get("corr_factor", 0.25),
            n_paths=self.n_corr_paths,
            seed=self.seed,
        )
        raw["correlation"] = corr_att.run()

        # 5. Overfitting detector
        logger.info("Running overfitting detector...")
        ov_det = OverfittingDetector(pnl_series=trade_pnl)
        raw["overfitting"] = ov_det.run(hypothesis_pnl=hypothesis_pnl)

        # 6. Adversarial market path
        adv_path_pnl = float("nan")
        if self.run_adv_market:
            logger.info("Searching for adversarial price path...")
            adv_mkt = AdversarialMarket(seed=self.seed)
            raw["adversarial_path"] = adv_mkt.run()
            adv_path_pnl = raw["adversarial_path"].pnl

        # Build report
        report = self._compile_report(raw, adv_path_pnl)
        logger.info("Adversarial report complete. Risk score: %.1f / 10",
                    report.overall_risk_score)
        return report

    # ------------------------------------------------------------------
    # Report compilation
    # ------------------------------------------------------------------

    def _compile_report(self, raw: dict, adv_path_pnl: float) -> AdversarialReport:
        """Aggregate all raw results into an AdversarialReport."""

        timestamp = datetime.utcnow().isoformat()

        # Sensitivity ranking from worst-case finder
        wc: SensitivityResult = raw["worst_case"]
        sens_ranked = sorted(
            wc.sensitivity_matrix.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )

        # Overfitting summary
        ov: OverfittingReport = raw["overfitting"]
        overfit_summary = {
            v.hypothesis_id: v.is_overfit
            for v in ov.hypothesis_validations
        }

        # Critical correlation
        corr: CorrelationResult = raw["correlation"]
        critical_corr = corr.critical_corr

        # Worst regime
        regime_results: Dict[str, StressResult] = raw["regime"]
        regime_worst = min(regime_results, key=lambda k: regime_results[k].pnl_p5)

        # Top 5 scenarios
        top5 = self._build_top5(raw, adv_path_pnl)

        # Guardrails
        guardrails = self._build_guardrails(raw, corr, regime_results)

        # Risk score
        risk_score = self._compute_risk_score(raw, corr, ov)

        return AdversarialReport(
            timestamp=timestamp,
            top_scenarios=top5,
            sensitivity_ranking=sens_ranked,
            overfitting_summary=overfit_summary,
            guardrails=guardrails,
            critical_corr=critical_corr,
            regime_worst_case=regime_worst,
            adversarial_path_pnl=adv_path_pnl,
            overall_risk_score=risk_score,
            raw_results=raw,
        )

    def _build_top5(self, raw: dict, adv_path_pnl: float) -> List[WorstCaseScenario]:
        """Collect and rank the top 5 worst-case scenarios."""
        candidates: List[WorstCaseScenario] = []

        # Regime stress scenarios
        for rname, rs in raw["regime"].items():
            candidates.append(WorstCaseScenario(
                rank=0,
                scenario_type="REGIME",
                name=rs.description,
                estimated_pnl=rs.pnl_p5,
                estimated_probability=rs.blowup_probability,
                description=(
                    f"5th-pct P&L={rs.pnl_p5:+.4f}, "
                    f"max_dd={rs.max_drawdown_p5:+.2%}, "
                    f"blowup_prob={rs.blowup_probability:.2%}"
                ),
                source_module="regime_stress",
            ))

        # Correlation spike
        spike = raw["correlation"].spike_result
        candidates.append(WorstCaseScenario(
            rank=0,
            scenario_type="CORRELATION",
            name="Correlation spike to 0.95 (30 days)",
            estimated_pnl=spike.pnl_p5,
            estimated_probability=spike.blowup_probability,
            description=(
                f"Realized corr=0.95 for 30 days: "
                f"P&L p5={spike.pnl_p5:+.4f}, "
                f"blowup={spike.blowup_probability:.2%}"
            ),
            source_module="correlation_attack",
        ))

        # Fuzzer worst case
        fuz: FuzzerResult = raw["fuzzer"]
        candidates.append(WorstCaseScenario(
            rank=0,
            scenario_type="PARAMETER",
            name="Worst-case parameter combination (LHS fuzzer)",
            estimated_pnl=fuz.worst_case_pnl,
            estimated_probability=0.01,  # 1% of parameter space is this bad
            description=(
                f"Worst combo: {fuz.worst_case_params}. "
                f"P&L={fuz.worst_case_pnl:+.4f}"
            ),
            source_module="fuzzer",
        ))

        # Adversarial price path
        if math.isfinite(adv_path_pnl):
            candidates.append(WorstCaseScenario(
                rank=0,
                scenario_type="ADVERSARIAL_PATH",
                name="Adversarial price sequence (CMA-ES optimised)",
                estimated_pnl=adv_path_pnl,
                estimated_probability=0.005,
                description=(
                    f"Gradient-optimised price path that minimises P&L: "
                    f"{adv_path_pnl:+.6f}. Statistically plausible."
                ),
                source_module="adversarial_market",
            ))

        # Sort by P&L ascending (worst first)
        candidates.sort(key=lambda s: s.estimated_pnl)
        top5 = candidates[:5]
        for i, sc in enumerate(top5, 1):
            sc.rank = i
        return top5

    def _build_guardrails(
        self,
        raw: dict,
        corr: CorrelationResult,
        regimes: Dict[str, StressResult],
    ) -> List[Guardrail]:
        """Derive hard stops and circuit breakers from the analysis."""
        guardrails: List[Guardrail] = []

        # Drawdown-based halt
        worst_dd = min(r.max_drawdown_p5 for r in regimes.values())
        halt_dd  = round(abs(worst_dd) * 0.60, 2)  # trigger at 60% of worst case
        guardrails.append(Guardrail(
            name="Intraday drawdown halt",
            description=(
                f"Halt all new trade entries if intraday drawdown exceeds {halt_dd:.0%}."
            ),
            trigger=f"drawdown > {halt_dd:.0%} from intraday high",
            action="Halt new entries for 4 hours; review regime.",
            source="regime_stress",
            priority="HIGH",
        ))

        # Correlation circuit breaker
        safe_corr = min(corr.critical_corr * 0.80, 0.70)
        guardrails.append(Guardrail(
            name="Realized-correlation circuit breaker",
            description=(
                f"Reduce position sizes by 50% if realized 1-day correlation "
                f"exceeds {safe_corr:.2f}."
            ),
            trigger=f"realized_1d_corr > {safe_corr:.2f}",
            action="Cut position sizes by 50% until correlation normalises.",
            source="correlation_attack",
            priority="HIGH",
        ))

        # Parameter drift alert
        wc: SensitivityResult = raw["worst_case"]
        most_dangerous = wc.most_damaging_param
        guardrails.append(Guardrail(
            name=f"{most_dangerous} parameter drift alert",
            description=(
                f"Alert if {most_dangerous} deviates more than 30% from baseline."
            ),
            trigger=f"|{most_dangerous} - baseline| / baseline > 0.30",
            action="Flag for IAE review; do not apply without revalidation.",
            source="worst_case_finder",
            priority="MEDIUM",
        ))

        # Overfitting circuit breaker
        ov: OverfittingReport = raw["overfitting"]
        if not ov.cpcv_result.is_robust:
            guardrails.append(Guardrail(
                name="CPCV robustness alert",
                description=(
                    "CPCV indicates strategy is not robust across time periods. "
                    "Require 2x normal walk-forward evidence before applying new hypotheses."
                ),
                trigger="CPCV P25 Sharpe < 0",
                action="Double the minimum OOS trades required before deploying hypotheses.",
                source="overfitting_detector",
                priority="MEDIUM",
            ))

        return guardrails

    def _compute_risk_score(
        self,
        raw: dict,
        corr: CorrelationResult,
        ov: OverfittingReport,
    ) -> float:
        """
        Compute an overall risk score in [0, 10].

        Contributions:
        - Blowup probability under worst regime (0-3 pts).
        - Distance of critical_corr from current corr_factor (0-3 pts).
        - CPCV P25 Sharpe (0-2 pts).
        - Overfitting rate (0-2 pts).
        """
        score = 0.0

        # Blowup probability
        max_blowup = max(r.blowup_probability for r in raw["regime"].values())
        score += min(max_blowup * 10, 3.0)

        # Correlation proximity to critical
        current_corr = self.current_params.get("corr_factor", 0.25)
        corr_gap     = max(corr.critical_corr - current_corr, 0.0)
        score += max(0, 3.0 - corr_gap * 15)

        # CPCV Sharpe
        p25_sharpe = ov.cpcv_result.sharpe_p25
        if p25_sharpe < 0:
            score += 2.0
        elif p25_sharpe < 0.5:
            score += 1.0

        # Overfitting rate
        score += min(ov.overall_overfit_score * 4, 2.0)

        return round(min(score, 10.0), 2)

    def save(self, report: AdversarialReport, path: Path) -> None:
        """Save the report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Adversarial report saved to %s", path)
