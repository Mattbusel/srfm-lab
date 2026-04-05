"""
microstructure/hypothesis_generator.py

Generates hypotheses from microstructure evidence.

Two modes
---------
1. CONFIRMATION mode:
   If the microstructure models CONFIRM the existing IAE entry-hour blocking
   (hour 1 UTC has 3× or wider Roll spread than hourly average), generate a
   HIGH-CONFIDENCE hypothesis providing structural evidence for the rule.
   This strengthens the hypothesis in the debate system with a causal story.

2. DISCOVERY mode:
   If the models find NEW hours with bad microstructure not currently blocked,
   generate a new hypothesis to block those hours.  This is automatic
   discovery of new entry-timing rules from first-principles market structure
   evidence — not from PnL pattern mining.

Why this is valuable
--------------------
Traditional IAE hypothesis generation finds patterns in PnL data.
This generator finds patterns in market structure data first, then
asks whether the PnL would benefit from acting on that structure.
This reverses the causal arrow: instead of "we notice PnL is bad at hour X"
(which could be noise), we say "we know spreads are structurally wide at
hour X, THEREFORE entries will have bad fills THEREFORE we block it."
The causal chain is clean: microstructure → execution quality → PnL.

Output
------
Generates hypothesis.Hypothesis objects and posts them to the IAE API.
Sets hypothesis.parameters with:
    blocked_hours: [list of hours to add to entry block]
    evidence_type: "microstructure_structural"
    confidence_source: "roll_spread_hourly_analysis"
    spread_ratio_evidence: {hour: spread_ratio}
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType

from microstructure.models.intraday_patterns import (
    IntradayMicrostructureProfile,
    IntradayPatternAnalyzer,
)

logger = logging.getLogger(__name__)

IAE_API = "http://localhost:8767"

# Hours already blocked by default IAE config — used to detect NEW discoveries
DEFAULT_BLOCKED_HOURS = {0, 1, 22, 23}

# Spread ratio threshold to trigger hypothesis generation
CONFIRMATION_SPREAD_RATIO = 3.0   # 3× daily average = confirmed bad
DISCOVERY_SPREAD_RATIO = 2.5      # 2.5× = worth flagging as new discovery
MIN_BAR_COUNT = 15                # minimum hourly observations for reliability


@dataclass
class MicrostructureHypothesisOutput:
    """Output of one hypothesis generation event."""
    hypothesis: Hypothesis
    evidence_summary: str
    hours_affected: list[int]
    max_spread_ratio: float
    is_confirmation: bool     # True = confirms existing rule, False = new discovery


class MicrostructureHypothesisGenerator:
    """
    Generates trading hypotheses grounded in microstructure evidence.

    Parameters
    ----------
    api_base          : IAE API base URL.
    blocked_hours     : Set of hours already blocked (to classify new vs confirm).
    dry_run           : If True, don't post to API (return hypothesis only).
    """

    def __init__(
        self,
        api_base: str = IAE_API,
        blocked_hours: set[int] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.blocked_hours = blocked_hours or DEFAULT_BLOCKED_HOURS.copy()
        self.dry_run = dry_run
        self.analyzer = IntradayPatternAnalyzer()

    def analyse_and_generate(
        self,
        symbol: str,
        profile: IntradayMicrostructureProfile,
    ) -> list[MicrostructureHypothesisOutput]:
        """
        Analyse a microstructure profile and generate applicable hypotheses.

        Parameters
        ----------
        symbol  : The symbol the profile covers.
        profile : IntradayMicrostructureProfile from intraday_patterns.py.

        Returns
        -------
        List of generated hypotheses (may be empty if no signal found).
        """
        outputs: list[MicrostructureHypothesisOutput] = []

        # Classify each hour as: confirmed bad, newly discovered bad, or ok
        confirmed_bad: dict[int, float] = {}    # hour → spread_ratio
        newly_bad: dict[int, float] = {}

        for hour, hp in profile.profiles.items():
            if hp.bar_count < MIN_BAR_COUNT:
                continue
            sr = hp.spread_ratio

            if sr >= CONFIRMATION_SPREAD_RATIO and hour in self.blocked_hours:
                confirmed_bad[hour] = sr

            elif sr >= DISCOVERY_SPREAD_RATIO and hour not in self.blocked_hours:
                newly_bad[hour] = sr

        # --- Generate confirmation hypothesis --------------------------
        if confirmed_bad:
            output = self._build_confirmation_hypothesis(
                symbol, confirmed_bad, profile
            )
            if output:
                outputs.append(output)
                self._post_hypothesis(output.hypothesis)

        # --- Generate discovery hypothesis(es) ------------------------
        if newly_bad:
            output = self._build_discovery_hypothesis(
                symbol, newly_bad, profile
            )
            if output:
                outputs.append(output)
                self._post_hypothesis(output.hypothesis)

        return outputs

    # ------------------------------------------------------------------
    # Hypothesis builders
    # ------------------------------------------------------------------

    def _build_confirmation_hypothesis(
        self,
        symbol: str,
        confirmed_hours: dict[int, float],
        profile: IntradayMicrostructureProfile,
    ) -> MicrostructureHypothesisOutput | None:
        if not confirmed_hours:
            return None

        worst_hour = max(confirmed_hours, key=lambda h: confirmed_hours[h])
        max_ratio = confirmed_hours[worst_hour]
        hours_list = sorted(confirmed_hours.keys())

        description = (
            f"Microstructure evidence CONFIRMS entry-hour block for {symbol}. "
            f"Hours {hours_list} show Roll spread {max_ratio:.1f}x daily average. "
            f"Hour {worst_hour} is worst: spread {confirmed_hours[worst_hour]:.1f}x. "
            f"Causal chain: thin Asian-session books → wide effective spread → "
            f"poor execution → adverse fill quality. Structural basis for block is confirmed."
        )

        hyp = Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=f"microstructure::{symbol}::hour_spread",
            parameters={
                "blocked_hours": hours_list,
                "evidence_type": "microstructure_structural_confirmation",
                "confidence_source": "roll_spread_hourly_analysis",
                "spread_ratio_evidence": {str(h): round(r, 3) for h, r in confirmed_hours.items()},
                "analysis_symbol": symbol,
                "bar_counts": {
                    str(h): profile.profiles[h].bar_count
                    for h in hours_list
                    if h in profile.profiles
                },
            },
            predicted_sharpe_delta=0.05,   # conservative — this confirms an existing rule
            predicted_dd_delta=-0.01,
            novelty_score=0.3,             # not novel — confirming known rule
            description=description,
        )

        evidence_summary = (
            f"Confirmed {len(confirmed_hours)} bad hours for {symbol}. "
            f"Max spread ratio {max_ratio:.2f}x at hour {worst_hour} UTC."
        )

        return MicrostructureHypothesisOutput(
            hypothesis=hyp,
            evidence_summary=evidence_summary,
            hours_affected=hours_list,
            max_spread_ratio=max_ratio,
            is_confirmation=True,
        )

    def _build_discovery_hypothesis(
        self,
        symbol: str,
        new_hours: dict[int, float],
        profile: IntradayMicrostructureProfile,
    ) -> MicrostructureHypothesisOutput | None:
        if not new_hours:
            return None

        worst_hour = max(new_hours, key=lambda h: new_hours[h])
        max_ratio = new_hours[worst_hour]
        hours_list = sorted(new_hours.keys())

        description = (
            f"NEW microstructure discovery for {symbol}: "
            f"hours {hours_list} have structurally wide spreads "
            f"(up to {max_ratio:.1f}x daily average) but are NOT currently blocked. "
            f"Hypothesis: adding these hours to the entry block will improve "
            f"execution quality and win rate. "
            f"Evidence is structural (Roll spread analysis), not PnL-mined."
        )

        # Estimate novelty by how far the worst spread ratio exceeds the threshold
        novelty = min(0.95, 0.5 + (max_ratio - DISCOVERY_SPREAD_RATIO) / 4.0)

        hyp = Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=f"microstructure::{symbol}::new_bad_hours",
            parameters={
                "blocked_hours": hours_list,
                "evidence_type": "microstructure_structural_discovery",
                "confidence_source": "roll_spread_hourly_analysis",
                "spread_ratio_evidence": {str(h): round(r, 3) for h, r in new_hours.items()},
                "analysis_symbol": symbol,
                "bar_counts": {
                    str(h): profile.profiles[h].bar_count
                    for h in hours_list
                    if h in profile.profiles
                },
            },
            predicted_sharpe_delta=0.10,
            predicted_dd_delta=-0.02,
            novelty_score=novelty,
            description=description,
        )

        evidence_summary = (
            f"Discovered {len(new_hours)} new blocked-hour candidates for {symbol}. "
            f"Max spread ratio {max_ratio:.2f}x at hour {worst_hour} UTC."
        )

        return MicrostructureHypothesisOutput(
            hypothesis=hyp,
            evidence_summary=evidence_summary,
            hours_affected=hours_list,
            max_spread_ratio=max_ratio,
            is_confirmation=False,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def _post_hypothesis(self, hypothesis: Hypothesis) -> None:
        if self.dry_run:
            logger.info("[DRY RUN] Would post hypothesis: %s", hypothesis.description[:80])
            return

        url = f"{self.api_base}/hypotheses"
        body = json.dumps(hypothesis.to_dict()).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    logger.info(
                        "Posted microstructure hypothesis %s",
                        hypothesis.hypothesis_id[:8],
                    )
                else:
                    logger.warning(
                        "POST /hypotheses returned %d for %s",
                        resp.status,
                        hypothesis.hypothesis_id[:8],
                    )
        except urllib.error.URLError as exc:
            logger.warning("Could not post hypothesis: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error posting hypothesis: %s", exc)
