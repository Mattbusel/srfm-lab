"""
hypothesis_generator.py -- Generates testable strategy improvement hypotheses
from post-trade data for LARSA v18.

Analyzes P&L patterns across regime dimensions and automatically produces
structured hypotheses with supporting evidence and confidence estimates.
Confirmed hypotheses are forwarded to the IAE (Incremental Adaptation Engine)
at POST :8780/hypotheses/new.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from post_trade_analyzer import (
    PostTradeRecord,
    load_trades_from_db,
    _sharpe,
    _win_rate,
    _avg,
    _bucket_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IAE endpoint
# ---------------------------------------------------------------------------

IAE_URL = "http://127.0.0.1:8780/hypotheses/new"
IAE_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GeneratedHypothesis:
    """
    A testable hypothesis auto-generated from post-trade data.

    Fields
    ------
    hypothesis_id : unique identifier
    condition : human-readable description of the condition tested
    predicted_improvement : plain-English description of the expected improvement
    improvement_pct : estimated percentage improvement in avg P&L or win rate
    confidence : float in [0, 1] -- fraction of sub-samples that support the hypothesis
    supporting_evidence_n : number of trades in the supporting cohort
    p_value : statistical significance of the observed difference
    effect_size : Cohen's d or similar standardized effect size
    testable_signal_code : Python code snippet that could be added to LARSA to test this
    hypothesis_type : "filter" | "exit_timing" | "position_sizing" | "entry_timing"
    created_at : UTC timestamp
    iae_submitted : whether hypothesis was sent to IAE
    """

    hypothesis_id: str
    condition: str
    predicted_improvement: str
    improvement_pct: float
    confidence: float
    supporting_evidence_n: int
    p_value: float
    effect_size: float
    testable_signal_code: str
    hypothesis_type: str = "filter"
    created_at: datetime = field(default_factory=datetime.utcnow)
    iae_submitted: bool = False

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "condition": self.condition,
            "predicted_improvement": self.predicted_improvement,
            "improvement_pct": round(self.improvement_pct, 4),
            "confidence": round(self.confidence, 4),
            "supporting_evidence_n": self.supporting_evidence_n,
            "p_value": round(self.p_value, 6),
            "effect_size": round(self.effect_size, 4),
            "testable_signal_code": self.testable_signal_code,
            "hypothesis_type": self.hypothesis_type,
            "created_at": self.created_at.isoformat(),
            "iae_submitted": self.iae_submitted,
        }

    def __str__(self) -> str:
        return (
            f"[{self.hypothesis_id}] {self.condition}\n"
            f"  Improvement: {self.predicted_improvement} "
            f"(+{self.improvement_pct:.1f}%)\n"
            f"  Confidence: {self.confidence:.2f}, p={self.p_value:.4f}, "
            f"n={self.supporting_evidence_n}"
        )


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def _cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _ttest_pvalue(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 1.0
    try:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        return 1.0


def _bootstrap_confidence(
    group_a: list[float],
    group_b: list[float],
    n_boot: int = 500,
) -> float:
    """
    Bootstrap fraction of resamples where mean(a) > mean(b).
    Returns confidence in [0, 1] that group_a has higher mean P&L.
    """
    if not group_a or not group_b:
        return 0.0
    arr_a = np.array(group_a)
    arr_b = np.array(group_b)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample_a = rng.choice(arr_a, size=len(arr_a), replace=True)
        sample_b = rng.choice(arr_b, size=len(arr_b), replace=True)
        if sample_a.mean() > sample_b.mean():
            count += 1
    return count / n_boot


def _make_id(prefix: str, counter: int) -> str:
    return f"HYP-{prefix.upper()[:3]}-{counter:04d}"


# ---------------------------------------------------------------------------
# HypothesisGenerator
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """
    Auto-generates testable hypotheses from post-trade data for LARSA v18.

    Strategies analyzed:
    1. NAV omega filter threshold tuning
    2. Hurst-conditional hold-time extension
    3. Event calendar filter accuracy (good vs bad filtered trades)
    4. BH mass entry threshold optimization
    5. GARCH vol regime position sizing
    6. Entry reason performance gaps

    Usage::

        hg = HypothesisGenerator(trades)
        hypotheses = hg.generate()
        hg.submit_confirmed(hypotheses, confidence_threshold=0.7)
    """

    def __init__(
        self,
        trades: list[PostTradeRecord] | None = None,
        db_path: str | None = None,
        min_group_size: int = 10,
        significance_threshold: float = 0.1,
        confidence_threshold: float = 0.65,
    ):
        if trades is not None:
            self.trades = trades
        elif db_path is not None:
            self.trades = load_trades_from_db(db_path)
        else:
            self.trades = []

        self.min_group_size = min_group_size
        self.significance_threshold = significance_threshold
        self.confidence_threshold = confidence_threshold
        self._counter = 0
        logger.info(
            "HypothesisGenerator: %d trades loaded", len(self.trades)
        )

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return _make_id(prefix, self._counter)

    # ------------------------------------------------------------------
    # Top-level generate()
    # ------------------------------------------------------------------

    def generate(
        self,
        trades: list[PostTradeRecord] | None = None,
    ) -> list[GeneratedHypothesis]:
        """
        Run all hypothesis generation routines and return a deduplicated list
        sorted by confidence descending.

        Parameters
        ----------
        trades : optional override of the instance trade list

        Returns list of GeneratedHypothesis objects.
        """
        if trades is not None:
            original = self.trades
            self.trades = trades
        else:
            original = None

        hypotheses: list[GeneratedHypothesis] = []

        generators = [
            self._hyp_nav_omega_filter,
            self._hyp_hurst_hold_extension,
            self._hyp_event_calendar_filter_accuracy,
            self._hyp_bh_mass_threshold,
            self._hyp_garch_vol_position_sizing,
            self._hyp_entry_reason_selection,
            self._hyp_exit_timing_by_regime,
            self._hyp_symbol_performance_gaps,
        ]

        for gen_fn in generators:
            try:
                batch = gen_fn()
                hypotheses.extend(batch)
            except Exception as exc:
                logger.warning("Hypothesis generator %s failed: %s", gen_fn.__name__, exc)

        # Deduplicate by hypothesis_id (should be unique but guard anyway)
        seen: set[str] = set()
        unique: list[GeneratedHypothesis] = []
        for h in hypotheses:
            if h.hypothesis_id not in seen:
                seen.add(h.hypothesis_id)
                unique.append(h)

        unique.sort(key=lambda h: h.confidence, reverse=True)
        logger.info("Generated %d hypotheses", len(unique))

        if original is not None:
            self.trades = original

        return unique

    # ------------------------------------------------------------------
    # Individual hypothesis generators
    # ------------------------------------------------------------------

    def _hyp_nav_omega_filter(self) -> list[GeneratedHypothesis]:
        """
        Test whether trades entered when nav_omega < threshold have
        meaningfully different P&L than those above threshold.
        """
        hyps = []
        for threshold in [0.3, 0.5, 0.7]:
            below = [t.pnl_pct for t in self.trades if t.nav_omega_at_entry < threshold]
            above = [t.pnl_pct for t in self.trades if t.nav_omega_at_entry >= threshold]
            if len(below) < self.min_group_size or len(above) < self.min_group_size:
                continue

            p_value = _ttest_pvalue(above, below)
            d = _cohens_d(above, below)
            conf = _bootstrap_confidence(above, below)
            improvement_pct = (_avg(above) - _avg(below)) / (abs(_avg(below)) + 1e-8) * 100.0

            if p_value < self.significance_threshold and abs(d) > 0.2:
                code = (
                    f"# NAV omega filter at {threshold:.2f}\n"
                    f"if nav_omega < {threshold:.2f}:\n"
                    f"    skip_entry = True  # avg P&L {_avg(below):.4f}% vs {_avg(above):.4f}%"
                )
                hyps.append(GeneratedHypothesis(
                    hypothesis_id=self._next_id("nav"),
                    condition=f"nav_omega_at_entry < {threshold:.2f}",
                    predicted_improvement=(
                        f"Filtering out trades with nav_omega < {threshold:.2f} "
                        f"removes low-quality entries averaging {_avg(below):.3f}% P&L "
                        f"(vs {_avg(above):.3f}% above threshold)"
                    ),
                    improvement_pct=improvement_pct,
                    confidence=conf,
                    supporting_evidence_n=len(below) + len(above),
                    p_value=p_value,
                    effect_size=d,
                    testable_signal_code=code,
                    hypothesis_type="filter",
                ))
        return hyps

    def _hyp_hurst_hold_extension(self) -> list[GeneratedHypothesis]:
        """
        Test whether holding positions longer when Hurst > 0.6 captures more P&L.
        Compares trades with Hurst > 0.6 vs <= 0.6, focusing on hold_bars effect.
        """
        hyps = []
        trending = [t for t in self.trades if t.hurst_at_entry > 0.6]
        non_trending = [t for t in self.trades if t.hurst_at_entry <= 0.6]

        if len(trending) < self.min_group_size:
            return hyps

        # Among trending trades, do long holders outperform short holders?
        median_bars = float(np.median([t.hold_bars for t in trending]))
        long_hold = [t.pnl_pct for t in trending if t.hold_bars > median_bars]
        short_hold = [t.pnl_pct for t in trending if t.hold_bars <= median_bars]

        if len(long_hold) < 5 or len(short_hold) < 5:
            return hyps

        p_value = _ttest_pvalue(long_hold, short_hold)
        d = _cohens_d(long_hold, short_hold)
        conf = _bootstrap_confidence(long_hold, short_hold)
        improvement_pct = (_avg(long_hold) - _avg(short_hold)) / (abs(_avg(short_hold)) + 1e-8) * 100.0

        extra_bars = round(median_bars * 0.5)
        code = (
            f"# Hold extension for trending regime\n"
            f"if hurst > 0.6 and hold_bars < {int(median_bars) + extra_bars}:\n"
            f"    continue_hold = True  "
            f"# Long holds: {_avg(long_hold):.4f}% vs short: {_avg(short_hold):.4f}%"
        )
        hyps.append(GeneratedHypothesis(
            hypothesis_id=self._next_id("hst"),
            condition=f"hurst_at_entry > 0.6 AND hold_bars > {int(median_bars)}",
            predicted_improvement=(
                f"Holding BTC/crypto positions for >{int(median_bars)} bars when "
                f"Hurst > 0.6 captures {improvement_pct:.1f}% more P&L "
                f"(+{_avg(long_hold) - _avg(short_hold):.4f}% avg)"
            ),
            improvement_pct=improvement_pct,
            confidence=conf,
            supporting_evidence_n=len(long_hold) + len(short_hold),
            p_value=p_value,
            effect_size=d,
            testable_signal_code=code,
            hypothesis_type="exit_timing",
        ))
        return hyps

    def _hyp_event_calendar_filter_accuracy(self) -> list[GeneratedHypothesis]:
        """
        Analyze trades blocked by the event calendar vs passed trades.
        Flag if the filter removes more good trades than bad ones.
        """
        hyps = []
        blocked = [t for t in self.trades if t.event_filtered]
        passed = [t for t in self.trades if not t.event_filtered]

        if len(blocked) < self.min_group_size or len(passed) < self.min_group_size:
            return hyps

        blocked_pnls = [t.pnl_pct for t in blocked]
        passed_pnls = [t.pnl_pct for t in passed]
        blocked_wins = [t.pnl_pct for t in blocked if t.pnl_pct > 0]
        blocked_losses = [t.pnl_pct for t in blocked if t.pnl_pct <= 0]

        p_value = _ttest_pvalue(passed_pnls, blocked_pnls)
        d = _cohens_d(passed_pnls, blocked_pnls)
        conf = _bootstrap_confidence(passed_pnls, blocked_pnls)
        improvement_pct = (_avg(blocked_pnls) - _avg(passed_pnls)) / (abs(_avg(passed_pnls)) + 1e-8) * 100.0

        # Check if event filter is removing positive-expectancy trades
        if _avg(blocked_pnls) > 0 and _avg(passed_pnls) > 0:
            # Both groups are positive -- filter may be too aggressive
            code = (
                f"# Event calendar over-filtering check\n"
                f"# Blocked trades: avg={_avg(blocked_pnls):.4f}%, "
                f"win_rate={_win_rate(blocked_pnls):.2f}\n"
                f"# Review event_calendar.filter() -- may be removing +expectancy trades\n"
                f"if event_flag and bh_mass > 0.7:  # strong signal overrides soft events\n"
                f"    event_filter_override = True"
            )
            hyps.append(GeneratedHypothesis(
                hypothesis_id=self._next_id("evt"),
                condition="event_filtered=True AND avg blocked P&L > 0",
                predicted_improvement=(
                    f"Event calendar filter blocks {len(blocked)} trades averaging "
                    f"{_avg(blocked_pnls):.3f}% P&L -- {_win_rate(blocked_pnls)*100:.0f}% "
                    f"are winning trades. Relaxing filter for high BH mass could recover alpha."
                ),
                improvement_pct=abs(improvement_pct),
                confidence=conf,
                supporting_evidence_n=len(blocked),
                p_value=p_value,
                effect_size=d,
                testable_signal_code=code,
                hypothesis_type="filter",
            ))

        return hyps

    def _hyp_bh_mass_threshold(self) -> list[GeneratedHypothesis]:
        """
        Test different BH mass entry thresholds to find the sweet spot.
        """
        hyps = []
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        best_threshold = None
        best_sharpe = -999.0

        for thr in thresholds:
            above = [t.pnl_pct for t in self.trades if t.bh_mass_at_entry >= thr]
            if len(above) < self.min_group_size:
                continue
            s = _sharpe(above)
            if s > best_sharpe:
                best_sharpe = s
                best_threshold = thr

        if best_threshold is None:
            return hyps

        all_pnls = [t.pnl_pct for t in self.trades]
        current_pnls = [
            t.pnl_pct for t in self.trades if t.bh_mass_at_entry >= 0.5
        ]

        if len(current_pnls) < self.min_group_size:
            return hyps

        optimal_pnls = [
            t.pnl_pct for t in self.trades
            if t.bh_mass_at_entry >= best_threshold
        ]

        p_value = _ttest_pvalue(optimal_pnls, current_pnls)
        d = _cohens_d(optimal_pnls, current_pnls)
        conf = _bootstrap_confidence(optimal_pnls, current_pnls)
        improvement_pct = (
            (_avg(optimal_pnls) - _avg(current_pnls)) / (abs(_avg(current_pnls)) + 1e-8) * 100.0
        )

        code = (
            f"# BH mass threshold optimization\n"
            f"BH_MASS_ENTRY_THRESHOLD = {best_threshold:.2f}  "
            f"# Changed from 0.50, Sharpe={best_sharpe:.2f}\n"
            f"if bh_mass < BH_MASS_ENTRY_THRESHOLD:\n"
            f"    skip_entry = True"
        )
        hyps.append(GeneratedHypothesis(
            hypothesis_id=self._next_id("bhm"),
            condition=f"bh_mass_at_entry >= {best_threshold:.2f} (vs current 0.50)",
            predicted_improvement=(
                f"Raising BH mass entry threshold to {best_threshold:.2f} "
                f"improves Sharpe to {best_sharpe:.2f} "
                f"({improvement_pct:+.1f}% avg P&L improvement)"
            ),
            improvement_pct=improvement_pct,
            confidence=conf,
            supporting_evidence_n=len(optimal_pnls),
            p_value=p_value,
            effect_size=d,
            testable_signal_code=code,
            hypothesis_type="entry_timing",
        ))
        return hyps

    def _hyp_garch_vol_position_sizing(self) -> list[GeneratedHypothesis]:
        """
        Test whether position sizing inversely proportional to GARCH vol
        improves risk-adjusted returns.
        """
        hyps = []
        low_vol = [t for t in self.trades if t.garch_vol_at_entry < 0.015]
        high_vol = [t for t in self.trades if t.garch_vol_at_entry >= 0.03]

        if len(low_vol) < self.min_group_size or len(high_vol) < self.min_group_size:
            return hyps

        lv_pnls = [t.pnl_pct for t in low_vol]
        hv_pnls = [t.pnl_pct for t in high_vol]

        # Vol-adjusted P&L: pnl / garch_vol (Sharpe per unit of vol)
        lv_adj = [t.pnl_pct / max(t.garch_vol_at_entry, 1e-6) for t in low_vol]
        hv_adj = [t.pnl_pct / max(t.garch_vol_at_entry, 1e-6) for t in high_vol]

        p_value = _ttest_pvalue(lv_pnls, hv_pnls)
        d = _cohens_d(lv_pnls, hv_pnls)
        conf = _bootstrap_confidence(lv_pnls, hv_pnls)

        # Estimate improvement from vol-scaling
        lv_sharpe = _sharpe(lv_pnls)
        hv_sharpe = _sharpe(hv_pnls)
        combined_naive = _sharpe(lv_pnls + hv_pnls)
        scaled_hv = [p * 0.5 for p in hv_pnls]  # halved size in high vol
        combined_scaled = _sharpe(lv_pnls + scaled_hv)
        improvement_pct = (combined_scaled - combined_naive) / (abs(combined_naive) + 1e-8) * 100.0

        code = (
            "# GARCH vol-scaled position sizing\n"
            "TARGET_VOL = 0.015  # annualized daily vol target\n"
            "vol_scalar = TARGET_VOL / max(garch_vol, TARGET_VOL)\n"
            "position_size = base_position_size * min(vol_scalar, 2.0)"
        )
        hyps.append(GeneratedHypothesis(
            hypothesis_id=self._next_id("gch"),
            condition="garch_vol_at_entry >= 0.030 (high vol regime)",
            predicted_improvement=(
                f"Vol-scaled position sizing (halving size when GARCH vol >= 0.030) "
                f"improves combined Sharpe by {improvement_pct:+.1f}%. "
                f"Low vol Sharpe={lv_sharpe:.2f}, high vol Sharpe={hv_sharpe:.2f}."
            ),
            improvement_pct=improvement_pct,
            confidence=conf,
            supporting_evidence_n=len(low_vol) + len(high_vol),
            p_value=p_value,
            effect_size=d,
            testable_signal_code=code,
            hypothesis_type="position_sizing",
        ))
        return hyps

    def _hyp_entry_reason_selection(self) -> list[GeneratedHypothesis]:
        """
        Compare P&L across entry reasons; flag underperforming entry signals.
        """
        hyps = []
        reason_groups: dict[str, list[float]] = {}
        for t in self.trades:
            reason_groups.setdefault(t.entry_reason, []).append(t.pnl_pct)

        if len(reason_groups) < 2:
            return hyps

        all_pnls = [t.pnl_pct for t in self.trades]
        avg_all = _avg(all_pnls)

        for reason, pnls in reason_groups.items():
            if len(pnls) < self.min_group_size:
                continue
            other_pnls = [t.pnl_pct for t in self.trades if t.entry_reason != reason]
            if len(other_pnls) < self.min_group_size:
                continue

            p_value = _ttest_pvalue(other_pnls, pnls)
            d = _cohens_d(other_pnls, pnls)
            conf = _bootstrap_confidence(other_pnls, pnls)
            avg_reason = _avg(pnls)
            improvement_pct = (avg_all - avg_reason) / (abs(avg_reason) + 1e-8) * 100.0

            if avg_reason < avg_all * 0.5 and p_value < self.significance_threshold:
                code = (
                    f"# Disable underperforming entry reason: {reason}\n"
                    f"DISABLED_ENTRY_REASONS = ['{reason}']\n"
                    f"if entry_reason in DISABLED_ENTRY_REASONS:\n"
                    f"    skip_entry = True  "
                    f"# avg={avg_reason:.4f}% vs portfolio avg={avg_all:.4f}%"
                )
                hyps.append(GeneratedHypothesis(
                    hypothesis_id=self._next_id("ent"),
                    condition=f"entry_reason == '{reason}'",
                    predicted_improvement=(
                        f"Disabling entry reason '{reason}' (avg {avg_reason:.3f}% P&L) "
                        f"vs portfolio avg {avg_all:.3f}% could improve overall performance "
                        f"by ~{improvement_pct:.1f}%"
                    ),
                    improvement_pct=improvement_pct,
                    confidence=conf,
                    supporting_evidence_n=len(pnls),
                    p_value=p_value,
                    effect_size=d,
                    testable_signal_code=code,
                    hypothesis_type="filter",
                ))

        return hyps

    def _hyp_exit_timing_by_regime(self) -> list[GeneratedHypothesis]:
        """
        For each regime, check if extending or shortening hold time improves P&L.
        Focuses on trades where exit_efficiency < 0.6 (gave back >40% of MFE).
        """
        hyps = []
        poor_exits = [
            t for t in self.trades
            if t.mfe > 0 and t.exit_efficiency < 0.6
        ]
        good_exits = [
            t for t in self.trades
            if t.mfe > 0 and t.exit_efficiency >= 0.6
        ]

        if len(poor_exits) < self.min_group_size or len(good_exits) < self.min_group_size:
            return hyps

        # Compare hold bars between good and poor exits
        poor_bars = [t.hold_bars for t in poor_exits]
        good_bars = [t.hold_bars for t in good_exits]
        poor_pnls = [t.pnl_pct for t in poor_exits]
        good_pnls = [t.pnl_pct for t in good_exits]

        p_value = _ttest_pvalue(good_pnls, poor_pnls)
        d = _cohens_d(good_pnls, poor_pnls)
        conf = _bootstrap_confidence(good_pnls, poor_pnls)
        avg_good_bars = _avg(good_bars)
        avg_poor_bars = _avg(poor_bars)
        improvement_pct = (_avg(good_pnls) - _avg(poor_pnls)) / (abs(_avg(poor_pnls)) + 1e-8) * 100.0

        direction = "shorter" if avg_good_bars < avg_poor_bars else "longer"
        target_bars = int(avg_good_bars)

        code = (
            f"# Exit timing improvement -- target {target_bars} bars\n"
            f"# Poor exits (eff<0.6) avg {avg_poor_bars:.1f} bars, "
            f"good exits avg {avg_good_bars:.1f} bars\n"
            f"if hold_bars >= {target_bars} and exit_efficiency_est < 0.6:\n"
            f"    force_exit = True  # gave back too much MFE"
        )
        hyps.append(GeneratedHypothesis(
            hypothesis_id=self._next_id("ext"),
            condition=f"exit_efficiency < 0.60 (gave back >40% of MFE)",
            predicted_improvement=(
                f"Poor exits (eff<0.6) hold {avg_poor_bars:.1f} bars on avg vs "
                f"{avg_good_bars:.1f} for good exits. Using {direction} hold times "
                f"(target ~{target_bars} bars) could improve avg P&L by "
                f"{improvement_pct:.1f}%"
            ),
            improvement_pct=improvement_pct,
            confidence=conf,
            supporting_evidence_n=len(poor_exits) + len(good_exits),
            p_value=p_value,
            effect_size=d,
            testable_signal_code=code,
            hypothesis_type="exit_timing",
        ))
        return hyps

    def _hyp_symbol_performance_gaps(self) -> list[GeneratedHypothesis]:
        """
        Identify symbols with significantly below-average performance.
        """
        hyps = []
        symbol_groups: dict[str, list[float]] = {}
        for t in self.trades:
            symbol_groups.setdefault(t.symbol, []).append(t.pnl_pct)

        if len(symbol_groups) < 2:
            return hyps

        all_pnls = [t.pnl_pct for t in self.trades]
        avg_all = _avg(all_pnls)

        for sym, pnls in symbol_groups.items():
            if len(pnls) < self.min_group_size:
                continue
            other_pnls = [t.pnl_pct for t in self.trades if t.symbol != sym]
            if len(other_pnls) < self.min_group_size:
                continue

            p_value = _ttest_pvalue(other_pnls, pnls)
            d = _cohens_d(other_pnls, pnls)
            conf = _bootstrap_confidence(other_pnls, pnls)
            avg_sym = _avg(pnls)
            improvement_pct = (avg_all - avg_sym) / (abs(avg_sym) + 1e-8) * 100.0

            if avg_sym < avg_all * 0.3 and p_value < self.significance_threshold:
                code = (
                    f"# Symbol filter: remove underperformer\n"
                    f"EXCLUDED_SYMBOLS = ['{sym}']\n"
                    f"if symbol in EXCLUDED_SYMBOLS:\n"
                    f"    skip_entry = True  "
                    f"# avg={avg_sym:.4f}% vs all={avg_all:.4f}%"
                )
                hyps.append(GeneratedHypothesis(
                    hypothesis_id=self._next_id("sym"),
                    condition=f"symbol == '{sym}'",
                    predicted_improvement=(
                        f"Symbol '{sym}' averages {avg_sym:.3f}% P&L per trade vs "
                        f"portfolio avg {avg_all:.3f}%. Excluding it could improve "
                        f"overall performance by ~{improvement_pct:.1f}%"
                    ),
                    improvement_pct=improvement_pct,
                    confidence=conf,
                    supporting_evidence_n=len(pnls),
                    p_value=p_value,
                    effect_size=d,
                    testable_signal_code=code,
                    hypothesis_type="filter",
                ))

        return hyps

    # ------------------------------------------------------------------
    # IAE submission
    # ------------------------------------------------------------------

    def submit_to_iae(
        self,
        hypothesis: GeneratedHypothesis,
        url: str = IAE_URL,
    ) -> bool:
        """
        POST a confirmed hypothesis to the IAE endpoint.

        Returns True on success, False on failure (network error, timeout, etc).
        """
        payload = json.dumps(hypothesis.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=IAE_TIMEOUT_S) as resp:
                status = resp.getcode()
                if 200 <= status < 300:
                    hypothesis.iae_submitted = True
                    logger.info("Submitted hypothesis %s to IAE", hypothesis.hypothesis_id)
                    return True
                logger.warning(
                    "IAE returned HTTP %d for hypothesis %s",
                    status, hypothesis.hypothesis_id,
                )
                return False
        except urllib.error.URLError as exc:
            logger.warning(
                "IAE submission failed for %s: %s",
                hypothesis.hypothesis_id, exc,
            )
            return False

    def submit_confirmed(
        self,
        hypotheses: list[GeneratedHypothesis],
        confidence_threshold: float | None = None,
        url: str = IAE_URL,
    ) -> dict:
        """
        Submit all hypotheses above the confidence threshold to the IAE.

        Returns summary dict with submitted_n, failed_n, skipped_n.
        """
        thr = confidence_threshold or self.confidence_threshold
        submitted = 0
        failed = 0
        skipped = 0
        for h in hypotheses:
            if h.confidence < thr:
                skipped += 1
                continue
            if self.submit_to_iae(h, url=url):
                submitted += 1
            else:
                failed += 1

        result = {
            "submitted_n": submitted,
            "failed_n": failed,
            "skipped_n": skipped,
            "confidence_threshold": thr,
        }
        logger.info("IAE submission complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(
        self, hypotheses: list[GeneratedHypothesis]
    ) -> pd.DataFrame:
        """Convert list of hypotheses to DataFrame for inspection."""
        if not hypotheses:
            return pd.DataFrame()
        return pd.DataFrame([h.to_dict() for h in hypotheses])

    def to_json(
        self,
        hypotheses: list[GeneratedHypothesis],
        indent: int = 2,
    ) -> str:
        """Serialize hypotheses to JSON string."""
        return json.dumps([h.to_dict() for h in hypotheses], indent=indent, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Hypothesis generator for LARSA v18"
    )
    parser.add_argument("db", help="Path to trades SQLite database")
    parser.add_argument(
        "--confidence", type=float, default=0.65,
        help="Minimum confidence to print / submit",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Submit confirmed hypotheses to IAE at :8780",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON"
    )
    args = parser.parse_args()

    hg = HypothesisGenerator(db_path=args.db)
    hyps = hg.generate()

    confirmed = [h for h in hyps if h.confidence >= args.confidence]

    if args.json:
        print(hg.to_json(confirmed))
    else:
        print(f"\nGenerated {len(hyps)} hypotheses, "
              f"{len(confirmed)} above confidence={args.confidence:.2f}\n")
        for h in confirmed:
            print(h)
            print()

    if args.submit:
        result = hg.submit_confirmed(hyps, args.confidence)
        print(f"\nIAE submission: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
