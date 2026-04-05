"""
hypothesis/templates/entry_timing_template.py

Converts a time_of_day MinedPattern into entry-timing hypotheses.
Hypothesis: "only enter between hour X and hour Y on days D"
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType, MinedPattern


# ---------------------------------------------------------------------------
# Helper: naive Sharpe-delta estimate from effect size + p-value
# ---------------------------------------------------------------------------

def _estimate_sharpe_delta(effect_size: float, p_value: float, ci_lower: float, ci_upper: float) -> float:
    """
    Estimate the expected change in annualised Sharpe ratio if we apply the filter.

    Intuition:
    - A large Cohen's-d effect_size means meaningful separation of return distributions.
    - Low p_value increases our confidence.
    - We dampen by ci_width to penalise uncertain estimates.
    - We cap at ±1.5 to avoid fantasy numbers.

    Returns a signed float: positive = Sharpe improves.
    """
    if p_value <= 0:
        p_value = 1e-10
    if p_value >= 1:
        p_value = 1 - 1e-10

    # Confidence factor: log-odds of significance (saturates near 0/1)
    confidence = -math.log10(p_value) / 10.0   # p=0.01 → 0.2, p=0.001 → 0.3
    confidence = min(confidence, 1.0)

    # CI penalty: wide CI = uncertain = discount
    ci_width = max(ci_upper - ci_lower, 1e-6)
    ci_penalty = 1.0 / (1.0 + ci_width)

    raw_delta = effect_size * confidence * ci_penalty
    return float(max(min(raw_delta, 1.5), -1.5))


def _estimate_dd_delta(effect_size: float, regime_context: dict) -> float:
    """
    Drawdown delta: positive = drawdown shrinks (good).
    Time-of-day filters typically reduce drawdown by restricting entry to good windows.
    """
    base = 0.0
    if effect_size > 0:
        # Every 0.1 unit of effect_size → roughly 0.5% DD improvement (heuristic)
        base = effect_size * 0.05
    if regime_context.get("high_vol_hours"):
        base += 0.02
    return float(min(base, 0.15))


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class EntryTimingTemplate:
    """
    Generates entry-timing hypotheses from a time_of_day MinedPattern.

    The pattern evidence is expected to contain:
        best_hours: list[int]           — UTC hours with positive edge
        best_days:  list[int]           — 0=Mon … 6=Sun
        hourly_sharpe: dict[str, float] — hour_str → sharpe
        session: str                    — "asia" | "london" | "ny" | ...
    """

    PATTERN_TYPE = "time_of_day"

    # Minimum effect size to bother generating a hypothesis
    MIN_EFFECT_SIZE = 0.05
    # Minimum confidence in best_hours list before we trust it
    MIN_SIGNIFICANT_HOURS = 1

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_handle(self, pattern: MinedPattern) -> bool:
        return pattern.pattern_type == self.PATTERN_TYPE

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        """
        Return a list of Hypothesis objects derived from the pattern.
        May return multiple hypotheses (one per instrument or session split).
        """
        if not self.can_handle(pattern):
            raise ValueError(f"EntryTimingTemplate cannot handle pattern type '{pattern.pattern_type}'")

        if abs(pattern.effect_size) < self.MIN_EFFECT_SIZE:
            return []

        hypotheses: list[Hypothesis] = []

        # --- Extract evidence fields ---
        best_hours: list[int] = pattern.evidence.get("best_hours", [])
        best_days: list[int] = pattern.evidence.get("best_days", list(range(7)))
        hourly_sharpe: dict[str, float] = pattern.evidence.get("hourly_sharpe", {})
        session: str = pattern.evidence.get("session", "unknown")

        if not best_hours:
            best_hours = self._infer_best_hours(hourly_sharpe)

        if len(best_hours) < self.MIN_SIGNIFICANT_HOURS:
            return []

        hour_start, hour_end = self._hours_to_window(best_hours)

        # --- Primary hypothesis: per-instrument time filter ---
        for instrument in pattern.instruments:
            h = self._build_primary_hypothesis(
                pattern=pattern,
                instrument=instrument,
                hour_start=hour_start,
                hour_end=hour_end,
                days_of_week=best_days,
                session=session,
            )
            hypotheses.append(h)

        # --- Secondary: avoid worst hours (inverse filter) ---
        worst_hours = self._infer_worst_hours(hourly_sharpe, best_hours)
        if worst_hours:
            h_avoid = self._build_avoid_hypothesis(
                pattern=pattern,
                worst_hours=worst_hours,
                days_of_week=best_days,
            )
            hypotheses.append(h_avoid)

        # --- Tertiary: session-specific hypothesis if session is named ---
        if session not in ("unknown", ""):
            h_session = self._build_session_hypothesis(
                pattern=pattern,
                session=session,
                hour_start=hour_start,
                hour_end=hour_end,
                days_of_week=best_days,
            )
            hypotheses.append(h_session)

        return hypotheses

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_primary_hypothesis(
        self,
        pattern: MinedPattern,
        instrument: str,
        hour_start: int,
        hour_end: int,
        days_of_week: list[int],
        session: str,
    ) -> Hypothesis:
        sharpe_delta = _estimate_sharpe_delta(
            pattern.effect_size, pattern.p_value, pattern.ci_lower, pattern.ci_upper
        )
        dd_delta = _estimate_dd_delta(pattern.effect_size, pattern.regime_context)

        parameters: dict[str, Any] = {
            "entry_hour_start": hour_start,
            "entry_hour_end": hour_end,
            "days_of_week": days_of_week,
            "instruments": [instrument],
            "session_label": session,
            "filter_mode": "allow",   # allow trades only in window
        }

        description = (
            f"[{instrument}] Only enter between {hour_start:02d}:00–{hour_end:02d}:00 UTC "
            f"on days {days_of_week} (session: {session}). "
            f"Effect size: {pattern.effect_size:.3f}, p={pattern.p_value:.4f}."
        )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=pattern.pattern_id,
            parameters=parameters,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern),
            description=description,
        )

    def _build_avoid_hypothesis(
        self,
        pattern: MinedPattern,
        worst_hours: list[int],
        days_of_week: list[int],
    ) -> Hypothesis:
        avoid_start, avoid_end = self._hours_to_window(worst_hours)

        sharpe_delta = _estimate_sharpe_delta(
            pattern.effect_size * 0.7,  # slightly discounted for inverse filter
            pattern.p_value,
            pattern.ci_lower,
            pattern.ci_upper,
        )
        dd_delta = _estimate_dd_delta(pattern.effect_size * 0.7, pattern.regime_context)

        parameters: dict[str, Any] = {
            "entry_hour_start": avoid_start,
            "entry_hour_end": avoid_end,
            "days_of_week": days_of_week,
            "instruments": pattern.instruments,
            "filter_mode": "block",   # block entries in this window
        }

        description = (
            f"Block entries between {avoid_start:02d}:00–{avoid_end:02d}:00 UTC "
            f"(worst-hour filter). Parent pattern {pattern.pattern_id[:8]}."
        )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=pattern.pattern_id,
            parameters=parameters,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern) * 0.8,
            description=description,
        )

    def _build_session_hypothesis(
        self,
        pattern: MinedPattern,
        session: str,
        hour_start: int,
        hour_end: int,
        days_of_week: list[int],
    ) -> Hypothesis:
        sharpe_delta = _estimate_sharpe_delta(
            pattern.effect_size * 0.85,
            pattern.p_value,
            pattern.ci_lower,
            pattern.ci_upper,
        )
        dd_delta = _estimate_dd_delta(pattern.effect_size * 0.85, pattern.regime_context)

        parameters: dict[str, Any] = {
            "entry_hour_start": hour_start,
            "entry_hour_end": hour_end,
            "days_of_week": days_of_week,
            "instruments": pattern.instruments,
            "session_label": session,
            "filter_mode": "session_allow",
        }

        description = (
            f"Session filter: only enter during '{session}' session "
            f"({hour_start:02d}:00–{hour_end:02d}:00 UTC). All instruments."
        )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=pattern.pattern_id,
            parameters=parameters,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern) * 0.9,
            description=description,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_best_hours(hourly_sharpe: dict[str, float]) -> list[int]:
        """Pick hours above median Sharpe from the hourly_sharpe map."""
        if not hourly_sharpe:
            return []
        values = list(hourly_sharpe.values())
        if not values:
            return []
        median_sharpe = sorted(values)[len(values) // 2]
        good = [int(h) for h, s in hourly_sharpe.items() if s > median_sharpe]
        return sorted(good)

    @staticmethod
    def _infer_worst_hours(hourly_sharpe: dict[str, float], best_hours: list[int]) -> list[int]:
        """Pick hours that are below zero Sharpe and not in best_hours."""
        if not hourly_sharpe:
            return []
        worst = [
            int(h)
            for h, s in hourly_sharpe.items()
            if s < 0 and int(h) not in best_hours
        ]
        return sorted(worst)

    @staticmethod
    def _hours_to_window(hours: list[int]) -> tuple[int, int]:
        """
        Convert a list of good hours to a [start, end) window.
        Handles wrap-around (e.g. [22, 23, 0, 1]).
        Returns (start_hour, end_hour) in [0, 24).
        """
        if not hours:
            return (0, 24)

        hours_sorted = sorted(set(hours))
        if len(hours_sorted) == 1:
            h = hours_sorted[0]
            return (h, (h + 1) % 24)

        # Detect wraparound: gaps > 12 hours suggest the window crosses midnight
        gaps = [(hours_sorted[i + 1] - hours_sorted[i]) for i in range(len(hours_sorted) - 1)]
        max_gap_idx = gaps.index(max(gaps))
        if max(gaps) > 12:
            # wraparound: start = element after the big gap, end = element before it
            start = hours_sorted[max_gap_idx + 1]
            end = (hours_sorted[max_gap_idx] + 1) % 24
        else:
            start = hours_sorted[0]
            end = (hours_sorted[-1] + 1) % 24

        return (start, end)

    @staticmethod
    def _novelty(pattern: MinedPattern) -> float:
        """
        Compute a rough novelty score for this pattern.
        More unusual pattern_type contexts get higher novelty.
        """
        base = 0.5
        # Unusual regime context boosts novelty
        if pattern.regime_context:
            base += min(len(pattern.regime_context) * 0.05, 0.3)
        # High effect size in time_of_day is relatively rare → boost
        if abs(pattern.effect_size) > 0.5:
            base += 0.1
        return min(base, 1.0)
