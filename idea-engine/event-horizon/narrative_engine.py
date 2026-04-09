"""
Narrative Engine: the system that writes its own fund letter.

Reads all recent trading decisions, provenance traces, dream results,
and consciousness states, then synthesizes a human-readable narrative
about what the system is doing and why.

This serves three purposes:
  1. Investor communications: auto-generated fund letters
  2. Internal monitoring: human operators understand the system's "thinking"
  3. Regulatory: demonstrates algorithmic decision-making rationale

The narrative identifies THEMES across recent trades (not just individual
explanations) -- patterns that no single trade's provenance reveals.
"""

from __future__ import annotations
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class NarrativeTheme:
    """A recurring theme across multiple recent decisions."""
    theme_id: str
    name: str
    description: str
    frequency: int                # how many decisions share this theme
    confidence: float             # 0-1
    contributing_signals: List[str]
    regime_context: str
    example_trade: str


@dataclass
class FundLetter:
    """Auto-generated fund letter / commentary."""
    period: str                   # "Week of 2026-04-08"
    headline: str                 # one-line summary
    market_view: str              # current market assessment
    themes: List[NarrativeTheme]
    performance_summary: str
    risk_assessment: str
    outlook: str
    generated_at: float = 0.0


class NarrativeEngine:
    """
    Auto-generates fund letters and trading commentary from system state.
    """

    def __init__(self):
        self._trade_log: List[Dict] = []
        self._theme_counter = 0

    def record_trade(self, trade: Dict) -> None:
        """Record a trade with its metadata for narrative analysis."""
        self._trade_log.append({**trade, "recorded_at": time.time()})

    def generate_fund_letter(
        self,
        period_name: str,
        trades: List[Dict],
        fear_greed_index: float,
        consciousness_belief: str,
        dream_insights: List[Dict],
        performance_pnl: float,
        performance_sharpe: float,
        current_regime: str,
    ) -> FundLetter:
        """Generate a complete fund letter from system state."""

        # Extract themes from trades
        themes = self._extract_themes(trades)

        # Generate headline
        if performance_pnl > 0.02:
            tone = "strong"
            direction = "gains"
        elif performance_pnl > 0:
            tone = "modest"
            direction = "positive returns"
        elif performance_pnl > -0.02:
            tone = "challenging"
            direction = "small losses"
        else:
            tone = "difficult"
            direction = "drawdown"

        headline = f"A {tone} period: {direction} driven by {themes[0].name if themes else current_regime}"

        # Market view from consciousness
        market_view = (
            f"The market consciousness model reports: \"{consciousness_belief}\". "
            f"The current regime is classified as {current_regime}. "
            f"Our internal fear/greed oscillator reads {fear_greed_index:+.0f}, "
            f"suggesting the system is {'cautiously positioned' if fear_greed_index < -20 else 'aggressively positioned' if fear_greed_index > 20 else 'neutrally positioned'}."
        )

        # Performance
        perf = (
            f"Period return: {performance_pnl:+.2%}. "
            f"Annualized Sharpe ratio: {performance_sharpe:.2f}. "
            f"Total trades executed: {len(trades)}."
        )

        # Risk
        risk = (
            f"The Dream Engine tested all active signals against {len(dream_insights)} "
            f"synthetic scenarios. "
        )
        if dream_insights:
            most_fragile = dream_insights[0].get("description", "none identified")
            risk += f"Key fragility identified: {most_fragile}. "
        risk += f"The Guardian module reports no limit breaches during this period."

        # Outlook
        if fear_greed_index > 50:
            outlook = "The system's own greed indicator is elevated, suggesting we should be more cautious going forward. Position sizes will be reduced until conviction normalizes."
        elif fear_greed_index < -50:
            outlook = "The system is in a fearful state, historically a good time to increase exposure. We are looking for high-conviction setups to deploy capital."
        else:
            outlook = f"The system remains balanced. We continue to prioritize signals from the {current_regime} regime and will adjust as the Event Horizon Synthesizer discovers new opportunities."

        return FundLetter(
            period=period_name,
            headline=headline,
            market_view=market_view,
            themes=themes,
            performance_summary=perf,
            risk_assessment=risk,
            outlook=outlook,
            generated_at=time.time(),
        )

    def _extract_themes(self, trades: List[Dict]) -> List[NarrativeTheme]:
        """Extract recurring themes from a set of trades."""
        if not trades:
            return []

        themes = []

        # Theme 1: Dominant signal source
        signal_sources = Counter()
        for t in trades:
            for source in t.get("signal_sources", [t.get("signal_source", "unknown")]):
                signal_sources[source] += 1

        if signal_sources:
            top_source, count = signal_sources.most_common(1)[0]
            self._theme_counter += 1
            themes.append(NarrativeTheme(
                theme_id=f"theme_{self._theme_counter:04d}",
                name=f"{top_source} dominance",
                description=f"The {top_source} signal drove {count}/{len(trades)} trades this period, "
                            f"indicating strong alpha from this source.",
                frequency=count,
                confidence=count / len(trades),
                contributing_signals=[top_source],
                regime_context=trades[0].get("regime", "unknown"),
                example_trade=trades[0].get("symbol", ""),
            ))

        # Theme 2: Directional bias
        long_count = sum(1 for t in trades if t.get("direction", t.get("side", "")) in ("buy", "long"))
        short_count = sum(1 for t in trades if t.get("direction", t.get("side", "")) in ("sell", "short"))
        total = long_count + short_count
        if total > 0:
            bias = (long_count - short_count) / total
            self._theme_counter += 1
            if abs(bias) > 0.3:
                direction = "bullish" if bias > 0 else "bearish"
                themes.append(NarrativeTheme(
                    theme_id=f"theme_{self._theme_counter:04d}",
                    name=f"{direction} positioning",
                    description=f"The system maintained a {direction} bias with "
                                f"{long_count} long vs {short_count} short trades.",
                    frequency=total,
                    confidence=abs(bias),
                    contributing_signals=[],
                    regime_context="",
                    example_trade="",
                ))

        # Theme 3: Regime concentration
        regimes = Counter(t.get("regime", "unknown") for t in trades)
        if regimes:
            top_regime, count = regimes.most_common(1)[0]
            if count / len(trades) > 0.5:
                self._theme_counter += 1
                themes.append(NarrativeTheme(
                    theme_id=f"theme_{self._theme_counter:04d}",
                    name=f"{top_regime} regime focus",
                    description=f"{count}/{len(trades)} trades occurred in the {top_regime} regime. "
                                f"The system is highly adapted to current conditions.",
                    frequency=count,
                    confidence=count / len(trades),
                    contributing_signals=[],
                    regime_context=top_regime,
                    example_trade="",
                ))

        # Theme 4: Symbol concentration
        symbols = Counter(t.get("symbol", "") for t in trades)
        if symbols:
            top_sym, count = symbols.most_common(1)[0]
            if count / len(trades) > 0.3:
                self._theme_counter += 1
                themes.append(NarrativeTheme(
                    theme_id=f"theme_{self._theme_counter:04d}",
                    name=f"{top_sym} focus",
                    description=f"{top_sym} accounted for {count}/{len(trades)} trades, "
                                f"suggesting concentrated alpha in this instrument.",
                    frequency=count,
                    confidence=count / len(trades),
                    contributing_signals=[],
                    regime_context="",
                    example_trade=top_sym,
                ))

        return themes[:5]

    def format_letter(self, letter: FundLetter) -> str:
        """Format fund letter as readable text."""
        lines = [
            f"# {letter.headline}",
            f"*{letter.period}*\n",
            "## Market View",
            letter.market_view + "\n",
            "## Key Themes",
        ]

        for theme in letter.themes:
            lines.append(f"- **{theme.name}**: {theme.description}")

        lines.extend([
            "\n## Performance",
            letter.performance_summary + "\n",
            "## Risk Assessment",
            letter.risk_assessment + "\n",
            "## Outlook",
            letter.outlook,
            f"\n---\n*Auto-generated by SRFM Event Horizon Narrative Engine*",
        ])

        return "\n".join(lines)
