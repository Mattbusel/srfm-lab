"""
experiment_report.py
--------------------
Generate comprehensive A/B test reports with:
  * Executive summary (winner, magnitude, confidence)
  * Breakdown by symbol, hour, and market regime
  * ASCII visualisations (no matplotlib dependency)
  * Actionable recommendations

All output is plain text or Markdown — no external rendering needed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..experiments.ab_test import ABTest
from ..experiments.significance_tester import SignificanceTester, SignificanceResult
from ..versioning.strategy_version import StrategyVersion


# ---------------------------------------------------------------------------
# ASCII chart helpers
# ---------------------------------------------------------------------------

def _sparkline(values: list[float], width: int = 40) -> str:
    """Draw a horizontal ASCII sparkline."""
    if not values:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v or 1.0
    chars = [blocks[min(8, int((v - min_v) / rng * 8))] for v in values[-width:]]
    return "".join(chars)


def _bar_chart(
    labels: list[str],
    values: list[float],
    width: int = 30,
    prefix: str = "",
) -> str:
    """Horizontal bar chart. Values can be any sign."""
    if not values:
        return "(no data)"
    max_abs = max(abs(v) for v in values) or 1.0
    lines: list[str] = []
    max_label = max(len(lb) for lb in labels)
    for label, val in zip(labels, values):
        filled = int(abs(val) / max_abs * width)
        bar = ("+" if val >= 0 else "-") * filled
        lines.append(f"{prefix}{label:<{max_label}}  {bar:<{width}}  {val:+.4f}")
    return "\n".join(lines)


def _histogram(values: list[float], bins: int = 15, width: int = 40) -> str:
    """ASCII histogram."""
    if not values:
        return "(no data)"
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) or 1
    lines: list[str] = []
    for i, count in enumerate(counts):
        bar = "█" * int(count / max_count * width)
        lo, hi = edges[i], edges[i + 1]
        lines.append(f"  [{lo:+.4f},{hi:+.4f}) {bar} {count}")
    return "\n".join(lines)


def _equity_chart(equity_a: list[float], equity_b: list[float], width: int = 60) -> str:
    """Dual equity curve ASCII chart."""
    if not equity_a and not equity_b:
        return "(no data)"
    all_vals = equity_a + equity_b
    min_v, max_v = min(all_vals), max(all_vals)
    rng = max_v - min_v or 1.0
    # Downsample to width
    def sample(vals: list[float]) -> list[float]:
        if len(vals) <= width:
            return vals
        step = len(vals) / width
        return [vals[int(i * step)] for i in range(width)]
    sa = sample(equity_a) if equity_a else []
    sb = sample(equity_b) if equity_b else []
    height = 10
    lines: list[str] = []
    for row in range(height, -1, -1):
        threshold = min_v + (row / height) * rng
        line = ""
        n = max(len(sa), len(sb))
        for i in range(min(n, width)):
            va = sa[i] if i < len(sa) else None
            vb = sb[i] if i < len(sb) else None
            ca = va is not None and va >= threshold
            cb = vb is not None and vb >= threshold
            if ca and cb:
                line += "+"
            elif ca:
                line += "A"
            elif cb:
                line += "B"
            else:
                line += " "
        label = f"${threshold:>12,.0f} |"
        lines.append(f"{label}{line}")
    lines.append(" " * 14 + "-" * min(n, width))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExperimentReport
# ---------------------------------------------------------------------------

class ExperimentReport:
    """
    Generates a comprehensive A/B experiment report.

    Parameters
    ----------
    test        : completed ABTest
    version_a   : StrategyVersion for the control
    version_b   : StrategyVersion for the challenger
    sig_result  : pre-computed SignificanceResult (computed if not provided)
    """

    def __init__(
        self,
        test: ABTest,
        version_a: StrategyVersion | None = None,
        version_b: StrategyVersion | None = None,
        sig_result: SignificanceResult | None = None,
    ) -> None:
        self.test       = test
        self.version_a  = version_a
        self.version_b  = version_b

        if sig_result is None:
            tester = SignificanceTester()
            self.sig_result = tester.test(
                test.daily_pnl_a, test.daily_pnl_b,
                trades_a=len(test.trades_a),
                trades_b=len(test.trades_b),
            )
        else:
            self.sig_result = sig_result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        sections = [
            self._header(),
            self._executive_summary(),
            self._statistical_details(),
            self._equity_curves(),
            self._return_distributions(),
            self._symbol_breakdown(),
            self._hourly_breakdown(),
            self._regime_breakdown(),
            self._recommendation(),
        ]
        return "\n\n".join(s for s in sections if s)

    def to_markdown(self) -> str:
        text = self.to_text()
        # Wrap section headers as markdown H2
        lines = text.split("\n")
        md_lines: list[str] = []
        for line in lines:
            if line.startswith("=") and len(line) > 10:
                continue  # separator line
            if line.isupper() and len(line) > 5 and not line.startswith(" "):
                md_lines.append(f"## {line}")
            else:
                md_lines.append(line)
        return "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _header(self) -> str:
        t = self.test
        a_name = self.version_a.description[:40] if self.version_a else t.version_a_id[:8]
        b_name = self.version_b.description[:40] if self.version_b else t.version_b_id[:8]
        return (
            "=" * 72 + "\n"
            f"A/B EXPERIMENT REPORT\n"
            f"Test ID    : {t.test_id[:8]}\n"
            f"Control  A : {t.version_a_id[:8]}  — {a_name}\n"
            f"Challenger B: {t.version_b_id[:8]}  — {b_name}\n"
            f"Period     : {t.start_date}  →  {t.end_date or 'open'}\n"
            f"Allocation : A={t.allocation_a:.0%}  /  B={t.allocation_b:.0%}\n"
            f"Status     : {t.status.value}  ({t.stop_reason})\n"
            + "=" * 72
        )

    def _executive_summary(self) -> str:
        sig = self.sig_result
        na, nb = self.test.n_trades()
        wr_a, wr_b = self.test.win_rate()
        mp_a, mp_b = self.test.mean_pnl()
        sha, shb = self.test.sharpe()

        winner_str = {
            "A": "CONTROL (A) WINS",
            "B": "CHALLENGER (B) WINS",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }.get(sig.winner, sig.winner)

        lines = [
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"RESULT: {winner_str}",
            f"  Confidence     : {'SIGNIFICANT' if sig.is_significant() else 'NOT SIGNIFICANT'}",
            f"  Min p-value    : {sig.min_p:.4f}  (threshold: {sig.threshold:.4f})",
            f"  Effect size    : Cohen's d = {sig.cohens_d:+.3f}",
            "",
            f"  {'Metric':<20} {'Control A':>12} {'Challenger B':>14}",
            f"  {'-'*20} {'-'*12} {'-'*14}",
            f"  {'Trades':<20} {na:>12d} {nb:>14d}",
            f"  {'Win Rate':<20} {wr_a:>11.1%} {wr_b:>13.1%}",
            f"  {'Mean P&L/trade':<20} {mp_a:>+11.2f} {mp_b:>+13.2f}",
            f"  {'Sharpe (ann.)':<20} {sha:>+11.3f} {shb:>+13.3f}",
        ]

        pnl_a_total = sum(t.get("pnl", 0) for t in self.test.trades_a)
        pnl_b_total = sum(t.get("pnl", 0) for t in self.test.trades_b)
        lines += [
            f"  {'Total P&L':<20} {pnl_a_total:>+11,.0f} {pnl_b_total:>+13,.0f}",
        ]
        return "\n".join(lines)

    def _statistical_details(self) -> str:
        sig = self.sig_result
        return "\n".join([
            "STATISTICAL DETAILS",
            "-" * 40,
            f"  Welch t-test p-value     : {sig.p_ttest:.4f}",
            f"  Mann-Whitney U p-value   : {sig.p_mannwhitney:.4f}",
            f"  Bootstrap permutation p  : {sig.p_bootstrap:.4f}",
            f"  Cohen's d (effect size)  : {sig.cohens_d:+.4f}",
            f"  Sufficient data          : {sig.sufficient_data}",
            f"  Reason                   : {sig.reason}",
        ])

    def _equity_curves(self) -> str:
        # Build cumulative equity from daily P&L
        eq_a = self._cumulative_equity(self.test.daily_pnl_a)
        eq_b = self._cumulative_equity(self.test.daily_pnl_b)
        chart = _equity_chart(eq_a, eq_b, width=55)
        spark_a = _sparkline(self.test.daily_pnl_a)
        spark_b = _sparkline(self.test.daily_pnl_b)
        return "\n".join([
            "EQUITY CURVES  (A=control, B=challenger, +=both)",
            chart,
            f"  A daily P&L: {spark_a}",
            f"  B daily P&L: {spark_b}",
        ])

    def _return_distributions(self) -> str:
        lines = ["RETURN DISTRIBUTIONS"]
        if self.test.daily_pnl_a:
            lines.append("  Control A:")
            lines.append(_histogram(self.test.daily_pnl_a))
        if self.test.daily_pnl_b:
            lines.append("  Challenger B:")
            lines.append(_histogram(self.test.daily_pnl_b))
        return "\n".join(lines)

    def _symbol_breakdown(self) -> str:
        def pnl_by_sym(trades: list[dict]) -> dict[str, float]:
            d: dict[str, float] = defaultdict(float)
            for t in trades:
                d[t.get("symbol", "?")] += float(t.get("pnl", 0))
            return dict(d)

        sym_a = pnl_by_sym(self.test.trades_a)
        sym_b = pnl_by_sym(self.test.trades_b)
        all_syms = sorted(set(sym_a) | set(sym_b))

        lines = ["BY SYMBOL (total P&L)"]
        labels = all_syms
        vals_a = [sym_a.get(s, 0.0) for s in all_syms]
        vals_b = [sym_b.get(s, 0.0) for s in all_syms]

        lines.append("  Control A:")
        lines.append(_bar_chart(labels, vals_a, prefix="    "))
        lines.append("  Challenger B:")
        lines.append(_bar_chart(labels, vals_b, prefix="    "))
        return "\n".join(lines)

    def _hourly_breakdown(self) -> str:
        def pnl_by_hour(trades: list[dict]) -> dict[int, float]:
            d: dict[int, float] = defaultdict(float)
            for t in trades:
                ts = t.get("timestamp", "")
                try:
                    hour = int(ts[11:13])
                except (ValueError, IndexError):
                    hour = 0
                d[hour] += float(t.get("pnl", 0))
            return dict(d)

        h_a = pnl_by_hour(self.test.trades_a)
        h_b = pnl_by_hour(self.test.trades_b)
        hours = sorted(set(h_a) | set(h_b))

        if not hours:
            return ""
        labels = [f"{h:02d}h" for h in hours]
        vals_a = [h_a.get(h, 0.0) for h in hours]
        vals_b = [h_b.get(h, 0.0) for h in hours]

        lines = ["BY HOUR OF DAY (total P&L)"]
        lines.append("  Control A:")
        lines.append(_bar_chart(labels, vals_a, prefix="    "))
        lines.append("  Challenger B:")
        lines.append(_bar_chart(labels, vals_b, prefix="    "))
        return "\n".join(lines)

    def _regime_breakdown(self) -> str:
        def pnl_by_mode(trades: list[dict]) -> dict[str, float]:
            d: dict[str, float] = defaultdict(float)
            for t in trades:
                d[t.get("mode", "UNKNOWN")] += float(t.get("pnl", 0))
            return dict(d)

        m_a = pnl_by_mode(self.test.trades_a)
        m_b = pnl_by_mode(self.test.trades_b)
        modes = sorted(set(m_a) | set(m_b))

        if not modes:
            return ""
        labels = modes
        vals_a = [m_a.get(m, 0.0) for m in modes]
        vals_b = [m_b.get(m, 0.0) for m in modes]

        lines = ["BY MARKET REGIME / MODE"]
        lines.append("  Control A:")
        lines.append(_bar_chart(labels, vals_a, prefix="    "))
        lines.append("  Challenger B:")
        lines.append(_bar_chart(labels, vals_b, prefix="    "))
        return "\n".join(lines)

    def _recommendation(self) -> str:
        sig = self.sig_result
        na, nb = self.test.n_trades()

        if not sig.sufficient_data:
            rec = (
                "RECOMMENDATION: CONTINUE TESTING\n"
                f"  Need at least {50} trades per variant.\n"
                f"  Currently A={na}, B={nb}. Run longer."
            )
        elif sig.winner == "B" and sig.is_significant():
            rec = (
                "RECOMMENDATION: PROMOTE CHALLENGER B\n"
                "  Challenger B showed statistically and practically significant\n"
                f"  improvement (p={sig.min_p:.4f}, d={sig.cohens_d:.3f}).\n"
                "  Begin gradual rollout: 10% → 30% → 50% → 100% over 4 weeks."
            )
        elif sig.winner == "A" and sig.is_significant():
            rec = (
                "RECOMMENDATION: KEEP CONTROL A\n"
                f"  Control A outperformed challenger (p={sig.min_p:.4f}, d={sig.cohens_d:.3f}).\n"
                "  Challenger B should be archived. Generate new IAE ideas."
            )
        else:
            rec = (
                "RECOMMENDATION: INCONCLUSIVE — RUN LONGER OR REDESIGN\n"
                f"  p={sig.min_p:.4f}, d={sig.cohens_d:.3f}. No practical difference detected.\n"
                "  Options: (1) run more trades, (2) test a larger parameter shift,\n"
                "  (3) focus A/B on a specific regime or symbol subset."
            )
        return rec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cumulative_equity(daily_pnl: list[float], start: float = 1_000_000.0) -> list[float]:
        equity = [start]
        for pnl in daily_pnl:
            equity.append(equity[-1] + pnl)
        return equity
