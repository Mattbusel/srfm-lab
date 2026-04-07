"""
attribution_report.py -- P&L attribution across LARSA v18 signal layers.

Uses a Brinson-Hood-Beebower (BHB) style incremental attribution model.
Each layer is evaluated against the previous layer's baseline to isolate
the marginal contribution of each signal filter.

Signal stack (in order):
  1. BH mass only (baseline)
  2. + CF cross-filter
  3. + Hurst damper
  4. + QuatNav gate
  5. + ML signal gate
  6. + Event calendar filter
  7. + RL exit override

The waterfall shows cumulative P&L build-up through the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

from trade_journal import JournalEntry


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LayerAttribution:
    """Attribution result for a single signal layer."""
    layer_name: str
    trades_included: int    # trades that passed through up to this layer
    trades_excluded: int    # trades removed by this layer vs previous
    gross_pnl: float        # total P&L of included trades
    delta_pnl: float        # incremental P&L vs previous layer
    delta_pnl_pct: float    # delta_pnl as % of baseline
    win_rate: float         # win rate of included trades
    avg_pnl: float
    # Decomposed into allocation and selection effects (BHB)
    allocation_effect: float  # effect from changing trade count
    selection_effect: float   # effect from trade quality change


@dataclass
class AttributionResult:
    """Full attribution result across all signal layers."""
    layers: List[LayerAttribution]
    baseline_pnl: float       # BH mass only
    final_pnl: float          # after all filters
    total_attribution: float  # sum of all deltas (should == final - baseline)
    unexplained: float        # rounding / interaction effects
    summary: Dict[str, float] = field(default_factory=dict)  # layer_name -> delta

    def attribution_table(self) -> List[Dict[str, Any]]:
        """Return attribution as a list of dicts for tabular display."""
        rows = []
        cumulative = self.baseline_pnl
        for layer in self.layers:
            cumulative += layer.delta_pnl
            rows.append({
                "layer": layer.layer_name,
                "delta_pnl": layer.delta_pnl,
                "cumulative_pnl": cumulative,
                "delta_pct": layer.delta_pnl_pct,
                "win_rate": layer.win_rate,
                "trades": layer.trades_included,
                "excluded": layer.trades_excluded,
            })
        return rows


# ---------------------------------------------------------------------------
# Layer simulation functions
# ---------------------------------------------------------------------------

# Each simulate_fn below takes a list of JournalEntry and returns a filtered
# sub-list representing the trades that would have been taken at that layer.
# In production these would re-run the strategy logic; here we use the
# flag fields on JournalEntry to reconstruct the signal states.


def _simulate_bh_only(trades: List[JournalEntry]) -> List[JournalEntry]:
    """Baseline: only BH mass signal is active. Return all BH-active trades."""
    return [t for t in trades if t.bh_active]


def _simulate_bh_plus_cf(trades: List[JournalEntry]) -> List[JournalEntry]:
    """BH + CF cross-filter: remove trades where CF was unfavorable (cf_direction != side)."""
    def cf_pass(t: JournalEntry) -> bool:
        if not t.bh_active:
            return False
        if t.side == "long" and t.cf_direction < 0:
            return False
        if t.side == "short" and t.cf_direction > 0:
            return False
        return True
    return [t for t in trades if cf_pass(t)]


def _simulate_bh_cf_hurst(trades: List[JournalEntry]) -> List[JournalEntry]:
    """BH + CF + Hurst damper: exclude mean-reverting regime when trending expected."""
    prev = _simulate_bh_plus_cf(trades)
    return [
        t for t in prev
        if not (t.was_hurst_damped and t.hurst_regime == "mean-reverting")
    ]


def _simulate_bh_cf_hurst_nav(trades: List[JournalEntry]) -> List[JournalEntry]:
    """Add QuatNav gate: exclude NAV-gated trades."""
    prev = _simulate_bh_cf_hurst(trades)
    return [t for t in prev if not t.was_nav_gated]


def _simulate_bh_cf_hurst_nav_ml(trades: List[JournalEntry]) -> List[JournalEntry]:
    """Add ML signal gate."""
    prev = _simulate_bh_cf_hurst_nav(trades)
    return [t for t in prev if not t.was_ml_filtered]


def _simulate_bh_cf_hurst_nav_ml_event(
    trades: List[JournalEntry],
) -> List[JournalEntry]:
    """Add event calendar filter."""
    prev = _simulate_bh_cf_hurst_nav_ml(trades)
    return [t for t in prev if not t.was_event_calendar_filtered]


def _simulate_full(trades: List[JournalEntry]) -> List[JournalEntry]:
    """Full strategy: include RL exit override (all trades that completed)."""
    # After event filter, RL exit just changes the exit timing, not inclusion
    return _simulate_bh_cf_hurst_nav_ml_event(trades)


# Default layer stack
DEFAULT_SIMULATE_FNS: List[Tuple[str, Callable]] = [
    ("BH mass (baseline)", _simulate_bh_only),
    ("+ CF cross-filter", _simulate_bh_plus_cf),
    ("+ Hurst damper", _simulate_bh_cf_hurst),
    ("+ QuatNav gate", _simulate_bh_cf_hurst_nav),
    ("+ ML signal", _simulate_bh_cf_hurst_nav_ml),
    ("+ Event calendar", _simulate_bh_cf_hurst_nav_ml_event),
    ("+ RL exit", _simulate_full),
]


# ---------------------------------------------------------------------------
# Attribution calculator
# ---------------------------------------------------------------------------


def _layer_stats(
    trades: List[JournalEntry],
) -> Tuple[float, float, float]:
    """Return (gross_pnl, win_rate, avg_pnl) for a set of trades."""
    if not trades:
        return 0.0, 0.0, 0.0
    gross_pnl = sum(t.net_pnl for t in trades)
    win_rate = sum(1 for t in trades if t.net_pnl > 0) / len(trades)
    avg_pnl = gross_pnl / len(trades)
    return gross_pnl, win_rate, avg_pnl


def _bhb_decompose(
    prev_trades: List[JournalEntry],
    curr_trades: List[JournalEntry],
    all_trades: List[JournalEntry],
) -> Tuple[float, float]:
    """
    BHB-style decomposition of delta_pnl into allocation and selection effects.

    Allocation effect: change in number of trades * benchmark avg_pnl
    Selection effect: number of included trades * change in avg_pnl

    Parameters
    ----------
    prev_trades : trades from previous layer
    curr_trades : trades from current layer
    all_trades  : full trade universe for benchmark avg

    Returns
    -------
    (allocation_effect, selection_effect)
    """
    n_prev = len(prev_trades)
    n_curr = len(curr_trades)

    _, _, avg_prev = _layer_stats(prev_trades)
    _, _, avg_curr = _layer_stats(curr_trades)
    _, _, avg_bench = _layer_stats(all_trades)

    # Allocation = (w_curr - w_prev) * avg_bench
    # where weight = n / total
    n_total = max(len(all_trades), 1)
    w_prev = n_prev / n_total
    w_curr = n_curr / n_total
    allocation = (w_curr - w_prev) * avg_bench * n_total

    # Selection = w_curr * (avg_curr - avg_prev)
    selection = w_curr * (avg_curr - avg_prev) * n_total

    return allocation, selection


# ---------------------------------------------------------------------------
# AttributionReport
# ---------------------------------------------------------------------------


class AttributionReport:
    """
    BHB-style P&L attribution across LARSA v18 signal layers.

    Usage::

        report = AttributionReport(trades)
        result = report.compute_layer_attribution()
        report.generate_waterfall_chart()
    """

    def __init__(
        self,
        trades: List[JournalEntry],
        simulate_fns: Optional[List[Tuple[str, Callable]]] = None,
    ):
        self.trades = trades
        self.simulate_fns = simulate_fns or DEFAULT_SIMULATE_FNS

    def compute_layer_attribution(
        self,
        simulate_fns: Optional[List[Tuple[str, Callable]]] = None,
    ) -> AttributionResult:
        """
        Compute incremental P&L attribution across all signal layers.

        Each layer is simulated to get the set of trades that would have
        been taken, and the incremental P&L vs the previous layer is computed.

        Returns AttributionResult with per-layer breakdowns.
        """
        fns = simulate_fns or self.simulate_fns
        all_trades = self.trades
        n_total = len(all_trades)

        layer_attrs: List[LayerAttribution] = []
        prev_trades: List[JournalEntry] = []
        prev_pnl = 0.0

        baseline_pnl = 0.0

        for i, (name, fn) in enumerate(fns):
            curr_trades = fn(all_trades)
            curr_pnl, win_rate, avg_pnl = _layer_stats(curr_trades)
            excluded = len(prev_trades) - len(curr_trades) if i > 0 else (n_total - len(curr_trades))
            delta_pnl = curr_pnl - prev_pnl

            if i == 0:
                baseline_pnl = curr_pnl
                delta_pnl = 0.0
                excluded = n_total - len(curr_trades)

            delta_pct = (delta_pnl / abs(baseline_pnl)) if baseline_pnl != 0 else 0.0

            alloc_eff, sel_eff = 0.0, 0.0
            if i > 0 and prev_trades:
                alloc_eff, sel_eff = _bhb_decompose(prev_trades, curr_trades, all_trades)

            layer_attrs.append(
                LayerAttribution(
                    layer_name=name,
                    trades_included=len(curr_trades),
                    trades_excluded=max(0, excluded),
                    gross_pnl=curr_pnl,
                    delta_pnl=delta_pnl,
                    delta_pnl_pct=delta_pct,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    allocation_effect=alloc_eff,
                    selection_effect=sel_eff,
                )
            )

            prev_trades = curr_trades
            prev_pnl = curr_pnl

        final_pnl = layer_attrs[-1].gross_pnl if layer_attrs else 0.0
        total_attr = sum(la.delta_pnl for la in layer_attrs[1:]) if len(layer_attrs) > 1 else 0.0
        unexplained = (final_pnl - baseline_pnl) - total_attr

        summary = {la.layer_name: la.delta_pnl for la in layer_attrs}

        return AttributionResult(
            layers=layer_attrs,
            baseline_pnl=baseline_pnl,
            final_pnl=final_pnl,
            total_attribution=total_attr,
            unexplained=unexplained,
            summary=summary,
        )

    def generate_waterfall_chart(
        self,
        result: Optional[AttributionResult] = None,
        title: str = "LARSA v18 -- P&L Attribution Waterfall",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 7),
    ) -> None:
        """
        Generate a matplotlib waterfall chart of cumulative P&L attribution.
        Each bar represents the incremental contribution of one signal layer.
        Silently skips if matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not available -- skipping waterfall chart")
            return

        if result is None:
            result = self.compute_layer_attribution()

        layers = result.layers
        n = len(layers)
        labels = [la.layer_name for la in layers]
        deltas = [layers[0].gross_pnl] + [la.delta_pnl for la in layers[1:]]

        # Compute running bottom positions for waterfall bars
        running = 0.0
        bottoms = []
        heights = []
        for i, delta in enumerate(deltas):
            if i == 0:
                # Baseline bar starts at 0
                bottoms.append(0.0)
                heights.append(delta)
                running = delta
            else:
                if delta >= 0:
                    bottoms.append(running)
                    heights.append(delta)
                else:
                    bottoms.append(running + delta)
                    heights.append(abs(delta))
                running += delta

        colors = []
        for i, delta in enumerate(deltas):
            if i == 0:
                colors.append("#58a6ff")   # baseline -- blue
            elif delta > 0:
                colors.append("#3fb950")   # positive -- green
            else:
                colors.append("#f85149")   # negative -- red

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        bars = ax.bar(
            range(n),
            heights,
            bottom=bottoms,
            color=colors,
            edgecolor="#30363d",
            linewidth=0.8,
            width=0.6,
        )

        # Connector lines between bars
        for i in range(1, n):
            prev_top = bottoms[i - 1] + heights[i - 1]
            ax.plot(
                [i - 0.7, i - 0.3],
                [prev_top, prev_top],
                color="#8b949e",
                linewidth=0.8,
                linestyle="--",
            )

        # Value labels on bars
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            bar_top = bottoms[i] + heights[i]
            if delta >= 0:
                y_label = bar_top + abs(result.final_pnl) * 0.01
                va = "bottom"
            else:
                y_label = bottoms[i] - abs(result.final_pnl) * 0.01
                va = "top"
            prefix = "+" if delta > 0 else ""
            ax.text(
                i, y_label,
                f"{prefix}${delta:,.0f}",
                ha="center", va=va,
                color="#c9d1d9", fontsize=8, fontweight="bold",
            )

        # Cumulative total marker
        ax.axhline(
            y=result.final_pnl,
            color="#f0883e",
            linestyle=":",
            linewidth=1.5,
            label=f"Final P&L: ${result.final_pnl:,.0f}",
        )
        ax.axhline(
            y=result.baseline_pnl,
            color="#8b949e",
            linestyle=":",
            linewidth=1.0,
            label=f"Baseline P&L: ${result.baseline_pnl:,.0f}",
        )

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=25, ha="right", color="#c9d1d9", fontsize=9)
        ax.set_ylabel("Cumulative P&L ($)", color="#8b949e")
        ax.set_title(title, color="#c9d1d9", fontsize=13)
        ax.tick_params(axis="y", colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        # Legend
        pos_patch = mpatches.Patch(color="#3fb950", label="Positive contribution")
        neg_patch = mpatches.Patch(color="#f85149", label="Negative contribution")
        base_patch = mpatches.Patch(color="#58a6ff", label="Baseline (BH only)")
        ax.legend(handles=[base_patch, pos_patch, neg_patch], loc="upper left",
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="#0d1117")
        else:
            plt.show()
        plt.close()

    def to_markdown(self, result: Optional[AttributionResult] = None) -> str:
        """Render attribution result as Markdown."""
        if result is None:
            result = self.compute_layer_attribution()

        lines = ["# LARSA v18 -- P&L Attribution Report", ""]
        lines.append(f"- Baseline P&L (BH mass only): **${result.baseline_pnl:,.2f}**")
        lines.append(f"- Final P&L (all filters): **${result.final_pnl:,.2f}**")
        lines.append(f"- Total attribution: **${result.total_attribution:,.2f}**")
        lines.append(f"- Unexplained residual: ${result.unexplained:,.2f}")
        lines.append("")
        lines.append("## Layer Attribution")
        lines.append(
            "| Layer | Trades | Excluded | Delta P&L | Cumul P&L | Win Rate | "
            "Alloc Effect | Select Effect |"
        )
        lines.append(
            "|-------|--------|----------|-----------|-----------|----------|"
            "-------------|---------------|"
        )

        cumul = 0.0
        for la in result.layers:
            cumul += la.delta_pnl if la.layer_name != result.layers[0].layer_name else la.gross_pnl
            # Fix: first row cumul = gross_pnl
        cumul = 0.0
        for i, la in enumerate(result.layers):
            if i == 0:
                cumul = la.gross_pnl
                delta_str = f"${la.gross_pnl:,.2f}"
            else:
                cumul += la.delta_pnl
                sign = "+" if la.delta_pnl >= 0 else ""
                delta_str = f"{sign}${la.delta_pnl:,.2f}"
            lines.append(
                f"| {la.layer_name} | {la.trades_included} | {la.trades_excluded} "
                f"| {delta_str} | ${cumul:,.2f} | {la.win_rate:.1%} "
                f"| ${la.allocation_effect:,.2f} | ${la.selection_effect:,.2f} |"
            )

        lines.append("")
        lines.append("## BHB Decomposition")
        lines.append("| Layer | Allocation Effect | Selection Effect | Total |")
        lines.append("|-------|-------------------|------------------|-------|")
        for la in result.layers[1:]:
            total_effect = la.allocation_effect + la.selection_effect
            lines.append(
                f"| {la.layer_name} | ${la.allocation_effect:,.2f} "
                f"| ${la.selection_effect:,.2f} | ${total_effect:,.2f} |"
            )

        return "\n".join(lines)

    def compute_signal_correlations(self) -> Dict[str, float]:
        """
        Compute pairwise correlation between signal values and trade P&L.
        Returns dict of signal_name -> Pearson correlation with net_pnl.
        """
        trades = self.trades
        if not trades:
            return {}

        pnl = [t.net_pnl for t in trades]
        signals = {
            "bh_mass_15m": [t.bh_mass_15m for t in trades],
            "bh_mass_1h": [t.bh_mass_1h for t in trades],
            "bh_mass_4h": [t.bh_mass_4h for t in trades],
            "cf_alignment": [t.cf_alignment for t in trades],
            "hurst_h": [t.hurst_h for t in trades],
            "nav_omega": [t.nav_omega for t in trades],
            "nav_geodesic": [t.nav_geodesic for t in trades],
            "ml_signal": [t.ml_signal for t in trades],
            "ml_confidence": [t.ml_confidence for t in trades],
            "granger_btc_corr": [t.granger_btc_corr for t in trades],
        }

        corrs: Dict[str, float] = {}
        for name, vals in signals.items():
            corrs[name] = _pearson_corr(vals, pnl)
        return corrs

    def compute_signal_information_coefficient(
        self, signal_name: str
    ) -> float:
        """
        Information Coefficient (IC) = rank correlation between signal and forward P&L.
        A well-known metric from quant factor research.
        """
        trades = self.trades
        if not trades or not hasattr(trades[0], signal_name):
            return 0.0

        signal_vals = [getattr(t, signal_name) for t in trades]
        pnl_vals = [t.net_pnl for t in trades]
        return _spearman_corr(signal_vals, pnl_vals)

    def compute_all_ics(self) -> Dict[str, float]:
        """Compute IC for all numeric signal fields."""
        signal_fields = [
            "bh_mass_15m", "bh_mass_1h", "bh_mass_4h",
            "cf_alignment", "hurst_h", "nav_omega", "nav_geodesic",
            "ml_signal", "ml_confidence", "granger_btc_corr", "granger_eth_corr",
        ]
        return {f: self.compute_signal_information_coefficient(f) for f in signal_fields}

    def top_contributing_trades(
        self,
        n: int = 10,
        layer_name: Optional[str] = None,
    ) -> List[JournalEntry]:
        """
        Return the top n contributing trades by abs(net_pnl),
        optionally filtered to a specific layer.
        """
        if layer_name:
            result = self.compute_layer_attribution()
            # Find which simulate_fn corresponds to the layer
            matching_fn = None
            for name, fn in self.simulate_fns:
                if name == layer_name:
                    matching_fn = fn
                    break
            trades = matching_fn(self.trades) if matching_fn else self.trades
        else:
            trades = self.trades

        return sorted(trades, key=lambda t: abs(t.net_pnl), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _pearson_corr(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    denom = dx * dy
    return num / denom if denom > 0 else 0.0


def _rank(values: List[float]) -> List[float]:
    """Return rank array (1-based) with average ranks for ties."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation coefficient."""
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    return _pearson_corr(rx, ry)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _rng
    import tempfile
    from trade_journal import TradeJournal, _make_sample_entry

    rng = _rng.Random(42)
    symbols = ["BTC", "ETH", "SOL", "AAPL", "NVDA"]

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_file = f.name

    with TradeJournal(db_file) as journal:
        for i in range(100):
            sym = rng.choice(symbols)
            pnl = rng.gauss(60, 350)
            e = _make_sample_entry(sym, pnl, rng.randint(3, 30))
            e.bh_active = rng.random() > 0.25
            e.cf_direction = rng.choice([-1, 0, 1])
            e.hurst_regime = rng.choice(["trending", "neutral", "mean-reverting"])
            e.was_cf_filtered = rng.random() > 0.7
            e.was_hurst_damped = rng.random() > 0.8
            e.was_nav_gated = rng.random() > 0.85
            e.was_ml_filtered = rng.random() > 0.9
            e.was_event_calendar_filtered = rng.random() > 0.92
            e.was_rl_exit = rng.random() > 0.85
            journal.add_entry(e)

        trades = journal.get_all()

    rpt = AttributionReport(trades)
    result = rpt.compute_layer_attribution()
    print(f"Baseline P&L: ${result.baseline_pnl:,.2f}")
    print(f"Final P&L:    ${result.final_pnl:,.2f}")
    print()
    for row in result.attribution_table():
        sign = "+" if row["delta_pnl"] >= 0 else ""
        print(
            f"  {row['layer']:<30} "
            f"delta={sign}${row['delta_pnl']:>9,.2f}  "
            f"cumul=${row['cumulative_pnl']:>9,.2f}  "
            f"WR={row['win_rate']:.1%}"
        )

    print()
    ics = rpt.compute_all_ics()
    print("Signal ICs:")
    for sig, ic in sorted(ics.items(), key=lambda x: -abs(x[1])):
        bar = "#" * int(abs(ic) * 40)
        print(f"  {sig:<25} IC={ic:+.4f}  {bar}")
