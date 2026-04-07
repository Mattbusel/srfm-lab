"""
performance_report.py -- Comprehensive performance reporting for LARSA v18.
Loads trades from SQLite, computes full metrics, exports HTML and CSV.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from trade_journal import TradeJournal, JournalEntry, DB_PATH_DEFAULT

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _annualise(returns: List[float], bars_per_year: int = 8760) -> float:
    """
    Annualise a list of per-trade returns (as fractions).
    Uses geometric compounding.
    """
    if not returns:
        return 0.0
    product = 1.0
    for r in returns:
        product *= 1.0 + r
    n = len(returns)
    if n == 0 or product <= 0:
        return 0.0
    return product ** (bars_per_year / n) - 1.0


def _sharpe(returns: List[float], rf_per_period: float = 0.0) -> float:
    """Compute Sharpe ratio from a list of per-trade returns."""
    if len(returns) < 2:
        return 0.0
    excess = [r - rf_per_period for r in returns]
    mean_e = sum(excess) / len(excess)
    var_e = sum((r - mean_e) ** 2 for r in excess) / (len(excess) - 1)
    std_e = math.sqrt(var_e) if var_e > 0 else 0.0
    return mean_e / std_e if std_e > 0 else 0.0


def _sortino(returns: List[float], rf_per_period: float = 0.0) -> float:
    """Compute Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    excess = [r - rf_per_period for r in returns]
    mean_e = sum(excess) / len(excess)
    downside = [r for r in excess if r < 0]
    if not downside:
        return float("inf")
    down_var = sum(r ** 2 for r in downside) / len(downside)
    down_std = math.sqrt(down_var)
    return mean_e / down_std if down_std > 0 else 0.0


def _max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """
    Return (max_drawdown_fraction, peak_idx, trough_idx).
    max_drawdown is expressed as a positive fraction (e.g. 0.20 = 20% DD).
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    best_peak_idx = 0
    best_trough_idx = 0
    for i, val in enumerate(equity_curve):
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            best_peak_idx = peak_idx
            best_trough_idx = i
    return max_dd, best_peak_idx, best_trough_idx


def _calmar(total_return: float, max_dd: float, years: float = 1.0) -> float:
    if max_dd == 0.0 or years == 0.0:
        return 0.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    return cagr / max_dd


def _build_equity_curve(entries: List[JournalEntry], start_nav: float = 100_000.0) -> List[float]:
    nav = start_nav
    curve = [nav]
    for e in sorted(entries, key=lambda x: x.exit_ts):
        nav += e.net_pnl
        curve.append(nav)
    return curve


def _monthly_key(ts: str) -> str:
    """Extract YYYY-MM from an ISO timestamp."""
    try:
        return ts[:7]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# DrawdownPeriod
# ---------------------------------------------------------------------------

@dataclass
class DrawdownPeriod:
    start_idx: int
    end_idx: int
    peak_idx: int
    trough_idx: int
    depth: float          # fraction e.g. 0.15 = 15%
    duration_bars: int
    recovery_bars: Optional[int]   # None if not yet recovered


def _find_all_drawdowns(equity_curve: List[float]) -> List[DrawdownPeriod]:
    """Identify all distinct drawdown periods in an equity curve."""
    periods: List[DrawdownPeriod] = []
    n = len(equity_curve)
    if n < 2:
        return periods

    in_dd = False
    peak_val = equity_curve[0]
    peak_idx = 0
    trough_val = equity_curve[0]
    trough_idx = 0
    dd_start = 0

    for i in range(1, n):
        val = equity_curve[i]
        if not in_dd:
            if val > peak_val:
                peak_val = val
                peak_idx = i
            elif val < peak_val:
                in_dd = True
                dd_start = peak_idx
                trough_val = val
                trough_idx = i
        else:
            if val < trough_val:
                trough_val = val
                trough_idx = i
            if val >= peak_val:
                # Recovered
                depth = (peak_val - trough_val) / peak_val if peak_val > 0 else 0.0
                periods.append(
                    DrawdownPeriod(
                        start_idx=dd_start,
                        end_idx=i,
                        peak_idx=dd_start,
                        trough_idx=trough_idx,
                        depth=depth,
                        duration_bars=i - dd_start,
                        recovery_bars=i - trough_idx,
                    )
                )
                in_dd = False
                peak_val = val
                peak_idx = i
                trough_val = val
                trough_idx = i

    # Still in drawdown at end
    if in_dd and trough_val < peak_val:
        depth = (peak_val - trough_val) / peak_val if peak_val > 0 else 0.0
        periods.append(
            DrawdownPeriod(
                start_idx=dd_start,
                end_idx=n - 1,
                peak_idx=dd_start,
                trough_idx=trough_idx,
                depth=depth,
                duration_bars=n - 1 - dd_start,
                recovery_bars=None,
            )
        )

    return periods


# ---------------------------------------------------------------------------
# PerformanceReport
# ---------------------------------------------------------------------------

class PerformanceReport:
    """
    Comprehensive performance reporter for LARSA v18.

    Loads all trades from the SQLite journal and computes a full suite
    of risk/return metrics, monthly breakdowns, symbol attribution, and
    drawdown analysis.
    """

    def __init__(
        self,
        db_path: Path | str = DB_PATH_DEFAULT,
        start_nav: float = 100_000.0,
        rf_annual: float = 0.05,
    ):
        self.db_path = Path(db_path)
        self.start_nav = start_nav
        self.rf_annual = rf_annual
        self._journal = TradeJournal(db_path)

    @property
    def entries(self) -> List[JournalEntry]:
        return self._journal.get_all()

    # -- Core computations --

    def _rf_per_trade(self, avg_hold_hours: float = 24.0) -> float:
        """Convert annual RF to per-trade RF given average hold duration."""
        hold_fraction = avg_hold_hours / (365 * 24)
        return (1.0 + self.rf_annual) ** hold_fraction - 1.0

    def _trade_returns(self, entries: List[JournalEntry]) -> List[float]:
        """Return list of per-trade returns as fractions of portfolio NAV at entry."""
        returns = []
        for e in entries:
            base = e.portfolio_nav_at_entry if e.portfolio_nav_at_entry > 0 else self.start_nav
            returns.append(e.net_pnl / base)
        return returns

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate the full performance report dictionary.

        Keys:
        - overview
        - monthly_breakdown
        - symbol_breakdown
        - signal_attribution
        - regime_analysis
        - drawdown_analysis
        - trade_statistics
        """
        entries = self.entries
        if not entries:
            return {"error": "No trades found in journal."}

        sorted_entries = sorted(entries, key=lambda e: e.exit_ts)
        equity_curve = _build_equity_curve(sorted_entries, self.start_nav)
        returns = self._trade_returns(sorted_entries)
        avg_hold_hours = (
            sum(e.hold_duration_hours() for e in sorted_entries) / len(sorted_entries)
        )
        rf_pt = self._rf_per_trade(avg_hold_hours)

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        max_dd, _, _ = _max_drawdown(equity_curve)

        # Estimate years from first/last trade
        try:
            t0 = datetime.fromisoformat(sorted_entries[0].entry_ts)
            t1 = datetime.fromisoformat(sorted_entries[-1].exit_ts)
            years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1 / 12)
        except Exception:
            years = 1.0

        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
        sharpe = _sharpe(returns, rf_pt)
        sortino = _sortino(returns, rf_pt)
        calmar = _calmar(total_return, max_dd, years)

        overview = {
            "total_trades": len(sorted_entries),
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "cagr": cagr,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "start_nav": equity_curve[0],
            "end_nav": equity_curve[-1],
            "years": years,
        }

        monthly_breakdown = self._monthly_breakdown(sorted_entries)
        symbol_breakdown = self._symbol_breakdown(sorted_entries)
        signal_attribution = self._signal_attribution(sorted_entries)
        regime_analysis = self._regime_analysis(sorted_entries)
        drawdown_analysis = self._drawdown_analysis(equity_curve)
        trade_stats = self._trade_statistics(sorted_entries, returns)

        return {
            "overview": overview,
            "monthly_breakdown": monthly_breakdown,
            "symbol_breakdown": symbol_breakdown,
            "signal_attribution": signal_attribution,
            "regime_analysis": regime_analysis,
            "drawdown_analysis": drawdown_analysis,
            "trade_statistics": trade_stats,
        }

    def _monthly_breakdown(
        self, entries: List[JournalEntry]
    ) -> Dict[str, Dict[str, Any]]:
        from collections import defaultdict
        monthly: Dict[str, List[JournalEntry]] = defaultdict(list)
        for e in entries:
            key = _monthly_key(e.exit_ts)
            monthly[key].append(e)

        result: Dict[str, Dict[str, Any]] = {}
        for month, sub in sorted(monthly.items()):
            wins = sum(1 for e in sub if e.net_pnl > 0)
            result[month] = {
                "trades": len(sub),
                "net_pnl": sum(e.net_pnl for e in sub),
                "win_rate": wins / len(sub),
                "gross_pnl": sum(e.pnl for e in sub),
                "avg_pnl": sum(e.net_pnl for e in sub) / len(sub),
            }
        return result

    def _symbol_breakdown(
        self, entries: List[JournalEntry]
    ) -> Dict[str, Dict[str, Any]]:
        from collections import defaultdict
        sym_map: Dict[str, List[JournalEntry]] = defaultdict(list)
        for e in entries:
            sym_map[e.symbol].append(e)

        result: Dict[str, Dict[str, Any]] = {}
        for sym, sub in sorted(sym_map.items()):
            wins = sum(1 for e in sub if e.net_pnl > 0)
            result[sym] = {
                "trades": len(sub),
                "net_pnl": sum(e.net_pnl for e in sub),
                "win_rate": wins / len(sub) if sub else 0.0,
                "avg_pnl": sum(e.net_pnl for e in sub) / len(sub),
                "pct_of_total_pnl": 0.0,  # filled in below
            }

        total_pnl = sum(v["net_pnl"] for v in result.values())
        for sym in result:
            result[sym]["pct_of_total_pnl"] = (
                result[sym]["net_pnl"] / total_pnl if total_pnl != 0 else 0.0
            )
        return result

    def _signal_attribution(
        self, entries: List[JournalEntry]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Attribute P&L to each signal layer using incremental bucketing.
        Each trade is assigned to its deepest active signal.
        """
        layers = {
            "bh_only": [],
            "cf_filtered": [],
            "hurst_damped": [],
            "nav_gated": [],
            "ml_filtered": [],
            "event_filtered": [],
            "rl_exit": [],
        }
        for e in entries:
            if e.was_rl_exit:
                layers["rl_exit"].append(e)
            elif e.was_event_calendar_filtered:
                layers["event_filtered"].append(e)
            elif e.was_ml_filtered:
                layers["ml_filtered"].append(e)
            elif e.was_nav_gated:
                layers["nav_gated"].append(e)
            elif e.was_hurst_damped:
                layers["hurst_damped"].append(e)
            elif e.was_cf_filtered:
                layers["cf_filtered"].append(e)
            else:
                layers["bh_only"].append(e)

        result: Dict[str, Dict[str, Any]] = {}
        for layer, sub in layers.items():
            if not sub:
                result[layer] = {"trades": 0, "net_pnl": 0.0, "win_rate": 0.0}
                continue
            wins = sum(1 for e in sub if e.net_pnl > 0)
            result[layer] = {
                "trades": len(sub),
                "net_pnl": sum(e.net_pnl for e in sub),
                "win_rate": wins / len(sub),
                "avg_pnl": sum(e.net_pnl for e in sub) / len(sub),
            }
        return result

    def _regime_analysis(
        self, entries: List[JournalEntry]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performance by (bh_active, hurst_regime, vol_regime) combinations.
        """
        from collections import defaultdict
        regime_map: Dict[str, List[JournalEntry]] = defaultdict(list)
        for e in entries:
            key = (
                f"bh={'ON' if e.bh_active else 'OFF'} | "
                f"hurst={e.hurst_regime} | "
                f"vol={e.vol_regime}"
            )
            regime_map[key].append(e)

        result: Dict[str, Dict[str, Any]] = {}
        for key, sub in sorted(regime_map.items()):
            wins = sum(1 for e in sub if e.net_pnl > 0)
            result[key] = {
                "trades": len(sub),
                "net_pnl": sum(e.net_pnl for e in sub),
                "win_rate": wins / len(sub),
                "avg_pnl": sum(e.net_pnl for e in sub) / len(sub),
                "avg_mfe": sum(e.mfe_pct for e in sub) / len(sub),
                "avg_mae": sum(e.mae_pct for e in sub) / len(sub),
            }
        return result

    def _drawdown_analysis(
        self, equity_curve: List[float]
    ) -> Dict[str, Any]:
        periods = _find_all_drawdowns(equity_curve)
        max_dd = max((p.depth for p in periods), default=0.0)
        avg_duration = (
            sum(p.duration_bars for p in periods) / len(periods) if periods else 0.0
        )
        avg_recovery = sum(
            p.recovery_bars for p in periods if p.recovery_bars is not None
        )
        recovered = [p for p in periods if p.recovery_bars is not None]
        avg_recovery_bars = avg_recovery / len(recovered) if recovered else None

        return {
            "num_drawdowns": len(periods),
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "avg_duration_bars": avg_duration,
            "avg_recovery_bars": avg_recovery_bars,
            "periods": [
                {
                    "start_idx": p.start_idx,
                    "trough_idx": p.trough_idx,
                    "end_idx": p.end_idx,
                    "depth": p.depth,
                    "depth_pct": p.depth * 100,
                    "duration_bars": p.duration_bars,
                    "recovery_bars": p.recovery_bars,
                }
                for p in periods
            ],
        }

    def _trade_statistics(
        self, entries: List[JournalEntry], returns: List[float]
    ) -> Dict[str, Any]:
        winners = [e for e in entries if e.net_pnl > 0]
        losers = [e for e in entries if e.net_pnl <= 0]
        total = len(entries)

        avg_win = sum(e.net_pnl for e in winners) / len(winners) if winners else 0.0
        avg_loss = (
            abs(sum(e.net_pnl for e in losers) / len(losers)) if losers else 0.0
        )
        win_rate = len(winners) / total
        loss_rate = 1.0 - win_rate
        expectancy = avg_win * win_rate - avg_loss * loss_rate

        gross_profit = sum(e.net_pnl for e in winners)
        gross_loss = abs(sum(e.net_pnl for e in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        hold_hours = [e.hold_duration_hours() for e in entries]
        avg_hold_h = sum(hold_hours) / total
        avg_hold_b = sum(e.hold_bars for e in entries) / total

        mfe_list = [e.mfe_pct for e in entries]
        mae_list = [e.mae_pct for e in entries]

        # Win/loss streak analysis
        streaks = _compute_streaks([e.net_pnl > 0 for e in entries])

        return {
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else float("inf"),
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "avg_hold_hours": avg_hold_h,
            "avg_hold_bars": avg_hold_b,
            "avg_mfe_pct": sum(mfe_list) / total,
            "avg_mae_pct": sum(mae_list) / total,
            "max_mfe_pct": max(mfe_list) if mfe_list else 0.0,
            "max_mae_pct": max(mae_list) if mae_list else 0.0,
            "max_win_streak": streaks["max_win"],
            "max_loss_streak": streaks["max_loss"],
            "current_streak": streaks["current"],
        }

    # -- Export --

    def export_html(self, output_path: Path | str) -> None:
        """
        Export full report as a standalone HTML file with Plotly charts.
        """
        output_path = Path(output_path)
        report = self.generate_full_report()
        if "error" in report:
            raise RuntimeError(report["error"])

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            has_plotly = True
        except ImportError:
            has_plotly = False

        entries = sorted(self.entries, key=lambda e: e.exit_ts)
        equity_curve = _build_equity_curve(entries, self.start_nav)

        parts: List[str] = []
        parts.append(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>LARSA v18 Performance Report</title>"
            "<style>"
            "body { font-family: Arial, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }"
            "h1, h2, h3 { color: #58a6ff; }"
            "table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }"
            "th, td { border: 1px solid #30363d; padding: 8px 12px; }"
            "th { background: #161b22; }"
            ".pos { color: #3fb950; } .neg { color: #f85149; }"
            ".metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }"
            ".metric-box { background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; }"
            ".metric-label { font-size: 0.8em; color: #8b949e; }"
            ".metric-value { font-size: 1.4em; font-weight: bold; }"
            "</style></head><body>"
        )
        parts.append("<h1>LARSA v18 -- Performance Report</h1>")

        ov = report["overview"]
        sign_cls = lambda v: "pos" if v >= 0 else "neg"
        parts.append("<div class='metric-grid'>")
        for label, val, fmt in [
            ("Total Return", ov["total_return_pct"], "{:.2f}%"),
            ("CAGR", ov["cagr_pct"], "{:.2f}%"),
            ("Sharpe", ov["sharpe"], "{:.3f}"),
            ("Sortino", ov["sortino"], "{:.3f}"),
            ("Calmar", ov["calmar"], "{:.3f}"),
            ("Max Drawdown", -ov["max_drawdown_pct"], "{:.2f}%"),
            ("Total Trades", ov["total_trades"], "{:.0f}"),
            ("End NAV", ov["end_nav"], "${:,.0f}"),
        ]:
            cls = sign_cls(val)
            parts.append(
                f"<div class='metric-box'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value {cls}'>{fmt.format(val)}</div>"
                f"</div>"
            )
        parts.append("</div>")

        # Equity curve chart
        if has_plotly:
            import plotly.graph_objects as go
            import plotly.io as pio
            trade_nums = list(range(len(equity_curve)))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trade_nums,
                    y=equity_curve,
                    mode="lines",
                    name="Equity Curve",
                    line=dict(color="#58a6ff", width=2),
                )
            )
            # Shade drawdown regions
            dd_info = report["drawdown_analysis"]["periods"]
            for dd in dd_info:
                if dd["depth"] > 0.01:
                    fig.add_vrect(
                        x0=dd["start_idx"],
                        x1=dd["end_idx"],
                        fillcolor="rgba(248,81,73,0.15)",
                        line_width=0,
                    )
            fig.update_layout(
                title="Equity Curve",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=400,
                xaxis_title="Trade #",
                yaxis_title="NAV ($)",
            )
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

            # Monthly P&L bar chart
            monthly = report["monthly_breakdown"]
            months = sorted(monthly.keys())
            monthly_pnl = [monthly[m]["net_pnl"] for m in months]
            monthly_colors = ["#3fb950" if v >= 0 else "#f85149" for v in monthly_pnl]
            fig2 = go.Figure(
                go.Bar(x=months, y=monthly_pnl, marker_color=monthly_colors)
            )
            fig2.update_layout(
                title="Monthly P&L",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=300,
            )
            parts.append(pio.to_html(fig2, full_html=False, include_plotlyjs=False))

            # Symbol P&L pie
            sym_data = report["symbol_breakdown"]
            syms = list(sym_data.keys())
            sym_pnls = [sym_data[s]["net_pnl"] for s in syms]
            fig3 = go.Figure(
                go.Pie(labels=syms, values=[abs(v) for v in sym_pnls], hole=0.4)
            )
            fig3.update_layout(
                title="Symbol P&L Distribution",
                paper_bgcolor="#0d1117",
                font_color="#c9d1d9",
                height=350,
            )
            parts.append(pio.to_html(fig3, full_html=False, include_plotlyjs=False))

        # Monthly breakdown table
        parts.append("<h2>Monthly Breakdown</h2>")
        parts.append(
            "<table><tr><th>Month</th><th>Trades</th>"
            "<th>Net P&L</th><th>Win Rate</th></tr>"
        )
        for month, info in sorted(report["monthly_breakdown"].items()):
            cls = sign_cls(info["net_pnl"])
            parts.append(
                f"<tr><td>{month}</td><td>{info['trades']}</td>"
                f"<td class='{cls}'>${info['net_pnl']:,.2f}</td>"
                f"<td>{info['win_rate']:.1%}</td></tr>"
            )
        parts.append("</table>")

        # Symbol breakdown table
        parts.append("<h2>Symbol Breakdown</h2>")
        parts.append(
            "<table><tr><th>Symbol</th><th>Trades</th>"
            "<th>Net P&L</th><th>Win Rate</th><th>% of Total</th></tr>"
        )
        for sym, info in sorted(
            report["symbol_breakdown"].items(),
            key=lambda x: -x[1]["net_pnl"],
        ):
            cls = sign_cls(info["net_pnl"])
            parts.append(
                f"<tr><td>{sym}</td><td>{info['trades']}</td>"
                f"<td class='{cls}'>${info['net_pnl']:,.2f}</td>"
                f"<td>{info['win_rate']:.1%}</td>"
                f"<td>{info['pct_of_total_pnl']:.1%}</td></tr>"
            )
        parts.append("</table>")

        # Trade statistics
        ts = report["trade_statistics"]
        parts.append("<h2>Trade Statistics</h2>")
        parts.append("<table>")
        for k, v in ts.items():
            if isinstance(v, float):
                disp = f"{v:.4f}"
            else:
                disp = str(v)
            parts.append(f"<tr><th>{k}</th><td>{disp}</td></tr>")
        parts.append("</table>")

        # Drawdown table
        dda = report["drawdown_analysis"]
        parts.append(f"<h2>Drawdown Analysis ({dda['num_drawdowns']} periods)</h2>")
        parts.append(
            "<table><tr><th>#</th><th>Start</th><th>Trough</th>"
            "<th>Depth %</th><th>Duration</th><th>Recovery</th></tr>"
        )
        for i, dd in enumerate(dda["periods"][:20]):
            parts.append(
                f"<tr><td>{i+1}</td><td>{dd['start_idx']}</td>"
                f"<td>{dd['trough_idx']}</td>"
                f"<td class='neg'>{dd['depth_pct']:.2f}%</td>"
                f"<td>{dd['duration_bars']} bars</td>"
                f"<td>{dd['recovery_bars'] or 'ongoing'}</td></tr>"
            )
        parts.append("</table>")

        parts.append("</body></html>")
        output_path.write_text("\n".join(parts), encoding="utf-8")

    def export_csv(self, output_path: Path | str) -> None:
        """Export all trades as a CSV file."""
        output_path = Path(output_path)
        entries = self.entries
        if not entries:
            return
        fieldnames = list(entries[0].to_dict().keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(e.to_dict())

    def close(self) -> None:
        self._journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# BenchmarkComparison
# ---------------------------------------------------------------------------

class BenchmarkComparison:
    """
    Compare strategy returns against benchmarks:
    - BTC buy-and-hold
    - 60/40 equity/bond portfolio proxy
    """

    @staticmethod
    def compute_information_ratio(
        strategy_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        """
        Information Ratio = mean(active return) / tracking_error.
        Active return = strategy_return - benchmark_return (per period).
        """
        if len(strategy_returns) != len(benchmark_returns):
            n = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:n]
            benchmark_returns = benchmark_returns[:n]
        if len(strategy_returns) < 2:
            return 0.0
        active = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        mean_a = sum(active) / len(active)
        var_a = sum((r - mean_a) ** 2 for r in active) / (len(active) - 1)
        te = math.sqrt(var_a) if var_a > 0 else 0.0
        return mean_a / te if te > 0 else 0.0

    @staticmethod
    def compute_beta(
        strategy_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        """
        Beta = Cov(strategy, benchmark) / Var(benchmark).
        """
        if len(strategy_returns) != len(benchmark_returns):
            n = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:n]
            benchmark_returns = benchmark_returns[:n]
        if len(strategy_returns) < 2:
            return 0.0

        n = len(strategy_returns)
        mean_s = sum(strategy_returns) / n
        mean_b = sum(benchmark_returns) / n
        cov = sum(
            (s - mean_s) * (b - mean_b)
            for s, b in zip(strategy_returns, benchmark_returns)
        ) / (n - 1)
        var_b = sum((b - mean_b) ** 2 for b in benchmark_returns) / (n - 1)
        return cov / var_b if var_b > 0 else 0.0

    @staticmethod
    def compute_alpha(
        strategy_returns: List[float],
        benchmark_returns: List[float],
        rf: float = 0.0,
    ) -> float:
        """
        Jensen's Alpha = mean(strategy) - [rf + beta * (mean(benchmark) - rf)].
        All returns are per-period.
        """
        if not strategy_returns or not benchmark_returns:
            return 0.0
        beta = BenchmarkComparison.compute_beta(strategy_returns, benchmark_returns)
        n = min(len(strategy_returns), len(benchmark_returns))
        mean_s = sum(strategy_returns[:n]) / n
        mean_b = sum(benchmark_returns[:n]) / n
        return mean_s - (rf + beta * (mean_b - rf))

    @staticmethod
    def btc_buyhold_returns(
        entries: List[JournalEntry],
        start_price: float = 40_000.0,
        end_price: float = 65_000.0,
    ) -> List[float]:
        """
        Simulate BTC buy-and-hold returns at the same time points as trades.
        Uses linear interpolation between start and end price.
        """
        n = len(entries)
        if n == 0:
            return []
        returns = []
        for i in range(n):
            frac = i / max(n - 1, 1)
            p0 = start_price + frac * (end_price - start_price)
            frac1 = (i + 1) / max(n - 1, 1)
            p1 = start_price + frac1 * (end_price - start_price)
            returns.append((p1 - p0) / p0 if p0 > 0 else 0.0)
        return returns

    @staticmethod
    def sixty_forty_returns(
        n_periods: int,
        equity_monthly_return: float = 0.008,
        bond_monthly_return: float = 0.003,
    ) -> List[float]:
        """
        Generate a synthetic 60/40 return series of length n_periods.
        """
        import random
        rng = random.Random(42)
        returns = []
        for _ in range(n_periods):
            eq = equity_monthly_return + rng.gauss(0, 0.04)
            bond = bond_monthly_return + rng.gauss(0, 0.01)
            returns.append(0.60 * eq + 0.40 * bond)
        return returns

    def full_comparison(
        self,
        entries: List[JournalEntry],
        start_nav: float = 100_000.0,
        btc_start: float = 40_000.0,
        btc_end: float = 65_000.0,
    ) -> Dict[str, Any]:
        """
        Return a comparison dict with strategy, BTC BH, and 60/40 metrics.
        """
        if not entries:
            return {}

        rf_per_trade = 0.0   # simplified
        strat_returns = [e.net_pnl / start_nav for e in entries]
        btc_returns = self.btc_buyhold_returns(entries, btc_start, btc_end)
        sixty_forty = self.sixty_forty_returns(len(entries))

        return {
            "strategy": {
                "sharpe": _sharpe(strat_returns),
                "sortino": _sortino(strat_returns),
                "total_return": sum(strat_returns),
            },
            "btc_buyhold": {
                "sharpe": _sharpe(btc_returns),
                "sortino": _sortino(btc_returns),
                "total_return": sum(btc_returns),
            },
            "sixty_forty": {
                "sharpe": _sharpe(sixty_forty),
                "sortino": _sortino(sixty_forty),
                "total_return": sum(sixty_forty),
            },
            "vs_btc": {
                "alpha": self.compute_alpha(strat_returns, btc_returns, rf_per_trade),
                "beta": self.compute_beta(strat_returns, btc_returns),
                "information_ratio": self.compute_information_ratio(
                    strat_returns, btc_returns
                ),
            },
            "vs_sixty_forty": {
                "alpha": self.compute_alpha(strat_returns, sixty_forty, rf_per_trade),
                "beta": self.compute_beta(strat_returns, sixty_forty),
                "information_ratio": self.compute_information_ratio(
                    strat_returns, sixty_forty
                ),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_streaks(outcomes: List[bool]) -> Dict[str, int]:
    """
    Given a list of True/False (win/loss), compute max win/loss streaks
    and current streak (positive=wins, negative=losses).
    """
    if not outcomes:
        return {"max_win": 0, "max_loss": 0, "current": 0}

    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for o in outcomes:
        if o:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)

    current = cur_win if outcomes[-1] else -cur_loss
    return {"max_win": max_win, "max_loss": max_loss, "current": current}


if __name__ == "__main__":
    import tempfile
    from trade_journal import _make_sample_entry
    import random

    rng = random.Random(0)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_file = f.name

    with TradeJournal(db_file) as journal:
        symbols = ["BTC", "ETH", "SOL", "AAPL", "NVDA"]
        base_ts = datetime(2024, 1, 1)
        for i in range(120):
            sym = rng.choice(symbols)
            pnl = rng.gauss(80, 400)
            bars = rng.randint(2, 48)
            e = _make_sample_entry(sym, pnl, bars)
            # Stagger timestamps
            ts = base_ts + timedelta(hours=i * 12)
            e.entry_ts = ts.isoformat()
            e.exit_ts = (ts + timedelta(hours=bars)).isoformat()
            e.portfolio_nav_at_entry = 100_000.0 + i * 50
            e.hurst_regime = rng.choice(["trending", "neutral", "mean-reverting"])
            e.vol_regime = rng.choice(["low", "med", "high"])
            e.bh_active = rng.random() > 0.3
            journal.add_entry(e)

    with PerformanceReport(db_file) as rpt:
        report = rpt.generate_full_report()
        ov = report["overview"]
        print(f"Total return: {ov['total_return_pct']:.2f}%")
        print(f"Sharpe: {ov['sharpe']:.3f}")
        print(f"Max DD: {ov['max_drawdown_pct']:.2f}%")
        print(f"Calmar: {ov['calmar']:.3f}")
