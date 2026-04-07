"""
risk_reporter.py # Risk reporting for SRFM.

Produces intraday snapshots, end-of-day reports, and weekly summaries.
Exports to JSON and formatted HTML with colored risk meters.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .drawdown_monitor import DrawdownMonitor, DrawdownWindow
from .margin_manager import MarginManager
from .var_monitor import VaRMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds for risk meter coloring
# ---------------------------------------------------------------------------

# (yellow_threshold, red_threshold) # values above red are "critical"
METER_THRESHOLDS: Dict[str, tuple] = {
    "var_utilization":    (0.60, 0.85),
    "margin_utilization": (0.70, 0.90),
    "drawdown_current":   (0.05, 0.10),
    "leverage":           (1.50, 2.50),
    "max_position_pct":   (0.15, 0.20),
}


def _meter_color(metric: str, value: float) -> str:
    """Return 'green', 'yellow', or 'red' based on thresholds."""
    thresholds = METER_THRESHOLDS.get(metric)
    if thresholds is None:
        return "green"
    yellow, red = thresholds
    if value >= red:
        return "red"
    if value >= yellow:
        return "yellow"
    return "green"


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class PositionSummary:
    symbol: str
    dollar_value: float
    pct_of_nav: float
    asset_class: str
    side: str                  # long | short


@dataclass
class RiskSnapshot:
    timestamp: datetime
    portfolio_var_99: float         # 99% 1-day VaR in dollars
    portfolio_cvar_99: float        # 99% CVaR in dollars
    max_position_pct: float         # largest single position as pct of NAV
    margin_utilization: float       # 0.0-1.0
    drawdown_current: float         # current drawdown from peak (fraction)
    drawdown_max: float             # max drawdown observed (fraction)
    leverage: float                 # gross leverage
    n_positions: int
    largest_position: str           # symbol of largest position
    var_utilization: float = 0.0    # portfolio_var / var_limit


@dataclass
class EODRiskReport:
    date: str                       # YYYY-MM-DD
    pnl: float
    realized_vol: float             # annualized realized volatility
    sharpe_ytd: float
    var_utilization: float
    limit_breaches: List[str]
    positions: List[PositionSummary]
    portfolio_var_99: float = 0.0
    portfolio_cvar_99: float = 0.0
    margin_utilization: float = 0.0
    drawdown_current: float = 0.0
    drawdown_max: float = 0.0
    leverage: float = 0.0
    nav: float = 0.0


@dataclass
class WeeklyRiskReport:
    week_ending: str                # YYYY-MM-DD (Friday)
    weekly_pnl: float
    weekly_return: float
    avg_daily_var: float
    max_daily_drawdown: float
    max_leverage_seen: float
    limit_breaches: List[str]
    daily_pnl_series: List[float]
    sharpe_weekly: float
    nav_end: float = 0.0


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

_HTML_STYLE = """
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
  h1, h2 { color: #c8a2c8; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
  th { background: #16213e; color: #c8a2c8; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #333; }
  tr:hover { background: #0f3460; }
  .green  { color: #4caf50; font-weight: bold; }
  .yellow { color: #ffeb3b; font-weight: bold; }
  .red    { color: #f44336; font-weight: bold; }
  .meter-bar { width: 120px; height: 14px; background: #333; border-radius: 4px; display: inline-block; vertical-align: middle; }
  .meter-fill { height: 100%; border-radius: 4px; }
  .section { margin-bottom: 32px; }
</style>
"""


def _meter_bar_html(metric: str, value: float, max_val: float = 1.0) -> str:
    color = _meter_color(metric, value)
    pct = min(100.0, value / max(max_val, 1e-10) * 100)
    return (
        f'<div class="meter-bar">'
        f'<div class="meter-fill" style="width:{pct:.1f}%;background:{"#4caf50" if color=="green" else "#ffeb3b" if color=="yellow" else "#f44336"};">'
        f'</div></div>&nbsp;<span class="{color}">{value:.2%}</span>'
    )


def _render_snapshot_html(snap: RiskSnapshot, title: str = "Intraday Risk Snapshot") -> str:
    rows = [
        ("Timestamp", snap.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Portfolio VaR 99%", f"${snap.portfolio_var_99:,.0f}"),
        ("Portfolio CVaR 99%", f"${snap.portfolio_cvar_99:,.0f}"),
        ("VaR Utilization", _meter_bar_html("var_utilization", snap.var_utilization)),
        ("Margin Utilization", _meter_bar_html("margin_utilization", snap.margin_utilization)),
        ("Current Drawdown", _meter_bar_html("drawdown_current", snap.drawdown_current, 0.15)),
        ("Max Drawdown", _meter_bar_html("drawdown_current", snap.drawdown_max, 0.20)),
        ("Gross Leverage", f'<span class="{_meter_color("leverage", snap.leverage)}">{snap.leverage:.2f}x</span>'),
        ("Largest Position", f'{snap.largest_position} ({snap.max_position_pct:.1%} NAV)'),
        ("Open Positions", str(snap.n_positions)),
    ]
    table_rows = "".join(
        f"<tr><td><b>{label}</b></td><td>{val}</td></tr>" for label, val in rows
    )
    return f"""
    <div class="section">
      <h2>{title}</h2>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{table_rows}</tbody></table>
    </div>
    """


def _render_positions_html(positions: List[PositionSummary]) -> str:
    if not positions:
        return "<p>No open positions.</p>"

    header = "<tr><th>Symbol</th><th>Side</th><th>Dollar Value</th><th>% NAV</th><th>Asset Class</th></tr>"
    rows = ""
    for pos in sorted(positions, key=lambda p: abs(p.dollar_value), reverse=True):
        color_cls = _meter_color("max_position_pct", pos.pct_of_nav)
        rows += (
            f"<tr>"
            f"<td>{pos.symbol}</td>"
            f"<td>{pos.side}</td>"
            f"<td>${pos.dollar_value:,.0f}</td>"
            f"<td class='{color_cls}'>{pos.pct_of_nav:.1%}</td>"
            f"<td>{pos.asset_class}</td>"
            f"</tr>"
        )
    return f"""
    <div class="section">
      <h2>Positions</h2>
      <table><thead>{header}</thead><tbody>{rows}</tbody></table>
    </div>
    """


def _render_eod_html(report: EODRiskReport) -> str:
    snap_rows = [
        ("Date", report.date),
        ("Daily P&L", f'<span class="{"green" if report.pnl >= 0 else "red"}">${report.pnl:,.0f}</span>'),
        ("Realized Volatility (Ann.)", f"{report.realized_vol:.1%}"),
        ("Sharpe YTD", f"{report.sharpe_ytd:.2f}"),
        ("Portfolio VaR 99%", f"${report.portfolio_var_99:,.0f}"),
        ("Portfolio CVaR 99%", f"${report.portfolio_cvar_99:,.0f}"),
        ("VaR Utilization", _meter_bar_html("var_utilization", report.var_utilization)),
        ("Margin Utilization", _meter_bar_html("margin_utilization", report.margin_utilization)),
        ("Current Drawdown", _meter_bar_html("drawdown_current", report.drawdown_current, 0.15)),
        ("Gross Leverage", f'{report.leverage:.2f}x'),
        ("Account NAV", f"${report.nav:,.0f}"),
    ]
    table_rows = "".join(
        f"<tr><td><b>{label}</b></td><td>{val}</td></tr>" for label, val in snap_rows
    )

    breach_html = ""
    if report.limit_breaches:
        breach_list = "".join(f'<li class="red">{b}</li>' for b in report.limit_breaches)
        breach_html = f'<div class="section"><h2>Limit Breaches</h2><ul>{breach_list}</ul></div>'
    else:
        breach_html = '<div class="section"><h2>Limit Breaches</h2><p class="green">None</p></div>'

    positions_html = _render_positions_html(report.positions)

    return f"""
    <div class="section">
      <h2>End-of-Day Risk Report # {report.date}</h2>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{table_rows}</tbody></table>
    </div>
    {breach_html}
    {positions_html}
    """


def _render_weekly_html(report: WeeklyRiskReport) -> str:
    weekly_ret_color = "green" if report.weekly_return >= 0 else "red"
    rows = [
        ("Week Ending", report.week_ending),
        ("Weekly P&L", f'<span class="{weekly_ret_color}">${report.weekly_pnl:,.0f}</span>'),
        ("Weekly Return", f'<span class="{weekly_ret_color}">{report.weekly_return:.2%}</span>'),
        ("Avg Daily VaR 99%", f"${report.avg_daily_var:,.0f}"),
        ("Max Daily Drawdown", _meter_bar_html("drawdown_current", report.max_daily_drawdown, 0.15)),
        ("Max Leverage Seen", f'{report.max_leverage_seen:.2f}x'),
        ("Sharpe (Week)", f'{report.sharpe_weekly:.2f}'),
        ("End NAV", f"${report.nav_end:,.0f}"),
    ]
    table_rows = "".join(
        f"<tr><td><b>{label}</b></td><td>{val}</td></tr>" for label, val in rows
    )

    daily_pnl_cells = "".join(
        f'<td class="{"green" if p >= 0 else "red"}">${p:,.0f}</td>'
        for p in report.daily_pnl_series
    )
    daily_pnl_row = f"<tr><td><b>Daily P&L</b></td>{daily_pnl_cells}</tr>"

    breach_html = ""
    if report.limit_breaches:
        breach_list = "".join(f'<li class="red">{b}</li>' for b in report.limit_breaches)
        breach_html = f'<div class="section"><h2>Limit Breaches This Week</h2><ul>{breach_list}</ul></div>'
    else:
        breach_html = '<div class="section"><h2>Limit Breaches This Week</h2><p class="green">None</p></div>'

    return f"""
    <div class="section">
      <h2>Weekly Risk Summary # w/e {report.week_ending}</h2>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{table_rows}</tbody></table>
      <table><tbody>{daily_pnl_row}</tbody></table>
    </div>
    {breach_html}
    """


def _full_html(body: str, title: str = "SRFM Risk Report") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  {_HTML_STYLE}
</head>
<body>
  <h1>{title}</h1>
  {body}
  <p style="color:#555;font-size:11px;">
    Generated by SRFM RiskReporter # {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
  </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Risk Reporter
# ---------------------------------------------------------------------------

class RiskReporter:
    """
    Aggregates data from DrawdownMonitor, MarginManager, and VaRMonitor
    to produce structured risk reports in JSON and HTML formats.
    """

    def __init__(
        self,
        var_monitor: VaRMonitor,
        margin_manager: MarginManager,
        drawdown_monitor: DrawdownMonitor,
        nav: float,
        var_limit: Optional[float] = None,
    ) -> None:
        self._var = var_monitor
        self._margin = margin_manager
        self._dd = drawdown_monitor
        self._nav = nav
        self._var_limit = var_limit or (nav * 0.02)   # default 2% NAV

        # Daily P&L history for Sharpe / vol computation
        self._daily_pnl: List[float] = []
        self._ytd_starting_nav: float = nav

    def update_nav(self, nav: float) -> None:
        self._nav = nav
        self._var.update_nav(nav)
        self._margin.update_nav(nav)

    def record_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl.append(pnl)

    # # Core snapshot -------------------------------------------------------

    def intraday_snapshot(self) -> RiskSnapshot:
        """Build a lightweight snapshot for real-time dashboards."""
        var_99 = self._var.portfolio_var(0.99)
        cvar_99 = self._var.portfolio_cvar(0.99)

        margin_snap = self._margin.snapshot()
        margin_util = margin_snap["margin_utilization"]
        leverage = margin_snap["gross_leverage"]
        n_pos = margin_snap["n_positions"]

        positions = self._margin._positions  # access internal dict
        nav = self._nav
        max_pos_pct = 0.0
        largest_sym = ""
        for sym, pos in positions.items():
            pct = abs(pos.market_value) / nav if nav > 0 else 0.0
            if pct > max_pos_pct:
                max_pos_pct = pct
                largest_sym = sym

        dd_current = self._dd.current_drawdown(DrawdownWindow.DAILY)
        dd_max = self._dd.max_drawdown(window=DrawdownWindow.ALL_TIME)
        var_util = var_99 / self._var_limit if self._var_limit > 0 else 0.0

        return RiskSnapshot(
            timestamp=datetime.now(timezone.utc),
            portfolio_var_99=var_99,
            portfolio_cvar_99=cvar_99,
            max_position_pct=max_pos_pct,
            margin_utilization=margin_util,
            drawdown_current=dd_current,
            drawdown_max=dd_max,
            leverage=leverage,
            n_positions=n_pos,
            largest_position=largest_sym,
            var_utilization=var_util,
        )

    # # End of day ----------------------------------------------------------

    def end_of_day_report(self, date: str) -> EODRiskReport:
        """Build a full end-of-day risk report."""
        snap = self.intraday_snapshot()

        daily_pnl_today = self._daily_pnl[-1] if self._daily_pnl else 0.0
        realized_vol = self._compute_realized_vol()
        sharpe_ytd = self._compute_sharpe_ytd()

        # Limit breaches
        breaches: List[str] = []
        if snap.var_utilization > 1.0:
            breaches.append(f"VaR limit breached: utilization {snap.var_utilization:.1%}")
        if snap.drawdown_current > 0.05:
            breaches.append(f"Daily drawdown {snap.drawdown_current:.1%} > 5% threshold")
        if snap.margin_utilization > 0.85:
            breaches.append(f"Margin utilization {snap.margin_utilization:.1%} > 85%")
        if snap.leverage > 3.0:
            breaches.append(f"Gross leverage {snap.leverage:.2f}x > 3x limit")
        if self._margin.is_margin_call():
            breaches.append("Margin call active")

        positions = self._build_position_summaries()

        return EODRiskReport(
            date=date,
            pnl=daily_pnl_today,
            realized_vol=realized_vol,
            sharpe_ytd=sharpe_ytd,
            var_utilization=snap.var_utilization,
            limit_breaches=breaches,
            positions=positions,
            portfolio_var_99=snap.portfolio_var_99,
            portfolio_cvar_99=snap.portfolio_cvar_99,
            margin_utilization=snap.margin_utilization,
            drawdown_current=snap.drawdown_current,
            drawdown_max=snap.drawdown_max,
            leverage=snap.leverage,
            nav=self._nav,
        )

    # # Weekly --------------------------------------------------------------

    def weekly_risk_summary(self, week_ending: str) -> WeeklyRiskReport:
        """Aggregate the last 5 days of daily P&L into a weekly summary."""
        recent_pnl = self._daily_pnl[-5:] if len(self._daily_pnl) >= 5 else list(self._daily_pnl)
        weekly_pnl = sum(recent_pnl)
        weekly_return = weekly_pnl / self._ytd_starting_nav if self._ytd_starting_nav > 0 else 0.0

        # Daily VaR average (use current as proxy if no history)
        avg_var = self._var.portfolio_var(0.99)

        # Max drawdown this week
        max_dd = self._dd.max_drawdown(lookback_days=7)

        # Sharpe on the week
        sharpe_weekly = self._compute_sharpe_from_series(recent_pnl)

        # Leverage # current as proxy; production would store daily
        leverage_now = self._margin.gross_leverage()

        # Limit breaches # look at EOD report for today
        today_snap = self.intraday_snapshot()
        week_breaches: List[str] = []
        if today_snap.var_utilization > 1.0:
            week_breaches.append(f"VaR limit: {today_snap.var_utilization:.1%}")
        if today_snap.drawdown_current > 0.05:
            week_breaches.append(f"Daily DD: {today_snap.drawdown_current:.1%}")

        return WeeklyRiskReport(
            week_ending=week_ending,
            weekly_pnl=weekly_pnl,
            weekly_return=weekly_return,
            avg_daily_var=avg_var,
            max_daily_drawdown=max_dd,
            max_leverage_seen=leverage_now,
            limit_breaches=week_breaches,
            daily_pnl_series=recent_pnl,
            sharpe_weekly=sharpe_weekly,
            nav_end=self._nav,
        )

    # # Export methods ------------------------------------------------------

    def export_to_json(self, report: Any, path: str) -> None:
        """Serialize a report dataclass to a JSON file."""
        def _default(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)  # type: ignore[arg-type]
            return str(obj)

        data = asdict(report) if hasattr(report, "__dataclass_fields__") else report
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_default)
        logger.info("Risk report exported to %s", path)

    def export_to_html(self, report: Any, path: str) -> None:
        """Export a report to a standalone HTML file with colored risk meters."""
        if isinstance(report, RiskSnapshot):
            body = _render_snapshot_html(report)
            title = "SRFM Intraday Risk Snapshot"
        elif isinstance(report, EODRiskReport):
            body = _render_eod_html(report)
            title = f"SRFM EOD Risk Report # {report.date}"
        elif isinstance(report, WeeklyRiskReport):
            body = _render_weekly_html(report)
            title = f"SRFM Weekly Risk Summary # w/e {report.week_ending}"
        else:
            body = f"<pre>{json.dumps(report, default=str, indent=2)}</pre>"
            title = "SRFM Risk Report"

        html = _full_html(body, title)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML risk report written to %s", path)

    # # Private helpers -----------------------------------------------------

    def _build_position_summaries(self) -> List[PositionSummary]:
        summaries = []
        nav = self._nav
        for sym, pos in self._margin._positions.items():
            summaries.append(
                PositionSummary(
                    symbol=sym,
                    dollar_value=pos.market_value,
                    pct_of_nav=abs(pos.market_value) / nav if nav > 0 else 0.0,
                    asset_class=pos.asset_class,
                    side="long" if pos.qty > 0 else "short",
                )
            )
        return sorted(summaries, key=lambda p: abs(p.dollar_value), reverse=True)

    def _compute_realized_vol(self) -> float:
        """
        Annualized realized volatility from daily P&L series.
        Returns 0.0 if fewer than 2 observations.
        """
        if len(self._daily_pnl) < 2 or self._nav <= 0:
            return 0.0
        returns = [p / self._nav for p in self._daily_pnl]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance * 252)

    def _compute_sharpe_ytd(self, risk_free_rate: float = 0.05) -> float:
        """
        Annualized Sharpe ratio from daily P&L series.
        risk_free_rate: annual risk-free rate (e.g. 0.05 = 5%).
        """
        if len(self._daily_pnl) < 2 or self._nav <= 0:
            return 0.0
        returns = [p / self._nav for p in self._daily_pnl]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        daily_rf = risk_free_rate / 252
        return (mean_r - daily_rf) / std * math.sqrt(252)

    def _compute_sharpe_from_series(
        self, pnl_series: List[float], risk_free_rate: float = 0.05
    ) -> float:
        if len(pnl_series) < 2 or self._nav <= 0:
            return 0.0
        returns = [p / self._nav for p in pnl_series]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        daily_rf = risk_free_rate / 252
        return (mean_r - daily_rf) / std * math.sqrt(252)
