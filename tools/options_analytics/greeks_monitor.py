"""
greeks_monitor.py — Live portfolio Greeks dashboard.

Features
--------
* Reads option positions from OptionOverlay state file + Alpaca REST
* Portfolio-level net delta, gamma, theta ($), vega, vanna
* Delta hedging signal when |net_delta| exceeds threshold
* Gamma scalping opportunity identification
* Theta decay calendar (30-day P&L projection)
* Scenario analysis: ±1%, ±5%, ±10% move → P&L impact (full re-price)
* Expiry countdown with intrinsic / extrinsic breakdown
* Rich terminal UI refreshing every 30 s
* JSON export for external dashboards

Usage
-----
    python greeks_monitor.py                        # live Rich dashboard
    python greeks_monitor.py --export out.json      # one-shot JSON export
    python greeks_monitor.py --positions pos.json   # load positions from file
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# Lazy imports for optional deps
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.live import Live
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from scipy.stats import norm
except ImportError:
    pass  # will fail at runtime but keep import clean

RISK_FREE_RATE = 0.0525
INSTRUMENTS = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT"]
DELTA_HEDGE_THRESHOLD = 0.10       # net delta units
GAMMA_SCALP_THRESHOLD = 0.005      # gamma per $ notional
REFRESH_INTERVAL = 30              # seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    symbol: str               # OCC option symbol
    underlying: str
    expiry: date
    strike: float
    option_type: str          # "call" | "put"
    quantity: int             # signed: positive = long
    avg_cost: float           # per contract (100 multiplier)
    # Filled by monitor:
    current_price: float = 0.0
    underlying_price: float = 0.0
    iv: float = 0.0


@dataclass
class PositionGreeks:
    symbol: str
    quantity: int
    delta: float              # per share (includes qty * 100)
    gamma: float
    theta: float              # $ per day
    vega: float               # $ per 1 vol pt
    vanna: float
    volga: float
    charm: float
    intrinsic: float          # $ total
    extrinsic: float          # $ total
    dte: int
    pnl: float                # mark-to-market


@dataclass
class PortfolioGreeks:
    timestamp: str
    net_delta: float
    net_gamma: float
    net_theta: float          # $ per day
    net_vega: float
    net_vanna: float
    position_count: int
    total_notional: float
    positions: List[PositionGreeks] = field(default_factory=list)
    hedge_signal: Optional[dict] = None
    gamma_opps: List[dict] = field(default_factory=list)
    scenarios: Dict[str, float] = field(default_factory=dict)
    theta_calendar: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Black-Scholes helpers (self-contained, no cross-module dep)
# ---------------------------------------------------------------------------

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, otype: str) -> float:
    if T <= 0:
        return max(S - K, 0.0) if otype == "call" else max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if otype == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_greeks_full(
    S: float, K: float, T: float, r: float, sigma: float, otype: str, qty: int
) -> dict:
    """All Greeks scaled by position quantity and 100 multiplier."""
    mult = qty * 100
    if T <= 0 or sigma <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0, vanna=0, volga=0, charm=0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)
    npd1 = norm.pdf(d1)
    disc = math.exp(-r * T)

    raw_delta = norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1.0
    raw_gamma = npd1 / (S * sigma * sqrt_T)
    raw_vega = S * npd1 * sqrt_T / 100.0  # per 1 vol pt
    if otype == "call":
        raw_theta = (
            -S * npd1 * sigma / (2 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        ) / 365.0
    else:
        raw_theta = (
            -S * npd1 * sigma / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        ) / 365.0
    raw_vanna = -npd1 * d2 / sigma
    raw_volga = raw_vega * 100 * d1 * d2 / sigma  # before /100 scaling
    raw_charm = -(
        npd1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    ) / 365.0

    return dict(
        delta=raw_delta * mult,
        gamma=raw_gamma * mult,
        theta=raw_theta * mult,          # $ per day
        vega=raw_vega * mult,
        vanna=raw_vanna * mult,
        volga=raw_volga * mult,
        charm=raw_charm * mult,
    )


def _bs_iv(mkt_price: float, S: float, K: float, T: float, r: float, otype: str) -> float:
    """Brentq IV solver. Returns 0.0 on failure."""
    from scipy.optimize import brentq
    if mkt_price <= 0 or T <= 0:
        return 0.0
    try:
        def obj(sig: float) -> float:
            return _bs_price(S, K, T, r, sig, otype) - mkt_price
        if obj(0.001) * obj(10.0) > 0:
            return 0.0
        return float(brentq(obj, 0.001, 10.0, xtol=1e-7, maxiter=200))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Alpaca position fetcher
# ---------------------------------------------------------------------------

class AlpacaClient:
    BROKER_URL = "https://api.alpaca.markets/v2"
    DATA_URL = "https://data.alpaca.markets/v2"

    def __init__(self):
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        })

    def get_positions(self) -> List[dict]:
        try:
            resp = self.session.get(f"{self.BROKER_URL}/positions", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    def get_latest_price(self, symbol: str) -> float:
        try:
            resp = self.session.get(
                f"{self.DATA_URL}/stocks/{symbol}/quotes/latest", timeout=8
            )
            resp.raise_for_status()
            q = resp.json().get("quote", {})
            return (q.get("bp", 0) + q.get("ap", 0)) / 2.0
        except Exception:
            return 0.0

    def get_option_snapshot(self, option_symbol: str) -> dict:
        try:
            resp = self.session.get(
                f"{self.DATA_URL}/options/snapshots/{option_symbol}",
                timeout=8,
            )
            resp.raise_for_status()
            snaps = resp.json().get("snapshots", {})
            return snaps.get(option_symbol, {})
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Position loader
# ---------------------------------------------------------------------------

def _parse_occ_symbol(sym: str) -> Optional[Tuple[str, date, str, float]]:
    """Parse OCC symbol → (underlying, expiry, option_type, strike)."""
    try:
        for i in range(len(sym) - 1, -1, -1):
            if sym[i] in ("C", "P") and i >= 3:
                underlying = sym[:i - 6].rstrip()
                date_str = sym[i - 6: i]
                otype = "call" if sym[i] == "C" else "put"
                strike = float(sym[i + 1:]) / 1000.0
                expiry = datetime.strptime(date_str, "%y%m%d").date()
                return underlying, expiry, otype, strike
    except Exception:
        pass
    return None


def load_positions_from_alpaca(client: AlpacaClient) -> List[OptionPosition]:
    """Load option positions from Alpaca broker account."""
    raw = client.get_positions()
    positions = []
    for pos in raw:
        sym = pos.get("symbol", "")
        asset_class = pos.get("asset_class", "")
        if asset_class != "us_option":
            continue
        parsed = _parse_occ_symbol(sym)
        if parsed is None:
            continue
        underlying, expiry, otype, strike = parsed
        qty = int(pos.get("qty", 0))
        avg_cost = float(pos.get("avg_entry_price", 0))
        positions.append(OptionPosition(
            symbol=sym,
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=otype,
            quantity=qty,
            avg_cost=avg_cost,
        ))
    return positions


def load_positions_from_file(path: str) -> List[OptionPosition]:
    """Load positions from a JSON file with list of position dicts."""
    with open(path) as f:
        data = json.load(f)
    positions = []
    for d in data:
        positions.append(OptionPosition(
            symbol=d["symbol"],
            underlying=d["underlying"],
            expiry=date.fromisoformat(d["expiry"]),
            strike=float(d["strike"]),
            option_type=d["option_type"],
            quantity=int(d["quantity"]),
            avg_cost=float(d.get("avg_cost", 0)),
        ))
    return positions


def load_overlay_state(path: str = "option_overlay_state.json") -> List[OptionPosition]:
    """Load positions from OptionOverlay state file if present."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        positions_raw = data.get("positions", [])
        return load_positions_from_file.__wrapped__(positions_raw) if positions_raw else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Greeks monitor core
# ---------------------------------------------------------------------------

class GreeksMonitor:
    """
    Computes and aggregates portfolio Greeks, generates signals,
    and exports to Rich terminal or JSON.
    """

    def __init__(
        self,
        positions: Optional[List[OptionPosition]] = None,
        positions_file: Optional[str] = None,
        r: float = RISK_FREE_RATE,
    ):
        self.r = r
        self.client = AlpacaClient()
        self._positions_file = positions_file
        self._static_positions = positions  # override

    def _get_positions(self) -> List[OptionPosition]:
        if self._static_positions:
            return list(self._static_positions)
        if self._positions_file:
            return load_positions_from_file(self._positions_file)
        # Try Alpaca
        pos = load_positions_from_alpaca(self.client)
        if not pos:
            # Try overlay state
            pos = load_overlay_state()
        return pos

    def _enrich_position(self, p: OptionPosition) -> OptionPosition:
        """Fetch current market prices and solve IV."""
        snap = self.client.get_option_snapshot(p.symbol)
        if snap:
            q = snap.get("latestQuote", {})
            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            p.current_price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        else:
            p.current_price = p.avg_cost  # fallback

        p.underlying_price = self.client.get_latest_price(p.underlying)
        if p.current_price > 0 and p.underlying_price > 0:
            T = max((p.expiry - date.today()).days / 365.0, 1e-6)
            p.iv = _bs_iv(p.current_price, p.underlying_price, p.strike, T, self.r, p.option_type)
        return p

    def compute(self) -> PortfolioGreeks:
        """Full portfolio Greeks computation."""
        positions = self._get_positions()
        if not positions:
            print("[GreeksMonitor] No positions loaded.")
            return PortfolioGreeks(
                timestamp=datetime.utcnow().isoformat(),
                net_delta=0, net_gamma=0, net_theta=0, net_vega=0,
                net_vanna=0, position_count=0, total_notional=0,
            )

        print(f"[GreeksMonitor] Enriching {len(positions)} positions...")
        pos_greeks_list: List[PositionGreeks] = []
        today = date.today()

        for p in positions:
            p = self._enrich_position(p)
            T = max((p.expiry - today).days / 365.0, 1e-6)
            dte = (p.expiry - today).days
            S = p.underlying_price or p.strike
            K = p.strike
            sigma = p.iv if p.iv > 0 else 0.25  # fallback to 25%

            g = _bs_greeks_full(S, K, T, self.r, sigma, p.option_type, p.quantity)
            price_now = _bs_price(S, K, T, self.r, sigma, p.option_type)
            intrinsic_per_share = max(S - K, 0) if p.option_type == "call" else max(K - S, 0)
            extrinsic_per_share = max(price_now - intrinsic_per_share, 0)

            pnl = (p.current_price - p.avg_cost) * p.quantity * 100

            pg = PositionGreeks(
                symbol=p.symbol,
                quantity=p.quantity,
                delta=g["delta"],
                gamma=g["gamma"],
                theta=g["theta"],
                vega=g["vega"],
                vanna=g["vanna"],
                volga=g["volga"],
                charm=g["charm"],
                intrinsic=intrinsic_per_share * abs(p.quantity) * 100,
                extrinsic=extrinsic_per_share * abs(p.quantity) * 100,
                dte=dte,
                pnl=pnl,
            )
            pos_greeks_list.append(pg)

        net_delta = sum(pg.delta for pg in pos_greeks_list)
        net_gamma = sum(pg.gamma for pg in pos_greeks_list)
        net_theta = sum(pg.theta for pg in pos_greeks_list)
        net_vega = sum(pg.vega for pg in pos_greeks_list)
        net_vanna = sum(pg.vanna for pg in pos_greeks_list)
        total_notional = sum(
            abs(p.current_price * p.quantity * 100) for p in positions
        )

        # Delta hedge signal
        hedge_signal = None
        if abs(net_delta) > DELTA_HEDGE_THRESHOLD * total_notional / 100:
            direction = "sell" if net_delta > 0 else "buy"
            hedge_signal = {
                "action": f"{direction} underlying to neutralize delta",
                "net_delta": round(net_delta, 4),
                "shares_to_trade": round(-net_delta),
                "urgency": "high" if abs(net_delta) > 2 * DELTA_HEDGE_THRESHOLD * total_notional / 100 else "medium",
            }

        # Gamma scalping opportunities
        gamma_opps = []
        for pg in pos_greeks_list:
            gamma_per_notional = abs(pg.gamma) / max(total_notional, 1)
            if gamma_per_notional > GAMMA_SCALP_THRESHOLD / 100:
                gamma_opps.append({
                    "symbol": pg.symbol,
                    "gamma": round(pg.gamma, 6),
                    "dte": pg.dte,
                    "note": "High gamma - consider scalp hedging",
                })

        # Scenario analysis: re-price at S ± moves
        scenarios = self._scenario_analysis(positions, today)

        # Theta decay calendar
        theta_calendar = self._theta_calendar(positions, today)

        return PortfolioGreeks(
            timestamp=datetime.utcnow().isoformat(),
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            net_vanna=net_vanna,
            position_count=len(positions),
            total_notional=total_notional,
            positions=pos_greeks_list,
            hedge_signal=hedge_signal,
            gamma_opps=gamma_opps,
            scenarios=scenarios,
            theta_calendar=theta_calendar,
        )

    def _scenario_analysis(
        self, positions: List[OptionPosition], today: date
    ) -> Dict[str, float]:
        scenarios: Dict[str, float] = {}
        moves = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]
        labels = ["-10%", "-5%", "-1%", "+1%", "+5%", "+10%"]

        for move, label in zip(moves, labels):
            total_pnl = 0.0
            for p in positions:
                if p.underlying_price <= 0:
                    continue
                S_new = p.underlying_price * (1 + move)
                T = max((p.expiry - today).days / 365.0, 1e-6)
                sigma = p.iv if p.iv > 0 else 0.25
                price_new = _bs_price(S_new, p.strike, T, self.r, sigma, p.option_type)
                price_now = _bs_price(p.underlying_price, p.strike, T, self.r, sigma, p.option_type)
                total_pnl += (price_new - price_now) * p.quantity * 100
            scenarios[label] = round(total_pnl, 2)

        return scenarios

    def _theta_calendar(
        self, positions: List[OptionPosition], today: date
    ) -> List[dict]:
        """Project daily theta decay P&L for next 30 days."""
        calendar = []
        for day_offset in range(1, 31):
            future_date = today + timedelta(days=day_offset)  # noqa: F821
            total_theta = 0.0
            for p in positions:
                if p.underlying_price <= 0:
                    continue
                T_future = max((p.expiry - future_date).days / 365.0, 1e-6)
                T_now = max((p.expiry - today).days / 365.0, 1e-6)
                sigma = p.iv if p.iv > 0 else 0.25
                S = p.underlying_price
                K = p.strike

                price_future = _bs_price(S, K, T_future, self.r, sigma, p.option_type)
                price_now = _bs_price(S, K, T_now, self.r, sigma, p.option_type)
                total_theta += (price_future - price_now) * p.quantity * 100

            calendar.append({
                "date": str(future_date),
                "day": day_offset,
                "cumulative_theta_pnl": round(total_theta, 2),
            })

        return calendar


# ---------------------------------------------------------------------------
# Rich terminal display
# ---------------------------------------------------------------------------

def _color_val(val: float, positive_good: bool = True) -> str:
    """Return Rich markup color string for a value."""
    if val > 0:
        color = "green" if positive_good else "red"
    elif val < 0:
        color = "red" if positive_good else "green"
    else:
        color = "white"
    return f"[{color}]{val:+.4f}[/{color}]"


def _color_dollar(val: float) -> str:
    color = "green" if val >= 0 else "red"
    return f"[{color}]${val:+,.2f}[/{color}]"


def _build_portfolio_panel(pg: PortfolioGreeks) -> "Panel":
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="bold white", min_width=16)
    t.add_column("Value", justify="right", min_width=14)

    t.add_row("Net Delta (Δ)", _color_val(pg.net_delta, positive_good=True))
    t.add_row("Net Gamma (Γ)", _color_val(pg.net_gamma, positive_good=True))
    t.add_row("Net Theta (Θ/day)", _color_dollar(pg.net_theta))
    t.add_row("Net Vega (ν)", _color_val(pg.net_vega, positive_good=True))
    t.add_row("Net Vanna", _color_val(pg.net_vanna))
    t.add_row("Positions", f"[white]{pg.position_count}[/white]")
    t.add_row("Notional", f"[white]${pg.total_notional:,.0f}[/white]")

    return Panel(t, title="Portfolio Greeks", border_style="blue")


def _build_positions_table(pg: PortfolioGreeks) -> "Table":
    from rich.table import Table
    from rich import box

    t = Table(
        title="Positions",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        show_lines=True,
    )
    t.add_column("Symbol", min_width=22)
    t.add_column("Qty", justify="right")
    t.add_column("DTE", justify="right")
    t.add_column("Delta", justify="right")
    t.add_column("Gamma", justify="right")
    t.add_column("Theta/day", justify="right")
    t.add_column("Vega", justify="right")
    t.add_column("Intrinsic", justify="right")
    t.add_column("Extrinsic", justify="right")
    t.add_column("P&L", justify="right")

    for p in pg.positions:
        t.add_row(
            p.symbol,
            str(p.quantity),
            str(p.dte),
            _color_val(p.delta),
            _color_val(p.gamma, positive_good=True),
            _color_dollar(p.theta),
            _color_val(p.vega),
            f"${p.intrinsic:,.2f}",
            f"${p.extrinsic:,.2f}",
            _color_dollar(p.pnl),
        )

    return t


def _build_scenario_table(scenarios: Dict[str, float]) -> "Table":
    from rich.table import Table
    from rich import box

    t = Table(title="Scenario Analysis", box=box.SIMPLE, header_style="bold yellow")
    t.add_column("Move", justify="center")
    t.add_column("P&L Impact", justify="right")

    for label, pnl in scenarios.items():
        t.add_row(label, _color_dollar(pnl))

    return t


def _build_theta_calendar_table(calendar: List[dict]) -> "Table":
    from rich.table import Table
    from rich import box

    t = Table(title="Theta Decay (30d)", box=box.SIMPLE, header_style="bold yellow")
    t.add_column("Date")
    t.add_column("Day", justify="right")
    t.add_column("Cumulative Decay", justify="right")

    for row in calendar[::5]:  # every 5 days to keep compact
        t.add_row(row["date"], str(row["day"]), _color_dollar(row["cumulative_theta_pnl"]))

    return t


def _render_dashboard(monitor: GreeksMonitor) -> "Table":
    from rich.table import Table
    from rich.columns import Columns
    from rich.console import Console
    from rich import box

    pg = monitor.compute()

    console = Console()
    ts = f"[dim]Last update: {pg.timestamp}[/dim]"

    console.rule(f"[bold blue]Options Greeks Monitor[/bold blue]  {ts}")
    console.print(_build_portfolio_panel(pg))

    if pg.hedge_signal:
        console.print(
            f"[bold red]HEDGE SIGNAL:[/bold red] {pg.hedge_signal['action']} "
            f"| Net Δ={pg.hedge_signal['net_delta']:+.3f} "
            f"| Shares: {pg.hedge_signal['shares_to_trade']:+.0f} "
            f"| Urgency: {pg.hedge_signal['urgency'].upper()}"
        )

    if pg.gamma_opps:
        console.print("[bold yellow]Gamma Scalp Opportunities:[/bold yellow]")
        for opp in pg.gamma_opps:
            console.print(f"  {opp['symbol']} γ={opp['gamma']:.5f} DTE={opp['dte']} — {opp['note']}")

    if pg.positions:
        console.print(_build_positions_table(pg))

    console.print(
        Columns([_build_scenario_table(pg.scenarios), _build_theta_calendar_table(pg.theta_calendar)])
    )

    return pg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Live portfolio Greeks monitor")
    parser.add_argument("--positions", default=None, help="JSON positions file")
    parser.add_argument("--export", default=None, help="Export JSON snapshot and exit")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL, help="Refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    monitor = GreeksMonitor(positions_file=args.positions)

    if args.export or args.once:
        pg = monitor.compute()
        out = {
            "timestamp": pg.timestamp,
            "net_delta": pg.net_delta,
            "net_gamma": pg.net_gamma,
            "net_theta": pg.net_theta,
            "net_vega": pg.net_vega,
            "net_vanna": pg.net_vanna,
            "position_count": pg.position_count,
            "total_notional": pg.total_notional,
            "hedge_signal": pg.hedge_signal,
            "gamma_opps": pg.gamma_opps,
            "scenarios": pg.scenarios,
            "theta_calendar": pg.theta_calendar,
            "positions": [asdict(p) for p in pg.positions],
        }
        if args.export:
            with open(args.export, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"[GreeksMonitor] Exported to {args.export}")
        else:
            print(json.dumps(out, indent=2, default=str))
        return

    if not HAS_RICH:
        print("[GreeksMonitor] rich not installed. Run: pip install rich")
        print("[GreeksMonitor] Falling back to one-shot JSON output.")
        pg = monitor.compute()
        print(json.dumps(asdict(pg), indent=2, default=str))
        return

    from rich.console import Console
    console = Console()

    try:
        while True:
            console.clear()
            _render_dashboard(monitor)
            if args.once:
                break
            console.print(f"\n[dim]Refreshing in {args.interval}s... (Ctrl+C to exit)[/dim]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Exiting Greeks Monitor.[/bold]")


if __name__ == "__main__":
    from datetime import timedelta  # needed in _theta_calendar
    _cli()
