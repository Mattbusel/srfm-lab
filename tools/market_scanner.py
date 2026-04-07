"""
market_scanner.py -- Real-time market scanner for SRFM signals.

Usage:
    python tools/market_scanner.py --universe crypto --top 10 --min-signal 0.3
    python tools/market_scanner.py --universe ES,NQ,YM,CL --top 5
    python tools/market_scanner.py --help

Fetches BH mass, Hurst, and regime signals for each symbol and ranks by
signal strength. Prints a Rich-formatted color-coded table.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
import requests
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIVE_TRADER_BASE = "http://live-trader:8080"
REQUEST_TIMEOUT = 3.0  # seconds

UNIVERSES: dict[str, List[str]] = {
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD",
               "XRP-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD"],
    "equity": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "GLD", "SLV"],
    "futures": ["ES", "NQ", "YM", "RTY", "CL", "GC", "SI", "ZB", "6E", "6J"],
    "mixed":   ["ES", "NQ", "BTC-USD", "ETH-USD", "GLD", "CL"],
}

console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    symbol: str
    signal_strength: float        # composite signal: -1 to 1
    bh_mass: float                # 0-5 scale
    hurst: float                  # 0-1
    regime: str                   # BULL / BEAR / SIDEWAYS / HV
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float            # (target - entry) / (entry - stop)
    confidence: float             # 0-1
    error: Optional[str] = field(default=None, repr=False)

    @property
    def direction(self) -> str:
        """Long / Short / Flat based on signal_strength."""
        if self.signal_strength > 0.15:
            return "LONG"
        if self.signal_strength < -0.15:
            return "SHORT"
        return "FLAT"

    @property
    def row_style(self) -> str:
        """Rich style name for the row."""
        if self.signal_strength > 0.6:
            return "bright_green"
        if self.signal_strength > 0.3:
            return "yellow"
        if self.signal_strength < 0:
            return "bright_red"
        return ""


# ---------------------------------------------------------------------------
# Wire-layer helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = REQUEST_TIMEOUT) -> Optional[dict]:
    """GET url and return parsed JSON, or None on failure."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _fetch_symbol_data(symbol: str) -> Optional[dict]:
    """Fetch combined signal data for one symbol from live-trader."""
    url = f"{LIVE_TRADER_BASE}/signals/symbol/{symbol}"
    return _get_json(url)


def _compute_signal_strength(bh_mass: float, hurst: float, regime: str) -> float:
    """
    Combine BH mass, Hurst exponent, and regime into a single signal score.
    Returns a float in approximately [-1, 1].
    """
    # BH mass component: active (>=1.5) pushes toward 1
    mass_score = min(bh_mass / 3.0, 1.0) if bh_mass >= 1.5 else -(1.0 - min(bh_mass / 1.5, 1.0)) * 0.3

    # Hurst component: trending (>0.58) is good, mean-reverting (<0.42) penalizes trend signals
    if hurst > 0.58:
        hurst_score = (hurst - 0.58) / 0.42  # 0 to 1
    elif hurst < 0.42:
        hurst_score = (hurst - 0.42) / 0.42  # -1 to 0
    else:
        hurst_score = 0.0

    # Regime component
    regime_map = {"BULL": 0.4, "BEAR": -0.4, "HV": 0.1, "SIDEWAYS": -0.1}
    regime_score = regime_map.get(regime, 0.0)

    # Weighted sum
    raw = 0.5 * mass_score + 0.3 * hurst_score + 0.2 * regime_score
    return max(-1.0, min(1.0, raw))


def _estimate_levels(price: float, signal_strength: float, regime: str) -> tuple[float, float, float]:
    """
    Estimate entry, stop, and target prices from current price and signal context.
    Returns (entry, stop, target).
    """
    # ATR proxy: use regime-based volatility assumption
    atr_pct = {"BULL": 0.005, "BEAR": 0.008, "HV": 0.015, "SIDEWAYS": 0.004}.get(regime, 0.006)
    atr = price * atr_pct

    entry = price
    if signal_strength > 0:
        stop = entry - 2.0 * atr
        target = entry + 3.0 * atr
    else:
        stop = entry + 2.0 * atr
        target = entry - 3.0 * atr

    rr = abs(target - entry) / max(abs(entry - stop), 1e-9)
    return entry, stop, target


# ---------------------------------------------------------------------------
# MarketScanner
# ---------------------------------------------------------------------------

class MarketScanner:
    """Scans a universe of symbols and ranks by SRFM signal strength."""

    def __init__(self, base_url: str = LIVE_TRADER_BASE, timeout: float = REQUEST_TIMEOUT):
        self._base = base_url
        self._timeout = timeout

    # -- single symbol -------------------------------------------------------

    def scan_symbol(self, symbol: str) -> ScanResult:
        """Fetch and score a single symbol. Returns ScanResult with error set on failure."""
        data = _get_json(f"{self._base}/signals/symbol/{symbol}", self._timeout)

        if data is None:
            # Return a placeholder result indicating fetch failure
            return ScanResult(
                symbol=symbol,
                signal_strength=0.0,
                bh_mass=0.0,
                hurst=0.5,
                regime="UNKNOWN",
                entry_price=0.0,
                stop_price=0.0,
                target_price=0.0,
                risk_reward=0.0,
                confidence=0.0,
                error="fetch failed",
            )

        bh_mass = float(data.get("bh_mass", 0.0))
        hurst = float(data.get("hurst", 0.5))
        regime = str(data.get("regime", "SIDEWAYS"))
        price = float(data.get("price", 0.0))
        confidence = float(data.get("confidence", 0.5))

        signal_strength = _compute_signal_strength(bh_mass, hurst, regime)
        entry, stop, target = _estimate_levels(price, signal_strength, regime)
        rr = abs(target - entry) / max(abs(entry - stop), 1e-9)

        return ScanResult(
            symbol=symbol,
            signal_strength=round(signal_strength, 4),
            bh_mass=round(bh_mass, 3),
            hurst=round(hurst, 4),
            regime=regime,
            entry_price=round(entry, 4),
            stop_price=round(stop, 4),
            target_price=round(target, 4),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
        )

    # -- universe scan -------------------------------------------------------

    def scan(self, universe: List[str]) -> List[ScanResult]:
        """Scan all symbols in the universe. Returns list of ScanResults."""
        results = []
        for sym in universe:
            results.append(self.scan_symbol(sym))
        return results

    # -- top opportunities ---------------------------------------------------

    def top_opportunities(self, universe: List[str], n: int = 10) -> List[ScanResult]:
        """Return the top-n symbols sorted by absolute signal strength."""
        results = self.scan(universe)
        # sort by abs signal_strength descending; exclude errors
        valid = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        valid.sort(key=lambda r: abs(r.signal_strength), reverse=True)
        return (valid + failed)[:n]


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def _hurst_style(h: float) -> str:
    if h > 0.58:
        return "green"
    if h >= 0.42:
        return "yellow"
    return "red"


def _regime_style(regime: str) -> str:
    return {"BULL": "bright_green", "BEAR": "bright_red", "HV": "bright_magenta",
            "SIDEWAYS": "yellow"}.get(regime, "white")


def _signal_bar(strength: float, width: int = 12) -> Text:
    """ASCII bar representation of signal strength."""
    filled = int(abs(strength) * width)
    if filled > width:
        filled = width
    bar = "█" * filled + "░" * (width - filled)
    color = "bright_green" if strength >= 0 else "bright_red"
    return Text(bar, style=color)


def print_scan_table(results: List[ScanResult], title: str = "SRFM Market Scanner") -> None:
    """Print a Rich-formatted color-coded table of scan results."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_blue",
        border_style="bright_black",
        show_lines=False,
    )

    table.add_column("Symbol", style="bold white", width=12)
    table.add_column("Signal", width=14, no_wrap=True)
    table.add_column("Strength", justify="right", width=8)
    table.add_column("BH Mass", justify="right", width=8)
    table.add_column("Hurst", justify="right", width=7)
    table.add_column("Regime", width=9)
    table.add_column("Dir", width=6)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("Stop", justify="right", width=10)
    table.add_column("Target", justify="right", width=10)
    table.add_column("R:R", justify="right", width=5)
    table.add_column("Conf", justify="right", width=6)

    for r in results:
        if r.error:
            table.add_row(
                r.symbol,
                Text("-- err --", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                Text("n/a", style="dim"),
                style="dim",
            )
            continue

        row_style = Style(color=r.row_style) if r.row_style else Style()

        strength_text = Text(f"{r.signal_strength:+.3f}", style=row_style)
        hurst_text = Text(f"{r.hurst:.4f}", style=_hurst_style(r.hurst))
        regime_text = Text(f"{r.regime:<8}", style=_regime_style(r.regime))
        dir_color = "bright_green" if r.direction == "LONG" else (
            "bright_red" if r.direction == "SHORT" else "yellow")
        dir_text = Text(r.direction, style=dir_color)

        entry_str = f"{r.entry_price:.4f}" if r.entry_price > 0 else "--"
        stop_str = f"{r.stop_price:.4f}" if r.stop_price > 0 else "--"
        target_str = f"{r.target_price:.4f}" if r.target_price > 0 else "--"

        table.add_row(
            Text(r.symbol, style="bold white"),
            _signal_bar(r.signal_strength),
            strength_text,
            Text(f"{r.bh_mass:.3f}", style="cyan"),
            hurst_text,
            regime_text,
            dir_text,
            Text(entry_str, style="white"),
            Text(stop_str, style="bright_red"),
            Text(target_str, style="bright_green"),
            Text(f"{r.risk_reward:.2f}", style="white"),
            Text(f"{r.confidence:.0%}", style="white"),
        )

    console.print(table)

    # Legend
    console.print(
        "[bright_green]Green[/bright_green] = signal > 0.6  "
        "[yellow]Yellow[/yellow] = 0.3-0.6  "
        "[bright_red]Red[/bright_red] = signal < 0 (potential short)  "
        "[dim]Hurst: green > 0.58 (trend) | yellow 0.42-0.58 | red < 0.42[/dim]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_universe(universe_arg: str) -> List[str]:
    """Resolve universe string to a list of symbols."""
    if universe_arg in UNIVERSES:
        return UNIVERSES[universe_arg]
    # treat as comma-separated symbols
    return [s.strip().upper() for s in universe_arg.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SRFM Real-time Market Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Available universes: " + ", ".join(UNIVERSES.keys()),
            "Or pass comma-separated symbols: --universe ES,NQ,BTC-USD",
        ]),
    )
    parser.add_argument(
        "--universe", default="futures",
        help="Named universe or comma-separated symbols (default: futures)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Show top N opportunities (default: 10)",
    )
    parser.add_argument(
        "--min-signal", type=float, default=0.0, dest="min_signal",
        help="Minimum absolute signal strength to include (default: 0.0)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan all symbols in universe, not just top N",
    )
    parser.add_argument(
        "--url", default=LIVE_TRADER_BASE,
        help=f"live-trader base URL (default: {LIVE_TRADER_BASE})",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Continuously refresh every 15 seconds (Ctrl-C to stop)",
    )

    args = parser.parse_args(argv)

    universe = _resolve_universe(args.universe)
    if not universe:
        console.print("[bright_red]Error:[/bright_red] empty universe -- nothing to scan")
        return 1

    scanner = MarketScanner(base_url=args.url)

    def run_once() -> None:
        console.rule(f"[bold bright_blue]SRFM Scanner[/bold bright_blue]  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if args.all:
            results = scanner.scan(universe)
            results.sort(key=lambda r: abs(r.signal_strength), reverse=True)
        else:
            results = scanner.top_opportunities(universe, n=args.top)

        if args.min_signal > 0:
            results = [r for r in results if r.error or abs(r.signal_strength) >= args.min_signal]

        if not results:
            console.print("[yellow]No results meet the min-signal threshold.[/yellow]")
            return

        title = (f"Top {len(results)} Opportunities -- universe: {args.universe}"
                 f"  ({len(universe)} symbols scanned)")
        print_scan_table(results, title=title)

    if args.watch:
        try:
            while True:
                run_once()
                console.print(f"[dim]Refreshing in 15s... Ctrl-C to stop[/dim]")
                time.sleep(15)
        except KeyboardInterrupt:
            console.print("\n[yellow]Scanner stopped.[/yellow]")
    else:
        run_once()

    return 0


if __name__ == "__main__":
    sys.exit(main())
