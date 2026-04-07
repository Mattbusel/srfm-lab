"""
tools/execution_quality_report.py
===================================
Analyzes execution quality from fills.

- Per-trade: slippage vs mid at order time, fill speed, commission paid
- By venue (Alpaca equity vs crypto): compare execution quality
- Size vs slippage scatter
- Market impact model validation (Almgren-Chriss prediction)

Usage:
    python tools/execution_quality_report.py --output exec_report.html
    python tools/execution_quality_report.py --db execution/live_trades.db
    python tools/execution_quality_report.py --show
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "execution" / "live_trades.db"

# ---------------------------------------------------------------------------
# Venue classification
# ---------------------------------------------------------------------------
CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE", "MATIC"}

# Almgren-Chriss parameters (simplified market impact)
# Impact = eta * sigma * sqrt(Q / ADV)
# where eta ~= 0.142 (permanent), sigma = daily vol, Q = order size, ADV = avg daily vol
AC_ETA = 0.142
AC_SIGMA_DEFAULT = 0.015   # 1.5% daily vol default
AC_ADV_DEFAULT = 1_000_000  # $1M default ADV


def _venue(symbol: str) -> str:
    sym_upper = symbol.upper()
    if any(sym_upper == c or sym_upper.startswith(c) for c in CRYPTO_SYMBOLS):
        return "CRYPTO"
    return "EQUITY"


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def load_fills(conn: sqlite3.Connection) -> list[dict]:
    """Load fill records from DB. Falls back through multiple table schemas."""
    # Try fills/executions table first
    for table in ("fills", "executions", "live_trades"):
        try:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY timestamp").fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


def load_trades(conn: sqlite3.Connection) -> list[dict]:
    """Load closed trades (for fall-through when fills not available)."""
    for table in ("trade_pnl", "trades"):
        try:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY entry_time").fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


# ---------------------------------------------------------------------------
# Slippage calculation
# ---------------------------------------------------------------------------

def compute_slippage(fill_price: float, mid_price: float, side: str) -> float:
    """
    Slippage = (fill_price - mid_price) for buys (positive = adverse)
              (mid_price - fill_price) for sells
    Returns slippage as a fraction of mid price.
    """
    if mid_price <= 0:
        return 0.0
    raw = fill_price - mid_price
    if side.lower() in ("sell", "short"):
        raw = -raw
    return raw / mid_price


def enrich_fill(fill: dict) -> dict:
    """Compute slippage and other quality metrics for one fill."""
    d = dict(fill)

    # Try to get fill and mid prices from various column names
    fill_price = float(
        d.get("fill_price") or d.get("price") or d.get("entry_price") or 0.0
    )
    mid_price = float(
        d.get("mid_price") or d.get("mid") or fill_price  # fallback to fill
    )
    side = str(d.get("side") or "buy")
    qty = float(d.get("qty") or d.get("quantity") or 0.0)
    commission = float(d.get("commission") or d.get("fee") or 0.0)
    symbol = str(d.get("symbol") or "UNKNOWN")

    # Fill speed (bars from signal to fill)
    fill_speed = int(d.get("fill_bars") or d.get("latency_bars") or 1)

    # Slippage
    slippage_frac = compute_slippage(fill_price, mid_price, side)
    slippage_bps = slippage_frac * 10_000

    # Notional
    notional = fill_price * qty

    # Almgren-Chriss predicted impact
    sigma = float(d.get("daily_vol") or AC_SIGMA_DEFAULT)
    adv = float(d.get("adv") or AC_ADV_DEFAULT)
    ac_impact_frac = AC_ETA * sigma * math.sqrt(notional / (adv + 1e-9))
    ac_impact_bps = ac_impact_frac * 10_000

    d.update({
        "fill_price": fill_price,
        "mid_price": mid_price,
        "slippage_frac": round(slippage_frac, 8),
        "slippage_bps": round(slippage_bps, 4),
        "fill_speed_bars": fill_speed,
        "commission": commission,
        "notional": round(notional, 4),
        "venue": _venue(symbol),
        "ac_predicted_bps": round(ac_impact_bps, 4),
        "ac_error_bps": round(slippage_bps - ac_impact_bps, 4),
    })
    return d


def enrich_all_fills(fills: list[dict]) -> list[dict]:
    return [enrich_fill(f) for f in fills]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = _safe_mean(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(var, 0.0))


def per_venue_stats(enriched: list[dict]) -> dict[str, dict]:
    venues: dict[str, list[dict]] = defaultdict(list)
    for f in enriched:
        venues[f["venue"]].append(f)

    result: dict[str, dict] = {}
    for venue, fills in venues.items():
        slippages = [f["slippage_bps"] for f in fills]
        speeds = [f["fill_speed_bars"] for f in fills]
        commissions = [f["commission"] for f in fills]
        result[venue] = {
            "n_fills": len(fills),
            "avg_slippage_bps": round(_safe_mean(slippages), 3),
            "median_slippage_bps": round(
                sorted(slippages)[len(slippages) // 2] if slippages else 0.0, 3
            ),
            "std_slippage_bps": round(_safe_std(slippages), 3),
            "avg_fill_speed_bars": round(_safe_mean(speeds), 2),
            "total_commission": round(sum(commissions), 4),
            "avg_commission": round(_safe_mean(commissions), 6),
        }
    return result


def size_vs_slippage(enriched: list[dict]) -> list[tuple[float, float]]:
    """Return (notional, slippage_bps) pairs for scatter analysis."""
    return [
        (f["notional"], f["slippage_bps"])
        for f in enriched
        if f["notional"] > 0
    ]


def ac_validation_stats(enriched: list[dict]) -> dict:
    """Compare actual slippage to Almgren-Chriss predictions."""
    errors = [f["ac_error_bps"] for f in enriched]
    predicted = [f["ac_predicted_bps"] for f in enriched]
    actual = [f["slippage_bps"] for f in enriched]

    if not errors:
        return {}

    mae = _safe_mean([abs(e) for e in errors])
    bias = _safe_mean(errors)  # positive = AC underestimates
    r2: float = 0.0
    if len(predicted) > 1:
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        mean_a = _safe_mean(actual)
        ss_tot = sum((a - mean_a) ** 2 for a in actual)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "n_fills": len(errors),
        "mae_bps": round(mae, 3),
        "bias_bps": round(bias, 3),
        "r2": round(r2, 4),
        "interpretation": (
            "AC underestimates impact" if bias > 1 else
            "AC overestimates impact" if bias < -1 else
            "AC model well-calibrated"
        ),
    }


def fill_speed_analysis(enriched: list[dict]) -> dict:
    """Histogram analysis of fill speed."""
    speeds = [f["fill_speed_bars"] for f in enriched]
    if not speeds:
        return {}
    instant = sum(1 for s in speeds if s <= 1)
    fast = sum(1 for s in speeds if 1 < s <= 3)
    slow = sum(1 for s in speeds if s > 3)
    return {
        "instant_pct": round(instant / len(speeds) * 100, 1),
        "fast_pct": round(fast / len(speeds) * 100, 1),
        "slow_pct": round(slow / len(speeds) * 100, 1),
        "avg_speed": round(_safe_mean(speeds), 2),
        "max_speed": max(speeds),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_slippage_distribution(enriched: list[dict], show: bool = False) -> Any:
    if not HAS_MPL or not enriched:
        return None

    venues = list({f["venue"] for f in enriched})
    fig, axes = plt.subplots(1, len(venues), figsize=(6 * len(venues), 5), squeeze=False)

    for idx, venue in enumerate(venues):
        slippages = [f["slippage_bps"] for f in enriched if f["venue"] == venue]
        ax = axes[0][idx]
        ax.hist(slippages, bins=30, color="steelblue", edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        mean_s = _safe_mean(slippages)
        ax.axvline(mean_s, color="green", linestyle="-", linewidth=1,
                   label=f"Mean: {mean_s:.2f}bps")
        ax.set_title(f"Slippage Distribution -- {venue}")
        ax.set_xlabel("Slippage (bps)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.suptitle("Execution Slippage by Venue", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_size_vs_slippage(pairs: list[tuple[float, float]], show: bool = False) -> Any:
    if not HAS_MPL or not pairs:
        return None

    notionals = [p[0] for p in pairs]
    slippages = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(notionals, slippages, alpha=0.5, s=20, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Notional ($)")
    ax.set_ylabel("Slippage (bps)")
    ax.set_title("Order Size vs Slippage")

    # Fit trend line
    if len(notionals) > 3:
        try:
            import math as _m
            log_n = [_m.log(max(n, 1)) for n in notionals]
            mx = sum(log_n) / len(log_n)
            my = sum(slippages) / len(slippages)
            slope = sum((x - mx) * (y - my) for x, y in zip(log_n, slippages))
            slope /= sum((x - mx) ** 2 for x in log_n) + 1e-9
            intercept = my - slope * mx
            x_range = [min(notionals), max(notionals)]
            y_range = [slope * _m.log(max(x, 1)) + intercept for x in x_range]
            ax.plot(x_range, y_range, color="red", linewidth=1.5,
                    linestyle="--", label="Log trend")
            ax.legend(fontsize=8)
        except Exception:
            pass

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_ac_validation(enriched: list[dict], show: bool = False) -> Any:
    if not HAS_MPL or not enriched:
        return None

    predicted = [f["ac_predicted_bps"] for f in enriched]
    actual = [f["slippage_bps"] for f in enriched]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predicted, actual, alpha=0.5, s=18, color="steelblue")
    lo = min(min(predicted), min(actual), 0) - 0.5
    hi = max(max(predicted), max(actual)) + 0.5
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1,
            label="Perfect prediction")
    ax.set_xlabel("AC Predicted Impact (bps)")
    ax.set_ylabel("Actual Slippage (bps)")
    ax.set_title("Almgren-Chriss Model Validation")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_fill_speed_hist(enriched: list[dict], show: bool = False) -> Any:
    if not HAS_MPL or not enriched:
        return None

    speeds = [f["fill_speed_bars"] for f in enriched]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(speeds, bins=range(1, max(speeds) + 2), color="darkorange",
            edgecolor="black", linewidth=0.5, align="left", rwidth=0.8)
    ax.set_xlabel("Fill Speed (bars from signal)")
    ax.set_ylabel("Frequency")
    ax.set_title("Fill Speed Distribution")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: Any) -> str:
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_html_report(
    enriched: list[dict],
    venue_stats: dict[str, dict],
    ac_stats: dict,
    speed_stats: dict,
    figs: list[Any],
) -> str:
    venue_rows = ""
    for venue, stats in venue_stats.items():
        venue_rows += (
            f"<tr><td>{venue}</td>"
            f"<td>{stats['n_fills']}</td>"
            f"<td>{stats['avg_slippage_bps']:.3f}</td>"
            f"<td>{stats['median_slippage_bps']:.3f}</td>"
            f"<td>{stats['std_slippage_bps']:.3f}</td>"
            f"<td>{stats['avg_fill_speed_bars']:.2f}</td>"
            f"<td>{stats['total_commission']:.4f}</td></tr>"
        )

    worst = sorted(enriched, key=lambda f: -abs(f.get("slippage_bps", 0)))[:15]
    worst_rows = ""
    for f in worst:
        color = "#f8d7da" if f["slippage_bps"] > 5 else "#ffffff"
        worst_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{f.get('symbol', '?')}</td>"
            f"<td>{f.get('venue', '?')}</td>"
            f"<td>{f.get('fill_price', 0):.4f}</td>"
            f"<td>{f.get('mid_price', 0):.4f}</td>"
            f"<td>{f['slippage_bps']:.3f}</td>"
            f"<td>{f.get('notional', 0):.2f}</td>"
            f"<td>{f.get('fill_speed_bars', 0)}</td>"
            f"<td>{f.get('commission', 0):.6f}</td></tr>"
        )

    imgs_html = ""
    for fig in figs:
        if fig is not None:
            try:
                b64 = _fig_to_base64(fig)
                imgs_html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:16px 0;">'
            except Exception:
                pass

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Execution Quality Report</title>
<style>
  body {{font-family: Arial, sans-serif; margin: 24px; color: #222;}}
  h1 {{color: #1a3c6b;}} h2 {{color: #2c6e9b; border-bottom: 2px solid #eee; padding-bottom:4px;}}
  table {{border-collapse: collapse; width:100%; margin-bottom:20px;}}
  th {{background:#2c6e9b; color:white; padding:6px 10px; text-align:left;}}
  td {{padding:5px 10px; border-bottom:1px solid #ddd;}}
  .card {{background:#f0f4f8; padding:12px; border-radius:6px; margin-bottom:12px;}}
</style>
</head><body>
<h1>Execution Quality Report</h1>
<p>Total fills analysed: {len(enriched)}</p>

<h2>Charts</h2>
{imgs_html}

<h2>By Venue</h2>
<table>
  <tr><th>Venue</th><th>N Fills</th><th>Avg Slip (bps)</th><th>Median Slip</th>
      <th>Std Slip</th><th>Avg Fill Speed</th><th>Total Commission</th></tr>
{venue_rows}
</table>

<h2>Almgren-Chriss Model Validation</h2>
<div class="card">
  N: {ac_stats.get('n_fills', 0)} |
  MAE: {ac_stats.get('mae_bps', 'n/a')} bps |
  Bias: {ac_stats.get('bias_bps', 'n/a')} bps |
  R2: {ac_stats.get('r2', 'n/a')} |
  <b>{ac_stats.get('interpretation', '')}</b>
</div>

<h2>Fill Speed</h2>
<div class="card">
  Instant (1 bar): {speed_stats.get('instant_pct', 0):.1f}% |
  Fast (2-3 bars): {speed_stats.get('fast_pct', 0):.1f}% |
  Slow (4+ bars): {speed_stats.get('slow_pct', 0):.1f}% |
  Avg: {speed_stats.get('avg_speed', 0):.2f} bars |
  Max: {speed_stats.get('max_speed', 0)} bars
</div>

<h2>Worst Fills (by absolute slippage)</h2>
<table>
  <tr><th>Symbol</th><th>Venue</th><th>Fill</th><th>Mid</th><th>Slip (bps)</th>
      <th>Notional</th><th>Fill Speed</th><th>Commission</th></tr>
{worst_rows}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Text fallback
# ---------------------------------------------------------------------------

def print_text_report(venue_stats: dict[str, dict],
                       ac_stats: dict,
                       speed_stats: dict) -> None:
    print("\n=== EXECUTION QUALITY REPORT ===\n")
    print("--- By Venue ---")
    fmt = "{:<10} {:>8} {:>12} {:>12} {:>14} {:>14}"
    print(fmt.format("Venue", "N", "Avg Slip bps", "Median Slip", "Avg Fill Speed", "Total Comm"))
    print("-" * 72)
    for venue, s in venue_stats.items():
        print(fmt.format(
            venue, s["n_fills"],
            f"{s['avg_slippage_bps']:.3f}",
            f"{s['median_slippage_bps']:.3f}",
            f"{s['avg_fill_speed_bars']:.2f}",
            f"{s['total_commission']:.4f}",
        ))

    print("\n--- Almgren-Chriss Validation ---")
    for k, v in ac_stats.items():
        if k != "interpretation":
            print(f"  {k}: {v}")
    print(f"  {ac_stats.get('interpretation', '')}")

    print("\n--- Fill Speed ---")
    print(f"  Instant: {speed_stats.get('instant_pct', 0):.1f}%  "
          f"Fast: {speed_stats.get('fast_pct', 0):.1f}%  "
          f"Slow: {speed_stats.get('slow_pct', 0):.1f}%  "
          f"Avg: {speed_stats.get('avg_speed', 0):.2f} bars")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _synthetic_fills(n: int = 200) -> list[dict]:
    import random
    rng = random.Random(55)
    symbols = ["BTC", "ETH", "SOL", "AAPL", "SPY", "QQQ"]
    fills: list[dict] = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ts = base + timedelta(hours=i * 4)
        sym = rng.choice(symbols)
        side = rng.choice(["buy", "sell"])
        qty = round(rng.uniform(0.01, 10.0), 4)
        mid = rng.uniform(50, 50000)
        slip_frac = rng.gauss(0.0002, 0.001)
        fill = mid * (1 + slip_frac if side == "buy" else 1 - slip_frac)
        fills.append({
            "id": i + 1,
            "symbol": sym,
            "side": side,
            "qty": qty,
            "fill_price": round(fill, 4),
            "mid_price": round(mid, 4),
            "price": round(fill, 4),
            "commission": round(qty * fill * 0.0001, 6),
            "fill_bars": rng.randint(1, 5),
            "timestamp": ts.isoformat(),
        })
    return fills


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze execution quality from fills"
    )
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to SQLite trades/fills DB")
    p.add_argument("--output", default=None,
                   help="Output HTML report path")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    db_path = Path(args.db)
    conn = _connect(db_path)

    raw_fills: list[dict] = []
    if conn:
        raw_fills = load_fills(conn)
        if not raw_fills:
            # Fall back to trades table with synthetic slippage
            raw_trades = load_trades(conn)
            for t in raw_trades:
                t.setdefault("mid_price", float(t.get("entry_price") or 0) * 1.0002)
                t.setdefault("fill_bars", 1)
            raw_fills = raw_trades
        conn.close()

    if not raw_fills:
        print("[exec_quality] No fills/trades found -- using synthetic demo",
              file=sys.stderr)
        raw_fills = _synthetic_fills(200)

    print(f"[exec_quality] Loaded {len(raw_fills)} fills")
    enriched = enrich_all_fills(raw_fills)

    venue_stats = per_venue_stats(enriched)
    sv_pairs = size_vs_slippage(enriched)
    ac_stats = ac_validation_stats(enriched)
    speed_stats = fill_speed_analysis(enriched)

    print(f"[exec_quality] Venues: {list(venue_stats.keys())}")
    for venue, s in venue_stats.items():
        print(f"  {venue}: avg_slip={s['avg_slippage_bps']:.3f}bps  "
              f"n={s['n_fills']}")

    figs: list[Any] = []
    if HAS_MPL:
        figs.append(plot_slippage_distribution(enriched))
        figs.append(plot_size_vs_slippage(sv_pairs))
        figs.append(plot_ac_validation(enriched))
        figs.append(plot_fill_speed_hist(enriched))
        if args.show:
            plt.show()

    if args.output:
        print(f"[exec_quality] Writing report to {args.output} ...")
        html = build_html_report(enriched, venue_stats, ac_stats, speed_stats, figs)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"[exec_quality] Report saved: {args.output}")
    else:
        print_text_report(venue_stats, ac_stats, speed_stats)

    for fig in figs:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
