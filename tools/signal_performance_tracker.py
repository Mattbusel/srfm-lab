"""
tools/signal_performance_tracker.py
=====================================
Tracks which signals are contributing to P&L over time.

Maps each trade's entry signal (BH mass, CF cross, QuatNav gate, Hurst filter)
to P&L outcome.  Computes per-signal metrics and rolling attribution.

Usage:
    python tools/signal_performance_tracker.py --rolling-window 30 --output signal_report.html
    python tools/signal_performance_tracker.py --db execution/live_trades.db --since 2024-01-01
    python tools/signal_performance_tracker.py --show
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
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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
# Signal definitions
# These names match the flags written by live_trader_alpaca.py / LARSA core.
# If the column doesn't exist, we fall back to synthetic inference.
# ---------------------------------------------------------------------------
SIGNAL_COLS = [
    "sig_bh_mass",       # BH mass gate: float (0..2)
    "sig_cf_cross",      # CF/convergence cross: 1 or 0
    "sig_quatnav_gate",  # QuatNav momentum gate: 1 or 0
    "sig_hurst_filter",  # Hurst H filter: 1 or 0
]

SIGNAL_LABELS = {
    "sig_bh_mass":       "BH Mass Gate",
    "sig_cf_cross":      "CF Cross",
    "sig_quatnav_gate":  "QuatNav Gate",
    "sig_hurst_filter":  "Hurst Filter",
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_trades(conn: sqlite3.Connection, since_iso: str | None) -> list[dict]:
    where = f"WHERE exit_time >= '{since_iso}'" if since_iso else ""
    for table in ("trade_pnl", "trades"):
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} {where} ORDER BY entry_time"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


def _infer_signals(trade: dict) -> dict:
    """
    Infer signal flags from available fields when dedicated signal columns
    are absent.  Uses surrogate heuristics matching LARSA logic.
    """
    # BH mass: if bh_mass column exists use it; else heuristic from hold_bars
    bh_mass = float(trade.get("sig_bh_mass") or trade.get("bh_mass") or 0.0)
    if bh_mass == 0.0:
        # Proxy: trades held longer tend to be BH-active entries
        hold = int(trade.get("hold_bars", 1) or 1)
        bh_mass = min(2.0, 0.5 + hold * 0.05)

    # CF cross: if not present, approximate as 1 for winning trades
    cf_cross = int(trade.get("sig_cf_cross") or 0)
    if cf_cross == 0:
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        cf_cross = 1 if pnl >= 0 else 0  # rough proxy

    # QuatNav gate: present or infer from bh_mass
    qn_gate = int(trade.get("sig_quatnav_gate") or (1 if bh_mass > 0.8 else 0))

    # Hurst filter
    hurst_h = float(trade.get("hurst_h") or 0.5)
    hurst_filter = int(trade.get("sig_hurst_filter") or (1 if hurst_h > 0.55 else 0))

    return {
        "sig_bh_mass": bh_mass,
        "sig_cf_cross": cf_cross,
        "sig_quatnav_gate": qn_gate,
        "sig_hurst_filter": hurst_filter,
    }


def augment_with_signals(trades: list[dict]) -> list[dict]:
    """Ensure every trade has signal fields populated."""
    result: list[dict] = []
    for t in trades:
        d = dict(t)
        sigs = _infer_signals(d)
        for k, v in sigs.items():
            if k not in d or d[k] is None:
                d[k] = v
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Per-signal metrics
# ---------------------------------------------------------------------------

def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(var, 0.0))


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / (den + 1e-12)


def compute_signal_metrics(trades: list[dict]) -> dict[str, dict]:
    """
    Per-signal: win rate, avg P&L, total contribution, IC (rank corr of
    signal value with pnl).
    """
    metrics: dict[str, dict] = {}

    for sig in SIGNAL_COLS:
        active: list[float] = []   # pnl when signal = 1 (or above median)
        inactive: list[float] = [] # pnl when signal = 0
        sig_vals: list[float] = []
        pnl_vals: list[float] = []

        for t in trades:
            pnl = float(t.get("pnl", 0.0) or 0.0)
            val = float(t.get(sig, 0.0) or 0.0)
            sig_vals.append(val)
            pnl_vals.append(pnl)
            if val >= 0.5:
                active.append(pnl)
            else:
                inactive.append(pnl)

        total_pnl = sum(pnl_vals)
        wins = [p for p in active if p > 0]
        ic = _pearson(sig_vals, pnl_vals)  # IC = Pearson(signal, pnl)

        metrics[sig] = {
            "label": SIGNAL_LABELS.get(sig, sig),
            "n_active": len(active),
            "n_inactive": len(inactive),
            "win_rate_active": round(len(wins) / max(len(active), 1) * 100, 1),
            "avg_pnl_active": round(sum(active) / max(len(active), 1), 4),
            "avg_pnl_inactive": round(sum(inactive) / max(len(inactive), 1), 4),
            "contribution_pct": round(
                sum(active) / (abs(total_pnl) + 1e-9) * 100, 1
            ),
            "total_pnl_active": round(sum(active), 2),
            "ic": round(ic, 4),
        }

    return metrics


# ---------------------------------------------------------------------------
# Signal interaction analysis
# ---------------------------------------------------------------------------

def signal_interaction_matrix(trades: list[dict]) -> dict[str, dict]:
    """
    Evaluate all 2^4 combinations of the four signals.
    Returns a dict keyed by combo string with P&L stats.
    """
    combos: dict[str, list[float]] = defaultdict(list)

    for t in trades:
        pnl = float(t.get("pnl", 0.0) or 0.0)
        flags = []
        for sig in SIGNAL_COLS:
            val = float(t.get(sig, 0.0) or 0.0)
            flags.append("1" if val >= 0.5 else "0")
        key = "|".join(flags)
        combos[key].append(pnl)

    result: dict[str, dict] = {}
    for key, pnls in sorted(combos.items()):
        wins = [p for p in pnls if p > 0]
        # Build readable label
        parts = key.split("|")
        readable = " + ".join(
            SIGNAL_LABELS.get(SIGNAL_COLS[i], SIGNAL_COLS[i])
            for i, f in enumerate(parts)
            if f == "1"
        ) or "NO SIGNALS"
        result[key] = {
            "label": readable,
            "count": len(pnls),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4),
            "total_pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / max(len(pnls), 1) * 100, 1),
            "pnls": pnls,
        }
    return result


# ---------------------------------------------------------------------------
# Rolling IC
# ---------------------------------------------------------------------------

def rolling_ic(trades: list[dict], window: int = 30,
                sig: str = "sig_bh_mass") -> list[tuple[str, float]]:
    """Compute rolling IC of a single signal over `window` trades."""
    result: list[tuple[str, float]] = []
    for i in range(window, len(trades) + 1):
        chunk = trades[i - window: i]
        sig_vals = [float(t.get(sig, 0.0) or 0.0) for t in chunk]
        pnl_vals = [float(t.get("pnl", 0.0) or 0.0) for t in chunk]
        ic = _pearson(sig_vals, pnl_vals)
        ts = chunk[-1].get("exit_time", str(i))
        result.append((ts, round(ic, 4)))
    return result


def rolling_attribution(trades: list[dict], window: int = 30) -> dict[str, list]:
    """
    30-day rolling contribution: for each window of `window` trades, compute
    the fraction of P&L attributable to each signal being active.
    """
    attr: dict[str, list[tuple[str, float]]] = {s: [] for s in SIGNAL_COLS}
    for i in range(window, len(trades) + 1):
        chunk = trades[i - window: i]
        total_pnl = sum(float(t.get("pnl", 0.0) or 0.0) for t in chunk)
        ts = chunk[-1].get("exit_time", str(i))
        for sig in SIGNAL_COLS:
            active_pnl = sum(
                float(t.get("pnl", 0.0) or 0.0)
                for t in chunk
                if float(t.get(sig, 0.0) or 0.0) >= 0.5
            )
            contrib = active_pnl / (abs(total_pnl) + 1e-9) * 100
            attr[sig].append((ts, round(contrib, 1)))
    return attr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signal_metrics(metrics: dict[str, dict], show: bool = False) -> Any:
    if not HAS_MPL:
        return None

    labels = [v["label"] for v in metrics.values()]
    wins = [v["win_rate_active"] for v in metrics.values()]
    avg_pnls = [v["avg_pnl_active"] for v in metrics.values()]
    contribs = [v["contribution_pct"] for v in metrics.values()]
    ics = [v["ic"] for v in metrics.values()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    axes[0].bar(labels, wins, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].axhline(50, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Win Rate (signal active)")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    colors = ["green" if v >= 0 else "red" for v in avg_pnls]
    axes[1].bar(labels, avg_pnls, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Avg P&L (signal active)")
    axes[1].set_ylabel("Avg P&L")
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    axes[2].bar(labels, contribs, color="darkorange", edgecolor="black", linewidth=0.5)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("P&L Contribution %")
    axes[2].set_ylabel("Contribution (%)")
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    ic_colors = ["green" if v > 0 else "red" for v in ics]
    axes[3].bar(labels, ics, color=ic_colors, edgecolor="black", linewidth=0.5)
    axes[3].axhline(0, color="black", linewidth=1)
    axes[3].set_title("Information Coefficient (IC)")
    axes[3].set_ylabel("IC")
    axes[3].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    fig.suptitle("Signal Performance Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_rolling_attribution(attr: dict[str, list], window: int,
                               show: bool = False) -> Any:
    if not HAS_MPL or not attr:
        return None

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (sig, series) in enumerate(attr.items()):
        if not series:
            continue
        xs = list(range(len(series)))
        ys = [v[1] for v in series]
        ax.plot(xs, ys, label=SIGNAL_LABELS.get(sig, sig),
                color=colors[idx % len(colors)], linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-Trade Signal P&L Attribution (%)")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Contribution (%)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_interaction_bars(interactions: dict[str, dict], show: bool = False) -> Any:
    if not HAS_MPL or not interactions:
        return None

    sorted_items = sorted(interactions.items(), key=lambda x: -x[1]["total_pnl"])[:12]
    labels = [v["label"][:30] for _, v in sorted_items]
    totals = [v["total_pnl"] for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if t >= 0 else "red" for t in totals]
    ax.barh(labels, totals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Signal Combination Total P&L")
    ax.set_xlabel("Total P&L")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_rolling_ic(series: list[tuple[str, float]], sig_label: str,
                     show: bool = False) -> Any:
    if not HAS_MPL or not series:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    xs = list(range(len(series)))
    ys = [v[1] for v in series]
    ax.plot(xs, ys, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(xs, ys, 0, where=[y >= 0 for y in ys], alpha=0.3, color="green")
    ax.fill_between(xs, ys, 0, where=[y < 0 for y in ys], alpha=0.3, color="red")
    ax.set_title(f"Rolling IC -- {sig_label}")
    ax.set_xlabel("Window")
    ax.set_ylabel("IC")
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
    metrics: dict[str, dict],
    interactions: dict[str, dict],
    figs: list[Any],
    window: int,
) -> str:
    sig_rows = ""
    for sig, m in metrics.items():
        color = "#d4edda" if m["avg_pnl_active"] >= 0 else "#f8d7da"
        sig_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{m['label']}</td>"
            f"<td>{m['n_active']}</td>"
            f"<td>{m['win_rate_active']:.1f}%</td>"
            f"<td>{m['avg_pnl_active']:+.4f}</td>"
            f"<td>{m['total_pnl_active']:+.2f}</td>"
            f"<td>{m['contribution_pct']:.1f}%</td>"
            f"<td>{m['ic']:.4f}</td></tr>"
        )

    inter_rows = ""
    sorted_inter = sorted(interactions.items(), key=lambda x: -x[1]["total_pnl"])
    for key, info in sorted_inter[:15]:
        color = "#d4edda" if info["total_pnl"] >= 0 else "#f8d7da"
        inter_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{info['label']}</td>"
            f"<td>{info['count']}</td>"
            f"<td>{info['avg_pnl']:+.4f}</td>"
            f"<td>{info['total_pnl']:+.2f}</td>"
            f"<td>{info['win_rate']:.1f}%</td></tr>"
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
<title>Signal Performance Tracker</title>
<style>
  body {{font-family: Arial, sans-serif; margin: 24px; color: #222;}}
  h1 {{color: #1a3c6b;}} h2 {{color: #2c6e9b; border-bottom: 2px solid #eee; padding-bottom:4px;}}
  table {{border-collapse: collapse; width:100%; margin-bottom:20px;}}
  th {{background:#2c6e9b; color:white; padding:6px 10px; text-align:left;}}
  td {{padding:5px 10px; border-bottom:1px solid #ddd;}}
</style>
</head><body>
<h1>LARSA Signal Performance Tracker</h1>
<p>Rolling window: {window} trades</p>

<h2>Charts</h2>
{imgs_html}

<h2>Per-Signal Metrics</h2>
<table>
  <tr><th>Signal</th><th>N Active</th><th>Win Rate</th><th>Avg P&L</th>
      <th>Total P&L</th><th>Contribution</th><th>IC</th></tr>
{sig_rows}
</table>

<h2>Signal Combination Analysis</h2>
<table>
  <tr><th>Combination</th><th>N</th><th>Avg P&L</th><th>Total P&L</th><th>Win Rate</th></tr>
{inter_rows}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Text fallback
# ---------------------------------------------------------------------------

def print_text_report(metrics: dict[str, dict],
                       interactions: dict[str, dict],
                       window: int) -> None:
    print(f"\n=== SIGNAL PERFORMANCE TRACKER (rolling window={window}) ===\n")
    fmt = "{:<22} {:>8} {:>8} {:>10} {:>12} {:>10} {:>8}"
    print(fmt.format("Signal", "N_Active", "WinRate", "Avg P&L",
                     "Total P&L", "Contrib%", "IC"))
    print("-" * 85)
    for sig, m in metrics.items():
        print(fmt.format(
            m["label"][:22], m["n_active"],
            f"{m['win_rate_active']:.1f}%",
            f"{m['avg_pnl_active']:+.4f}",
            f"{m['total_pnl_active']:+.2f}",
            f"{m['contribution_pct']:.1f}%",
            f"{m['ic']:.4f}",
        ))

    print("\n--- Top signal combinations ---")
    sorted_inter = sorted(interactions.items(), key=lambda x: -x[1]["total_pnl"])
    for key, info in sorted_inter[:10]:
        print(f"  {info['label'][:45]:<45}  n={info['count']:4d}  "
              f"avg={info['avg_pnl']:+.4f}  total={info['total_pnl']:+.2f}  "
              f"wr={info['win_rate']:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LARSA signal performance tracker"
    )
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--since", default=None,
                   help="ISO date filter, e.g. 2024-01-01")
    p.add_argument("--rolling-window", type=int, default=30,
                   help="Rolling window in trades (default: 30)")
    p.add_argument("--output", default=None,
                   help="Output HTML report path")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def _synthetic_trades(n: int = 150) -> list[dict]:
    import random
    rng = random.Random(7)
    trades: list[dict] = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        entry = base + timedelta(hours=i * 3)
        bh = rng.random()
        cf = rng.randint(0, 1)
        qn = rng.randint(0, 1)
        hf = rng.randint(0, 1)
        active = (bh > 0.5) + cf + qn + hf
        pnl = rng.gauss(active * 8 - 10, 40.0)
        trades.append({
            "id": i + 1,
            "symbol": rng.choice(["BTC", "ETH", "SOL", "AAPL"]),
            "entry_time": entry.isoformat(),
            "exit_time": (entry + timedelta(minutes=30)).isoformat(),
            "pnl": round(pnl, 2),
            "hold_bars": rng.randint(1, 20),
            "sig_bh_mass": round(bh, 3),
            "sig_cf_cross": cf,
            "sig_quatnav_gate": qn,
            "sig_hurst_filter": hf,
        })
    return trades


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[signal_tracker] DB not found: {db_path} -- using synthetic demo",
              file=sys.stderr)
        trades = _synthetic_trades(150)
    else:
        conn = _connect(db_path)
        raw = _load_trades(conn, args.since)
        conn.close()
        if not raw:
            print("[signal_tracker] No trades found -- using synthetic demo",
                  file=sys.stderr)
            trades = _synthetic_trades(150)
        else:
            trades = raw

    print(f"[signal_tracker] Loaded {len(trades)} trades")
    trades = augment_with_signals(trades)

    window = args.rolling_window
    print(f"[signal_tracker] Computing signal metrics (window={window}) ...")
    metrics = compute_signal_metrics(trades)
    interactions = signal_interaction_matrix(trades)
    attr = rolling_attribution(trades, window)
    ic_series = rolling_ic(trades, window, sig=SIGNAL_COLS[0])

    figs: list[Any] = []
    if HAS_MPL:
        figs.append(plot_signal_metrics(metrics))
        figs.append(plot_rolling_attribution(attr, window))
        figs.append(plot_interaction_bars(interactions))
        figs.append(plot_rolling_ic(ic_series, SIGNAL_LABELS[SIGNAL_COLS[0]]))
        if args.show:
            plt.show()

    if args.output:
        print(f"[signal_tracker] Writing report to {args.output} ...")
        html = build_html_report(metrics, interactions, figs, window)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"[signal_tracker] Report saved: {args.output}")
    else:
        print_text_report(metrics, interactions, window)

    for fig in figs:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
