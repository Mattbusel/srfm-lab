"""
tools/regime_diagnostic.py
===========================
Comprehensive regime diagnostic tool -- analyzes the relationship between
regime states and LARSA strategy performance.

Usage:
    python tools/regime_diagnostic.py --since 2024-01-01 --output regime_report.html
    python tools/regime_diagnostic.py --db path/to/custom.db --since 2024-06-01
    python tools/regime_diagnostic.py --show  # display plots interactively
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy imports
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
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "execution" / "live_trades.db"
_BARS_DB = _REPO / "data" / "bars.db"

# ---------------------------------------------------------------------------
# Regime label helpers
# ---------------------------------------------------------------------------

BH_LABELS = {True: "BH_ACTIVE", False: "BH_QUIET"}
HURST_LABELS = {
    "trending": "TRENDING",
    "random": "RANDOM",
    "mean_rev": "MEAN_REV",
}
VOL_LABELS = {
    "high": "VOL_HIGH",
    "normal": "VOL_NORM",
    "low": "VOL_LOW",
}


def _classify_hurst(h: float) -> str:
    if h > 0.6:
        return "trending"
    if h < 0.4:
        return "mean_rev"
    return "random"


def _classify_vol(garch_vol: float, vol_p25: float, vol_p75: float) -> str:
    if garch_vol > vol_p75:
        return "high"
    if garch_vol < vol_p25:
        return "low"
    return "normal"


def _hurst_rs(prices: list[float], min_len: int = 20) -> float:
    """Estimate Hurst exponent via rescaled range (R/S) analysis."""
    n = len(prices)
    if n < min_len:
        return 0.5
    if HAS_NUMPY:
        arr = np.array(prices, dtype=float)
        log_n: list[float] = []
        log_rs: list[float] = []
        for seg in range(2, min(n // 4 + 1, 16)):
            chunks = n // seg
            if chunks < 4:
                continue
            rs_vals: list[float] = []
            for i in range(seg):
                sub = arr[i * chunks: (i + 1) * chunks]
                mean_s = float(np.mean(sub))
                dev = np.cumsum(sub - mean_s)
                r = float(np.max(dev) - np.min(dev))
                s = float(np.std(sub, ddof=1))
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                log_n.append(math.log(chunks))
                log_rs.append(math.log(sum(rs_vals) / len(rs_vals)))
        if len(log_n) > 1:
            x_arr = np.array(log_n)
            y_arr = np.array(log_rs)
            slope = float(np.polyfit(x_arr, y_arr, 1)[0])
            return max(0.0, min(1.0, slope))
    return 0.5


def _garch_vol_simple(returns: list[float], omega: float = 1e-6,
                       alpha: float = 0.09, beta: float = 0.90) -> float:
    """Simple GARCH(1,1) variance estimate, returns annualised vol."""
    if not returns:
        return 0.0
    var = omega / (1 - alpha - beta + 1e-9)
    for r in returns:
        var = omega + alpha * r ** 2 + beta * var
    return math.sqrt(max(var, 0.0)) * math.sqrt(252 * 26)  # 15m bars / day ~26


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_trades(conn: sqlite3.Connection, since_iso: str | None) -> list[dict]:
    where = f"WHERE exit_time >= '{since_iso}'" if since_iso else ""
    try:
        rows = conn.execute(
            f"SELECT * FROM trade_pnl {where} ORDER BY entry_time"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        # Fallback: trades table instead of trade_pnl
        try:
            rows = conn.execute(
                f"""SELECT id, symbol, entry_time, exit_time,
                           entry_price, exit_price, qty,
                           (exit_price - entry_price) * qty AS pnl,
                           COALESCE(hold_bars, 0) AS hold_bars
                    FROM trades {where} ORDER BY entry_time"""
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []


def _load_bars_for_symbol(bars_conn: sqlite3.Connection | None,
                           symbol: str, ts: str,
                           lookback: int = 100) -> list[dict]:
    """Load up to `lookback` bars ending at or before `ts` for a symbol."""
    if bars_conn is None:
        return []
    try:
        rows = bars_conn.execute(
            """SELECT ts, open, high, low, close, volume
               FROM bars
               WHERE symbol = ? AND ts <= ?
               ORDER BY ts DESC LIMIT ?""",
            (symbol, ts, lookback)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    except sqlite3.OperationalError:
        return []


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_trade(trade: dict, bars: list[dict],
                 vol_p25: float = 0.01, vol_p75: float = 0.03) -> dict:
    """Attach regime features to a trade dict."""
    if len(bars) < 20:
        trade.update({
            "bh_mass": 0.0,
            "bh_active": False,
            "hurst_h": 0.5,
            "hurst_regime": "random",
            "garch_vol": 0.0,
            "vol_regime": "normal",
            "nav_omega": 0.0,
        })
        return trade

    closes = [b["close"] for b in bars if b.get("close")]
    if not closes:
        closes = [1.0]

    # Hurst
    h = _hurst_rs(closes)

    # GARCH vol from returns
    rets = [
        (closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-9)
        for i in range(1, len(closes))
    ]
    gvol = _garch_vol_simple(rets)

    # BH mass -- simple surrogate: ratio of small moves
    if len(rets) >= 10:
        small = sum(1 for r in rets[-20:] if abs(r) < 0.001)
        bh_mass = small / min(20, len(rets))
    else:
        bh_mass = 0.0
    bh_active = bh_mass > 0.5

    # nav_omega: scalar measure of momentum alignment (simplified)
    if len(closes) >= 10:
        nav_omega = float((closes[-1] - closes[-10]) / (closes[-10] + 1e-9))
    else:
        nav_omega = 0.0

    trade.update({
        "bh_mass": round(bh_mass, 4),
        "bh_active": bh_active,
        "hurst_h": round(h, 4),
        "hurst_regime": _classify_hurst(h),
        "garch_vol": round(gvol, 6),
        "vol_regime": _classify_vol(gvol, vol_p25, vol_p75),
        "nav_omega": round(nav_omega, 6),
    })
    return trade


def enrich_all_trades(trades: list[dict], bars_db_path: Path) -> list[dict]:
    """Enrich every trade with regime state at entry."""
    bars_conn = None
    if bars_db_path.exists():
        try:
            bars_conn = _connect(bars_db_path)
        except Exception:
            pass

    # Compute global vol percentiles from first pass if we have bar data
    all_vols: list[float] = []
    if bars_conn:
        for t in trades[:200]:  # sample to set thresholds
            sym = t.get("symbol", "BTC")
            ts = t.get("entry_time", "")
            bars = _load_bars_for_symbol(bars_conn, sym, ts)
            closes = [b["close"] for b in bars if b.get("close")]
            rets = [
                (closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-9)
                for i in range(1, len(closes))
            ]
            all_vols.append(_garch_vol_simple(rets))

    if len(all_vols) >= 4:
        sorted_v = sorted(all_vols)
        p25 = sorted_v[len(sorted_v) // 4]
        p75 = sorted_v[3 * len(sorted_v) // 4]
    else:
        p25, p75 = 0.01, 0.03

    enriched: list[dict] = []
    for t in trades:
        sym = t.get("symbol", "BTC")
        ts = t.get("entry_time", "")
        bars = _load_bars_for_symbol(bars_conn, sym, ts)
        enriched.append(enrich_trade(dict(t), bars, p25, p75))

    if bars_conn:
        bars_conn.close()
    return enriched


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(variance, 0.0))


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    std = _safe_std(pnls)
    return mean / (std + 1e-9) * math.sqrt(252)


def pnl_by_regime(trades: list[dict]) -> dict[str, dict]:
    """Group P&L stats by regime label."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        pnl = t.get("pnl", 0.0) or 0.0
        bh = "BH_ACTIVE" if t.get("bh_active") else "BH_QUIET"
        hr = HURST_LABELS.get(t.get("hurst_regime", "random"), "RANDOM")
        vr = VOL_LABELS.get(t.get("vol_regime", "normal"), "VOL_NORM")
        groups[bh].append(pnl)
        groups[hr].append(pnl)
        groups[vr].append(pnl)
        combo = f"{bh}|{hr}|{vr}"
        groups[combo].append(pnl)

    result: dict[str, dict] = {}
    for label, pnls in sorted(groups.items()):
        wins = [p for p in pnls if p > 0]
        result[label] = {
            "count": len(pnls),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4),
            "total_pnl": round(sum(pnls), 4),
            "win_rate": round(len(wins) / max(len(pnls), 1) * 100, 1),
            "sharpe": round(_sharpe(pnls), 3),
            "pnls": pnls,
        }
    return result


def transition_matrix(trades: list[dict]) -> dict[tuple[str, str], dict]:
    """Build regime transition matrix with post-transition P&L."""
    transitions: dict[tuple[str, str], list[float]] = defaultdict(list)
    prev_regime = None
    for t in trades:
        bh = "BH_ACTIVE" if t.get("bh_active") else "BH_QUIET"
        hr = t.get("hurst_regime", "random")
        regime = f"{bh}|{hr}"
        if prev_regime is not None and prev_regime != regime:
            pnl = t.get("pnl", 0.0) or 0.0
            transitions[(prev_regime, regime)].append(pnl)
        prev_regime = regime

    result: dict[tuple[str, str], dict] = {}
    for (src, dst), pnls in sorted(transitions.items(), key=lambda x: -len(x[1])):
        result[(src, dst)] = {
            "count": len(pnls),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4),
        }
    return result


def timing_analysis(trades: list[dict]) -> dict:
    """Compare entry-regime distribution vs full backtest distribution."""
    entry_dist: dict[str, int] = defaultdict(int)
    for t in trades:
        regime = ("BH_ACTIVE" if t.get("bh_active") else "BH_QUIET")
        entry_dist[regime] += 1

    total = max(sum(entry_dist.values()), 1)
    pct_dist = {k: round(v / total * 100, 1) for k, v in entry_dist.items()}

    # Ideal: BH_ACTIVE should dominate entries (we filter for it)
    bh_active_pct = pct_dist.get("BH_ACTIVE", 0.0)
    assessment = (
        "GOOD -- majority of entries in BH_ACTIVE regime"
        if bh_active_pct >= 55
        else "POOR -- too many entries in BH_QUIET regime"
    )
    return {
        "entry_distribution": pct_dist,
        "bh_active_entry_pct": bh_active_pct,
        "assessment": assessment,
    }


def hold_time_by_regime(trades: list[dict]) -> dict[str, list]:
    """Return (hold_bars, pnl) pairs grouped by regime."""
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for t in trades:
        hold = int(t.get("hold_bars", 0) or 0)
        pnl = float(t.get("pnl", 0.0) or 0.0)
        bh = "BH_ACTIVE" if t.get("bh_active") else "BH_QUIET"
        hr = t.get("hurst_regime", "random")
        groups[f"{bh}|{hr}"].append((hold, pnl))
    return dict(groups)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _violin_or_box(ax: Any, data: list[list[float]], labels: list[str],
                    title: str) -> None:
    """Draw violin plot if enough data, else box plot."""
    filtered = [(d, l) for d, l in zip(data, labels) if len(d) >= 3]
    if not filtered:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    fdata, flabels = zip(*filtered)
    try:
        ax.violinplot(list(fdata), showmedians=True)
    except Exception:
        ax.boxplot(list(fdata))
    ax.set_xticks(range(1, len(flabels) + 1))
    ax.set_xticklabels(list(flabels), rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("P&L")


def plot_regime_pnl(regime_stats: dict[str, dict], show: bool = False) -> Any:
    """Plot P&L distributions per regime."""
    if not HAS_MPL:
        return None

    # Only top-level regimes (no combo)
    top_keys = [k for k in regime_stats if "|" not in k or k.count("|") == 0]
    combo_keys = [k for k in regime_stats if k.count("|") == 2]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    top_data = [regime_stats[k]["pnls"] for k in top_keys]
    _violin_or_box(ax1, top_data, top_keys, "P&L by Top-Level Regime")

    ax2 = fig.add_subplot(gs[0, 1])
    combo_data = [regime_stats[k]["pnls"] for k in combo_keys[:8]]
    combo_labels = [k.replace("|", "\n") for k in combo_keys[:8]]
    _violin_or_box(ax2, combo_data, combo_labels, "P&L by Combo Regime")

    ax3 = fig.add_subplot(gs[1, 0])
    avg_pnls = [regime_stats[k]["avg_pnl"] for k in top_keys]
    colors = ["green" if v >= 0 else "red" for v in avg_pnls]
    ax3.bar(top_keys, avg_pnls, color=colors, edgecolor="black", linewidth=0.5)
    ax3.axhline(0, color="black", linewidth=1)
    ax3.set_title("Avg P&L by Regime")
    ax3.set_xticklabels(top_keys, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Avg P&L")

    ax4 = fig.add_subplot(gs[1, 1])
    win_rates = [regime_stats[k]["win_rate"] for k in top_keys]
    ax4.barh(top_keys, win_rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax4.axvline(50, color="red", linewidth=1, linestyle="--", label="50%")
    ax4.set_title("Win Rate by Regime")
    ax4.set_xlabel("Win Rate (%)")
    ax4.legend(fontsize=8)

    fig.suptitle("LARSA Regime Performance Diagnostic", fontsize=13, fontweight="bold")

    if show:
        plt.show()
    return fig


def plot_transition_matrix(transitions: dict[tuple[str, str], dict],
                             show: bool = False) -> Any:
    """Visualise the regime transition matrix."""
    if not HAS_MPL or not transitions:
        return None

    regimes = sorted({r for pair in transitions for r in pair})
    n = len(regimes)
    if n == 0:
        return None

    matrix_count = [[0] * n for _ in range(n)]
    matrix_pnl = [[0.0] * n for _ in range(n)]
    idx = {r: i for i, r in enumerate(regimes)}
    for (src, dst), info in transitions.items():
        i, j = idx.get(src, -1), idx.get(dst, -1)
        if i >= 0 and j >= 0:
            matrix_count[i][j] = info["count"]
            matrix_pnl[i][j] = info["avg_pnl"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if HAS_NUMPY:
        cnt_arr = np.array(matrix_count, dtype=float)
        pnl_arr = np.array(matrix_pnl, dtype=float)
        im1 = ax1.imshow(cnt_arr, cmap="Blues")
        im2 = ax2.imshow(pnl_arr, cmap="RdYlGn")
        for ax, arr, im, title in [
            (ax1, cnt_arr, im1, "Transition Frequency"),
            (ax2, pnl_arr, im2, "Post-Transition Avg P&L"),
        ]:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            short = [r[:12] for r in regimes]
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            ax.set_title(title)
            ax.set_xlabel("To Regime")
            ax.set_ylabel("From Regime")
            fig.colorbar(im, ax=ax, shrink=0.8)
            for i2 in range(n):
                for j2 in range(n):
                    ax.text(j2, i2, f"{arr[i2, j2]:.1f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if arr[i2, j2] > arr.max() * 0.6 else "black")

    fig.suptitle("Regime Transition Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_scatter_hold_time(hold_data: dict[str, list], show: bool = False) -> Any:
    """Scatter of hold bars vs P&L by regime."""
    if not HAS_MPL or not hold_data:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    patches = []
    for idx, (regime, pairs) in enumerate(hold_data.items()):
        if not pairs:
            continue
        holds = [p[0] for p in pairs]
        pnls = [p[1] for p in pairs]
        color = colors_cycle[idx % len(colors_cycle)]
        ax.scatter(holds, pnls, alpha=0.5, s=18, color=color, label=regime)
        patches.append(Patch(color=color, label=regime[:20]))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hold Time (bars)")
    ax.set_ylabel("P&L")
    ax.set_title("Hold Time vs P&L by Regime")
    if patches:
        ax.legend(handles=patches, fontsize=7, loc="best")
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
    trades: list[dict],
    regime_stats: dict[str, dict],
    transitions: dict[tuple[str, str], dict],
    timing: dict,
    hold_data: dict,
    figs: list[Any],
) -> str:
    """Build a self-contained HTML report."""
    rows_html = ""
    combo_keys = sorted(
        [(k, v) for k, v in regime_stats.items() if k.count("|") == 2],
        key=lambda x: -abs(x[1]["total_pnl"])
    )
    for label, stats in combo_keys[:20]:
        color = "#d4edda" if stats["avg_pnl"] >= 0 else "#f8d7da"
        rows_html += (
            f"<tr style='background:{color}'>"
            f"<td>{label}</td>"
            f"<td>{stats['count']}</td>"
            f"<td>{stats['avg_pnl']:.4f}</td>"
            f"<td>{stats['total_pnl']:.2f}</td>"
            f"<td>{stats['win_rate']:.1f}%</td>"
            f"<td>{stats['sharpe']:.3f}</td>"
            f"</tr>"
        )

    trans_rows = ""
    for (src, dst), info in list(transitions.items())[:15]:
        color = "#d4edda" if info["avg_pnl"] >= 0 else "#f8d7da"
        trans_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{src}</td><td>{dst}</td>"
            f"<td>{info['count']}</td>"
            f"<td>{info['avg_pnl']:.4f}</td></tr>"
        )

    imgs_html = ""
    for fig in figs:
        if fig is not None:
            try:
                b64 = _fig_to_base64(fig)
                imgs_html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:16px 0;">'
            except Exception:
                pass

    timing_html = ""
    for regime, pct in timing.get("entry_distribution", {}).items():
        timing_html += f"<li><b>{regime}</b>: {pct:.1f}%</li>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LARSA Regime Diagnostic Report</title>
<style>
  body {{font-family: Arial, sans-serif; margin: 24px; color: #222;}}
  h1 {{color: #1a3c6b;}} h2 {{color: #2c6e9b; border-bottom: 2px solid #eee; padding-bottom:4px;}}
  table {{border-collapse: collapse; width:100%; margin-bottom:20px;}}
  th {{background:#2c6e9b; color:white; padding:6px 10px; text-align:left;}}
  td {{padding:5px 10px; border-bottom:1px solid #ddd;}}
  .summary {{background:#f0f4f8; padding:12px; border-radius:6px; margin-bottom:20px;}}
  ul {{margin:0; padding-left:20px;}}
</style>
</head><body>
<h1>LARSA Regime Diagnostic Report</h1>
<div class="summary">
  <b>Total trades analysed:</b> {len(trades)}<br>
  <b>BH_ACTIVE entry pct:</b> {timing.get('bh_active_entry_pct', 0):.1f}%<br>
  <b>Timing assessment:</b> {timing.get('assessment', 'n/a')}
</div>

<h2>Charts</h2>
{imgs_html}

<h2>What Regime Made Money? (Top Combos)</h2>
<table>
  <tr><th>Regime Combo</th><th>Trades</th><th>Avg P&L</th><th>Total P&L</th>
      <th>Win Rate</th><th>Sharpe</th></tr>
{rows_html}
</table>

<h2>Regime Transition Analysis</h2>
<table>
  <tr><th>From</th><th>To</th><th>Count</th><th>Post-Transition Avg P&L</th></tr>
{trans_rows}
</table>

<h2>Entry Regime Timing</h2>
<ul>{timing_html}</ul>
<p>{timing.get('assessment', '')}</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Text fallback report
# ---------------------------------------------------------------------------

def print_text_report(regime_stats: dict[str, dict],
                       transitions: dict[tuple[str, str], dict],
                       timing: dict) -> None:
    print("\n=== LARSA REGIME DIAGNOSTIC REPORT ===\n")

    print("--- What regime made money? (top combos) ---")
    combos = sorted(
        [(k, v) for k, v in regime_stats.items() if k.count("|") == 2],
        key=lambda x: -x[1]["total_pnl"]
    )
    fmt = "{:<45} {:>6} {:>10} {:>10} {:>8} {:>8}"
    print(fmt.format("Regime", "N", "Avg P&L", "Total", "WinRate", "Sharpe"))
    print("-" * 95)
    for label, stats in combos[:15]:
        print(fmt.format(
            label[:45], stats["count"],
            f"{stats['avg_pnl']:+.4f}", f"{stats['total_pnl']:+.2f}",
            f"{stats['win_rate']:.1f}%", f"{stats['sharpe']:.3f}"
        ))

    print("\n--- Regime transitions (top 10) ---")
    for (src, dst), info in list(transitions.items())[:10]:
        print(f"  {src:30} --> {dst:30}  n={info['count']:3d}  avg_pnl={info['avg_pnl']:+.4f}")

    print("\n--- Entry timing ---")
    for regime, pct in timing.get("entry_distribution", {}).items():
        print(f"  {regime}: {pct:.1f}%")
    print(f"  Assessment: {timing.get('assessment', 'n/a')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LARSA regime diagnostic -- analyse regime/performance relationship"
    )
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to SQLite trades DB (default: execution/live_trades.db)")
    p.add_argument("--bars-db", default=str(_BARS_DB),
                   help="Path to bars SQLite DB for regime enrichment")
    p.add_argument("--since", default=None,
                   help="Filter trades since ISO date, e.g. 2024-01-01")
    p.add_argument("--output", default=None,
                   help="Output HTML report path, e.g. regime_report.html")
    p.add_argument("--show", action="store_true",
                   help="Show matplotlib plots interactively")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[regime_diagnostic] DB not found: {db_path}", file=sys.stderr)
        print("[regime_diagnostic] Creating synthetic demo data ...", file=sys.stderr)
        trades = _synthetic_trades(200)
    else:
        conn = _connect(db_path)
        trades = _load_trades(conn, args.since)
        conn.close()
        if not trades:
            print("[regime_diagnostic] No trades found, using synthetic demo data.",
                  file=sys.stderr)
            trades = _synthetic_trades(200)

    print(f"[regime_diagnostic] Loaded {len(trades)} trades")

    bars_db_path = Path(args.bars_db)
    print("[regime_diagnostic] Enriching trades with regime data ...")
    enriched = enrich_all_trades(trades, bars_db_path)

    print("[regime_diagnostic] Computing regime analytics ...")
    regime_stats = pnl_by_regime(enriched)
    transitions = transition_matrix(enriched)
    timing = timing_analysis(enriched)
    hold_data = hold_time_by_regime(enriched)

    figs: list[Any] = []
    if HAS_MPL:
        figs.append(plot_regime_pnl(regime_stats, show=False))
        figs.append(plot_transition_matrix(transitions, show=False))
        figs.append(plot_scatter_hold_time(hold_data, show=False))
        if args.show:
            plt.show()

    if args.output:
        print(f"[regime_diagnostic] Writing report to {args.output} ...")
        html = build_html_report(enriched, regime_stats, transitions, timing,
                                  hold_data, figs)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"[regime_diagnostic] Report saved: {args.output}")
    else:
        print_text_report(regime_stats, transitions, timing)

    for fig in figs:
        if fig is not None:
            plt.close(fig)

    return 0


# ---------------------------------------------------------------------------
# Synthetic data for demo / tests
# ---------------------------------------------------------------------------

def _synthetic_trades(n: int = 100) -> list[dict]:
    import random
    rng = random.Random(42)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades: list[dict] = []
    ts_offset = 0
    symbols = ["BTC", "ETH", "SOL", "AAPL"]
    for i in range(n):
        ts_offset += rng.randint(30, 480)  # minutes
        entry_dt = base_ts.replace(minute=0) if ts_offset == 0 else base_ts
        from datetime import timedelta
        entry_dt = base_ts + timedelta(minutes=ts_offset)
        exit_dt = entry_dt + timedelta(minutes=rng.randint(15, 120))
        pnl = rng.gauss(5.0, 50.0)
        trades.append({
            "id": i + 1,
            "symbol": rng.choice(symbols),
            "entry_time": entry_dt.isoformat(),
            "exit_time": exit_dt.isoformat(),
            "entry_price": rng.uniform(100, 50000),
            "exit_price": rng.uniform(100, 50000),
            "qty": rng.uniform(0.01, 1.0),
            "pnl": round(pnl, 2),
            "hold_bars": rng.randint(1, 32),
        })
    return trades


if __name__ == "__main__":
    sys.exit(main())
