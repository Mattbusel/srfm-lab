"""
tools/param_impact_analyzer.py
================================
Analyzes how parameter changes (from IAE/Optuna) impacted live performance.

- Loads parameter update history from Elixir coordination layer (:8781/params/history)
- For each update: computes Sharpe before/after (4h, 24h windows)
- Attribution: which params consistently improve performance?
- Identifies params causing rollbacks, params at edge of their range
- Regime-conditioned impact: trending vs mean-reversion

Usage:
    python tools/param_impact_analyzer.py --lookback-days 90 --output param_report.html
    python tools/param_impact_analyzer.py --api http://localhost:8781 --lookback-days 30
    python tools/param_impact_analyzer.py --local-history params_history.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import urllib.request
import urllib.error
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
_DEFAULT_HISTORY = _REPO / "coordination" / "params_history.json"
_DEFAULT_API = "http://localhost:8781"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_params_history(api_url: str, timeout: float = 5.0) -> list[dict]:
    """Fetch parameter update history from Elixir coordination layer."""
    url = f"{api_url.rstrip('/')}/params/history"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("history", data.get("updates", []))
    except (urllib.error.URLError, Exception) as exc:
        print(f"[param_impact] Could not fetch from {url}: {exc}", file=sys.stderr)
    return []


def load_history_from_file(path: Path) -> list[dict]:
    """Load parameter history from a JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("history", data.get("updates", []))
    except Exception as exc:
        print(f"[param_impact] Failed to load {path}: {exc}", file=sys.stderr)
    return []


def load_trades_from_db(db_path: Path) -> list[dict]:
    """Load all trades from SQLite."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    trades: list[dict] = []
    for table in ("trade_pnl", "trades"):
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY exit_time"
            ).fetchall()
            trades = [dict(r) for r in rows]
            break
        except sqlite3.OperationalError:
            continue
    conn.close()
    return trades


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(var, 0.0))


def sharpe_window(trades: list[dict], start_iso: str, end_iso: str) -> float:
    """Compute Sharpe ratio from trades within [start_iso, end_iso]."""
    pnls = [
        float(t.get("pnl", 0.0) or 0.0)
        for t in trades
        if start_iso <= (t.get("exit_time") or "") <= end_iso
    ]
    if len(pnls) < 2:
        return float("nan")
    mean = sum(pnls) / len(pnls)
    std = _safe_std(pnls)
    if std == 0:
        return float("nan")
    return mean / std * math.sqrt(252 * 26)


def _add_hours(iso: str, hours: float) -> str:
    """Add `hours` to an ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)
    result = dt + timedelta(hours=hours)
    return result.isoformat()


def _sub_hours(iso: str, hours: float) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)
    result = dt - timedelta(hours=hours)
    return result.isoformat()


def enrich_update_with_performance(
    update: dict,
    trades: list[dict],
    windows: list[float] = (4.0, 24.0),
) -> dict:
    """
    For one parameter update event, compute Sharpe before and after
    for each time window.
    """
    ts = update.get("timestamp") or update.get("ts") or update.get("time", "")
    result = dict(update)
    result["_performance"] = {}

    for w in windows:
        before_start = _sub_hours(ts, w)
        after_end = _add_hours(ts, w)
        s_before = sharpe_window(trades, before_start, ts)
        s_after = sharpe_window(trades, ts, after_end)
        result["_performance"][f"{w}h"] = {
            "sharpe_before": round(s_before, 4) if not math.isnan(s_before) else None,
            "sharpe_after": round(s_after, 4) if not math.isnan(s_after) else None,
            "delta": (
                round(s_after - s_before, 4)
                if (not math.isnan(s_before) and not math.isnan(s_after))
                else None
            ),
        }
    return result


def param_attribution(enriched_updates: list[dict],
                        window: str = "24h") -> dict[str, dict]:
    """
    For each changed parameter, compute:
    - average Sharpe delta when that param was changed
    - fraction of updates where it improved performance
    - times it caused a rollback
    """
    # Group by changed parameter names
    per_param: dict[str, list[float]] = defaultdict(list)
    rollbacks: dict[str, int] = defaultdict(int)

    for upd in enriched_updates:
        perf = upd.get("_performance", {}).get(window, {})
        delta = perf.get("delta")
        if delta is None:
            continue

        # Extract changed parameters
        changed = upd.get("changed_params", upd.get("params", {}))
        if isinstance(changed, dict):
            param_names = list(changed.keys())
        elif isinstance(changed, list):
            param_names = [str(p) for p in changed]
        else:
            param_names = ["unknown"]

        for pname in param_names:
            per_param[pname].append(delta)
            was_rollback = upd.get("rollback", False) or delta < -0.5
            if was_rollback:
                rollbacks[pname] += 1

    result: dict[str, dict] = {}
    for pname, deltas in sorted(per_param.items()):
        improvements = sum(1 for d in deltas if d > 0)
        result[pname] = {
            "n_updates": len(deltas),
            "avg_delta": round(sum(deltas) / len(deltas), 4),
            "improvement_rate": round(improvements / len(deltas) * 100, 1),
            "n_rollbacks": rollbacks.get(pname, 0),
            "max_delta": round(max(deltas), 4),
            "min_delta": round(min(deltas), 4),
            "deltas": deltas,
        }
    return result


def find_edge_params(enriched_updates: list[dict],
                      pct_threshold: float = 0.10) -> list[dict]:
    """
    Identify parameters whose current value is within `pct_threshold`
    fraction of their historical range boundaries.
    """
    # Collect value history per param
    param_history: dict[str, list[float]] = defaultdict(list)
    current_vals: dict[str, float] = {}

    for upd in sorted(enriched_updates,
                       key=lambda u: u.get("timestamp", u.get("ts", ""))):
        changed = upd.get("changed_params", upd.get("params", {}))
        if isinstance(changed, dict):
            for k, v in changed.items():
                try:
                    fv = float(v)
                    param_history[k].append(fv)
                    current_vals[k] = fv
                except (TypeError, ValueError):
                    pass

    edge_params: list[dict] = []
    for pname, vals in param_history.items():
        if len(vals) < 3:
            continue
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span == 0:
            continue
        current = current_vals.get(pname, vals[-1])
        dist_lo = (current - lo) / span
        dist_hi = (hi - current) / span
        at_edge = dist_lo < pct_threshold or dist_hi < pct_threshold
        edge_params.append({
            "param": pname,
            "current": round(current, 6),
            "min_seen": round(lo, 6),
            "max_seen": round(hi, 6),
            "dist_from_min_pct": round(dist_lo * 100, 1),
            "dist_from_max_pct": round(dist_hi * 100, 1),
            "at_edge": at_edge,
            "recommendation": (
                "EXPAND RANGE" if at_edge else "range OK"
            ),
        })
    return sorted(edge_params, key=lambda x: min(x["dist_from_min_pct"],
                                                   x["dist_from_max_pct"]))


def regime_conditioned_impact(
    enriched_updates: list[dict],
    trades: list[dict],
    window_h: float = 24.0,
) -> dict[str, dict]:
    """
    For each update, classify the post-update regime (trending vs MR)
    and compute regime-conditioned Sharpe delta.
    """
    trending_deltas: list[float] = []
    mr_deltas: list[float] = []

    for upd in enriched_updates:
        ts = upd.get("timestamp", upd.get("ts", ""))
        perf = upd.get("_performance", {}).get(f"{window_h}h", {})
        delta = perf.get("delta")
        if delta is None:
            continue

        # Classify regime from post-update trades
        after_end = _add_hours(ts, window_h)
        post_trades = [
            t for t in trades
            if ts <= (t.get("exit_time") or "") <= after_end
        ]
        avg_hold = (
            sum(int(t.get("hold_bars", 4) or 4) for t in post_trades)
            / max(len(post_trades), 1)
        )
        # Proxy: long hold = trending, short hold = MR
        if avg_hold > 8:
            trending_deltas.append(delta)
        else:
            mr_deltas.append(delta)

    def _stats(d: list[float]) -> dict:
        if not d:
            return {"n": 0, "avg_delta": None, "pct_improved": None}
        imp = sum(1 for x in d if x > 0)
        return {
            "n": len(d),
            "avg_delta": round(sum(d) / len(d), 4),
            "pct_improved": round(imp / len(d) * 100, 1),
        }

    return {
        "trending": _stats(trending_deltas),
        "mean_rev": _stats(mr_deltas),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_param_attribution(attribution: dict[str, dict],
                             show: bool = False) -> Any:
    if not HAS_MPL or not attribution:
        return None

    items = sorted(attribution.items(), key=lambda x: -abs(x[1]["avg_delta"]))[:15]
    labels = [i[0][:20] for i in items]
    avg_deltas = [i[1]["avg_delta"] for i in items]
    impr_rates = [i[1]["improvement_rate"] for i in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["green" if v >= 0 else "red" for v in avg_deltas]
    ax1.barh(labels, avg_deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_title("Avg Sharpe Delta per Parameter Change")
    ax1.set_xlabel("Avg Sharpe Delta (24h)")

    ax2.barh(labels, impr_rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.axvline(50, color="red", linestyle="--", linewidth=1)
    ax2.set_title("Improvement Rate per Parameter")
    ax2.set_xlabel("% Updates That Improved Sharpe")

    fig.suptitle("Parameter Impact Attribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_sharpe_before_after(enriched: list[dict],
                              window: str = "24h",
                              show: bool = False) -> Any:
    if not HAS_MPL or not enriched:
        return None

    befores: list[float] = []
    afters: list[float] = []
    for upd in enriched:
        perf = upd.get("_performance", {}).get(window, {})
        b = perf.get("sharpe_before")
        a = perf.get("sharpe_after")
        if b is not None and a is not None:
            befores.append(b)
            afters.append(a)

    if not befores:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(befores, afters, alpha=0.6, s=30, color="steelblue")
    lo = min(min(befores), min(afters)) - 0.1
    hi = max(max(befores), max(afters)) + 0.1
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Sharpe Before ({window})")
    ax.set_ylabel(f"Sharpe After ({window})")
    ax.set_title(f"Sharpe Before vs After Parameter Update ({window})")
    improved = sum(1 for b, a in zip(befores, afters) if a > b)
    ax.text(0.02, 0.98,
            f"Improved: {improved}/{len(befores)} ({improved/max(len(befores),1)*100:.0f}%)",
            transform=ax.transAxes, va="top", fontsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_edge_params(edge_params: list[dict], show: bool = False) -> Any:
    if not HAS_MPL or not edge_params:
        return None

    params = [e["param"][:18] for e in edge_params[:12]]
    dist_lo = [e["dist_from_min_pct"] for e in edge_params[:12]]
    dist_hi = [e["dist_from_max_pct"] for e in edge_params[:12]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(params))
    ax.bar([i - 0.2 for i in x], dist_lo, width=0.4,
            label="Dist from Min (%)", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar([i + 0.2 for i in x], dist_hi, width=0.4,
            label="Dist from Max (%)", color="darkorange", edgecolor="black", linewidth=0.5)
    ax.axhline(10, color="red", linestyle="--", linewidth=1, label="10% edge threshold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(params, rotation=30, ha="right", fontsize=8)
    ax.set_title("Parameters Near Range Boundaries")
    ax.set_ylabel("Distance from Boundary (%)")
    ax.legend(fontsize=8)
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
    enriched_updates: list[dict],
    attribution: dict[str, dict],
    edge_params: list[dict],
    regime_cond: dict[str, dict],
    figs: list[Any],
    window: str = "24h",
) -> str:
    attr_rows = ""
    for pname, stats in sorted(
        attribution.items(), key=lambda x: -abs(x[1]["avg_delta"])
    )[:20]:
        color = "#d4edda" if stats["avg_delta"] >= 0 else "#f8d7da"
        attr_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{pname}</td>"
            f"<td>{stats['n_updates']}</td>"
            f"<td>{stats['avg_delta']:+.4f}</td>"
            f"<td>{stats['improvement_rate']:.1f}%</td>"
            f"<td>{stats['n_rollbacks']}</td>"
            f"<td>{stats['max_delta']:+.4f}</td>"
            f"<td>{stats['min_delta']:+.4f}</td></tr>"
        )

    edge_rows = ""
    for e in edge_params[:15]:
        color = "#fff3cd" if e["at_edge"] else "#ffffff"
        edge_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{e['param']}</td>"
            f"<td>{e['current']}</td>"
            f"<td>{e['min_seen']}</td>"
            f"<td>{e['max_seen']}</td>"
            f"<td>{e['dist_from_min_pct']:.1f}%</td>"
            f"<td>{e['dist_from_max_pct']:.1f}%</td>"
            f"<td><b>{e['recommendation']}</b></td></tr>"
        )

    imgs_html = ""
    for fig in figs:
        if fig is not None:
            try:
                b64 = _fig_to_base64(fig)
                imgs_html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:16px 0;">'
            except Exception:
                pass

    rc_trending = regime_cond.get("trending", {})
    rc_mr = regime_cond.get("mean_rev", {})

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Parameter Impact Analyzer</title>
<style>
  body {{font-family: Arial, sans-serif; margin: 24px; color: #222;}}
  h1 {{color: #1a3c6b;}} h2 {{color: #2c6e9b; border-bottom: 2px solid #eee; padding-bottom:4px;}}
  table {{border-collapse: collapse; width:100%; margin-bottom:20px;}}
  th {{background:#2c6e9b; color:white; padding:6px 10px; text-align:left;}}
  td {{padding:5px 10px; border-bottom:1px solid #ddd;}}
  .grid {{display:grid; grid-template-columns:1fr 1fr; gap:20px;}}
  .card {{background:#f0f4f8; padding:14px; border-radius:6px;}}
</style>
</head><body>
<h1>Parameter Impact Analyzer</h1>
<p>Analysis window: {window} | Total updates: {len(enriched_updates)}</p>

<h2>Regime-Conditioned Impact</h2>
<div class="grid">
  <div class="card">
    <b>Trending Regime</b><br>
    N: {rc_trending.get('n', 0)} |
    Avg delta: {rc_trending.get('avg_delta') or 'n/a'} |
    Improved: {rc_trending.get('pct_improved') or 'n/a'}%
  </div>
  <div class="card">
    <b>Mean-Reversion Regime</b><br>
    N: {rc_mr.get('n', 0)} |
    Avg delta: {rc_mr.get('avg_delta') or 'n/a'} |
    Improved: {rc_mr.get('pct_improved') or 'n/a'}%
  </div>
</div>

<h2>Charts</h2>
{imgs_html}

<h2>Parameter Attribution (Sharpe {window} window)</h2>
<table>
  <tr><th>Parameter</th><th>N Updates</th><th>Avg Delta</th><th>Impr Rate</th>
      <th>Rollbacks</th><th>Max Delta</th><th>Min Delta</th></tr>
{attr_rows}
</table>

<h2>Parameters Near Range Boundaries</h2>
<table>
  <tr><th>Parameter</th><th>Current</th><th>Min Seen</th><th>Max Seen</th>
      <th>Dist Min</th><th>Dist Max</th><th>Recommendation</th></tr>
{edge_rows}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Text fallback
# ---------------------------------------------------------------------------

def print_text_report(attribution: dict[str, dict],
                       edge_params: list[dict],
                       regime_cond: dict[str, dict]) -> None:
    print("\n=== PARAMETER IMPACT ANALYZER ===\n")
    print("--- Attribution (24h Sharpe delta) ---")
    fmt = "{:<25} {:>8} {:>10} {:>10} {:>10}"
    print(fmt.format("Parameter", "N", "Avg Delta", "Impr%", "Rollbacks"))
    print("-" * 68)
    for pname, stats in sorted(
        attribution.items(), key=lambda x: -x[1]["avg_delta"]
    )[:15]:
        print(fmt.format(
            pname[:25], stats["n_updates"],
            f"{stats['avg_delta']:+.4f}",
            f"{stats['improvement_rate']:.1f}%",
            str(stats["n_rollbacks"]),
        ))

    print("\n--- Parameters at edge of range ---")
    for e in edge_params[:10]:
        flag = " *** EXPAND RANGE ***" if e["at_edge"] else ""
        print(f"  {e['param']:<25} current={e['current']:.6f}  "
              f"range=[{e['min_seen']:.6f}, {e['max_seen']:.6f}]  "
              f"dist_lo={e['dist_from_min_pct']:.1f}%{flag}")

    print("\n--- Regime-conditioned impact ---")
    for regime, stats in regime_cond.items():
        n = stats.get("n", 0)
        avg_d = stats.get("avg_delta")
        pct = stats.get("pct_improved")
        print(f"  {regime:<12} n={n}  avg_delta={avg_d}  improved={pct}%")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _synthetic_history(n: int = 60) -> list[dict]:
    import random
    rng = random.Random(77)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    params = ["bh_mass_threshold", "hurst_min", "garch_omega", "atr_mult",
              "cf_decay", "quatnav_lambda", "risk_pdr", "hold_max_bars"]
    result: list[dict] = []
    for i in range(n):
        ts = base + timedelta(hours=i * 36)
        n_changed = rng.randint(1, 3)
        changed = {
            rng.choice(params): round(rng.uniform(0.001, 2.0), 4)
            for _ in range(n_changed)
        }
        result.append({
            "timestamp": ts.isoformat(),
            "changed_params": changed,
            "rollback": rng.random() < 0.15,
            "source": rng.choice(["optuna", "iae", "manual"]),
        })
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze parameter change impact on LARSA performance"
    )
    p.add_argument("--api", default=_DEFAULT_API,
                   help=f"Elixir coordination API base URL (default: {_DEFAULT_API})")
    p.add_argument("--local-history", default=None,
                   help="Path to local JSON parameter history file")
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to trades SQLite DB")
    p.add_argument("--lookback-days", type=int, default=90,
                   help="Lookback window in days (default: 90)")
    p.add_argument("--window", default="24h",
                   choices=["4h", "24h"],
                   help="Performance window (default: 24h)")
    p.add_argument("--output", default=None,
                   help="Output HTML report path")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load parameter history
    history: list[dict] = []
    if args.local_history:
        history = load_history_from_file(Path(args.local_history))
    if not history:
        history = fetch_params_history(args.api)
    if not history:
        print("[param_impact] No history found -- using synthetic demo", file=sys.stderr)
        history = _synthetic_history(60)

    # Filter by lookback
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.lookback_days)).isoformat()
    history = [
        h for h in history
        if (h.get("timestamp") or h.get("ts") or "9") >= cutoff
    ]
    print(f"[param_impact] {len(history)} parameter updates in lookback window")

    # Load trades
    trades = load_trades_from_db(Path(args.db))
    if not trades:
        print("[param_impact] No trades in DB -- performance windows will be empty",
              file=sys.stderr)

    # Enrich updates with performance windows
    windows = [4.0, 24.0]
    print("[param_impact] Enriching updates with performance windows ...")
    enriched = [enrich_update_with_performance(u, trades, windows) for u in history]

    # Analytics
    attribution = param_attribution(enriched, window=args.window)
    edge_params = find_edge_params(enriched)
    regime_cond = regime_conditioned_impact(enriched, trades)

    figs: list[Any] = []
    if HAS_MPL:
        figs.append(plot_param_attribution(attribution))
        figs.append(plot_sharpe_before_after(enriched, window=args.window))
        figs.append(plot_edge_params(edge_params))
        if args.show:
            plt.show()

    if args.output:
        print(f"[param_impact] Writing report to {args.output} ...")
        html = build_html_report(enriched, attribution, edge_params,
                                  regime_cond, figs, window=args.window)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"[param_impact] Report saved: {args.output}")
    else:
        print_text_report(attribution, edge_params, regime_cond)

    for fig in figs:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
