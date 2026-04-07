"""
tools/correlation_drift_monitor.py
=====================================
Monitors pairwise correlations across the instrument universe for regime shifts.

- Rolling 20-day / 60-day pairwise Pearson correlations
- Alerts for correlation regime changes
- Outputs rolling correlation heatmap (static or animated)

Usage:
    python tools/correlation_drift_monitor.py --window 20 60 --alert-threshold 0.85
    python tools/correlation_drift_monitor.py --csv data/prices.csv
    python tools/correlation_drift_monitor.py --output corr_report.html
"""

from __future__ import annotations

import argparse
import csv
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
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "data" / "bars.db"

INSTRUMENTS = ["BTC", "ETH", "SOL", "AAPL", "SPY", "QQQ"]

# Alert thresholds
BTCETH_DECOUPLE_THRESHOLD = 0.70
HIGH_CORR_THRESHOLD = 0.85
DXY_BTC_SIGN_CHANGE = True  # monitor sign changes

CORR_REGIMES = {
    "LOW_CORR": "Diversified -- low systematic risk",
    "MEDIUM_CORR": "Normal correlation regime",
    "HIGH_CORR": "Crowded trade -- elevated systematic risk",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices_from_db(db_path: Path,
                         symbols: list[str],
                         window_days: int = 120) -> dict[str, list[tuple[str, float]]]:
    """Load (ts, close) for each symbol from bars SQLite DB."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    result: dict[str, list[tuple[str, float]]] = {}
    for sym in symbols:
        try:
            rows = conn.execute(
                "SELECT ts, close FROM bars WHERE symbol=? AND ts>=? ORDER BY ts",
                (sym, cutoff)
            ).fetchall()
            result[sym] = [(r["ts"], float(r["close"])) for r in rows]
        except sqlite3.OperationalError:
            pass
    conn.close()
    return result


def load_prices_from_csv(csv_path: Path) -> dict[str, list[tuple[str, float]]]:
    """
    Load prices from CSV.  Accepts two formats:
    1) date, BTC, ETH, SOL, ...  (wide format)
    2) date, symbol, close        (long format)
    """
    result: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "symbol" in cols or "Symbol" in cols:
            # Long format
            sym_col = "symbol" if "symbol" in cols else "Symbol"
            date_col = next((c for c in cols if c.lower() in ("date", "ts", "time")), cols[0])
            close_col = next((c for c in cols if c.lower() == "close"), cols[-1])
            for row in reader:
                sym = row.get(sym_col, "")
                ts = row.get(date_col, "")
                try:
                    close = float(row.get(close_col, 0))
                    if sym and close > 0:
                        result[sym].append((ts, close))
                except ValueError:
                    pass
        else:
            # Wide format: date + one col per symbol
            date_col = cols[0]
            sym_cols = cols[1:]
            for row in reader:
                ts = row.get(date_col, "")
                for sym in sym_cols:
                    try:
                        val = float(row.get(sym, 0) or 0)
                        if val > 0:
                            result[sym].append((ts, val))
                    except ValueError:
                        pass
    return dict(result)


# ---------------------------------------------------------------------------
# Correlation calculations
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / (den + 1e-12)


def returns_from_prices(prices: list[float]) -> list[float]:
    return [
        (prices[i] - prices[i - 1]) / (prices[i - 1] + 1e-9)
        for i in range(1, len(prices))
    ]


def align_series(
    data_a: list[tuple[str, float]],
    data_b: list[tuple[str, float]],
) -> tuple[list[float], list[float]]:
    """Align two (ts, price) series by timestamp intersection."""
    dict_b = {ts: p for ts, p in data_b}
    aligned: list[tuple[float, float]] = []
    for ts, pa in data_a:
        if ts in dict_b:
            aligned.append((pa, dict_b[ts]))
    if not aligned:
        return [], []
    ps_a = [x[0] for x in aligned]
    ps_b = [x[1] for x in aligned]
    return ps_a, ps_b


def rolling_pairwise_corr(
    prices: dict[str, list[tuple[str, float]]],
    symbols: list[str],
    window: int,
) -> list[tuple[str, dict[str, float]]]:
    """
    Compute pairwise correlations using a rolling window (in bars).
    Returns list of (ts, {pair: corr}) dicts sorted by time.
    """
    # Build aligned return matrix per timestamp
    # First, find common timestamps
    if not prices:
        return []

    common_syms = [s for s in symbols if s in prices and len(prices[s]) > window]
    if len(common_syms) < 2:
        return []

    # Use first symbol's timestamps as index
    ref_sym = common_syms[0]
    ref_ts = [ts for ts, _ in prices[ref_sym]]

    # For each symbol, build a price dict
    sym_price_dicts: dict[str, dict[str, float]] = {}
    for sym in common_syms:
        sym_price_dicts[sym] = {ts: p for ts, p in prices[sym]}

    # Compute pairwise rolling
    records: list[tuple[str, dict[str, float]]] = []

    for end_i in range(window, len(ref_ts)):
        ts_window = ref_ts[end_i - window: end_i + 1]
        ts_end = ref_ts[end_i]

        pair_corrs: dict[str, float] = {}
        all_valid: list[float] = []

        for i, sym_a in enumerate(common_syms):
            for sym_b in common_syms[i + 1:]:
                ps_a = [sym_price_dicts[sym_a].get(ts, 0) for ts in ts_window]
                ps_b = [sym_price_dicts[sym_b].get(ts, 0) for ts in ts_window]
                # Remove zeros (missing data)
                pairs = [(a, b) for a, b in zip(ps_a, ps_b) if a > 0 and b > 0]
                if len(pairs) < 10:
                    continue
                rets_a = returns_from_prices([p[0] for p in pairs])
                rets_b = returns_from_prices([p[1] for p in pairs])
                c = _pearson(rets_a, rets_b)
                if not math.isnan(c):
                    key = f"{sym_a}-{sym_b}"
                    pair_corrs[key] = round(c, 4)
                    all_valid.append(c)

        if pair_corrs:
            pair_corrs["_avg"] = round(sum(all_valid) / len(all_valid), 4)
            records.append((ts_end, pair_corrs))

    return records


def classify_corr_regime(avg_corr: float,
                           threshold: float = HIGH_CORR_THRESHOLD) -> str:
    if avg_corr > threshold:
        return "HIGH_CORR"
    if avg_corr > 0.5:
        return "MEDIUM_CORR"
    return "LOW_CORR"


def generate_alerts(
    rolling_records: list[tuple[str, dict[str, float]]],
    btceth_threshold: float = BTCETH_DECOUPLE_THRESHOLD,
    high_corr_threshold: float = HIGH_CORR_THRESHOLD,
) -> list[dict]:
    """Scan rolling records for alert conditions."""
    alerts: list[dict] = []
    prev_dxy_btc_sign: int | None = None

    for ts, corrs in rolling_records:
        # BTC-ETH decoupling
        btceth = corrs.get("BTC-ETH")
        if btceth is not None and btceth < btceth_threshold:
            alerts.append({
                "ts": ts,
                "type": "BTC_ETH_DECOUPLE",
                "value": btceth,
                "threshold": btceth_threshold,
                "message": f"BTC-ETH correlation {btceth:.3f} below {btceth_threshold} -- unusual decoupling",
            })

        # High average correlation
        avg = corrs.get("_avg", 0.0)
        if avg > high_corr_threshold:
            alerts.append({
                "ts": ts,
                "type": "HIGH_AVG_CORR",
                "value": avg,
                "threshold": high_corr_threshold,
                "message": f"Average pairwise correlation {avg:.3f} -- crowded trade risk",
            })

        # DXY-BTC sign change (if available)
        dxy_btc = corrs.get("DXY-BTC")
        if dxy_btc is not None:
            sign = 1 if dxy_btc >= 0 else -1
            if prev_dxy_btc_sign is not None and sign != prev_dxy_btc_sign:
                alerts.append({
                    "ts": ts,
                    "type": "DXY_BTC_SIGN_CHANGE",
                    "value": dxy_btc,
                    "threshold": 0.0,
                    "message": f"DXY-BTC correlation sign changed to {dxy_btc:.3f} -- dollar regime shift",
                })
            prev_dxy_btc_sign = sign

    return alerts


def latest_regime_summary(
    rolling_records: list[tuple[str, dict[str, float]]],
    window_label: str,
    threshold: float,
) -> dict:
    """Summarise the most recent correlation state."""
    if not rolling_records:
        return {"regime": "UNKNOWN", "avg_corr": None, "window": window_label}
    ts, corrs = rolling_records[-1]
    avg = corrs.get("_avg", 0.0)
    regime = classify_corr_regime(avg, threshold)
    return {
        "regime": regime,
        "description": CORR_REGIMES[regime],
        "avg_corr": avg,
        "window": window_label,
        "ts": ts,
        "all_pairs": {k: v for k, v in corrs.items() if not k.startswith("_")},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    rolling_records: list[tuple[str, dict[str, float]]],
    symbols: list[str],
    window_label: str,
    show: bool = False,
) -> Any:
    if not HAS_MPL or not rolling_records:
        return None

    # Use the most recent snapshot
    _, corrs = rolling_records[-1]
    common_syms = [s for s in symbols if any(
        s in k for k in corrs if "-" in k
    )]
    # Build matrix
    n = len(common_syms)
    if n < 2:
        return None
    matrix = [[1.0] * n for _ in range(n)]
    idx = {s: i for i, s in enumerate(common_syms)}
    for key, val in corrs.items():
        if key.startswith("_"):
            continue
        parts = key.split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        ia = idx.get(a, -1)
        ib = idx.get(b, -1)
        if ia >= 0 and ib >= 0:
            matrix[ia][ib] = val
            matrix[ib][ia] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        import numpy as np
        mat = np.array(matrix)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    except ImportError:
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                color = "green" if val > 0 else "red"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        im = None

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(common_syms)
    ax.set_yticklabels(common_syms)
    ax.set_title(f"Correlation Heatmap ({window_label})")
    if im is not None:
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_rolling_avg_corr(
    records_by_window: dict[str, list[tuple[str, dict[str, float]]]],
    threshold: float,
    show: bool = False,
) -> Any:
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["steelblue", "darkorange", "green"]
    for idx, (label, records) in enumerate(records_by_window.items()):
        if not records:
            continue
        xs = list(range(len(records)))
        ys = [r[1].get("_avg", 0.0) for r in records]
        ax.plot(xs, ys, label=f"Avg Corr ({label})",
                color=colors[idx % len(colors)], linewidth=1.5)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"Alert threshold ({threshold})")
    ax.axhline(BTCETH_DECOUPLE_THRESHOLD, color="orange", linestyle=":",
               linewidth=1, label=f"BTC-ETH decouple ({BTCETH_DECOUPLE_THRESHOLD})")
    ax.set_title("Rolling Average Pairwise Correlation")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Avg Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_pair_history(
    records: list[tuple[str, dict[str, float]]],
    pair: str,
    window_label: str,
    show: bool = False,
) -> Any:
    if not HAS_MPL or not records:
        return None

    ts_vals = [(ts, corrs[pair]) for ts, corrs in records if pair in corrs]
    if not ts_vals:
        return None

    fig, ax = plt.subplots(figsize=(11, 3))
    xs = list(range(len(ts_vals)))
    ys = [v[1] for v in ts_vals]
    ax.plot(xs, ys, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(xs, ys, 0, where=[y >= 0 for y in ys], alpha=0.2, color="green")
    ax.fill_between(xs, ys, 0, where=[y < 0 for y in ys], alpha=0.2, color="red")
    ax.set_title(f"{pair} Rolling Correlation ({window_label})")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# HTML / text reports
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: Any) -> str:
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_html_report(
    summaries: list[dict],
    alerts: list[dict],
    figs: list[Any],
) -> str:
    summary_html = ""
    for s in summaries:
        color_map = {"HIGH_CORR": "#f8d7da", "MEDIUM_CORR": "#fff3cd", "LOW_CORR": "#d4edda"}
        color = color_map.get(s.get("regime", ""), "#fff")
        summary_html += f"""
<div style="background:{color}; padding:12px; border-radius:6px; margin-bottom:10px;">
  <b>{s.get('window', '')} window</b> --
  Regime: <b>{s.get('regime', '')}</b> ({s.get('description', '')})<br>
  Avg correlation: {s.get('avg_corr', 'n/a'):.3f if isinstance(s.get('avg_corr'), float) else 'n/a'}<br>
  As of: {s.get('ts', 'n/a')[:19]}
</div>"""

    alert_rows = ""
    for a in alerts[-30:]:
        alert_type_colors = {
            "HIGH_AVG_CORR": "#f8d7da",
            "BTC_ETH_DECOUPLE": "#fff3cd",
            "DXY_BTC_SIGN_CHANGE": "#cce5ff",
        }
        color = alert_type_colors.get(a["type"], "#fff")
        alert_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{a['ts'][:19]}</td>"
            f"<td>{a['type']}</td>"
            f"<td>{a['value']:.4f}</td>"
            f"<td>{a['message']}</td></tr>"
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
<title>Correlation Drift Monitor</title>
<style>
  body {{font-family: Arial, sans-serif; margin: 24px; color: #222;}}
  h1 {{color: #1a3c6b;}} h2 {{color: #2c6e9b; border-bottom: 2px solid #eee; padding-bottom:4px;}}
  table {{border-collapse: collapse; width:100%; margin-bottom:20px;}}
  th {{background:#2c6e9b; color:white; padding:6px 10px; text-align:left;}}
  td {{padding:5px 10px; border-bottom:1px solid #ddd;}}
</style>
</head><body>
<h1>Correlation Drift Monitor</h1>

<h2>Current Regime Summary</h2>
{summary_html}

<h2>Charts</h2>
{imgs_html}

<h2>Alerts (last 30)</h2>
<table>
  <tr><th>Timestamp</th><th>Type</th><th>Value</th><th>Message</th></tr>
{alert_rows}
</table>
</body></html>"""


def print_text_report(summaries: list[dict], alerts: list[dict]) -> None:
    print("\n=== CORRELATION DRIFT MONITOR ===\n")
    for s in summaries:
        avg = s.get("avg_corr")
        avg_str = f"{avg:.3f}" if isinstance(avg, float) else "n/a"
        print(f"Window: {s.get('window')}  Regime: {s.get('regime')}  "
              f"Avg corr: {avg_str}  ({s.get('description', '')})")

    if alerts:
        print(f"\n--- Alerts ({len(alerts)} total, showing last 10) ---")
        for a in alerts[-10:]:
            print(f"  [{a['ts'][:19]}] {a['type']}: {a['message']}")
    else:
        print("\nNo alerts.")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _synthetic_prices(symbols: list[str], n_bars: int = 200) -> dict[str, list[tuple[str, float]]]:
    import random
    rng = random.Random(123)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result: dict[str, list[tuple[str, float]]] = {}
    for sym in symbols:
        price = rng.uniform(100, 50000)
        series: list[tuple[str, float]] = []
        for i in range(n_bars):
            ts = (base + timedelta(hours=i * 24)).isoformat()[:10]
            price *= (1 + rng.gauss(0.0, 0.02))
            price = max(price, 1.0)
            series.append((ts, round(price, 4)))
        result[sym] = series
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor pairwise correlations for regime shifts"
    )
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to bars SQLite DB")
    p.add_argument("--csv", default=None,
                   help="Path to CSV price file")
    p.add_argument("--symbols", nargs="+", default=INSTRUMENTS,
                   help="Instrument list")
    p.add_argument("--window", nargs="+", type=int, default=[20, 60],
                   help="Rolling window(s) in bars (default: 20 60)")
    p.add_argument("--alert-threshold", type=float, default=HIGH_CORR_THRESHOLD,
                   help=f"Alert threshold for avg correlation (default: {HIGH_CORR_THRESHOLD})")
    p.add_argument("--output", default=None,
                   help="Output HTML report path")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load price data
    prices: dict[str, list[tuple[str, float]]] = {}
    if args.csv:
        prices = load_prices_from_csv(Path(args.csv))
    if not prices:
        prices = load_prices_from_db(Path(args.db), args.symbols,
                                      window_days=max(args.window) * 2)
    if not prices:
        print("[corr_monitor] No price data found -- using synthetic demo", file=sys.stderr)
        prices = _synthetic_prices(args.symbols, n_bars=max(args.window) * 4)

    print(f"[corr_monitor] Loaded {len(prices)} symbols")

    windows = args.window
    records_by_window: dict[str, list] = {}
    summaries: list[dict] = []
    all_alerts: list[dict] = []

    for w in windows:
        label = f"{w}d"
        print(f"[corr_monitor] Computing rolling {label} correlations ...")
        records = rolling_pairwise_corr(prices, args.symbols, window=w)
        records_by_window[label] = records

        alerts = generate_alerts(records, high_corr_threshold=args.alert_threshold)
        all_alerts.extend(alerts)

        summary = latest_regime_summary(records, label, args.alert_threshold)
        summaries.append(summary)
        print(f"[corr_monitor]   {label}: regime={summary['regime']}  "
              f"avg_corr={summary.get('avg_corr') or 'n/a'}  alerts={len(alerts)}")

    # Deduplicate and sort alerts
    all_alerts.sort(key=lambda a: a["ts"])

    figs: list[Any] = []
    if HAS_MPL:
        figs.append(plot_rolling_avg_corr(records_by_window, args.alert_threshold))
        # Heatmap for first window
        first_label = f"{windows[0]}d"
        figs.append(plot_correlation_heatmap(
            records_by_window.get(first_label, []),
            args.symbols, first_label,
        ))
        # BTC-ETH pair history
        for label, records in records_by_window.items():
            fig = plot_pair_history(records, "BTC-ETH", label)
            if fig is not None:
                figs.append(fig)
        if args.show:
            plt.show()

    if args.output:
        print(f"[corr_monitor] Writing report to {args.output} ...")
        html = build_html_report(summaries, all_alerts, figs)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"[corr_monitor] Report saved: {args.output}")
    else:
        print_text_report(summaries, all_alerts)

    for fig in figs:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
