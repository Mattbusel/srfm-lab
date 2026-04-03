# -*- coding: utf-8 -*-
"""
bh_pulse.py — BH mass pulse chart.

Visualizes Black Hole mass over time — shows when BH forms, how long it lasts,
and the mass trajectory.

Usage:
    python tools/bh_pulse.py --csv data/NDX_hourly_poly.csv --cf 0.005
    python tools/bh_pulse.py --synthetic --n-bars 3000
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np

_ARENA_DIR = os.path.dirname(__file__)
sys.path.insert(0, _ARENA_DIR)
from arena_v2 import run_v2, load_ohlcv, generate_synthetic

CHART_WIDTH = 72
CHART_HEIGHT = 12
FORMATION_THRESHOLD = 1.5
BLOCK = "█"
ACTIVE_CHAR = "■"
INACTIVE_CHAR = " "


def _sparkline(values: list, width: int, height: int, threshold: float) -> list:
    """Return list of strings — ASCII bar chart rows (top to bottom)."""
    if not values:
        return []
    # Downsample to width
    n = len(values)
    if n > width:
        # Average pool
        chunk = n / width
        sampled = []
        for i in range(width):
            start = int(i * chunk)
            end = int((i + 1) * chunk)
            sampled.append(sum(values[start:end]) / max(1, end - start))
    else:
        sampled = values + [0.0] * (width - n)

    max_val = max(sampled) if max(sampled) > 0 else 1.0
    rows = []
    for h in range(height, 0, -1):
        row_thresh = max_val * h / height
        row = ""
        for v in sampled:
            row += BLOCK if v >= row_thresh else " "
        rows.append(row)
    return rows, sampled, max_val


def _active_bar(bh_active_sampled: list) -> str:
    return "".join(ACTIVE_CHAR if a else INACTIVE_CHAR for a in bh_active_sampled)


def _parse_year(date_str: str) -> int:
    import re
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else 0


def _axis_label(n_bars: int, width: int, dates: list) -> str:
    """Build time-axis label string."""
    if not dates or not dates[0]:
        # Synthetic
        return " " * width
    # Try to get year markers
    years_at = {}
    for i, d in enumerate(dates):
        y = _parse_year(d)
        col = int(i * width / max(len(dates), 1))
        if y and y not in years_at:
            years_at[col] = str(y)
    label = [" "] * width
    for col, yr in years_at.items():
        for j, ch in enumerate(yr):
            if col + j < width:
                label[col + j] = ch
    return "".join(label)


def _duration_histogram(durations: list) -> list:
    """ASCII histogram of BH active durations."""
    if not durations:
        return []
    max_dur = max(durations)
    bins = [0, 5, 10, 20, 40, 80, 160, 999999]
    bin_labels = ["1-5", "6-10", "11-20", "21-40", "41-80", "81-160", "161+"]
    counts = [0] * (len(bins) - 1)
    for d in durations:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                counts[i] += 1
                break
    max_c = max(counts) if counts else 1
    bar_w = 20
    lines = ["BH ACTIVE DURATION HISTOGRAM (bars):"]
    for label, cnt in zip(bin_labels, counts):
        filled = int(cnt * bar_w / max_c)
        lines.append(f"  {label:>7}h: {'█' * filled:<{bar_w}} {cnt}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="LARSA BH Mass Pulse Chart")
    parser.add_argument("--csv", help="Price CSV file")
    parser.add_argument("--ticker", default="NDX")
    parser.add_argument("--cf", type=float, default=0.005)
    parser.add_argument("--bh-form", type=float, default=1.5)
    parser.add_argument("--bh-decay", type=float, default=0.95)
    parser.add_argument("--max-leverage", type=float, default=0.65)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-bars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp", default="", help="Arena experiment flags e.g. ABCD")
    parser.add_argument("--last-n", type=int, default=0, help="Only chart last N bars")
    parser.add_argument("--out", default="results/bh_pulse.md")
    args = parser.parse_args()

    if args.synthetic:
        bars = generate_synthetic(args.n_bars, args.seed)
        ticker = "SYNTH"
    elif args.csv:
        bars = load_ohlcv(args.csv)
        ticker = args.ticker
        print(f"  Loaded {len(bars)} bars from {args.csv}")
    else:
        parser.error("Provide --csv or --synthetic")

    cfg = {
        "cf": args.cf,
        "bh_form": args.bh_form,
        "bh_collapse": 1.0,
        "bh_decay": args.bh_decay,
    }

    print(f"  Running arena_v2 (verbose) cf={args.cf} ...")
    broker, bar_log = run_v2(bars, cfg, args.max_leverage, args.exp, verbose=True)

    if not bar_log:
        print("ERROR: bar_log is empty.")
        sys.exit(1)

    # Optionally restrict to last N
    if args.last_n and args.last_n < len(bar_log):
        bar_log = bar_log[-args.last_n:]

    n = len(bar_log)
    print(f"  Got {n} bar records.")

    masses = [e.get("bh_mass", 0.0) for e in bar_log]
    actives = [bool(e.get("bh_active", False)) for e in bar_log]
    dates = [e.get("date", "") for e in bar_log]

    # ── Active runs ───────────────────────────────────────────────────────────
    durations = []
    in_run = False
    run_len = 0
    for a in actives:
        if a:
            in_run = True
            run_len += 1
        else:
            if in_run:
                durations.append(run_len)
            in_run = False
            run_len = 0
    if in_run and run_len > 0:
        durations.append(run_len)

    # ── Per-year formation count ──────────────────────────────────────────────
    year_formations: dict = defaultdict(int)
    prev_active = False
    for i, (a, d) in enumerate(zip(actives, dates)):
        if a and not prev_active:
            y = _parse_year(d)
            if y:
                year_formations[y] += 1
        prev_active = a

    # ── Build chart ───────────────────────────────────────────────────────────
    rows, sampled_mass, max_mass = _sparkline(masses, CHART_WIDTH, CHART_HEIGHT, FORMATION_THRESHOLD)

    # Downsample actives to CHART_WIDTH
    n_vals = len(actives)
    sampled_active = []
    if n_vals > CHART_WIDTH:
        chunk = n_vals / CHART_WIDTH
        for i in range(CHART_WIDTH):
            start = int(i * chunk)
            end = int((i + 1) * chunk)
            chunk_active = actives[start:end]
            sampled_active.append(any(chunk_active))
    else:
        sampled_active = actives + [False] * (CHART_WIDTH - n_vals)

    active_line = _active_bar(sampled_active)
    threshold_row = max(0, CHART_HEIGHT - 1 - int((FORMATION_THRESHOLD / max_mass) * CHART_HEIGHT))
    axis_label = _axis_label(n, CHART_WIDTH, dates)

    # ── Format output ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"BH MASS PULSE CHART — {ticker} ({n} bars, cf={args.cf})")
    lines.append("=" * (CHART_WIDTH + 12))

    for i, row in enumerate(rows):
        y_val = max_mass * (CHART_HEIGHT - i) / CHART_HEIGHT
        marker = " ← formation threshold" if i == threshold_row else ""
        lines.append(f"{y_val:5.2f} |{row}{marker}")

    lines.append("      +" + "─" * CHART_WIDTH)
    lines.append("      " + axis_label)
    lines.append("")
    lines.append(f"Formation threshold: {FORMATION_THRESHOLD:.1f} " + "─" * 30)
    lines.append(f"BH active: {active_line}")
    lines.append("")

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_active = sum(1 for a in actives if a)
    total_inactive = n - total_active
    pct_active = 100.0 * total_active / n if n else 0.0
    lines.append(f"BH STATS:")
    lines.append(f"  Total bars:         {n:,}")
    lines.append(f"  BH active bars:     {total_active:,} ({pct_active:.1f}%)")
    lines.append(f"  BH inactive bars:   {total_inactive:,} ({100-pct_active:.1f}%)")
    lines.append(f"  Formation events:   {len(durations)}")
    if durations:
        lines.append(f"  Avg active run:     {sum(durations)/len(durations):.1f} bars")
        lines.append(f"  Max active run:     {max(durations)} bars")
        lines.append(f"  Min active run:     {min(durations)} bars")
    lines.append(f"  Max BH mass seen:   {max(masses):.4f}")
    lines.append(f"  Mean BH mass:       {sum(masses)/len(masses):.4f}")

    if year_formations:
        lines.append("")
        lines.append("BH FORMATIONS PER YEAR:")
        for yr in sorted(year_formations):
            lines.append(f"  {yr}: {year_formations[yr]} formations")

    lines.append("")
    lines += _duration_histogram(durations)

    output = "\n".join(lines)
    print()
    print(output)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("```\n" + output + "\n```\n")
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
