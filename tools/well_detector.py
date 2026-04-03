"""
well_detector.py — Standalone BH well detection on historical price data.

No trading.  Purely runs SRFM physics and outputs every formation/collapse event.
Useful for: calibrating BH parameters on new instruments before building a strategy.

Usage:
    python tools/well_detector.py --csv data/ES_hourly.csv --ticker ES
    python tools/well_detector.py --csv data/GC_hourly.csv --bh-form 1.2 --bh-collapse 0.35
    python tools/well_detector.py --csv data/ZB_hourly.csv --plot

CSV format expected: date, open, high, low, close, volume  (header row required)

Output:
    - Console table of all well events
    - Optional timeline plot
    - Optional CSV export to results/wells_<ticker>.csv
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from srfm_core import MinkowskiClassifier, BlackHoleDetector, BHState, Causal


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class Bar:
    date: str
    close: float
    ret: float = 0.0


def load_csv(path: str) -> List[Bar]:
    bars: List[Bar] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        prev_close = None
        for row in reader:
            # Try common column name variants
            close = float(row.get("close") or row.get("Close") or row.get("CLOSE") or 0)
            date  = row.get("date") or row.get("Date") or row.get("DATE") or row.get("time") or ""
            if prev_close and prev_close > 0:
                ret = (close - prev_close) / prev_close
            else:
                ret = 0.0
            bars.append(Bar(date=date, close=close, ret=ret))
            prev_close = close
    return bars


# ─── Well event record ────────────────────────────────────────────────────────

@dataclass
class WellEvent:
    formed_at: str
    collapsed_at: str
    direction: int        # +1 long, -1 short
    mass_peak: float
    duration_bars: int
    hawking_temp: float
    price_at_formation: float
    price_at_collapse: float
    price_move_pct: float


def direction_str(d: int) -> str:
    return "LONG" if d == 1 else "SHORT" if d == -1 else "FLAT"


# ─── Detector ─────────────────────────────────────────────────────────────────

def detect_wells(
    bars: List[Bar],
    cf: float,
    bh_form: float,
    bh_collapse: float,
    mass_decay: float,
) -> List[WellEvent]:
    mink = MinkowskiClassifier(cf=cf)
    bh   = BlackHoleDetector(
        bh_form_threshold=bh_form,
        bh_collapse_threshold=bh_collapse,
        mass_decay=mass_decay,
    )

    events: List[WellEvent] = []

    formation_bar:  Optional[Bar] = None
    peak_mass:      float = 0.0
    bars_in_well:   int   = 0
    well_direction: int   = 0
    in_well: bool = False

    for bar in bars:
        if bar.ret == 0.0:
            continue

        causal = mink.update(bar.ret)
        state  = bh.update(causal, bar.ret)

        if state == BHState.ACTIVE and not in_well:
            in_well = True
            formation_bar = bar
            peak_mass = bh.mass
            bars_in_well = 0
            well_direction = bh.direction

        elif state == BHState.ACTIVE and in_well:
            bars_in_well += 1
            if bh.mass > peak_mass:
                peak_mass = bh.mass
                well_direction = bh.direction

        elif state == BHState.COLLAPSE and in_well:
            in_well = False
            if formation_bar:
                p0 = formation_bar.close
                p1 = bar.close
                move = (p1 - p0) / p0 if p0 > 0 else 0.0
                temp = 1.0 / (8.0 * math.pi * peak_mass) if peak_mass > 0 else float("inf")
                events.append(WellEvent(
                    formed_at=formation_bar.date,
                    collapsed_at=bar.date,
                    direction=well_direction,
                    mass_peak=peak_mass,
                    duration_bars=bars_in_well,
                    hawking_temp=temp,
                    price_at_formation=p0,
                    price_at_collapse=p1,
                    price_move_pct=move * 100,
                ))
            formation_bar = None

    return events


# ─── Output ───────────────────────────────────────────────────────────────────

def print_events(events: List[WellEvent], ticker: str):
    if not events:
        print("No BH wells detected.")
        return

    print(f"\n{'─'*110}")
    print(f"  BH Well Report — {ticker}   ({len(events)} events)")
    print(f"{'─'*110}")
    header = (
        f"{'Formed':<20} {'Collapsed':<20} {'Dir':<6} {'MassPeak':>9} "
        f"{'Bars':>5} {'T_H':>8} {'P@Form':>9} {'P@Coll':>9} {'Move%':>7}"
    )
    print(header)
    print("─" * 110)
    for e in events:
        print(
            f"{e.formed_at:<20} {e.collapsed_at:<20} {direction_str(e.direction):<6} "
            f"{e.mass_peak:>9.4f} {e.duration_bars:>5} {e.hawking_temp:>8.5f} "
            f"{e.price_at_formation:>9.2f} {e.price_at_collapse:>9.2f} {e.price_move_pct:>+7.2f}%"
        )
    print("─" * 110)
    long_wells  = [e for e in events if e.direction == 1]
    short_wells = [e for e in events if e.direction == -1]
    avg_dur = sum(e.duration_bars for e in events) / len(events)
    avg_mass = sum(e.mass_peak for e in events) / len(events)
    print(f"\n  Total: {len(events)}  |  Long: {len(long_wells)}  |  Short: {len(short_wells)}")
    print(f"  Avg duration: {avg_dur:.1f} bars  |  Avg peak mass: {avg_mass:.4f}")
    print()


def save_csv(events: List[WellEvent], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "formed_at", "collapsed_at", "direction", "mass_peak",
            "duration_bars", "hawking_temp", "price_at_formation",
            "price_at_collapse", "price_move_pct",
        ])
        for e in events:
            writer.writerow([
                e.formed_at, e.collapsed_at, direction_str(e.direction),
                f"{e.mass_peak:.6f}", e.duration_bars, f"{e.hawking_temp:.6f}",
                f"{e.price_at_formation:.4f}", f"{e.price_at_collapse:.4f}",
                f"{e.price_move_pct:.4f}",
            ])
    print(f"CSV saved → {path}")


def plot_wells(bars: List[Bar], events: List[WellEvent], ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    dates  = [b.date for b in bars]
    closes = [b.close for b in bars]
    date_index = {d: i for i, d in enumerate(dates)}

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(range(len(closes)), closes, color="black", linewidth=0.8, label="Price")

    for e in events:
        i0 = date_index.get(e.formed_at, -1)
        i1 = date_index.get(e.collapsed_at, -1)
        if i0 < 0 or i1 < 0:
            continue
        color = "green" if e.direction == 1 else "red"
        ax.axvspan(i0, i1, alpha=0.2, color=color)

    long_patch  = mpatches.Patch(color="green", alpha=0.4, label="Long well")
    short_patch = mpatches.Patch(color="red",   alpha=0.4, label="Short well")
    ax.legend(handles=[ax.lines[0], long_patch, short_patch])
    ax.set_title(f"BH Well Timeline — {ticker}", fontweight="bold")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out = f"results/wells_{ticker}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SRFM BH well detector")
    parser.add_argument("--csv",        required=True, help="Path to price CSV")
    parser.add_argument("--ticker",     default="UNKNOWN", help="Ticker symbol (for labels)")
    parser.add_argument("--cf",         type=float, default=1.0, help="Minkowski speed-of-light factor")
    parser.add_argument("--bh-form",    type=float, default=1.5, help="BH formation threshold")
    parser.add_argument("--bh-collapse",type=float, default=0.4, help="BH collapse threshold")
    parser.add_argument("--mass-decay", type=float, default=0.92, help="Per-bar mass decay")
    parser.add_argument("--plot",       action="store_true", help="Generate timeline plot")
    parser.add_argument("--save-csv",   action="store_true", help="Export events to CSV")
    args = parser.parse_args()

    print(f"Loading {args.csv}...", end=" ")
    bars = load_csv(args.csv)
    print(f"{len(bars)} bars")

    events = detect_wells(
        bars,
        cf=args.cf,
        bh_form=args.bh_form,
        bh_collapse=args.bh_collapse,
        mass_decay=args.mass_decay,
    )

    print_events(events, args.ticker)

    if args.save_csv:
        save_csv(events, f"results/wells_{args.ticker}.csv")

    if args.plot:
        plot_wells(bars, events, args.ticker)


if __name__ == "__main__":
    main()
