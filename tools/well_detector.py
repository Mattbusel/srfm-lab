"""
well_detector.py — Standalone BH well detection on historical price CSV.

No trading. Runs SRFM physics (MinkowskiClassifier + BlackHoleDetector)
and outputs every formation/collapse event with timestamps and stats.

Usage:
    python tools/well_detector.py --csv data/synthetic_ES_hourly.csv --ticker ES --cf 0.001
    python tools/well_detector.py --csv data/GC_hourly.csv --bh-form 1.2 --bh-collapse 0.8 --plot
    python tools/well_detector.py --csv data/ES.csv --ticker ES --save-csv

CSV format required: date, open, high, low, close, volume  (header row)
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from srfm_core import MinkowskiClassifier, BlackHoleDetector


# ---- Data loading -------------------------------------------------------

@dataclass
class Bar:
    date:  str
    close: float


def load_csv(path: str) -> List[Bar]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            close = float(row.get("close") or row.get("Close") or row.get("CLOSE") or 0)
            date  = row.get("date") or row.get("Date") or row.get("DATE") or row.get("time") or ""
            if close > 0:
                bars.append(Bar(date=date, close=close))
    return bars


# ---- Well event record --------------------------------------------------

@dataclass
class WellEvent:
    formed_at:          str
    collapsed_at:       str
    direction:          int    # +1 long, -1 short
    mass_peak:          float
    duration_bars:      int
    hawking_temp:       float
    price_at_formation: float
    price_at_collapse:  float
    price_move_pct:     float


def _direction_str(d: int) -> str:
    return "LONG" if d == 1 else "SHORT" if d == -1 else "FLAT"


# ---- Detector -----------------------------------------------------------

def detect_wells(
    bars:        List[Bar],
    cf:          float,
    bh_form:     float,
    bh_collapse: float,
    mass_decay:  float,
) -> List[WellEvent]:
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(bh_form=bh_form, bh_collapse=bh_collapse, bh_decay=mass_decay)

    events:       List[WellEvent] = []
    formation_bar: Optional[Bar]  = None
    peak_mass:    float = 0.0
    bars_in_well: int   = 0
    in_well:      bool  = False

    prev_close: Optional[float] = None
    for bar in bars:
        if prev_close is None:
            prev_close = bar.close
            mc.update(bar.close)
            continue

        bit    = mc.update(bar.close)
        active = bh.update(bit, bar.close, prev_close)

        if active and not in_well:
            in_well       = True
            formation_bar = bar
            peak_mass     = bh.bh_mass
            bars_in_well  = 0

        elif active and in_well:
            bars_in_well += 1
            if bh.bh_mass > peak_mass:
                peak_mass = bh.bh_mass

        elif not active and in_well:
            in_well = False
            if formation_bar:
                p0   = formation_bar.close
                p1   = bar.close
                move = (p1 - p0) / p0 * 100 if p0 > 0 else 0.0
                temp = 1.0 / (8.0 * math.pi * peak_mass) if peak_mass > 0 else float("inf")
                events.append(WellEvent(
                    formed_at=formation_bar.date,
                    collapsed_at=bar.date,
                    direction=bh.bh_dir,
                    mass_peak=peak_mass,
                    duration_bars=bars_in_well,
                    hawking_temp=temp,
                    price_at_formation=p0,
                    price_at_collapse=p1,
                    price_move_pct=move,
                ))
            formation_bar = None
            peak_mass     = 0.0
            bars_in_well  = 0

        prev_close = bar.close

    return events


# ---- Output -------------------------------------------------------------

def print_events(events: List[WellEvent], ticker: str):
    if not events:
        print("No BH wells detected.")
        return
    w = 108
    print(f"\n{'-'*w}")
    print(f"  BH Well Report -- {ticker}   ({len(events)} events)")
    print(f"{'-'*w}")
    hdr = (f"{'Formed':<20} {'Collapsed':<20} {'Dir':<6} {'MassPeak':>9} "
           f"{'Bars':>5} {'T_H':>9} {'P@Form':>9} {'P@Coll':>9} {'Move%':>7}")
    print(hdr)
    print("-" * w)
    for e in events:
        print(f"{e.formed_at:<20} {e.collapsed_at:<20} "
              f"{_direction_str(e.direction):<6} {e.mass_peak:>9.4f} "
              f"{e.duration_bars:>5} {e.hawking_temp:>9.5f} "
              f"{e.price_at_formation:>9.2f} {e.price_at_collapse:>9.2f} "
              f"{e.price_move_pct:>+7.2f}%")
    print("-" * w)
    long_w  = [e for e in events if e.direction ==  1]
    short_w = [e for e in events if e.direction == -1]
    avg_dur  = sum(e.duration_bars for e in events)  / len(events)
    avg_mass = sum(e.mass_peak     for e in events)  / len(events)
    avg_move = sum(abs(e.price_move_pct) for e in events) / len(events)
    print(f"\n  Total: {len(events)}  |  Long: {len(long_w)}  |  Short: {len(short_w)}")
    print(f"  Avg duration : {avg_dur:.1f} bars")
    print(f"  Avg peak mass: {avg_mass:.4f}")
    print(f"  Avg |move|   : {avg_move:.2f}%")
    long_moves  = [e.price_move_pct for e in long_w]
    short_moves = [e.price_move_pct for e in short_w]
    if long_moves:
        print(f"  Best long    : {max(long_moves):+.2f}%  Worst long: {min(long_moves):+.2f}%")
    if short_moves:
        print(f"  Best short   : {min(short_moves):+.2f}%  Worst short: {max(short_moves):+.2f}%")
    print()


def save_csv(events: List[WellEvent], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "formed_at", "collapsed_at", "direction", "mass_peak",
            "duration_bars", "hawking_temp", "price_at_formation",
            "price_at_collapse", "price_move_pct",
        ])
        for e in events:
            writer.writerow([
                e.formed_at, e.collapsed_at, _direction_str(e.direction),
                f"{e.mass_peak:.6f}", e.duration_bars, f"{e.hawking_temp:.6f}",
                f"{e.price_at_formation:.4f}", f"{e.price_at_collapse:.4f}",
                f"{e.price_move_pct:.4f}",
            ])
    print(f"CSV saved -> {path}")


def plot_wells(bars: List[Bar], events: List[WellEvent], ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    dates  = [b.date  for b in bars]
    closes = [b.close for b in bars]
    di     = {d: i for i, d in enumerate(dates)}

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(range(len(closes)), closes, color="black", linewidth=0.7, label="Price")

    for e in events:
        i0 = di.get(e.formed_at,    -1)
        i1 = di.get(e.collapsed_at, -1)
        if i0 >= 0 and i1 >= 0:
            color = "green" if e.direction == 1 else "red"
            ax.axvspan(i0, i1, alpha=0.18, color=color)

    long_p  = mpatches.Patch(color="green", alpha=0.4, label="Long well")
    short_p = mpatches.Patch(color="red",   alpha=0.4, label="Short well")
    ax.legend(handles=[ax.lines[0], long_p, short_p])
    ax.set_title(f"BH Well Timeline -- {ticker}  ({len(events)} wells)", fontweight="bold")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = f"results/wells_{ticker}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")
    plt.show()


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM BH well detector")
    parser.add_argument("--csv",         required=True)
    parser.add_argument("--ticker",      default="UNKNOWN")
    parser.add_argument("--cf",          type=float, default=0.001)
    parser.add_argument("--bh-form",     type=float, default=1.5)
    parser.add_argument("--bh-collapse", type=float, default=1.0)
    parser.add_argument("--mass-decay",  type=float, default=0.95)
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--save-csv",    action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.csv}...", end=" ")
    bars = load_csv(args.csv)
    print(f"{len(bars)} bars")

    events = detect_wells(bars, args.cf, args.bh_form, args.bh_collapse, args.mass_decay)
    print_events(events, args.ticker)

    if args.save_csv:
        save_csv(events, f"results/wells_{args.ticker}.csv")

    if args.plot:
        plot_wells(bars, events, args.ticker)


if __name__ == "__main__":
    main()
