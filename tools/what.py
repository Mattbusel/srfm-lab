#!/usr/bin/env python
"""what.py — Give a date, get full strategy context.

Usage:
    python tools/what.py 2024-10-14
"""
import sys
import json
import os
import csv
from datetime import datetime, timezone, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIMES_CSV   = os.path.join(LAB, "results", "regimes_ES.csv")
ANALYSIS_JSON = os.path.join(LAB, "research", "trade_analysis_data.json")
ANALYSIS_V3   = os.path.join(LAB, "research", "trade_analysis_v3_data.json")

RULE = "━" * 53

def parse_dt(s):
    """Parse ISO or date-only string to aware datetime (UTC)."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def load_regimes():
    rows = []
    try:
        with open(REGIMES_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = parse_dt(row["date"])
                if dt:
                    rows.append({"dt": dt, "regime": row["regime"],
                                 "confidence": float(row.get("confidence", 0))})
    except FileNotFoundError:
        pass
    return rows


def find_regime(rows, target_dt):
    if not rows:
        return None
    best = min(rows, key=lambda r: abs((r["dt"] - target_dt).total_seconds()))
    return best


def load_analysis(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_active_wells(wells, target_dt):
    active = []
    for w in wells:
        start = parse_dt(w.get("start", ""))
        end   = parse_dt(w.get("end", ""))
        if start and end and start <= target_dt <= end:
            active.append(w)
    return active


def interpolate_equity(equity_curve, target_dt):
    """Find equity value nearest to target date."""
    if not equity_curve:
        return None, None
    parsed = []
    for entry in equity_curve:
        dt = parse_dt(entry[0])
        if dt:
            parsed.append((dt, entry[1]))
    if not parsed:
        return None, None
    parsed.sort(key=lambda x: x[0])

    # Find bounding points
    before = [(dt, v) for dt, v in parsed if dt <= target_dt]
    after  = [(dt, v) for dt, v in parsed if dt > target_dt]

    if not before and not after:
        return None, None
    if not before:
        return after[0][1], after[0][0]
    if not after:
        return before[-1][1], before[-1][0]

    # Interpolate
    t0, v0 = before[-1]
    t1, v1 = after[0]
    span = (t1 - t0).total_seconds()
    if span == 0:
        return v0, t0
    frac = (target_dt - t0).total_seconds() / span
    value = v0 + (v1 - v0) * frac
    return value, t0


def find_nearby_wins(wells, target_dt, n=1):
    """Find nearest wins before and after target date."""
    wins = [w for w in wells if w.get("is_win") and w.get("net_pnl", 0) > 50000]
    before = sorted([w for w in wins if parse_dt(w["end"]) < target_dt],
                    key=lambda w: parse_dt(w["end"]), reverse=True)
    after  = sorted([w for w in wins if parse_dt(w["start"]) > target_dt],
                    key=lambda w: parse_dt(w["start"]))
    return before[:n], after[:n]


def days_between(dt1, dt2):
    return abs((dt2 - dt1).days)


def pct_gain(value, baseline=1_000_000):
    return (value / baseline - 1) * 100


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/what.py YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    date_str = sys.argv[1]
    target_dt = parse_dt(date_str)
    if not target_dt:
        print(f"Could not parse date: {date_str}", file=sys.stderr)
        sys.exit(1)
    # normalize to noon UTC
    target_dt = target_dt.replace(hour=12, minute=0, second=0)

    # Load data
    regime_rows = load_regimes()
    data_v1 = load_analysis(ANALYSIS_JSON)
    data_v3 = load_analysis(ANALYSIS_V3)

    print(RULE)
    print(f"  {date_str} — Strategy Context")
    print(RULE)

    # Regime
    reg = find_regime(regime_rows, target_dt)
    if reg:
        print(f"  Regime:     {reg['regime']} (confidence {reg['confidence']:.2f})")
    else:
        print("  Regime:     (no regime data)")

    # Multi-instrument regime (check if NQ/YM regimes files exist)
    reg_parts = []
    for ticker in ["ES", "NQ", "YM"]:
        path = os.path.join(LAB, "results", f"regimes_{ticker}.csv")
        rows = []
        if os.path.exists(path):
            try:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dt = parse_dt(row["date"])
                        if dt:
                            rows.append({"dt": dt, "regime": row["regime"]})
            except Exception:
                pass
        if rows:
            best = min(rows, key=lambda r: abs((r["dt"] - target_dt).total_seconds()))
            reg_parts.append(f"{ticker} regime: {best['regime']}")
    if reg_parts:
        print(f"  {' | '.join(reg_parts)}")

    print()

    # Wells active that day (v1)
    if data_v1:
        wells_v1 = data_v1.get("wells", [])
        active = find_active_wells(wells_v1, target_dt)
        print(f"  Wells active that day:")
        if active:
            for w in active:
                instr = ", ".join(w.get("instruments", []))
                dirs  = ", ".join(w.get("directions", []))
                start = w.get("start", "?")[:10]
                net   = w.get("net_pnl", 0)
                peak  = w.get("pnl_pct", 0)
                status = "[OPEN]" if w.get("is_win") is None else ("[WIN]" if w.get("is_win") else "[LOSS]")
                pnl_str = f"+${net:,.0f}" if net >= 0 else f"-${abs(net):,.0f}"
                print(f"    {instr:<4} {dirs:<4} started {start}  pnl_pct={peak:.2f}  {status}")
                if not w.get("is_win") and net < -50000:
                    print(f"    [!] This well lost ${net:,.0f}")
        else:
            print("    None — no active wells")

    print()

    # Equity values
    baseline = 1_000_000
    if data_v1:
        ec_v1 = data_v1.get("equity_curve", [])
        val_v1, _ = interpolate_equity(ec_v1, target_dt)
        if val_v1:
            print(f"  v1 portfolio that day:  ~${val_v1:,.0f}  ({pct_gain(val_v1, baseline):+.1f}% from start)")
    if data_v3:
        ec_v3 = data_v3.get("equity_curve", [])
        val_v3, _ = interpolate_equity(ec_v3, target_dt)
        if val_v3:
            print(f"  v3 portfolio that day:  ~${val_v3:,.0f}  ({pct_gain(val_v3, baseline):+.1f}% from start)")

    print()

    # Flat periods
    if data_v1:
        flat = data_v1.get("flat_periods", [])
        nearby_flat = []
        for fp in flat:
            start = parse_dt(fp.get("start", ""))
            end   = parse_dt(fp.get("end", ""))
            if start and end:
                if start <= target_dt <= end:
                    nearby_flat.append(fp)
        print("  Nearby flat periods:")
        if nearby_flat:
            for fp in nearby_flat:
                print(f"    {fp.get('start','?')[:10]} → {fp.get('end','?')[:10]}")
        else:
            print("    None — strategy was active")

    print()

    # Days since last big win / until next
    if data_v1:
        wells_v1 = data_v1.get("wells", [])
        before_wins, after_wins = find_nearby_wins(wells_v1, target_dt)
        if before_wins:
            w = before_wins[0]
            end_dt = parse_dt(w["end"])
            days = days_between(end_dt, target_dt)
            instr = ", ".join(w.get("instruments", []))
            dirs  = ", ".join(w.get("directions", []))
            net   = w.get("net_pnl", 0)
            print(f"  Days since last big win: {days} ({end_dt.date()} {instr} {dirs} +${net:,.0f})")
        if after_wins:
            w = after_wins[0]
            start_dt = parse_dt(w["start"])
            days = days_between(target_dt, start_dt)
            instr = ", ".join(w.get("instruments", []))
            dirs  = ", ".join(w.get("directions", []))
            net   = w.get("net_pnl", 0)
            print(f"  Days until next big win: {days} ({start_dt.date()} {instr} {dirs} +${net:,.0f})")

    print(RULE)


if __name__ == "__main__":
    main()
