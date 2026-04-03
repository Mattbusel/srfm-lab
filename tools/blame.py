#!/usr/bin/env python
"""blame.py — Date range → ranked trades. "What made/lost money in this period?"

Usage:
    python tools/blame.py --from 2024-01-01 --to 2024-12-31 --csv trades.csv
    cat trades.csv | python tools/blame.py --from 2024-01-01 --to 2024-12-31
"""
import sys
import csv
import os
import io
import argparse
from datetime import datetime, date

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

RULE = "━" * 53


def parse_date(s):
    s = s.strip()
    # Strip trailing Z or timezone offset for ISO strings
    for suffix in ("+00:00", "Z"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
                "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def parse_csv_trades(stream_or_path):
    """Parse trades CSV. Tries multiple common column name conventions."""
    if isinstance(stream_or_path, str):
        f = open(stream_or_path, newline="", encoding="utf-8-sig")
    else:
        content = stream_or_path.read()
        f = io.StringIO(content)

    reader = csv.DictReader(f)
    headers = [h.strip().lower() for h in (reader.fieldnames or [])]

    def col(*candidates):
        for c in candidates:
            for h in (reader.fieldnames or []):
                if h.strip().lower() == c.lower():
                    return h
        return None

    date_col   = col("entry time", "date", "entry date", "entrydate", "trade date", "closedate", "close date", "exit time")
    pnl_col    = col("p&l", "net profit", "net_profit", "netprofit", "pnl", "net pnl", "profit")
    instr_col  = col("symbols", "symbol", "instrument", "ticker", "contract")
    dir_col    = col("direction", "type", "side", "trade type")

    trades = []
    for row in reader:
        if date_col:
            dt = parse_date(str(row.get(date_col, "") or ""))
        else:
            dt = None

        pnl = 0.0
        if pnl_col:
            raw = str(row.get(pnl_col, "0") or "0").replace("$", "").replace(",", "").strip()
            try:
                pnl = float(raw)
            except ValueError:
                continue

        instr = ""
        if instr_col:
            instr = str(row.get(instr_col, "")).strip()
            # Normalize: ES, NQ, YM from full contract names
            for base in ("ES", "NQ", "YM", "MNQ", "MES"):
                if base in instr.upper():
                    instr = base
                    break

        direction = ""
        if dir_col:
            direction = str(row.get(dir_col, "")).strip()
            d_upper = direction.upper()
            if any(k in d_upper for k in ("BUY", "LONG")):
                direction = "Buy"
            elif any(k in d_upper for k in ("SELL", "SHORT")):
                direction = "Sell"

        trades.append({"date": dt, "pnl": pnl, "instrument": instr, "direction": direction, "raw": row})

    if isinstance(stream_or_path, str):
        f.close()
    return trades


def filter_trades(trades, from_date, to_date, instrument=None):
    result = []
    for t in trades:
        if t["date"] is None:
            continue
        if from_date and t["date"] < from_date:
            continue
        if to_date and t["date"] > to_date:
            continue
        if instrument and t["instrument"].upper() != instrument.upper():
            continue
        result.append(t)
    return result


def side_by_side(left_lines, right_lines, gap=4):
    max_left = max((len(l) for l in left_lines), default=0)
    out = []
    for i in range(max(len(left_lines), len(right_lines))):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        out.append(f"{l:<{max_left}}{' '*gap}{r}")
    return out


def report(trades, from_date, to_date, top_n=5, label="", compare_trades=None):
    if not trades:
        print(RULE)
        print(f"  BLAME REPORT: {from_date} → {to_date}{' ' + label if label else ''}")
        print(RULE)
        print("  No trades found in this date range.")
        print(RULE)
        return

    total_pnl = sum(t["pnl"] for t in trades)
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n * 100 if n else 0

    sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    winners = sorted_by_pnl[:top_n]
    losers  = sorted(trades, key=lambda t: t["pnl"])[:top_n]

    pnl_sign = "+" if total_pnl >= 0 else ""

    print(RULE)
    print(f"  BLAME REPORT: {from_date} → {to_date}{' ' + label if label else ''}")
    print(f"  {n} trades  |  Net: {pnl_sign}${total_pnl:,.0f}  |  WR: {wr:.1f}%")
    print(RULE)

    def fmt_trade(t, winner=True):
        d = str(t["date"]) if t["date"] else "????"
        instr = t["instrument"] or "??"
        direc = t["direction"] or "??"
        pnl   = t["pnl"]
        sign  = "+" if pnl >= 0 else ""
        return f"  {d}  {instr:<4} {direc:<4} {sign}${pnl:,.0f}"

    # Build left/right columns
    win_lines  = ["  TOP WINNERS"] + [fmt_trade(t, True)  for t in winners]
    loss_lines = ["  TOP LOSERS"]  + [fmt_trade(t, False) for t in losers]

    paired = side_by_side(win_lines, loss_lines, gap=4)
    for line in paired:
        print(line)

    print(RULE)

    # By instrument
    instrs = sorted(set(t["instrument"] for t in trades if t["instrument"]))
    if instrs:
        instr_parts = []
        for ins in instrs:
            ins_trades = [t for t in trades if t["instrument"] == ins]
            ins_pnl = sum(t["pnl"] for t in ins_trades)
            sign = "+" if ins_pnl >= 0 else ""
            instr_parts.append(f"{ins} {sign}${ins_pnl/1000:.0f}k")
        print(f"  By Instrument:  {' '.join(instr_parts)}")

    # By direction
    dirs = sorted(set(t["direction"] for t in trades if t["direction"]))
    if dirs:
        dir_parts = []
        for d in dirs:
            d_trades = [t for t in trades if t["direction"] == d]
            d_pnl = sum(t["pnl"] for t in d_trades)
            sign = "+" if d_pnl >= 0 else ""
            dir_parts.append(f"{d} {sign}${d_pnl/1000:.0f}k")
        print(f"  By Direction:   {' '.join(dir_parts)}")

    # Win rate by instrument
    if instrs:
        wr_parts = []
        for ins in instrs:
            ins_trades = [t for t in trades if t["instrument"] == ins]
            ins_wins = sum(1 for t in ins_trades if t["pnl"] > 0)
            ins_wr = ins_wins / len(ins_trades) * 100 if ins_trades else 0
            wr_parts.append(f"{ins} {ins_wr:.0f}%")
        print(f"  Win rate:       {'  '.join(wr_parts)}")

    print(RULE)

    # Compare mode
    if compare_trades is not None:
        total_pnl2 = sum(t["pnl"] for t in compare_trades)
        n2 = len(compare_trades)
        wins2 = sum(1 for t in compare_trades if t["pnl"] > 0)
        wr2 = wins2 / n2 * 100 if n2 else 0
        diff = total_pnl2 - total_pnl
        sign = "+" if diff >= 0 else ""
        print(f"  COMPARE (v2):  {n2} trades  Net: ${total_pnl2:,.0f}  WR: {wr2:.1f}%")
        print(f"  Difference:    {sign}${diff:,.0f}")
        print(RULE)


def main():
    parser = argparse.ArgumentParser(description="Blame report — ranked trades for a date range")
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--to",   dest="to_date",   default=None)
    parser.add_argument("--csv",  dest="csv_path",  default=None)
    parser.add_argument("--top",  dest="top_n",     type=int, default=5)
    parser.add_argument("--instrument", dest="instrument", default=None)
    parser.add_argument("--compare", dest="compare_csv", default=None)
    args = parser.parse_args()

    from_date = parse_date(args.from_date) if args.from_date else None
    to_date   = parse_date(args.to_date)   if args.to_date   else None

    # Load trades
    if args.csv_path:
        trades = parse_csv_trades(args.csv_path)
    elif not sys.stdin.isatty():
        trades = parse_csv_trades(sys.stdin)
    else:
        print("Provide --csv PATH or pipe a CSV to stdin", file=sys.stderr)
        sys.exit(1)

    trades = filter_trades(trades, from_date, to_date, args.instrument)

    compare_trades = None
    if args.compare_csv:
        raw2 = parse_csv_trades(args.compare_csv)
        compare_trades = filter_trades(raw2, from_date, to_date, args.instrument)

    label = f"[{args.instrument}]" if args.instrument else ""
    report(trades, from_date or "all", to_date or "all", args.top_n, label, compare_trades)


if __name__ == "__main__":
    main()
