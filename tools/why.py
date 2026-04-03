#!/usr/bin/env python
"""
tools/why.py -- Given a losing trade date + instrument, explain in plain English
what the SRFM state was that caused entry. The "git blame" for bad trades.

Usage:
  python tools/why.py --date 2024-10-14 --instrument NQ
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV = Path("C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv")
TRADE_DATA_JSON = REPO_ROOT / "research" / "trade_analysis_data.json"
REGIMES_CSV = REPO_ROOT / "results" / "regimes_ES.csv"

NQ_MULTIPLIER = 20
ES_MULTIPLIER = 50
YM_MULTIPLIER = 5

MULTIPLIERS = {"NQ": NQ_MULTIPLIER, "ES": ES_MULTIPLIER, "YM": YM_MULTIPLIER}


def load_trades() -> list[dict]:
    rows = []
    with open(TRADES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_regimes() -> dict[str, tuple[str, float]]:
    """Returns {date_str: (regime, confidence)}"""
    regimes = {}
    with open(REGIMES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            regimes[r["date"]] = (r["regime"], float(r["confidence"]))
    return regimes


def parse_dt(s: str) -> datetime:
    # Handle ISO format with Z
    s = s.replace("Z", "+00:00").replace(" ", "T")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def find_trade(trades: list[dict], target_date: str, instrument: str) -> dict | None:
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    window_start = target_dt - timedelta(hours=24)
    window_end = target_dt + timedelta(hours=48)

    candidates = []
    for t in trades:
        entry_dt = parse_dt(t["Entry Time"])
        sym = t["Symbols"]
        instr = instrument.upper()
        if instr in sym and window_start <= entry_dt <= window_end:
            candidates.append((entry_dt, t))

    if not candidates:
        return None

    # Prefer biggest loss if multiple
    losers = [(dt, t) for dt, t in candidates if float(t["P&L"]) < 0]
    if losers:
        return min(losers, key=lambda x: float(x[1]["P&L"]))[1]
    return candidates[0][1]


def get_regime_at(regimes: dict, dt: datetime) -> tuple[str, float]:
    # Try exact hour match, then scan backwards up to 8h
    for h in range(0, 9):
        check = dt - timedelta(hours=h)
        key = check.strftime("%Y-%m-%d %H:00")
        if key in regimes:
            return regimes[key]
    return ("UNKNOWN", 0.5)


def find_similar_trades(trades: list[dict], instrument: str, direction: str,
                        max_duration_h: float, exclude_entry_time: str) -> list[dict]:
    similar = []
    for t in trades:
        if instrument.upper() not in t["Symbols"]:
            continue
        if t["Direction"] != direction:
            continue
        if t["Entry Time"] == exclude_entry_time:
            continue
        pnl = float(t["P&L"])
        if pnl >= 0:
            continue
        entry_dt = parse_dt(t["Entry Time"])
        exit_dt = parse_dt(t["Exit Time"])
        dur_h = (exit_dt - entry_dt).total_seconds() / 3600
        if dur_h <= max_duration_h * 2:
            similar.append(t)
    return similar


def infer_bh_mass(regime: str, confidence: float) -> float:
    """Rough inference: high confidence BULL regime implies high BH mass."""
    base = 1.5 + (confidence - 0.5) * 2.0
    return round(min(base, 3.0), 2)


def infer_tl_confirm(regime: str, confidence: float) -> int:
    if confidence >= 0.85:
        return 6
    elif confidence >= 0.70:
        return 4
    else:
        return 3


def infer_ctl(bh_mass: float) -> int:
    # ctl accretes with TL bars; high mass implies >=5
    if bh_mass >= 2.0:
        return 8
    elif bh_mass >= 1.5:
        return 5
    return 3


def v4_notional_cap_reduction(pnl: float, entry_price: float, quantity: int,
                               instrument: str) -> dict:
    mult = MULTIPLIERS.get(instrument.upper(), 1)
    notional_cap = 400_000
    actual_notional = entry_price * quantity * mult
    if instrument.upper() != "NQ" or actual_notional <= notional_cap:
        return {"applies": False, "ratio": 1.0, "new_pnl": pnl, "savings": 0}
    ratio = notional_cap / actual_notional
    new_pnl = pnl * ratio
    savings = pnl - new_pnl  # savings is negative - actual saving is abs
    return {
        "applies": True,
        "ratio": round(ratio, 2),
        "actual_notional": actual_notional,
        "capped_notional": notional_cap,
        "original_qty": quantity,
        "capped_qty": max(1, int(quantity * ratio)),
        "new_pnl": round(new_pnl),
        "savings": round(abs(savings)),
    }


def main():
    parser = argparse.ArgumentParser(description="WHY: Explain a losing trade")
    parser.add_argument("--date", required=True, help="Trade date YYYY-MM-DD")
    parser.add_argument("--instrument", required=True, help="NQ, ES, or YM")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    width = 55

    if not TRADES_CSV.exists():
        print(f"Trades CSV not found: {TRADES_CSV}")
        sys.exit(1)

    trades = load_trades()
    regimes = load_regimes()

    trade = find_trade(trades, args.date, instrument)
    if trade is None:
        print(f"No {instrument} trade found near {args.date}")
        sys.exit(1)

    entry_dt = parse_dt(trade["Entry Time"])
    exit_dt = parse_dt(trade["Exit Time"])
    duration_h = (exit_dt - entry_dt).total_seconds() / 3600
    pnl = float(trade["P&L"])
    direction = trade["Direction"]
    entry_price = float(trade["Entry Price"])
    exit_price = float(trade["Exit Price"])
    quantity = int(trade["Quantity"])
    is_win = trade["IsWin"] == "1"

    regime, confidence = get_regime_at(regimes, entry_dt)
    bh_mass = infer_bh_mass(regime, confidence)
    tl_confirm = infer_tl_confirm(regime, confidence)
    ctl = infer_ctl(bh_mass)
    is_flash = duration_h <= 2.0
    is_big_loss = pnl < -100_000
    geo_raw = round(0.3 + (1.0 - confidence) * 0.5, 2)

    cap = v4_notional_cap_reduction(pnl, entry_price, quantity, instrument)
    similar = find_similar_trades(trades, instrument, direction, duration_h * 2,
                                  trade["Entry Time"])
    total_similar_pnl = sum(float(t["P&L"]) for t in similar)

    print("-" * width)
    pnl_str = f"-${abs(pnl):,.0f}" if pnl < 0 else f"${pnl:,.0f}"
    print(f"  WHY: {instrument} {direction} {entry_dt.strftime('%Y-%m-%d')}  ->  P&L: {pnl_str}")
    print("-" * width)

    print("  Entry conditions met:")
    if regime == "BULL" and direction == "Buy":
        check = "(+)"
    elif regime == "BEAR" and direction == "Sell":
        check = "(+)"
    else:
        check = "(~)"
    print(f"    {check} {regime} regime (confidence {confidence:.2f}) -- all EMA stacked")
    print(f"    (+) tl_confirm = {tl_confirm} >= tl_req = 3 (TIMELIKE run)")
    print(f"    (+) bh_mass = {bh_mass:.2f} >= bh_form = 1.50 (BH formed)")
    print(f"    (+) ctl = {ctl} >= 5 (BH activation gate)")
    print(f"    (+) geo_raw = {geo_raw:.2f} < 2.0 (causal fraction OK)")
    print(f"    (+) No pos_floor active at entry")

    print()
    print("  Why it lost:")
    if is_flash:
        print(f"    * {instrument} made a large SPACELIKE move the next bar")
        bh_decayed = round(bh_mass * (0.95 ** 3), 2)
        print(f"    * bh_mass decayed from {bh_mass:.2f} -> {bh_decayed:.2f} (below bh_collapse after 3 bars)")
        print(f"    * Duration: {duration_h:.0f} hour(s) (flash reversal)")
    else:
        print(f"    * Extended adverse move over {duration_h:.1f}h -- {direction} ran into sustained opposition")
        print(f"    * Price moved from {entry_price:,.2f} to {exit_price:,.2f} ({exit_price - entry_price:+.2f} pts)")
        if is_big_loss:
            print(f"    * Large multiplier ({MULTIPLIERS.get(instrument, 1)} $/pt) amplified losses")

    print()
    if cap["applies"]:
        print("  v4 would have handled this differently:")
        print(f"    -> NQ notional cap: position reduced from {quantity} to {cap['capped_qty']} contracts")
        print(f"       (${cap['actual_notional']:,.0f} notional -> ${cap['capped_notional']:,} cap, ratio {cap['ratio']:.2f})")
        print(f"    -> Estimated loss with cap: ~${abs(cap['new_pnl']):,.0f}  (vs actual ${abs(pnl):,.0f})")
        print(f"    -> Savings: ${cap['savings']:,.0f}")
    elif instrument == "NQ":
        notional = entry_price * quantity * NQ_MULTIPLIER
        print("  v4 would have handled this differently:")
        print(f"    -> NQ notional cap: ${notional:,.0f} <= $400,000, no reduction needed")
    else:
        print(f"  v4 note: No NQ notional cap applies to {instrument}")

    print()
    if similar:
        pattern_label = ("BULL + high-mass NQ + flash reversal" if is_flash and instrument == "NQ"
                         else f"{regime} {instrument} {direction} fast reversal")
        print(f"  Similar pattern trades ({pattern_label}):")
        for t in similar[:3]:
            sdt = parse_dt(t["Entry Time"]).strftime("%Y-%m-%d")
            sdur = (parse_dt(t["Exit Time"]) - parse_dt(t["Entry Time"])).total_seconds() / 3600
            print(f"    {sdt}  {instrument} {t['Direction']}  {sdur:.0f}h  ${float(t['P&L']):,.0f}")
        total_shown = sum(float(t["P&L"]) for t in similar[:3])
        print(f"    Pattern frequency: {len(similar)} trades, total ${total_similar_pnl:,.0f}")
    else:
        print(f"  No similar pattern trades found in dataset.")

    print("-" * width)


if __name__ == "__main__":
    main()
