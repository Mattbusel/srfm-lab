"""
fetch_polygon.py — Download hourly index/futures data from Polygon.io.

Key capabilities by plan tier:
  Free (Indices plan): I:NDX, I:COMP, I:DJIA hourly/daily — from ~Feb 2023
  Paid (Stocks):       SPY, QQQ, DIA — full history
  Paid (Futures):      ES:XCME, NQ:XCME, YM:XCBT — full history

With a free Indices key, use index proxies:
  NQ futures  -> I:NDX  (Nasdaq-100 Index)
  ES futures  -> I:COMP (Nasdaq Composite, or yfinance for SPX)
  YM futures  -> I:DJIA (Dow Jones — if accessible)

Rate limit: 5 calls/minute. Script sleeps 13s between chunks.

Usage:
    # Free tier (indices from ~Feb 2023):
    python scripts/fetch_polygon.py --key KEY --ndx
    python scripts/fetch_polygon.py --key KEY --all-indices

    # Paid tier (full history):
    python scripts/fetch_polygon.py --key KEY --all --start 2018-01-01

Output: data/{TICKER}_hourly_poly.csv
"""

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Optional


def fetch_chunk(api_key: str, ticker: str, start: str, end: str, res: str = "hour") -> List[dict]:
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{res}/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
        return data.get("results", [])
    except Exception as e:
        print(f"  [ERROR] {ticker} {start}->{end}: {e}")
        return []


def fetch_chunked(
    api_key: str,
    ticker: str,
    start: str,
    end: str,
    res: str = "hour",
    chunk_weeks: int = 5,
    sleep_s: float = 13.0,
) -> List[dict]:
    """Fetch in chunks to avoid pagination and rate limits."""
    all_bars = []
    dt    = datetime.strptime(start, "%Y-%m-%d")
    dt_end = datetime.strptime(end, "%Y-%m-%d")
    chunk  = timedelta(weeks=chunk_weeks)

    while dt < dt_end:
        chunk_end = min(dt + chunk, dt_end)
        bars = fetch_chunk(api_key, ticker, dt.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), res)
        all_bars.extend(bars)
        if bars:
            t0 = datetime.fromtimestamp(bars[0]["t"]/1000, tz=timezone.utc)
            t1 = datetime.fromtimestamp(bars[-1]["t"]/1000, tz=timezone.utc)
            print(f"  {dt.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}: "
                  f"{len(bars)} bars ({t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')})")
        else:
            print(f"  {dt.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}: 0 bars")
        dt = chunk_end
        if dt < dt_end:
            time.sleep(sleep_s)

    return all_bars


def save_bars(bars: List[dict], path: str) -> List[dict]:
    rows = []
    for b in bars:
        ts = datetime.fromtimestamp(b["t"] / 1000, tz=timezone.utc)
        rows.append({
            "date":   ts.strftime("%Y-%m-%d %H:%M"),
            "open":   b.get("o", b["c"]),
            "high":   b.get("h", b["c"]),
            "low":    b.get("l", b["c"]),
            "close":  b["c"],
            "volume": b.get("v", 0),
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","open","high","low","close","volume"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_stats(rows: List[dict], ticker: str, cf: float):
    if len(rows) < 2:
        return
    closes = [float(r["close"]) for r in rows]
    rets   = sorted([abs(closes[i]-closes[i-1])/(closes[i-1]+1e-9) for i in range(1, len(closes))])
    sl_pct = sum(1 for r in rets if r > cf) / len(rets) * 100
    print(f"  {ticker}: {len(rows)} bars  {rows[0]['date']} -> {rows[-1]['date']}")
    print(f"  Median |return|: {rets[len(rets)//2]:.6f}")
    print(f"  % SPACELIKE (CF={cf}): {sl_pct:.1f}%")
    from srfm_core import MinkowskiClassifier, BlackHoleDetector
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(1.5, 1.0, 0.95)
    prev = None; max_mass = 0.0; wells = 0; was_active = False
    for row in rows:
        c = float(row["close"])
        if prev is None: prev = c; mc.update(c); continue
        bit    = mc.update(c)
        active = bh.update(bit, c, prev)
        if bh.bh_mass > max_mass: max_mass = bh.bh_mass
        if active and not was_active: wells += 1
        was_active = active; prev = c
    print(f"  Max BH mass: {max_mass:.4f}  Wells formed: {wells}")


FUTURES_MAP = {
    "ES": ("ES:XCME", 0.001),
    "NQ": ("NQ:XCME", 0.0012),
    "YM": ("YM:XCBT", 0.0008),
}

INDEX_MAP = {
    "NDX":  ("I:NDX",  0.0012),   # Nasdaq-100 (NQ proxy)
    "COMP": ("I:COMP", 0.0010),   # Nasdaq Composite (ES proxy)
    "DJIA": ("I:DJIA", 0.0008),   # Dow (YM proxy)
}


def main():
    parser = argparse.ArgumentParser(description="Polygon.io data fetcher")
    parser.add_argument("--key",         required=True)
    parser.add_argument("--all",         action="store_true", help="Fetch ES/NQ/YM futures (paid tier)")
    parser.add_argument("--all-indices", action="store_true", help="Fetch NDX/COMP/DJIA indices (free tier)")
    parser.add_argument("--ndx",         action="store_true", help="Fetch I:NDX only")
    parser.add_argument("--ticker",      help="Specific ticker symbol (e.g. ES, NDX)")
    parser.add_argument("--start",       default="2018-01-01")
    parser.add_argument("--end",         default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--res",         default="hour", choices=["minute","hour","day"])
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = [(name, sym, cf) for name, (sym, cf) in FUTURES_MAP.items()]
    if args.all_indices:
        targets = [(name, sym, cf) for name, (sym, cf) in INDEX_MAP.items()]
    if args.ndx:
        targets = [("NDX", "I:NDX", 0.0012)]
    if args.ticker:
        if args.ticker in FUTURES_MAP:
            sym, cf = FUTURES_MAP[args.ticker]
        elif args.ticker in INDEX_MAP:
            sym, cf = INDEX_MAP[args.ticker]
        else:
            sym, cf = args.ticker, 0.001
        targets = [(args.ticker, sym, cf)]

    if not targets:
        parser.error("Specify --all, --all-indices, --ndx, or --ticker")

    for i, (name, sym, cf) in enumerate(targets):
        if i > 0:
            print(f"Sleeping 13s...")
            time.sleep(13)
        print(f"\nFetching {name} ({sym}) {args.start} -> {args.end}...")
        bars = fetch_chunked(args.key, sym, args.start, args.end, args.res)
        if not bars:
            print(f"  No data returned.")
            continue
        path = f"data/{name}_hourly_poly.csv"
        rows = save_bars(bars, path)
        print(f"  Saved -> {path}")
        try:
            import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
            print_stats(rows, name, cf)
        except Exception:
            pass


if __name__ == "__main__":
    main()
