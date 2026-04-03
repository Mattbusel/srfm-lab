"""
fetch_polygon.py — Download real futures hourly data from Polygon.io free tier.

Free tier: 5 API calls/minute, unlimited historical data.
Requires a free API key from polygon.io.

Usage:
    python scripts/fetch_polygon.py --key YOUR_API_KEY --ticker ES --start 2020-01-01
    python scripts/fetch_polygon.py --key YOUR_API_KEY --all --start 2020-01-01

Polygon tickers for CME futures:
    ES  -> C:ES1!   (S&P 500 E-mini continuous)
    NQ  -> C:NQ1!   (Nasdaq-100 E-mini continuous)
    YM  -> C:YM1!   (Dow E-mini continuous)

Output: data/{TICKER}_hourly_poly.csv (RTH-filtered: 9:30-16:15 ET)
"""

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Optional

RTH_TICKERS = {
    "ES": "ES:XCME",    # Polygon format for ES futures
    "NQ": "NQ:XCME",
    "YM": "YM:XCBT",
}

# RTH = 09:30 to 16:15 ET (Eastern Time = UTC-5 in winter, UTC-4 in summer)
RTH_START_UTC = 14   # 9:30 ET = 14:30 UTC (roughly)
RTH_END_UTC   = 21   # 16:15 ET = 21:15 UTC


def polygon_aggs(
    api_key:    str,
    ticker:     str,
    from_date:  str,
    to_date:    str,
    timespan:   str = "hour",
    multiplier: int = 1,
    limit:      int = 50000,
) -> List[dict]:
    """
    Fetch aggregates from Polygon.io.
    Returns list of bars: {t, o, h, l, c, v} (timestamp in ms).
    """
    base = "https://api.polygon.io/v2/aggs/ticker"
    url  = (f"{base}/{ticker}/range/{multiplier}/{timespan}/"
            f"{from_date}/{to_date}"
            f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")

    bars = []
    while url:
        try:
            req  = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [ERROR] {e}")
            break

        results = data.get("results", [])
        bars.extend(results)
        url = None

        # Pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url + f"&apiKey={api_key}"
            time.sleep(12)  # respect free tier rate limit (5/min)

    return bars


def to_csv(bars: List[dict], path: str, rth_only: bool = True):
    rows = []
    for bar in bars:
        ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
        if rth_only and not (RTH_START_UTC <= ts.hour < RTH_END_UTC):
            continue
        rows.append({
            "date":   ts.strftime("%Y-%m-%d %H:%M"),
            "open":   bar.get("o", bar.get("c")),
            "high":   bar.get("h", bar.get("c")),
            "low":    bar.get("l", bar.get("c")),
            "close":  bar.get("c"),
            "volume": bar.get("v", 0),
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","open","high","low","close","volume"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} RTH bars -> {path}")
    return rows


def fetch_ticker(api_key: str, name: str, poly_ticker: str, start: str, end: str):
    print(f"\nFetching {name} ({poly_ticker}) {start} -> {end}...")
    bars = polygon_aggs(api_key, poly_ticker, start, end)
    if not bars:
        print(f"  [WARN] No data returned for {poly_ticker}")
        return
    print(f"  Raw bars: {len(bars)}")
    path = f"data/{name}_hourly_poly.csv"
    rows = to_csv(bars, path, rth_only=True)

    # Quick stats
    if rows:
        closes = [float(r["close"]) for r in rows]
        import math
        rets = [abs(closes[i]-closes[i-1])/(closes[i-1]+1e-9) for i in range(1, len(closes))]
        rets.sort()
        print(f"  Median |return|: {rets[len(rets)//2]:.6f}")
        print(f"  % SPACELIKE (CF=0.001): {sum(1 for r in rets if r>0.001)/len(rets)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Polygon.io futures data fetcher")
    parser.add_argument("--key",    required=True, help="Polygon.io API key")
    parser.add_argument("--ticker", help="Single ticker: ES, NQ, or YM")
    parser.add_argument("--all",    action="store_true", help="Fetch ES, NQ, YM")
    parser.add_argument("--start",  default="2020-01-01")
    parser.add_argument("--end",    default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    tickers = list(RTH_TICKERS.items()) if args.all else [(args.ticker, RTH_TICKERS[args.ticker])]

    for i, (name, poly_ticker) in enumerate(tickers):
        if i > 0:
            print("Sleeping 13s for rate limit...")
            time.sleep(13)
        fetch_ticker(args.key, name, poly_ticker, args.start, args.end)

    print("\nDone. Run arena on real data:")
    for name, _ in tickers:
        print(f"  python tools/arena.py --csv data/{name}_hourly_poly.csv --ticker {name}")


if __name__ == "__main__":
    main()
