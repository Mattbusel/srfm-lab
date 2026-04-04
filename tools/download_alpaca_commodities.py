"""
Download commodity ETF proxy data via Alpaca.
USO=CL, GLD=GC, TLT=ZB, UNG=NG, VIXY=VX
Saves to tools/data_cache/ as CSVs.
"""
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

API_KEY    = "PKAJISZM3NEO654DQSLPG35I33"
SECRET_KEY = "Eusa11jFafE5UPuJX8GQs7uHeZLXhdCcgzSFGJbsY5Z1"
BASE_URL   = "https://data.alpaca.markets/v2/stocks"

SYMBOLS = {
    "CL": "USO",
    "GC": "GLD",
    "ZB": "TLT",
    "NG": "UNG",
    "VX": "VIXY",
}

START_DATE = "2016-01-01"
CACHE_DIR  = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}


def fetch_bars(ticker: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    all_bars = []
    params = {
        "start":      start,
        "end":        end,
        "timeframe":  timeframe,
        "adjustment": "split",
        "feed":       "iex",
        "limit":      10000,
    }

    url = f"{BASE_URL}/{ticker}/bars"
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        bars = data.get("bars", [])
        if bars:
            all_bars.extend(bars)

        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token
        time.sleep(0.3)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df = df.rename(columns={"t": "datetime", "o": "Open", "h": "High",
                             "l": "Low", "c": "Close", "v": "Volume"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    df = df.between_time("09:30", "16:00")
    return df


def download_symbol(sym_label: str, ticker: str, timeframe: str):
    tf_tag     = "1h" if "Hour" in timeframe else "15m"
    cache_path = CACHE_DIR / f"{sym_label}_{tf_tag}.csv"

    if cache_path.exists():
        existing    = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if existing.index.tz is not None:
            existing.index = existing.index.tz_localize(None)
        last_date   = existing.index.max()
        fetch_start = (last_date + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"  [{tf_tag}] {ticker}: cache to {last_date.date()}, fetching newer...")
    else:
        existing    = pd.DataFrame()
        fetch_start = f"{START_DATE}T09:30:00Z"
        print(f"  [{tf_tag}] {ticker}: no cache, fetching from {START_DATE}...")

    fetch_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"         fetching {fetch_start[:10]} to {fetch_end[:10]}...", end=" ", flush=True)
    new_data = fetch_bars(ticker, timeframe, fetch_start, fetch_end)

    if new_data.empty:
        print("0 bars (up to date)")
        return existing

    print(f"{len(new_data)} bars")
    all_data = pd.concat([existing, new_data]).sort_index()
    all_data = all_data[~all_data.index.duplicated(keep="last")]
    all_data.to_csv(cache_path)
    print(f"         saved {len(all_data)} total bars [{all_data.index.min().date()} to {all_data.index.max().date()}]")
    return all_data


if __name__ == "__main__":
    print("Downloading commodity ETF proxies from Alpaca...\n")
    for label, ticker in SYMBOLS.items():
        print(f"{label} ({ticker}):")
        download_symbol(label, ticker, "1Hour")
        download_symbol(label, ticker, "15Min")
        print()
    print("Done.")
