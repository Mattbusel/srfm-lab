"""
Download hourly OHLCV data via Twelve Data API.
Saves to tools/data_cache/ as CSVs — only downloads missing date ranges.
Run once; backtester loads from cache after that.
"""
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

API_KEY  = "77fca7a68e094ce391da80169260878e"
BASE_URL = "https://api.twelvedata.com/time_series"

SYMBOLS = {
    "ES": "SPY",
    "NQ": "QQQ",
    "YM": "DIA",
}

START_DATE = "2010-01-01"
CACHE_DIR  = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Twelve Data: max 5000 bars per request, 8 req/min on free tier
CHUNK_YEARS = 2      # ~3,200 hourly bars per 2-year chunk — safely under 5000
SLEEP_SEC   = 8      # 8 sec between requests = ~7.5 req/min (under 8/min limit)


def fetch_chunk(symbol: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "symbol":     symbol,
        "interval":   "1h",
        "start_date": start,
        "end_date":   end,
        "outputsize": 5000,
        "format":     "JSON",
        "apikey":     API_KEY,
        "timezone":   "America/New_York",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if data.get("status") == "error":
        print(f"    API error: {data.get('message')}")
        return pd.DataFrame()

    values = data.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.columns = [c.capitalize() for c in df.columns]
    return df


def download_symbol(sym_label: str, ticker: str):
    cache_path = CACHE_DIR / f"{sym_label}_1h.csv"

    # Load existing cache
    if cache_path.exists():
        existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        last_date = existing.index.max()
        # Only fetch from day after last cached date
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  {ticker}: cache has data to {last_date.date()}, fetching from {fetch_start}")
    else:
        existing = pd.DataFrame()
        fetch_start = START_DATE
        print(f"  {ticker}: no cache, fetching from {fetch_start}")

    fetch_end   = datetime.today().strftime("%Y-%m-%d")
    start_dt    = datetime.strptime(fetch_start, "%Y-%m-%d")
    end_dt      = datetime.strptime(fetch_end,   "%Y-%m-%d")

    if start_dt >= end_dt:
        print(f"  {ticker}: cache is up to date")
        return existing

    # Chunk into 2-year windows
    chunks = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=365 * CHUNK_YEARS), end_dt)
        s = cursor.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    Fetching {ticker} {s} to {e}...", end=" ", flush=True)

        df = fetch_chunk(ticker, s, e)
        if not df.empty:
            # Keep regular hours only (9:30–16:00 ET)
            df = df.between_time("09:30", "16:00")
            chunks.append(df)
            print(f"{len(df)} bars")
        else:
            print("0 bars")

        cursor = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    if chunks:
        new_data = pd.concat(chunks)
        all_data = pd.concat([existing, new_data]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.to_csv(cache_path)
        print(f"  {ticker}: saved {len(all_data)} total bars to {cache_path.name}")
        return all_data
    else:
        print(f"  {ticker}: nothing new fetched")
        return existing


def load_cached(sym_label: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{sym_label}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Ensure tz-naive for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


if __name__ == "__main__":
    print("Downloading hourly data via Twelve Data...\n")
    for label, ticker in SYMBOLS.items():
        download_symbol(label, ticker)
        print()
    print("Done. Run local_backtest.py to use the cached data.")
