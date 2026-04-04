"""
data_loader.py — Data ingestion for Spacetime Arena.

Sources:
  - yfinance
  - Alpaca (using keys from live_trader_alpaca.py environment)
  - CSV (handles existing data/ CSV format)

Auto-resamples to produce daily / hourly / 15m DataFrames.
Caches results to spacetime/cache/ as parquet.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names and ensure standard OHLCV set."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]  # type: ignore[assignment]

    # Alias common variations
    renames: Dict[str, str] = {}
    for c in df.columns:
        if c in ("o",):
            renames[c] = "open"
        elif c in ("h",):
            renames[c] = "high"
        elif c in ("l",):
            renames[c] = "low"
        elif c in ("c",):
            renames[c] = "close"
        elif c in ("v", "vol"):
            renames[c] = "volume"
    df = df.rename(columns=renames)

    if "volume" not in df.columns:
        df["volume"] = 1_000.0

    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])
        elif "datetime" in df.columns:
            df.index = pd.to_datetime(df["datetime"])
            df = df.drop(columns=["datetime"])
        elif "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        else:
            df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(source: str, sym: str, interval: str, start: str, end: str) -> Path:
    tag = f"{source}_{sym}_{interval}_{start}_{end}".replace("/", "-").replace(":", "")
    return CACHE_DIR / f"{tag}.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning("Cache read failed %s: %s", path, e)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.warning("Cache write failed %s: %s", path, e)


# ---------------------------------------------------------------------------
# yfinance loader
# ---------------------------------------------------------------------------

def load_yfinance(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1h",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance.

    Parameters
    ----------
    ticker   : e.g. "SPY", "BTC-USD", "GC=F"
    start    : ISO date string "YYYY-MM-DD"
    end      : ISO date string "YYYY-MM-DD"
    interval : yfinance interval string: "1m","5m","15m","30m","60m","1h","1d","1wk","1mo"
    """
    cache_path = _cache_key("yf", ticker, interval, start, end)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            logger.info("yfinance cache hit: %s", cache_path.name)
            return cached

    import yfinance as yf  # type: ignore

    logger.info("Downloading %s %s from yfinance [%s to %s]", ticker, interval, start, end)
    raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")

    # yfinance returns multi-level columns when downloading single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = _normalize_cols(raw)
    df = _ensure_datetime_index(df)
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    if use_cache:
        _save_cache(df, cache_path)

    return df


# ---------------------------------------------------------------------------
# Alpaca loader
# ---------------------------------------------------------------------------

def _get_alpaca_keys() -> Tuple[str, str]:
    """Read API keys from environment or common config locations."""
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not key or not secret:
        # Try to read from live_trader_alpaca.py config file if present
        config_path = Path(__file__).parent.parent.parent / "lib" / "alpaca_keys.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("alpaca_keys", config_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                key    = getattr(mod, "API_KEY", key)
                secret = getattr(mod, "SECRET_KEY", secret)

    return key, secret


def load_alpaca(
    sym: str,
    start: str,
    end: str,
    timeframe: str = "1Hour",
    asset_class: str = "stock",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from Alpaca Markets.

    Parameters
    ----------
    sym         : symbol, e.g. "SPY", "BTC/USD"
    start       : ISO date string
    end         : ISO date string
    timeframe   : "1Min","5Min","15Min","1Hour","1Day"
    asset_class : "stock" or "crypto"
    """
    cache_path = _cache_key("alpaca", sym, timeframe, start, end)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            logger.info("Alpaca cache hit: %s", cache_path.name)
            return cached

    key, secret = _get_alpaca_keys()

    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
    from datetime import datetime as dt

    tf_map = {
        "1Min":  TimeFrame.Minute,
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day":  TimeFrame.Day,
    }
    tf_obj = tf_map.get(timeframe, TimeFrame.Hour)

    start_dt = dt.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt   = dt.fromisoformat(end).replace(tzinfo=timezone.utc)

    rows = []
    if asset_class == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient  # type: ignore
        from alpaca.data.requests import CryptoBarsRequest              # type: ignore

        client = CryptoHistoricalDataClient()
        req    = CryptoBarsRequest(symbol_or_symbols=[sym], timeframe=tf_obj,
                                   start=start_dt, end=end_dt)
        result = client.get_crypto_bars(req)
        bars   = result.get(sym, [])
        for b in bars:
            rows.append({"timestamp": b.timestamp, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": getattr(b, "volume", 0.0)})
    else:
        from alpaca.data.historical import StockHistoricalDataClient   # type: ignore
        from alpaca.data.requests import StockBarsRequest              # type: ignore

        client = StockHistoricalDataClient(key, secret)
        req    = StockBarsRequest(symbol_or_symbols=[sym], timeframe=tf_obj,
                                  start=start_dt, end=end_dt)
        result = client.get_stock_bars(req)
        bars   = result.get(sym, [])
        for b in bars:
            rows.append({"timestamp": b.timestamp, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": getattr(b, "volume", 0.0)})

    if not rows:
        raise ValueError(f"Alpaca returned no data for {sym}")

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.drop(columns=["timestamp"])
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])

    if use_cache:
        _save_cache(df, cache_path)

    return df


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str | Path, use_cache: bool = False) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Handles the existing data/ CSV format (date,open,high,low,close,volume)
    as well as common variations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_cols(df)

    # Try to find datetime column
    for col in ("date", "datetime", "timestamp", "time"):
        if col in df.columns:
            df.index = pd.to_datetime(df[col])
            df = df.drop(columns=[col])
            break

    df = _ensure_datetime_index(df)

    required = {"open", "high", "low", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df


# ---------------------------------------------------------------------------
# Auto-resample
# ---------------------------------------------------------------------------

def resample_ohlcv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to target frequency.

    Parameters
    ----------
    df     : normalized OHLCV DataFrame with DatetimeIndex
    target : pandas offset alias: "1D", "1h", "15min", etc.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(target).agg(agg).dropna(subset=["close"])
    return out


def get_multiframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given any-frequency OHLCV DataFrame, return (df_1d, df_1h, df_15m).
    """
    df_1d  = resample_ohlcv(df, "1D")
    df_1h  = resample_ohlcv(df, "1h")
    df_15m = resample_ohlcv(df, "15min")
    return df_1d, df_1h, df_15m
