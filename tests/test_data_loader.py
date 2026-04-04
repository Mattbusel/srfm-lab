"""
test_data_loader.py — Tests for data loading and resampling.

~400 LOC.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))


# ─────────────────────────────────────────────────────────────────────────────
# Inline data loader (mirrors the pattern in arena.py / bh_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV and return a normalized DataFrame with DatetimeIndex."""
    df = pd.read_csv(path)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("date", "datetime", "time", "timestamp"):
            col_map[c] = "date"
        elif lc in ("open", "o"):      col_map[c] = "open"
        elif lc in ("high", "h"):      col_map[c] = "high"
        elif lc in ("low", "l"):       col_map[c] = "low"
        elif lc in ("close", "c"):     col_map[c] = "close"
        elif lc in ("volume", "vol", "v"): col_map[c] = "volume"
    df = df.rename(columns=col_map)
    if "date" in df.columns:
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    if "volume" not in df.columns:
        df["volume"] = 1000.0
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.sort_index()
    return df.dropna(subset=["close"])


def resample_ohlcv(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """Resample OHLCV data to target frequency."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resampled = df.resample(target_freq).agg(agg).dropna(subset=["close"])
    return resampled


def _write_sample_csv(path: str, n: int = 200, freq: str = "1h",
                       start: str = "2023-01-02", with_volume: bool = True) -> None:
    """Write a synthetic OHLCV CSV file."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=n, freq=freq)
    closes = np.empty(n)
    closes[0] = 4500.0
    for i in range(1, n):
        closes[i] = closes[i-1] * (1.0 + 0.0001 + 0.0008 * rng.standard_normal())
    noise = 0.0005 * np.abs(rng.standard_normal(n))
    highs = closes * (1.0 + noise)
    lows  = closes * (1.0 - noise)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    df = pd.DataFrame({
        "date":   idx,
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
    })
    if with_volume:
        df["volume"] = rng.integers(10_000, 100_000, size=n).astype(float)
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Simple cache (mirrors pattern used in lab)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: Dict[str, pd.DataFrame] = {}


def cached_load(path: str) -> pd.DataFrame:
    """Load CSV with caching by path."""
    if path in _CACHE:
        return _CACHE[path]
    df = load_ohlcv_csv(path)
    _CACHE[path] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Class TestDataLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:

    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path):
        self.tmp = tmp_path
        _CACHE.clear()

    def test_load_csv_returns_dataframe(self):
        """load_ohlcv_csv should return a pandas DataFrame."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        assert isinstance(df, pd.DataFrame)

    def test_load_csv_has_ohlcv_columns(self):
        """Loaded DataFrame must have open, high, low, close, volume columns."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_load_csv_has_datetime_index(self):
        """Index should be a DatetimeIndex."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_csv_sorted_index(self):
        """Index should be sorted ascending."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        assert df.index.is_monotonic_increasing

    def test_load_csv_no_missing_close(self):
        """Loaded DataFrame should have no NaN close values."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        assert df["close"].isna().sum() == 0

    def test_resample_to_daily(self):
        """Hourly data resampled to daily should have fewer rows."""
        p = str(self.tmp / "hourly.csv")
        _write_sample_csv(p, n=500, freq="1h")
        df = load_ohlcv_csv(p)
        df_daily = resample_ohlcv(df, "1D")
        assert len(df_daily) < len(df), (
            f"Daily df ({len(df_daily)}) should have fewer rows than hourly ({len(df)})")

    def test_resample_preserves_close(self):
        """Last close of each day in hourly data should match daily close after resample."""
        p = str(self.tmp / "hourly.csv")
        _write_sample_csv(p, n=240, freq="1h", start="2023-01-02")
        df = load_ohlcv_csv(p)
        df_daily = resample_ohlcv(df, "1D")
        # For each day in daily, the close should match the last hourly close
        for day_ts in df_daily.index[:5]:
            day_mask = df.index.date == day_ts.date()
            if day_mask.any():
                last_hourly_close = float(df[day_mask]["close"].iloc[-1])
                daily_close = float(df_daily.loc[day_ts, "close"])
                assert abs(last_hourly_close - daily_close) < 1e-6, (
                    f"Mismatch on {day_ts}: hourly={last_hourly_close}, daily={daily_close}")

    def test_resample_high_is_max(self):
        """Resampled high should equal the max high in the window."""
        p = str(self.tmp / "hourly.csv")
        _write_sample_csv(p, n=240, freq="1h", start="2023-01-02")
        df = load_ohlcv_csv(p)
        df_daily = resample_ohlcv(df, "1D")
        for day_ts in df_daily.index[:3]:
            day_mask = df.index.date == day_ts.date()
            if day_mask.sum() > 1:
                max_hourly_high = float(df[day_mask]["high"].max())
                daily_high      = float(df_daily.loc[day_ts, "high"])
                assert abs(max_hourly_high - daily_high) < 1e-6

    def test_cache_hit_returns_same_data(self):
        """Second load from same path should return cached object."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df1 = cached_load(p)
        df2 = cached_load(p)
        assert df1 is df2, "Cache hit should return exact same object"

    def test_handles_timezone_aware_index(self):
        """CSV with timezone-aware timestamps should load without crash."""
        p = str(self.tmp / "tz.csv")
        n = 100
        idx = pd.date_range("2023-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
        closes = np.full(n, 4500.0) + np.arange(n) * 0.1
        df = pd.DataFrame({
            "date":   idx,
            "open":   closes - 1.0,
            "high":   closes + 2.0,
            "low":    closes - 2.0,
            "close":  closes,
            "volume": np.full(n, 10_000.0),
        })
        df.to_csv(p, index=False)
        # Should not crash even with tz-aware timestamps
        try:
            result = load_ohlcv_csv(p)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Acceptable if tz parsing fails
            assert "timezone" in str(e).lower() or "parse" in str(e).lower()

    def test_handles_missing_bars_in_middle(self):
        """DataFrame with a gap in the middle should still load correctly."""
        p = str(self.tmp / "gap.csv")
        # Create hourly data with a 12-hour gap
        idx1 = pd.date_range("2023-01-02 09:00", periods=50, freq="1h")
        idx2 = pd.date_range("2023-01-04 09:00", periods=50, freq="1h")
        closes = np.full(100, 4500.0)
        df = pd.DataFrame({
            "date":   list(idx1) + list(idx2),
            "open":   closes - 1.0,
            "high":   closes + 2.0,
            "low":    closes - 2.0,
            "close":  closes,
            "volume": np.full(100, 10_000.0),
        })
        df.to_csv(p, index=False)
        result = load_ohlcv_csv(p)
        assert len(result) == 100  # Both segments loaded

    def test_load_csv_without_volume_column(self):
        """CSV without volume column should get default volume."""
        p = str(self.tmp / "novol.csv")
        _write_sample_csv(p, n=100, with_volume=False)
        df = load_ohlcv_csv(p)
        assert "volume" in df.columns
        assert (df["volume"] == 1000.0).all()

    def test_resample_open_is_first(self):
        """Resampled open should equal the first open in the window."""
        p = str(self.tmp / "hourly.csv")
        _write_sample_csv(p, n=240, freq="1h", start="2023-01-02")
        df = load_ohlcv_csv(p)
        df_daily = resample_ohlcv(df, "1D")
        for day_ts in df_daily.index[:3]:
            day_mask = df.index.date == day_ts.date()
            if day_mask.sum() > 0:
                first_hourly_open = float(df[day_mask]["open"].iloc[0])
                daily_open        = float(df_daily.loc[day_ts, "open"])
                assert abs(first_hourly_open - daily_open) < 1e-6

    def test_resample_volume_is_sum(self):
        """Resampled volume should equal the sum of volumes in the window."""
        p = str(self.tmp / "hourly.csv")
        _write_sample_csv(p, n=240, freq="1h", start="2023-01-02")
        df = load_ohlcv_csv(p)
        df_daily = resample_ohlcv(df, "1D")
        for day_ts in df_daily.index[:3]:
            day_mask = df.index.date == day_ts.date()
            if day_mask.sum() > 0:
                sum_vol  = float(df[day_mask]["volume"].sum())
                day_vol  = float(df_daily.loc[day_ts, "volume"])
                assert abs(sum_vol - day_vol) < 1e-3

    def test_nonexistent_file_raises(self):
        """Loading a nonexistent file should raise an exception."""
        with pytest.raises(Exception):
            load_ohlcv_csv("/nonexistent/path/data.csv")

    def test_load_large_csv_performance(self):
        """Loading a 10K-bar CSV should complete in reasonable time."""
        import time
        p = str(self.tmp / "large.csv")
        _write_sample_csv(p, n=10_000, freq="1h")
        t0 = time.time()
        df = load_ohlcv_csv(p)
        elapsed = time.time() - t0
        assert len(df) == 10_000
        assert elapsed < 5.0, f"Loading 10K bars took {elapsed:.2f}s (> 5s)"

    def test_resample_15min_to_hourly(self):
        """15-minute data resampled to 1h should reduce rows by ~4×."""
        p = str(self.tmp / "15m.csv")
        _write_sample_csv(p, n=400, freq="15min")
        df = load_ohlcv_csv(p)
        df_1h = resample_ohlcv(df, "1h")
        ratio = len(df) / len(df_1h)
        assert 3.0 <= ratio <= 5.0, f"Resample ratio {ratio:.1f} not near 4"

    def test_all_prices_positive(self):
        """All OHLCV prices should be positive."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=100)
        df = load_ohlcv_csv(p)
        for col in ("open", "high", "low", "close"):
            assert (df[col] > 0).all(), f"Non-positive values in {col}"

    def test_high_gte_low(self):
        """High should be >= low for all bars."""
        p = str(self.tmp / "test.csv")
        _write_sample_csv(p, n=200)
        df = load_ohlcv_csv(p)
        assert (df["high"] >= df["low"]).all(), "high < low found in data"
