"""
conftest.py — shared pytest fixtures for the SRFM test suite.

All tests run from the project root (srfm-lab/).
lib/ and spacetime/engine/ are added to sys.path here.
"""

from __future__ import annotations

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_LIB  = _ROOT / "lib"
_ENGINE = _ROOT / "spacetime" / "engine"
for p in (_LIB, _ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from srfm_core import (
    MinkowskiClassifier,
    BlackHoleDetector,
    GeodesicAnalyzer,
    GravitationalLens,
    HawkingMonitor,
    MarketRegime,
)


# ── Helper: build a DataFrame ─────────────────────────────────────────────────

def _make_ohlcv_df(
    closes: np.ndarray,
    freq: str = "1h",
    start: str = "2021-01-01",
    vol_noise: float = 0.002,
    volume: float = 50_000.0,
) -> pd.DataFrame:
    """Convert a close-price array into an OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(999)
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq)
    noise = vol_noise * np.abs(rng.standard_normal(n))
    highs  = closes * (1.0 + noise)
    lows   = closes * (1.0 - noise)
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    vols   = np.full(n, volume)
    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": vols,
    }, index=idx)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: synthetic_trending
# 2000 hourly bars trending up at ~15% CAGR
# Design ensures BH activates by making moves small enough to be TIMELIKE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_trending() -> pd.DataFrame:
    """
    2000-bar hourly series trending up at ~15% CAGR.
    Per-bar drift = 15% / (252*6.5) ≈ 0.000091 with low noise (0.0008).
    Most bars will be TIMELIKE (beta < 1 at cf=0.001).
    Known BH activations: roughly every 20-40 bars after warmup.
    """
    rng = np.random.default_rng(42)
    n = 2000
    price = 4500.0
    # Annual drift 15% → hourly drift  (252 trading days × ~6.5 hours)
    hourly_drift = 0.15 / (252 * 6.5)
    sigma        = 0.0008          # sub-CF noise to keep beta < 1

    closes = np.empty(n)
    closes[0] = price
    for i in range(1, n):
        ret = hourly_drift + sigma * rng.standard_normal()
        ret = np.clip(ret, -0.015, 0.015)
        closes[i] = closes[i - 1] * (1.0 + ret)

    return _make_ohlcv_df(closes, freq="1h", start="2022-01-03", vol_noise=0.0005)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: synthetic_mean_reverting
# Ornstein-Uhlenbeck process — mean-reverting, BH rarely fires
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_mean_reverting() -> pd.DataFrame:
    """
    2000-bar OU process: dx = -theta*(x-mu)*dt + sigma*dW
    theta=0.05, mu=4500, sigma=8 (small relative to price → mostly TIMELIKE
    but mean-reversion prevents BH mass from building continuously).
    """
    rng = np.random.default_rng(123)
    n = 2000
    theta = 0.05
    mu_ou = 4500.0
    sigma = 6.0     # daily vol in price units
    dt    = 1.0

    x = mu_ou
    closes = np.empty(n)
    for i in range(n):
        closes[i] = x
        x = x + theta * (mu_ou - x) * dt + sigma * rng.standard_normal()
        x = max(3000.0, x)  # floor

    return _make_ohlcv_df(closes, freq="1h", start="2022-01-03", vol_noise=0.0003)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: synthetic_volatile
# GARCH-like jumpy price — lots of SPACELIKE bars
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_volatile() -> pd.DataFrame:
    """
    2000-bar GARCH(1,1)-like series with fat tails.
    Produces frequent SPACELIKE bars because sigma exceeds CF=0.001 often.
    """
    rng = np.random.default_rng(77)
    n = 2000
    price = 4500.0
    closes = np.empty(n)
    closes[0] = price
    h = 1e-4  # variance
    omega, alpha, beta_g = 2e-6, 0.12, 0.82

    for i in range(1, n):
        z   = rng.standard_normal()
        h   = omega + alpha * (closes[i-1] * z / closes[i-1])**2 + beta_g * h
        h   = np.clip(h, 1e-8, 1e-2)
        sig = math.sqrt(h)
        ret = sig * rng.standard_normal()
        # Occasionally inject a 3-sigma jump
        if rng.random() < 0.02:
            ret += rng.choice([-1, 1]) * 3 * sig
        closes[i] = closes[i-1] * max(0.5, 1.0 + np.clip(ret, -0.1, 0.1))

    return _make_ohlcv_df(closes, freq="1h", start="2022-01-03", vol_noise=0.002)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: synthetic_bearish
# Trending down — tests short-signal suppression and long_only flag
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_bearish() -> pd.DataFrame:
    """
    2000-bar hourly downtrend at ~-20% CAGR.
    long_only=True strategy should produce very few / zero trades.
    """
    rng = np.random.default_rng(55)
    n = 2000
    hourly_drift = -0.20 / (252 * 6.5)
    sigma        = 0.001
    price = 4500.0
    closes = np.empty(n)
    closes[0] = price
    for i in range(1, n):
        ret = hourly_drift + sigma * rng.standard_normal()
        closes[i] = closes[i-1] * max(0.01, 1.0 + np.clip(ret, -0.05, 0.05))

    return _make_ohlcv_df(closes, freq="1h", start="2022-01-03", vol_noise=0.0008)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: real_es_data
# Loads from data/ES_hourly_real.csv if available; otherwise returns trending
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_es_data() -> pd.DataFrame:
    """Load ES hourly data from data/ES_hourly_real.csv, or synthetic fallback."""
    csv_path = _ROOT / "data" / "ES_hourly_real.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = [c.lower().strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if c in ("o", "open"):   col_map[c] = "open"
            if c in ("h", "high"):   col_map[c] = "high"
            if c in ("l", "low"):    col_map[c] = "low"
            if c in ("c", "close"):  col_map[c] = "close"
            if c in ("v", "vol", "volume"): col_map[c] = "volume"
        df = df.rename(columns=col_map)
        if "volume" not in df.columns:
            df["volume"] = 50_000.0
        return df.sort_index().dropna(subset=["close"])
    # Fallback: synthetic trending
    rng = np.random.default_rng(42)
    n = 5000
    closes = np.empty(n)
    closes[0] = 4200.0
    for i in range(1, n):
        closes[i] = closes[i-1] * (1.0 + 0.0001 + 0.0008 * rng.standard_normal())
    return _make_ohlcv_df(closes, freq="1h", start="2021-01-04")


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: full_instrument_universe
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def full_instrument_universe() -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of {sym: DataFrame} for a representative cross-asset universe.
    Each series is 1000 synthetic hourly bars.
    """
    rng = np.random.default_rng(2024)
    configs = {
        "ES":    (4500.0, 0.0001, 0.0008),
        "NQ":    (15000.0, 0.00012, 0.0010),
        "YM":    (35000.0, 0.00009, 0.0007),
        "BTC":   (42000.0, 0.00020, 0.005),
        "ETH":   (2500.0,  0.00018, 0.006),
        "GC":    (1900.0,  0.00005, 0.0012),
        "CL":    (75.0,    0.00008, 0.008),
        "EURUSD":(1.08,    0.00002, 0.0003),
        "ZB":    (115.0,   0.00003, 0.0005),
    }
    universe = {}
    for sym, (start_p, drift, sigma) in configs.items():
        n = 1000
        closes = np.empty(n)
        closes[0] = start_p
        for i in range(1, n):
            closes[i] = closes[i-1] * max(1e-3, 1.0 + drift + sigma * rng.standard_normal())
        universe[sym] = _make_ohlcv_df(closes, freq="1h", start="2022-06-01")
    return universe


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: bh_state_fixture
# Pre-warmed BH state: MinkowskiClassifier + BlackHoleDetector at known mass
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bh_state_fixture():
    """
    Returns (mc, bh) pair pre-warmed so that bh.bh_mass is between
    bh_form (1.5) and 2.5. State is reproducible.
    """
    rng = np.random.default_rng(1337)
    mc  = MinkowskiClassifier(cf=0.001)
    bh  = BlackHoleDetector(bh_form=1.5, bh_collapse=1.0, bh_decay=0.95)

    price = 4500.0
    # Feed 30 timelike bars to build mass
    for _ in range(30):
        price = price * (1.0 + 0.0003 + 0.0002 * rng.standard_normal())
        bit = mc.update(price)
        prev = price / (1.0 + 0.0003)
        bh.update(bit, price, prev)

    return mc, bh, price


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: sample_trade_records
# A list of realistic trade dicts for MC testing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_trade_records() -> List[dict]:
    """50 synthetic trade dicts with realistic P&L distribution."""
    rng = np.random.default_rng(888)
    regimes = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]
    trades = []
    base_time = pd.Timestamp("2023-01-01")
    for i in range(50):
        win  = rng.random() < 0.58
        pnl  = float(rng.exponential(0.012)) if win else -float(rng.exponential(0.008))
        regime = regimes[i % 4]
        trades.append({
            "entry_time": base_time + pd.Timedelta(hours=i * 24),
            "exit_time":  base_time + pd.Timedelta(hours=i * 24 + 12),
            "pnl":   pnl * 1_000_000 * 0.25,
            "regime": regime,
            "tf_score": int(rng.integers(4, 8)),
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: positive_edge_trades / negative_edge_trades
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def positive_edge_trades() -> List[dict]:
    """80 trades with clear positive edge: 65% win rate, avg win 1.5× avg loss."""
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2023-01-01")
    trades = []
    for i in range(80):
        win = rng.random() < 0.65
        pnl = float(rng.exponential(1500.0)) if win else -float(rng.exponential(1000.0))
        trades.append({
            "entry_time": base + pd.Timedelta(hours=i * 8),
            "exit_time":  base + pd.Timedelta(hours=i * 8 + 4),
            "pnl": pnl,
            "regime": "BULL",
            "tf_score": 6,
        })
    return trades


@pytest.fixture(scope="session")
def negative_edge_trades() -> List[dict]:
    """60 trades with negative edge: 35% win rate."""
    rng = np.random.default_rng(22)
    base = pd.Timestamp("2023-01-01")
    trades = []
    for i in range(60):
        win = rng.random() < 0.35
        pnl = float(rng.exponential(800.0)) if win else -float(rng.exponential(1200.0))
        trades.append({
            "entry_time": base + pd.Timedelta(hours=i * 6),
            "exit_time":  base + pd.Timedelta(hours=i * 6 + 3),
            "pnl": pnl,
            "regime": "SIDEWAYS",
            "tf_score": 2,
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: minimal_flat_df
# Perfectly flat price (no moves) — no BH should fire
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_flat_df() -> pd.DataFrame:
    """500-bar hourly series with identical close prices."""
    n = 500
    closes = np.full(n, 4500.0)
    return _make_ohlcv_df(closes, freq="1h", start="2023-01-01", vol_noise=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Utility accessible to test modules
# ─────────────────────────────────────────────────────────────────────────────

def make_trending_closes(n: int = 500, drift: float = 0.0003, sigma: float = 0.0004,
                          seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    closes = np.empty(n)
    closes[0] = 4500.0
    for i in range(1, n):
        closes[i] = closes[i-1] * (1.0 + drift + sigma * rng.standard_normal())
    return closes


def make_mean_reverting_closes(n: int = 500, theta: float = 0.05,
                                 mu: float = 4500.0, sigma: float = 5.0,
                                 seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = mu
    closes = np.empty(n)
    for i in range(n):
        closes[i] = x
        x += theta * (mu - x) + sigma * rng.standard_normal()
        x = max(100.0, x)
    return closes
