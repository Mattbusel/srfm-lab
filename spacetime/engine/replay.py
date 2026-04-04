"""
replay.py — Signal replay engine for Spacetime Arena.

Given sym, date range, speed multiplier:
  - Load OHLCV data for that range
  - Replay bar by bar, computing full BH state at each bar
  - Yield events with full BH state
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Optional

import pandas as pd

_LIB = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(_LIB))

from srfm_core import (
    BlackHoleDetector,
    GeodesicAnalyzer,
    GravitationalLens,
    HawkingMonitor,
    MinkowskiClassifier,
    MarketRegime,
)
from regime import RegimeDetector

from .bh_engine import (
    GEEKY_DEFAULTS,
    INSTRUMENT_CONFIGS,
    TF_CAP,
    _ATR, _ADX, _BB, _EMA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous generator
# ---------------------------------------------------------------------------

def replay_bars(
    sym: str,
    df: pd.DataFrame,
    speed_mult: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    starting_equity: float = 1_000_000.0,
) -> Generator[Dict[str, Any], None, None]:
    """
    Replay BH state bar by bar.

    Yields one event dict per hourly bar:
    {
      bar_idx, timestamp, price, beta, is_timelike, bh_mass, bh_active,
      bh_dir, ctl, regime, position_frac, pos_floor, equity,
      bh_mass_1d, bh_mass_1h, bh_mass_15m, tf_score
    }
    """
    cfg = dict(INSTRUMENT_CONFIGS.get(sym.upper(), INSTRUMENT_CONFIGS["ES"]))
    cfg.update(GEEKY_DEFAULTS)
    if params:
        cfg.update(params)

    cf         = cfg["cf"]
    bh_form    = cfg["bh_form"]
    bh_coll    = cfg["bh_collapse"]
    bh_decay   = cfg["bh_decay"]

    # Normalize input
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("open", "o"): col_map[c] = "open"
        elif lc in ("high", "h"): col_map[c] = "high"
        elif lc in ("low", "l"): col_map[c] = "low"
        elif lc in ("close", "c"): col_map[c] = "close"
        elif lc in ("volume", "vol", "v"): col_map[c] = "volume"
    df = df.rename(columns=col_map)
    if "volume" not in df.columns:
        df["volume"] = 1000.0
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Resample to 1h for replay spine
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_1h  = df.resample("1h").agg(agg).dropna(subset=["close"])
    df_1d  = df.resample("1D").agg(agg).dropna(subset=["close"])
    df_15m = df.resample("15min").agg(agg).dropna(subset=["close"])

    # Physics
    mc_1d  = MinkowskiClassifier(cf=cf)
    mc_1h  = MinkowskiClassifier(cf=cf)
    mc_15m = MinkowskiClassifier(cf=cf * 0.4)
    bh_1d  = BlackHoleDetector(bh_form, bh_coll, bh_decay)
    bh_1h  = BlackHoleDetector(bh_form, bh_coll, bh_decay)
    bh_15m = BlackHoleDetector(bh_form, bh_coll, bh_decay)
    geo    = GeodesicAnalyzer(cf=cf)
    gl     = GravitationalLens()
    hw     = HawkingMonitor()
    reg    = RegimeDetector(atr_window=50)

    e12  = _EMA(12);  e26  = _EMA(26)
    e50  = _EMA(50);  e200 = _EMA(200)
    atr_ = _ATR(14);  adx_ = _ADX(14)
    bb_  = _BB(20)

    equity    = starting_equity
    pos_frac  = 0.0
    pos_floor = 0.0
    prev_1d_close: Optional[float] = None

    bar_idx = 0

    for day_ts, day_row in df_1d.iterrows():
        day_close = float(day_row["close"])
        day_high  = float(day_row["high"])
        day_low   = float(day_row["low"])

        bit_1d = mc_1d.update(day_close)
        prev_1d = prev_1d_close if prev_1d_close else day_close
        bh_1d.update(bit_1d, day_close, prev_1d)
        prev_1d_close = day_close

        ema12  = e12.update(day_close)
        ema26  = e26.update(day_close)
        ema50  = e50.update(day_close)
        ema200 = e200.update(day_close)
        atr_v  = atr_.update(day_high, day_low, day_close)
        adx_v  = adx_.update(day_high, day_low, day_close)
        bb_mid, bb_std = bb_.update(day_close)

        regime, _ = reg.update(day_close, ema12, ema26, ema50, ema200, adx_v, atr_v)
        ht = hw.update(day_close, bb_mid, bb_std)

        day_date = pd.Timestamp(day_ts).date()
        h_mask   = df_1h.index.date == day_date

        for h_ts, h_row in df_1h[h_mask].iterrows():
            h_close  = float(h_row["close"])
            h_high   = float(h_row["high"])
            h_low    = float(h_row["low"])
            h_vol    = float(h_row.get("volume", 1000.0))

            bit_1h = mc_1h.update(h_close)
            bh_1h.update(bit_1h, h_close,
                         float(df_1h["close"].iloc[max(0, df_1h.index.get_loc(h_ts) - 1)]))

            # 15m
            h_start = pd.Timestamp(h_ts)
            h_end   = h_start + pd.Timedelta(hours=1)
            m15_mask = (df_15m.index >= h_start) & (df_15m.index < h_end)
            for _, m_row in df_15m[m15_mask].iterrows():
                bit_15m = mc_15m.update(float(m_row["close"]))
                bh_15m.update(bit_15m, float(m_row["close"]), h_close)

            tf_score = (
                (4 if bh_1d.bh_active else 0)
                | (2 if bh_1h.bh_active else 0)
                | (1 if bh_15m.bh_active else 0)
            )

            ceiling   = TF_CAP.get(tf_score, 0.0)
            direction = bh_1d.bh_dir if bh_1d.bh_active else bh_1h.bh_dir
            target    = ceiling * (1.0 if direction > 0 else -1.0 if direction < 0 else 0.0)

            # pos_floor
            if tf_score >= 6 and abs(target) > 0.15 and bh_1h.ctl >= 5:
                pos_floor = max(pos_floor, 0.70 * abs(target))
            if pos_floor > 0 and tf_score >= 4 and not (pos_frac == 0.0):
                target     = max(target, pos_floor) if direction >= 0 else min(target, -pos_floor)
                pos_floor *= 0.95
            if tf_score < 4 or target == 0.0:
                pos_floor = 0.0

            if abs(target - pos_frac) > 0.02:
                bar_ret = (h_close - (df_1h["close"].iloc[max(0, df_1h.index.get_loc(h_ts) - 1)])) / h_close
                equity += pos_frac * equity * bar_ret
                pos_frac = target

            geo_dev, geo_slope, causal_frac, rapidity = geo.update(h_close, atr_v)
            gl.update(h_close, h_vol, bit_1h, mc_1h.tl_confirm, atr_v)

            yield {
                "bar_idx":      bar_idx,
                "timestamp":    str(h_ts),
                "price":        h_close,
                "beta":         mc_1h.beta,
                "is_timelike":  mc_1h.is_timelike,
                "bh_mass":      bh_1h.bh_mass,
                "bh_mass_1d":   bh_1d.bh_mass,
                "bh_mass_1h":   bh_1h.bh_mass,
                "bh_mass_15m":  bh_15m.bh_mass,
                "bh_active":    bh_1h.bh_active,
                "bh_dir":       bh_1h.bh_dir,
                "ctl":          bh_1h.ctl,
                "tf_score":     tf_score,
                "regime":       regime.name,
                "position_frac": pos_frac,
                "pos_floor":    pos_floor,
                "equity":       equity,
                "geo_dev":      geo_dev,
                "geo_slope":    geo_slope,
                "mu":           gl.mu,
                "ht":           ht,
            }
            bar_idx += 1


# ---------------------------------------------------------------------------
# Async generator (for WebSocket streaming)
# ---------------------------------------------------------------------------

async def async_replay_bars(
    sym: str,
    df: pd.DataFrame,
    speed_mult: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    starting_equity: float = 1_000_000.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async version of replay_bars, yielding with asyncio.sleep between bars."""
    delay = max(0.0, 1.0 / max(speed_mult, 0.001))

    for event in replay_bars(sym, df, speed_mult=speed_mult, params=params,
                              starting_equity=starting_equity):
        yield event
        if delay > 0:
            await asyncio.sleep(delay)
