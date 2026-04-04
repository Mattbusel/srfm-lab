"""
bh_engine.py — Universal Black Hole backtester engine for Spacetime Arena.

Supports any instrument with OHLCV data via pandas DataFrame.
3-timeframe support: daily, hourly, 15m.
Full trade record output with MFE/MAE, tf_score, regime, bh_mass_at_entry.
"""

from __future__ import annotations

import sys
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Resolve lib path
_LIB = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(_LIB))

from srfm_core import (
    MinkowskiClassifier,
    BlackHoleDetector,
    GeodesicAnalyzer,
    GravitationalLens,
    HawkingMonitor,
    MarketRegime,
)
from regime import RegimeDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument configs
# ---------------------------------------------------------------------------

INSTRUMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Equity indices
    "ES":  {"asset_class": "equity_index", "cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "NQ":  {"asset_class": "equity_index", "cf": 0.0012, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "YM":  {"asset_class": "equity_index", "cf": 0.0008, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "RTY": {"asset_class": "equity_index", "cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "SPY": {"asset_class": "equity_index", "cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "QQQ": {"asset_class": "equity_index", "cf": 0.0012, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # Commodities — energy
    "CL":  {"asset_class": "commodity",    "cf": 0.015,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "NG":  {"asset_class": "commodity",    "cf": 0.020,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # Commodities — metals
    "GC":  {"asset_class": "commodity",    "cf": 0.008,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "SI":  {"asset_class": "commodity",    "cf": 0.008,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # Bonds
    "ZB":  {"asset_class": "bond",         "cf": 0.003,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "ZN":  {"asset_class": "bond",         "cf": 0.003,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # Forex
    "EURUSD": {"asset_class": "forex",     "cf": 0.0005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "GBPUSD": {"asset_class": "forex",     "cf": 0.0005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "USDJPY": {"asset_class": "forex",     "cf": 0.0005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # Crypto
    "BTC":  {"asset_class": "crypto",      "cf": 0.005,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "ETH":  {"asset_class": "crypto",      "cf": 0.007,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "SOL":  {"asset_class": "crypto",      "cf": 0.010,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    # VIX
    "VIX":  {"asset_class": "volatility",  "cf": 0.025,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
}

# Geeky Orange Sheep defaults
GEEKY_DEFAULTS = {
    "bh_form": 1.5,
    "bh_collapse": 1.0,
    "bh_decay": 0.95,
}

TF_CAP = {7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0}

MIN_BARS = 100


# ---------------------------------------------------------------------------
# Internal indicator helpers (matching arena.py pattern)
# ---------------------------------------------------------------------------

class _EMA:
    def __init__(self, p: int) -> None:
        self.k = 2.0 / (p + 1)
        self.v: Optional[float] = None

    def update(self, x: float) -> float:
        self.v = x if self.v is None else x * self.k + self.v * (1 - self.k)
        return self.v  # type: ignore[return-value]


class _ATR:
    def __init__(self, p: int = 14) -> None:
        self.p = p
        self._prev: Optional[float] = None
        self._buf: List[float] = []
        self.v: Optional[float] = None

    def update(self, h: float, lo: float, c: float) -> float:
        tr = (h - lo) if self._prev is None else max(h - lo, abs(h - self._prev), abs(lo - self._prev))
        self._prev = c
        if self.v is None:
            self._buf.append(tr)
            if len(self._buf) >= self.p:
                self.v = sum(self._buf) / len(self._buf)
        else:
            self.v = (self.v * (self.p - 1) + tr) / self.p
        return self.v or tr

    @property
    def ready(self) -> bool:
        return self.v is not None


class _ADX:
    def __init__(self, p: int = 14) -> None:
        self.p = p
        self._atr = _ATR(p)
        self._ph: Optional[float] = None
        self._pl: Optional[float] = None
        self._pdm = self._ndm = 0.0
        self._dx_buf: List[float] = []
        self.v = 0.0

    def update(self, h: float, lo: float, c: float) -> float:
        atr = self._atr.update(h, lo, c)
        if self._ph is None:
            self._ph, self._pl = h, lo
            return 0.0
        pdm = max(0.0, h - self._ph)
        ndm = max(0.0, self._pl - lo)
        if pdm > ndm:
            ndm = 0.0
        elif ndm > pdm:
            pdm = 0.0
        else:
            pdm = ndm = 0.0
        k = 2.0 / (self.p + 1)
        self._pdm = pdm * k + self._pdm * (1 - k)
        self._ndm = ndm * k + self._ndm * (1 - k)
        if atr > 0:
            pdi, ndi = 100 * self._pdm / atr, 100 * self._ndm / atr
            d = pdi + ndi
            dx = 100 * abs(pdi - ndi) / d if d > 0 else 0.0
        else:
            dx = 0.0
        self._dx_buf.append(dx)
        if len(self._dx_buf) >= self.p:
            self.v = sum(self._dx_buf[-self.p :]) / self.p
        self._ph, self._pl = h, lo
        return self.v


class _BB:
    def __init__(self, p: int = 20) -> None:
        self.p = p
        self._buf: List[float] = []

    def update(self, c: float) -> Tuple[float, float]:
        self._buf.append(c)
        if len(self._buf) > self.p:
            self._buf.pop(0)
        mid = sum(self._buf) / len(self._buf)
        std = (sum((x - mid) ** 2 for x in self._buf) / len(self._buf)) ** 0.5
        return mid, std


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    entry_time: Any
    exit_time: Any
    sym: str
    entry_price: float
    exit_price: float
    pnl: float
    hold_bars: int
    mfe: float          # max favorable excursion (as fraction)
    mae: float          # max adverse excursion (as fraction, positive = loss)
    tf_score: int       # 0-7, number of active timeframes (bitmask)
    regime: str
    bh_mass_at_entry: float


@dataclass
class BarState:
    timestamp: Any
    price: float
    bh_active_1d: bool
    bh_active_1h: bool
    bh_active_15m: bool
    bh_mass_1d: float
    bh_mass_1h: float
    bh_mass_15m: float
    bh_dir_1d: int
    bh_dir_1h: int
    bh_dir_15m: int
    tf_score: int
    regime: str
    regime_confidence: float
    pos_floor: float
    position_frac: float


@dataclass
class BacktestResult:
    sym: str
    trades: List[TradeRecord]
    equity_curve: List[Tuple[Any, float]]
    bar_states: List[BarState]
    mass_series_1d: List[float]
    mass_series_1h: List[float]
    mass_series_15m: List[float]
    stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# 3-timeframe BH state (matching crypto_backtest_mc.py pattern)
# ---------------------------------------------------------------------------

class _BHState3TF:
    """Holds BH detectors for 1d / 1h / 15m timeframes."""

    def __init__(self, cf_1d: float, cf_1h: float, cf_15m: float,
                 bh_form: float, bh_collapse: float, bh_decay: float) -> None:
        self.bh_1d  = BlackHoleDetector(bh_form, bh_collapse, bh_decay)
        self.bh_1h  = BlackHoleDetector(bh_form, bh_collapse, bh_decay)
        self.bh_15m = BlackHoleDetector(bh_form, bh_collapse, bh_decay)

        self.mc_1d  = MinkowskiClassifier(cf=cf_1d)
        self.mc_1h  = MinkowskiClassifier(cf=cf_1h)
        self.mc_15m = MinkowskiClassifier(cf=cf_15m)

    def update_1d(self, close: float, prev_close: float) -> bool:
        bit = self.mc_1d.update(close)
        return self.bh_1d.update(bit, close, prev_close)

    def update_1h(self, close: float, prev_close: float) -> bool:
        bit = self.mc_1h.update(close)
        return self.bh_1h.update(bit, close, prev_close)

    def update_15m(self, close: float, prev_close: float) -> bool:
        bit = self.mc_15m.update(close)
        return self.bh_15m.update(bit, close, prev_close)

    @property
    def tf_score(self) -> int:
        d = 4 if self.bh_1d.bh_active else 0
        h = 2 if self.bh_1h.bh_active else 0
        m = 1 if self.bh_15m.bh_active else 0
        return d | h | m


# ---------------------------------------------------------------------------
# Regime helper
# ---------------------------------------------------------------------------

def _classify_regime_simple(bh_dir: int, bh_mass: float, atr_ratio: float) -> str:
    """Simple regime from daily BH direction and mass level."""
    if atr_ratio >= 1.5:
        return "HIGH_VOL"
    if bh_dir > 0 and bh_mass > 1.5:
        return "BULL"
    if bh_dir < 0 and bh_mass > 1.5:
        return "BEAR"
    return "SIDEWAYS"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class BHEngine:
    """
    Universal BH backtester engine.

    Parameters
    ----------
    sym        : instrument symbol (used for config lookup)
    long_only  : if True, only take long trades
    params     : override any config key (cf, bh_form, bh_collapse, bh_decay)
    starting_equity : portfolio starting capital
    """

    def __init__(
        self,
        sym: str,
        long_only: bool = True,
        params: Optional[Dict[str, Any]] = None,
        starting_equity: float = 1_000_000.0,
    ) -> None:
        self.sym = sym
        self.long_only = long_only
        self.starting_equity = starting_equity

        cfg = dict(INSTRUMENT_CONFIGS.get(sym.upper(), INSTRUMENT_CONFIGS["ES"]))
        cfg.update(GEEKY_DEFAULTS)
        if params:
            cfg.update(params)

        self.cfg = cfg
        # Scale CF per timeframe — daily moves ~5x hourly, hourly ~4x 15m
        base_cf = cfg["cf"]
        self.cf_1d  = cfg.get("cf_1d",  base_cf * 5.0)
        self.cf_1h  = cfg.get("cf_1h",  base_cf)
        self.cf_15m = cfg.get("cf_15m", base_cf * 0.35)

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on OHLCV DataFrame.

        DataFrame must have columns: open, high, low, close, volume
        Index: DatetimeIndex (any frequency; will be auto-resampled if needed).

        Returns BacktestResult.
        """
        if len(df) < MIN_BARS:
            raise ValueError(f"Insufficient data: {len(df)} bars < {MIN_BARS} required")

        df = self._normalize_df(df)
        df_1d, df_1h, df_15m = self._resample(df)

        return self._run_multiframe(df_1d, df_1h, df_15m)

    # ------------------------------------------------------------------
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure DatetimeIndex."""
        col_map = {}
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ("open", "o"):
                col_map[c] = "open"
            elif lc in ("high", "h"):
                col_map[c] = "high"
            elif lc in ("low", "l"):
                col_map[c] = "low"
            elif lc in ("close", "c"):
                col_map[c] = "close"
            elif lc in ("volume", "vol", "v"):
                col_map[c] = "volume"
        df = df.rename(columns=col_map)
        if "volume" not in df.columns:
            df["volume"] = 1000.0
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ------------------------------------------------------------------
    def _resample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Produce 1d, 1h, 15m DataFrames from input."""
        # Detect frequency
        if len(df) > 1:
            median_diff = pd.Series(df.index).diff().dropna().median()
            freq_minutes = median_diff.total_seconds() / 60
        else:
            freq_minutes = 60

        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

        if freq_minutes <= 1:
            df_15m = df.resample("15min").agg(agg).dropna(subset=["close"])
            df_1h  = df.resample("1h").agg(agg).dropna(subset=["close"])
            df_1d  = df.resample("1D").agg(agg).dropna(subset=["close"])
        elif freq_minutes <= 15:
            df_15m = df
            df_1h  = df.resample("1h").agg(agg).dropna(subset=["close"])
            df_1d  = df.resample("1D").agg(agg).dropna(subset=["close"])
        elif freq_minutes <= 60:
            df_1h  = df
            df_15m = df.resample("15min").agg(agg).dropna(subset=["close"])
            df_1d  = df.resample("1D").agg(agg).dropna(subset=["close"])
        else:
            df_1d  = df
            df_1h  = df.resample("1h").agg(agg).dropna(subset=["close"])
            df_15m = df.resample("15min").agg(agg).dropna(subset=["close"])

        return df_1d, df_1h, df_15m

    # ------------------------------------------------------------------
    def _run_multiframe(
        self,
        df_1d: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> BacktestResult:
        """Main backtest loop, 3-timeframe synchronized."""
        cfg = self.cfg
        bh_form     = cfg["bh_form"]
        bh_collapse = cfg["bh_collapse"]
        bh_decay    = cfg["bh_decay"]

        bh_1d  = BlackHoleDetector(bh_form, bh_collapse, bh_decay)
        bh_1h  = BlackHoleDetector(bh_form, bh_collapse, bh_decay)
        bh_15m = BlackHoleDetector(bh_form, bh_collapse, bh_decay)

        mc_1d  = MinkowskiClassifier(cf=self.cf_1d)
        mc_1h  = MinkowskiClassifier(cf=self.cf_1h)
        mc_15m = MinkowskiClassifier(cf=self.cf_15m)

        # Regime detector on daily
        reg = RegimeDetector(atr_window=50)
        e12  = _EMA(12);  e26  = _EMA(26)
        e50  = _EMA(50);  e200 = _EMA(200)
        atr_ = _ATR(14);  adx_ = _ADX(14)
        bb_  = _BB(20)

        mass_1d:  List[float] = []
        mass_1h:  List[float] = []
        mass_15m: List[float] = []
        bar_states: List[BarState] = []
        equity_curve: List[Tuple[Any, float]] = []
        trades: List[TradeRecord] = []

        equity  = self.starting_equity
        peak    = equity
        pos_frac = 0.0
        pos_floor = 0.0

        # Open trade tracking
        in_trade = False
        entry_time: Any = None
        entry_price = 0.0
        trade_entry_tf = 0
        trade_entry_regime = "SIDEWAYS"
        trade_entry_mass = 0.0
        trade_high_price = 0.0
        trade_low_price = 0.0
        trade_bars = 0

        prev_1d_close: Optional[float] = None
        atr_history: List[float] = []

        for day_ts, day_row in df_1d.iterrows():
            day_close = float(day_row["close"])
            day_high  = float(day_row["high"])
            day_low   = float(day_row["low"])

            # Daily BH
            bit_1d = mc_1d.update(day_close)
            prev_close_1d = prev_1d_close if prev_1d_close else day_close
            bh_1d.update(bit_1d, day_close, prev_close_1d)
            prev_1d_close = day_close

            # Daily indicators
            ema12  = e12.update(day_close)
            ema26  = e26.update(day_close)
            ema50  = e50.update(day_close)
            ema200 = e200.update(day_close)
            atr_v  = atr_.update(day_high, day_low, day_close)
            adx_v  = adx_.update(day_high, day_low, day_close)
            bb_mid, bb_std = bb_.update(day_close)

            atr_history.append(atr_v)
            atr_ratio = atr_v / (sum(atr_history[-50:]) / len(atr_history[-50:]) + 1e-9)

            regime, reg_conf = reg.update(day_close, ema12, ema26, ema50, ema200, adx_v, atr_v)
            regime_str = regime.name

            mass_1d.append(bh_1d.bh_mass)

            # Hourly bars for this day
            day_date = pd.Timestamp(day_ts).date()
            h1_mask = df_1h.index.date == day_date
            h1_today = df_1h[h1_mask]

            prev_h_close: Optional[float] = None

            for h_ts, h_row in h1_today.iterrows():
                h_close = float(h_row["close"])
                h_high  = float(h_row["high"])
                h_low   = float(h_row["low"])

                # Hourly BH
                bit_1h = mc_1h.update(h_close)
                prev_h = prev_h_close if prev_h_close else h_close
                bh_1h.update(bit_1h, h_close, prev_h)
                prev_h_close = h_close

                mass_1h.append(bh_1h.bh_mass)

                # 15m bars within this hour
                h_start = pd.Timestamp(h_ts)
                h_end   = h_start + pd.Timedelta(hours=1)
                m15_mask = (df_15m.index >= h_start) & (df_15m.index < h_end)
                m15_today = df_15m[m15_mask]

                prev_m_close: Optional[float] = None

                for m_ts, m_row in m15_today.iterrows():
                    m_close = float(m_row["close"])
                    bit_15m = mc_15m.update(m_close)
                    prev_m  = prev_m_close if prev_m_close else m_close
                    bh_15m.update(bit_15m, m_close, prev_m)
                    prev_m_close = m_close
                    mass_15m.append(bh_15m.bh_mass)

                # TF score at hourly bar
                tf_score = (
                    (4 if bh_1d.bh_active else 0)
                    | (2 if bh_1h.bh_active else 0)
                    | (1 if bh_15m.bh_active else 0)
                )

                # Position sizing
                ceiling = TF_CAP.get(tf_score, 0.0)

                # Direction from daily BH (most reliable)
                direction = bh_1d.bh_dir if bh_1d.bh_active else bh_1h.bh_dir
                if direction == 0 and bh_15m.bh_active:
                    direction = bh_15m.bh_dir

                if self.long_only and direction < 0:
                    direction = 0

                target_frac = ceiling * (1.0 if direction > 0 else (-1.0 if direction < 0 else 0.0))

                # pos_floor logic (from crypto_backtest_mc.py)
                if tf_score >= 6 and abs(target_frac) > 0.15 and bh_1h.ctl >= 5:
                    pos_floor = max(pos_floor, 0.70 * abs(target_frac))
                if pos_floor > 0 and tf_score >= 4 and not math.isclose(pos_frac, 0.0):
                    target_frac = max(target_frac, pos_floor) if direction >= 0 else min(target_frac, -pos_floor)
                    pos_floor *= 0.95
                if tf_score < 4 or math.isclose(target_frac, 0.0):
                    pos_floor = 0.0

                # Trade open / close logic
                delta = abs(target_frac - pos_frac)
                if delta > 0.02:
                    if in_trade and entry_price > 0:
                        # Close trade
                        exit_price = h_close
                        ret = (exit_price - entry_price) / entry_price
                        pnl = pos_frac * equity * ret
                        equity += pnl

                        # MFE / MAE
                        if pos_frac > 0:
                            mfe = (trade_high_price - entry_price) / entry_price
                            mae = (entry_price - trade_low_price) / entry_price
                        else:
                            mfe = (entry_price - trade_low_price) / entry_price
                            mae = (trade_high_price - entry_price) / entry_price

                        trades.append(TradeRecord(
                            entry_time=entry_time,
                            exit_time=h_ts,
                            sym=self.sym,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            hold_bars=trade_bars,
                            mfe=float(mfe),
                            mae=float(mae),
                            tf_score=trade_entry_tf,
                            regime=trade_entry_regime,
                            bh_mass_at_entry=trade_entry_mass,
                        ))
                        in_trade = False

                    if not math.isclose(target_frac, 0.0):
                        # Open new trade
                        entry_time         = h_ts
                        entry_price        = h_close
                        trade_entry_tf     = tf_score
                        trade_entry_regime = regime_str
                        trade_entry_mass   = bh_1d.bh_mass
                        trade_high_price   = h_close
                        trade_low_price    = h_close
                        trade_bars         = 0
                        in_trade           = True

                    pos_frac = target_frac

                if in_trade and entry_price > 0:
                    trade_high_price = max(trade_high_price, h_close)
                    trade_low_price  = min(trade_low_price, h_close)
                    trade_bars      += 1

                if equity > peak:
                    peak = equity

                bar_states.append(BarState(
                    timestamp=h_ts,
                    price=h_close,
                    bh_active_1d=bh_1d.bh_active,
                    bh_active_1h=bh_1h.bh_active,
                    bh_active_15m=bh_15m.bh_active,
                    bh_mass_1d=bh_1d.bh_mass,
                    bh_mass_1h=bh_1h.bh_mass,
                    bh_mass_15m=bh_15m.bh_mass,
                    bh_dir_1d=bh_1d.bh_dir,
                    bh_dir_1h=bh_1h.bh_dir,
                    bh_dir_15m=bh_15m.bh_dir,
                    tf_score=tf_score,
                    regime=regime_str,
                    regime_confidence=reg_conf,
                    pos_floor=pos_floor,
                    position_frac=pos_frac,
                ))

            equity_curve.append((day_ts, equity))

        # Close any open trade at end
        if in_trade and entry_price > 0 and bar_states:
            last_price = bar_states[-1].price
            ret = (last_price - entry_price) / entry_price
            pnl = pos_frac * equity * ret
            equity += pnl
            mfe = (trade_high_price - entry_price) / entry_price if pos_frac > 0 else (entry_price - trade_low_price) / entry_price
            mae = (entry_price - trade_low_price) / entry_price if pos_frac > 0 else (trade_high_price - entry_price) / entry_price
            trades.append(TradeRecord(
                entry_time=entry_time,
                exit_time=bar_states[-1].timestamp,
                sym=self.sym,
                entry_price=entry_price,
                exit_price=last_price,
                pnl=pnl,
                hold_bars=trade_bars,
                mfe=float(mfe),
                mae=float(mae),
                tf_score=trade_entry_tf,
                regime=trade_entry_regime,
                bh_mass_at_entry=trade_entry_mass,
            ))

        stats = self._compute_stats(trades, equity_curve, peak)

        return BacktestResult(
            sym=self.sym,
            trades=trades,
            equity_curve=equity_curve,
            bar_states=bar_states,
            mass_series_1d=mass_1d,
            mass_series_1h=mass_1h,
            mass_series_15m=mass_15m,
            stats=stats,
        )

    # ------------------------------------------------------------------
    def _compute_stats(
        self,
        trades: List[TradeRecord],
        equity_curve: List[Tuple[Any, float]],
        peak: float,
    ) -> Dict[str, Any]:
        if not equity_curve:
            return {}

        values = np.array([v for _, v in equity_curve])
        start  = values[0]
        end    = values[-1]
        dates  = [t for t, _ in equity_curve]

        years = 1.0
        if len(dates) >= 2:
            try:
                d0 = pd.Timestamp(dates[0])
                d1 = pd.Timestamp(dates[-1])
                years = max(0.01, (d1 - d0).days / 365.25)
            except Exception:
                pass

        cagr = (end / start) ** (1 / years) - 1 if end > 0 else -1.0

        pk   = np.maximum.accumulate(values)
        dd   = (values - pk) / (pk + 1e-9)
        max_dd = float(dd.min())

        rets = np.diff(values) / (values[:-1] + 1e-9)
        sharpe = float(rets.mean() / (rets.std() + 1e-12) * math.sqrt(252)) if len(rets) > 1 else 0.0

        pnls   = [t.pnl for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        pf = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")
        avg_hold = float(np.mean([t.hold_bars for t in trades])) if trades else 0.0

        return {
            "cagr":           round(float(cagr), 4),
            "sharpe":         round(sharpe, 3),
            "max_drawdown":   round(float(max_dd), 4),
            "win_rate":       round(win_rate, 4),
            "profit_factor":  round(float(pf), 3),
            "trade_count":    len(trades),
            "avg_hold_bars":  round(avg_hold, 1),
            "final_equity":   round(float(end), 2),
            "total_return":   round(float((end - start) / start), 4),
        }


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_backtest(
    sym: str,
    df: pd.DataFrame,
    long_only: bool = True,
    params: Optional[Dict[str, Any]] = None,
    starting_equity: float = 1_000_000.0,
) -> BacktestResult:
    """Run a BH backtest and return full results."""
    engine = BHEngine(sym, long_only=long_only, params=params, starting_equity=starting_equity)
    return engine.run(df)
