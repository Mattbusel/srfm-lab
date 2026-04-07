"""
tools/larsa_v18_backtest.py
===========================
LARSA v18 -- Production Backtesting Adapter

Standalone backtesting script that replicates the full LARSA v18 live trader
logic for historical data. Supports SQLite, CSV, and synthetic GBM data.

Usage:
    python tools/larsa_v18_backtest.py --start 2024-01-01 --end 2024-12-31

No em dashes used anywhere in this file.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    stream=__import__("sys").stderr,
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("larsa_v18_backtest")

_REPO_ROOT = Path(__file__).parents[1]


# =============================================================================
# CONFIG DATACLASS
# =============================================================================

@dataclass
class LARSAv18Config:
    """All strategy parameters matching live trader constants."""

    # BH physics
    BH_FORM: float = 1.92
    BH_CTL_MIN: int = 3
    BH_DECAY: float = 0.924
    BH_COLLAPSE: float = 0.992

    # Risk / sizing
    DAILY_RISK: float = 0.05
    CORR_NORMAL: float = 0.25
    CORR_STRESS: float = 0.60
    CORR_STRESS_THRESHOLD: float = 0.60
    CRYPTO_CAP_FRAC: float = 0.40
    EQUITY_CAP_FRAC: float = 0.20
    MIN_TRADE_FRAC: float = 0.003
    DELTA_MAX_FRAC: float = 0.40
    OU_FRAC: float = 0.08

    # GARCH
    GARCH_TARGET_VOL: float = 0.90
    GARCH_OMEGA: float = 1e-6
    GARCH_ALPHA: float = 0.10
    GARCH_BETA: float = 0.85
    GARCH_WARMUP: int = 30

    # Hurst
    HURST_WINDOW: int = 100
    HURST_MIN_BARS: int = 30
    HURST_TRENDING_THRESH: float = 0.58
    HURST_MR_THRESH: float = 0.42

    # QuatNav
    NAV_OMEGA_SCALE_K: float = 0.5
    NAV_GEO_ENTRY_GATE: float = 3.0
    NAV_EMA_ALPHA: float = 0.05

    # RL / ML
    RL_EXIT_ACTIVE: bool = True
    ML_SIGNAL_BOOST: float = 1.20
    ML_SIGNAL_BOOST_THRESH: float = 0.30
    ML_SIGNAL_SUPPRESS_THRESH: float = -0.30

    # Granger
    GRANGER_BOOST_ACTIVE: bool = True

    # Event calendar
    EVENT_CAL_ACTIVE: bool = True

    # Position management
    MIN_HOLD_MINUTES: int = 240
    DD_HALT_PCT: float = 0.10
    DD_RESUME_PCT: float = 0.05
    STALE_15M_MOVE: float = 0.002
    WINNER_PROTECTION_PCT: float = 0.005

    # Hour filters
    BLOCKED_ENTRY_HOURS_UTC: tuple = (1, 13, 14, 15, 17, 18)
    BOOST_ENTRY_HOURS_UTC: tuple = (3, 9, 16, 19)
    HOUR_BOOST_MULTIPLIER: float = 1.25

    # TF caps
    TF_CAP: dict = field(default_factory=lambda: {
        7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0
    })

    # Transaction costs
    CRYPTO_COST_BPS: float = 15.0
    EQUITY_COST_BPS: float = 5.0

    # Warmup
    WARMUP_BARS: int = 30

    # Instruments: cf thresholds
    INSTRUMENTS: dict = field(default_factory=lambda: {
        "BTC":  {"asset_class": "crypto", "cf_4h": 0.016, "cf_15m": 0.010, "cf_1h": 0.030},
        "ETH":  {"asset_class": "crypto", "cf_4h": 0.012, "cf_15m": 0.007, "cf_1h": 0.020},
        "XRP":  {"asset_class": "crypto", "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030},
        "AVAX": {"asset_class": "crypto", "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018},
        "LINK": {"asset_class": "crypto", "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018},
        "SPY":  {"asset_class": "equity", "cf_4h": 0.003, "cf_15m": 0.0003, "cf_1h": 0.001},
        "QQQ":  {"asset_class": "equity", "cf_4h": 0.004, "cf_15m": 0.0004, "cf_1h": 0.0012},
        "GLD":  {"asset_class": "equity", "cf_4h": 0.004, "cf_15m": 0.0004, "cf_1h": 0.0012},
        "NVDA": {"asset_class": "equity", "cf_4h": 0.012, "cf_15m": 0.0012, "cf_1h": 0.004},
        "AAPL": {"asset_class": "equity", "cf_4h": 0.006, "cf_15m": 0.0006, "cf_1h": 0.002},
        "TSLA": {"asset_class": "equity", "cf_4h": 0.015, "cf_15m": 0.0015, "cf_1h": 0.005},
    })

    # Feature flags
    USE_QUATNAV: bool = True
    USE_HURST: bool = True
    USE_ML: bool = True
    USE_RL: bool = True
    USE_GRANGER: bool = True
    USE_EVENT_CAL: bool = True


# =============================================================================
# LOW-LEVEL HELPERS
# =============================================================================

def _ema(prev: float | None, val: float, alpha: float) -> float:
    return val if prev is None else alpha * val + (1.0 - alpha) * prev


def _alpha(n: int) -> float:
    return 2.0 / (n + 1)


# =============================================================================
# BH PHYSICS ENGINE
# =============================================================================

class BHPhysicsEngine:
    """
    Black Hole physics engine -- exact replica of the live trader BH mass
    computation. Supports single-timeframe and multi-timeframe (15m/1h/4h)
    operation on historical bar sequences.

    ds^2 = weight_price*(dp/p)^2 + weight_vol*(dV/V)^2 + weight_sigma*(dsigma/sigma)^2

    Mass accumulates on timelike bars (ds^2 > 0, i.e. consolidation) and
    decays between BH events as per the Minkowski metric interpretation.
    """

    # Minkowski weights for the spacetime interval
    W_PRICE: float = 1.0
    W_VOL: float = 0.3
    W_SIGMA: float = 0.2

    def __init__(self, cf: float, cfg: LARSAv18Config) -> None:
        self.cf = cf
        self.cfg = cfg
        self.cf_scale: float = 1.0
        self.mass: float = 0.0
        self.active: bool = False
        self.bh_dir: int = 0
        self.ctl: int = 0
        self._prices: deque[float] = deque(maxlen=25)
        self._vols: deque[float] = deque(maxlen=25)
        self._sigmas: deque[float] = deque(maxlen=25)

    def ds_squared(self, p_prev: float, p_now: float,
                   v_prev: float, v_now: float,
                   s_prev: float, s_now: float) -> float:
        """Compute Minkowski spacetime interval ds^2."""
        dp_p = (p_now - p_prev) / (p_prev + 1e-12)
        dv_v = (v_now - v_prev) / (v_prev + 1e-12) if v_prev > 1e-9 else 0.0
        ds_s = (s_now - s_prev) / (s_prev + 1e-12) if s_prev > 1e-9 else 0.0
        return (self.W_PRICE * dp_p ** 2
                + self.W_VOL * dv_v ** 2
                + self.W_SIGMA * ds_s ** 2)

    def update(self, price: float, volume: float = 0.0, sigma: float = 0.0) -> None:
        """Update BH state with a new bar. Mirrors the live BHState.update() logic."""
        self._prices.append(float(price))
        self._vols.append(float(volume) if volume > 0 else 1.0)
        self._sigmas.append(float(sigma) if sigma > 0 else 1.0)

        if len(self._prices) < 2:
            return

        px = list(self._prices)
        beta = (abs(px[-1] - px[-2]) / (px[-2] + 1e-9)
                / (self.cf * self.cf_scale + 1e-9))

        was = self.active
        if beta < 1.0:
            # Timelike bar -- consolidating, accumulate mass
            self.ctl += 1
            self.mass = self.mass * 0.97 + 0.03 * min(2.0, 1.0 + self.ctl * 0.1)
        else:
            # Spacelike bar -- price moving, decay mass
            self.ctl = 0
            self.mass *= self.cfg.BH_DECAY

        if not was:
            self.active = (self.mass > self.cfg.BH_FORM
                           and self.ctl >= self.cfg.BH_CTL_MIN)
        else:
            self.active = (self.mass > self.cfg.BH_COLLAPSE
                           and self.ctl >= self.cfg.BH_CTL_MIN)

        if not was and self.active:
            lb = min(20, len(px) - 1)
            self.bh_dir = 1 if px[-1] > px[-1 - lb] else -1
        elif was and not self.active:
            self.bh_dir = 0

    def compute_on_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of OHLCV bars and return a DataFrame with columns:
        [mass, active, bh_dir, ctl, ds2].
        bars must have columns: open, high, low, close, volume (optional).
        """
        results = []
        prev_sigma = None
        sigma_ema = None

        for _, row in bars.iterrows():
            c = float(row["close"])
            v = float(row.get("volume", 1.0)) if "volume" in row.index else 1.0

            # Running vol proxy (annualised 15m return std as sigma)
            if len(self._prices) >= 2:
                ret = math.log(c / (list(self._prices)[-1] + 1e-12) + 1e-12)
                sigma_ema = _ema(sigma_ema, abs(ret), 0.1)
            else:
                sigma_ema = 0.001

            self.update(c, v, sigma_ema or 0.001)

            ds2 = 0.0
            if len(self._prices) >= 2:
                px = list(self._prices)
                vx = list(self._vols)
                sx = list(self._sigmas)
                ds2 = self.ds_squared(px[-2], px[-1],
                                      vx[-2], vx[-1],
                                      sx[-2], sx[-1])

            results.append({
                "mass": self.mass,
                "active": self.active,
                "bh_dir": self.bh_dir,
                "ctl": self.ctl,
                "ds2": ds2,
            })

        return pd.DataFrame(results, index=bars.index)

    def reset(self) -> None:
        self.mass = 0.0
        self.active = False
        self.bh_dir = 0
        self.ctl = 0
        self._prices.clear()
        self._vols.clear()
        self._sigmas.clear()


# =============================================================================
# INDICATOR HELPERS
# =============================================================================

class ATRTracker:
    """14-bar EMA of ATR."""

    def __init__(self) -> None:
        self.atr: float | None = None
        self.prev_c: float | None = None

    def update(self, h: float, l: float, c: float) -> None:
        tr = ((h - l) if self.prev_c is None
              else max(h - l, abs(h - self.prev_c), abs(l - self.prev_c)))
        self.atr = _ema(self.atr, tr, _alpha(14))
        self.prev_c = c


class GARCHTracker:
    """Online GARCH(1,1) vol forecaster -- annualised."""

    def __init__(self, omega: float = 1e-6, alpha: float = 0.10,
                 beta: float = 0.85, warmup: int = 30,
                 target_vol: float = 0.90) -> None:
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self._warmup = warmup
        self.target_vol = target_vol
        self._var: float | None = None
        self._returns: deque[float] = deque(maxlen=100)
        self.vol: float | None = None

    def update(self, ret: float) -> None:
        self._returns.append(ret)
        if len(self._returns) < self._warmup:
            return
        if self._var is None:
            self._var = float(np.var(list(self._returns))) + 1e-12
        else:
            self._var = self.omega + self.alpha * ret ** 2 + self.beta * self._var
        self.vol = math.sqrt(max(self._var, 1e-12) * 365)

    @property
    def vol_scale(self) -> float:
        if self.vol is None or self.vol <= 0:
            return 1.0
        return min(2.0, max(0.3, self.target_vol / self.vol))


class HurstEstimator:
    """Online Hurst exponent estimator using R/S (rescaled range) analysis."""

    def __init__(self, window: int = 100, min_bars: int = 30) -> None:
        self._window = window
        self._min_bars = min_bars
        self._prices: deque[float] = deque(maxlen=window)
        self.hurst: float | None = None
        self.regime_bias: str = "neutral"

    def update(self, price: float) -> None:
        self._prices.append(math.log(max(price, 1e-12)))
        if len(self._prices) < self._min_bars:
            return
        self.hurst = self._rs_hurst(list(self._prices))
        if self.hurst is not None:
            if self.hurst > 0.58:
                self.regime_bias = "trending"
            elif self.hurst < 0.42:
                self.regime_bias = "mean_reverting"
            else:
                self.regime_bias = "neutral"

    @staticmethod
    def _rs_hurst(log_prices: list[float]) -> float | None:
        try:
            rets = np.diff(log_prices)
            n = len(rets)
            if n < 10:
                return None
            ns: list[float] = []
            rs_vals: list[float] = []
            for sub_n in [max(8, n // 4), max(8, n // 2), n]:
                if sub_n > n:
                    continue
                chunk = rets[:sub_n]
                mean_r = np.mean(chunk)
                dev = np.cumsum(chunk - mean_r)
                R = dev.max() - dev.min()
                S = np.std(chunk, ddof=1)
                if S < 1e-12:
                    continue
                ns.append(math.log(sub_n))
                rs_vals.append(math.log(R / S))
            if len(ns) < 2:
                return None
            h = float(np.polyfit(ns, rs_vals, 1)[0])
            return max(0.05, min(0.95, h))
        except Exception:
            return None

    @property
    def is_trending(self) -> bool:
        return self.regime_bias == "trending"

    @property
    def is_mean_reverting(self) -> bool:
        return self.regime_bias == "mean_reverting"


class QuatNavState:
    """
    Pure-Python quaternion navigation state.
    Tracks angular velocity (rate of change of rotation in price-space)
    and geodesic deviation (unexpected path length in returns manifold).
    Used as a proxy for the Rust quat_nav_bridge module.
    """

    def __init__(self, ema_alpha: float = 0.05) -> None:
        self._alpha = ema_alpha
        self._prev_price: float | None = None
        self._prev_ret: float | None = None
        self._angular_velocity: float = 0.0
        self._geodesic_deviation: float = 0.0
        self.omega_ema: float | None = None
        self.geo_ema: float | None = None

    def update(self, price: float) -> tuple[float, float]:
        """Return (angular_velocity, geodesic_deviation)."""
        if self._prev_price is None or self._prev_price <= 0:
            self._prev_price = price
            return 0.0, 0.0

        ret = math.log(price / self._prev_price + 1e-12)
        self._prev_price = price

        # Angular velocity = rate of change of return direction
        if self._prev_ret is not None:
            delta = ret - self._prev_ret
            self._angular_velocity = abs(delta)
        self._prev_ret = ret

        # Geodesic deviation = cumulative absolute return deviation from EMA
        ret_ema = _ema(getattr(self, "_ret_ema", None), ret, 0.1)
        self._ret_ema = ret_ema
        self._geodesic_deviation = abs(ret - ret_ema)

        # Update EMA baselines
        if self.omega_ema is None:
            self.omega_ema = self._angular_velocity + 1e-9
            self.geo_ema = self._geodesic_deviation + 1e-9
        else:
            self.omega_ema = _ema(self.omega_ema, self._angular_velocity, self._alpha)
            self.geo_ema = _ema(self.geo_ema, self._geodesic_deviation, self._alpha)

        return self._angular_velocity, self._geodesic_deviation


class CFCrossDetector:
    """
    CF cross detection -- EMA momentum crossover using per-symbol cf thresholds.
    Fires when price momentum exceeds the cf threshold for the timeframe.
    """

    def __init__(self, cf: float) -> None:
        self.cf = cf
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self.cross_up: bool = False
        self.cross_down: bool = False

    def update(self, price: float) -> None:
        fast_a = _alpha(8)
        slow_a = _alpha(21)
        prev_fast = self._ema_fast
        prev_slow = self._ema_slow
        self._ema_fast = _ema(self._ema_fast, price, fast_a)
        self._ema_slow = _ema(self._ema_slow, price, slow_a)
        if prev_fast is None or prev_slow is None:
            self.cross_up = False
            self.cross_down = False
            return
        # Cross detection: fast crosses slow with momentum > cf threshold
        momentum = abs(self._ema_fast - self._ema_slow) / (self._ema_slow + 1e-12)
        self.cross_up = (prev_fast <= prev_slow and self._ema_fast > self._ema_slow
                         and momentum > self.cf)
        self.cross_down = (prev_fast >= prev_slow and self._ema_fast < self._ema_slow
                           and momentum > self.cf)

    @property
    def momentum(self) -> float:
        if self._ema_fast is None or self._ema_slow is None:
            return 0.0
        return (self._ema_fast - self._ema_slow) / (self._ema_slow + 1e-12)


class OUDetector:
    """Ornstein-Uhlenbeck mean-reversion detector."""

    def __init__(self, window: int = 50, entry_z: float = 1.5,
                 exit_z: float = 0.05) -> None:
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._prices: deque[float] = deque(maxlen=window)
        self.mean = self.std = self.zscore = self.half_life = None

    def update(self, price: float) -> None:
        self._prices.append(math.log(price + 1e-12))
        if len(self._prices) < 20:
            return
        px = np.array(self._prices)
        self.mean = float(np.mean(px))
        self.std = float(np.std(px)) + 1e-9
        self.zscore = (px[-1] - self.mean) / self.std
        y, x = px[1:], px[:-1]
        if len(x) > 5:
            rho = float(np.corrcoef(x, y)[0, 1])
            rho = max(-0.9999, min(0.9999, rho))
            self.half_life = (-math.log(2) / math.log(abs(rho) + 1e-12)
                              if rho < 1 else 999)

    @property
    def long_signal(self) -> bool:
        return (self.zscore is not None and self.zscore < -self.entry_z
                and self.half_life is not None and 2 < self.half_life < 120)

    @property
    def short_signal(self) -> bool:
        return (self.zscore is not None and self.zscore > self.entry_z
                and self.half_life is not None and 2 < self.half_life < 120)

    @property
    def exit_signal(self) -> bool:
        return self.zscore is not None and abs(self.zscore) < self.exit_z


# =============================================================================
# WAVE-4 MODULES
# =============================================================================

class EventCalendarFilter:
    """Reduces sizing 50% in the +-2h window around high-impact macro events."""

    _WINDOW = timedelta(hours=2)
    _CAL_FILE = _REPO_ROOT / "config" / "event_calendar.json"

    def __init__(self) -> None:
        self._events: list[datetime] = self._build_events()

    def _build_events(self) -> list[datetime]:
        events: list[datetime] = []
        for year in range(2020, 2030):
            for month in [3, 5, 6, 7, 9, 11, 12, 1]:
                try:
                    events.append(datetime(year, month, 15, 18, 0, tzinfo=timezone.utc))
                except ValueError:
                    pass
        try:
            if self._CAL_FILE.exists():
                data = json.loads(self._CAL_FILE.read_text())
                for entry in data.get("events", []):
                    dt = datetime.fromisoformat(entry["time"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    events.append(dt)
        except Exception:
            pass
        return events

    def position_multiplier(self, bar_time: datetime) -> float:
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        for ev in self._events:
            if abs(bar_time - ev) <= self._WINDOW:
                return 0.5
        return 1.0


class NetworkSignalTracker:
    """Rolling 30-day Granger causality proxy (BTC correlation)."""

    WINDOW = 30
    CORR_THRESH = 0.30
    BOOST = 1.20

    def __init__(self, syms: list[str]) -> None:
        self._syms = syms
        self._btc_rets: deque[float] = deque(maxlen=self.WINDOW + 1)
        self._alt_rets: dict[str, deque[float]] = {
            s: deque(maxlen=self.WINDOW) for s in syms if s != "BTC"
        }
        self._granger_active: set[str] = set()

    def update_daily(self, daily_rets: dict[str, float]) -> None:
        btc_ret = daily_rets.get("BTC")
        if btc_ret is None:
            return
        self._btc_rets.append(btc_ret)
        for s, q in self._alt_rets.items():
            if s in daily_rets:
                q.append(daily_rets[s])
        if len(self._btc_rets) < self.WINDOW + 1:
            return
        btc_arr = np.array(list(self._btc_rets))[-self.WINDOW:]
        self._granger_active.clear()
        for s, q in self._alt_rets.items():
            if len(q) < self.WINDOW:
                continue
            alt_arr = np.array(list(q))
            n = min(len(btc_arr), len(alt_arr))
            if n < 10:
                continue
            b, a = btc_arr[-n:], alt_arr[-n:]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            try:
                corr = float(np.corrcoef(b, a)[0, 1])
            except Exception:
                corr = 0.0
            if abs(corr) > self.CORR_THRESH:
                self._granger_active.add(s)

    def boost_multiplier(self, sym: str, btc_bh_active: bool) -> float:
        if sym == "BTC":
            return 1.0
        if btc_bh_active and sym in self._granger_active:
            return self.BOOST
        return 1.0


class _SGDLogistic:
    """Online logistic regressor with SGD + L2 regularization."""

    def __init__(self, n_feat: int = 6, alpha: float = 0.01, lam: float = 1e-4) -> None:
        self.w = np.zeros(n_feat)
        self.b = 0.0
        self.alpha = alpha
        self.lam = lam

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def train_one(self, feats: list[float], label: float) -> None:
        x = np.array(feats[-6:] if len(feats) >= 6 else feats + [0.0] * (6 - len(feats)))
        z = float(self.w @ x) + self.b
        p = self._sigmoid(z)
        err = p - label
        self.w = self.w * (1.0 - self.alpha * self.lam) - self.alpha * err * x
        self.b -= self.alpha * err

    def predict(self, feats: list[float]) -> float:
        x = np.array(feats[-6:] if len(feats) >= 6 else feats + [0.0] * (6 - len(feats)))
        z = float(self.w @ x) + self.b
        return math.tanh(z)


class MLSignalModule:
    """Per-instrument online logistic regression signal. 30-bar warmup."""

    _N_WARMUP = 30

    def __init__(self) -> None:
        self._models: dict[str, _SGDLogistic] = {}
        self._counts: dict[str, int] = {}
        self._rets: dict[str, deque[float]] = {}

    def _ensure(self, sym: str) -> None:
        if sym not in self._models:
            self._models[sym] = _SGDLogistic()
            self._counts[sym] = 0
            self._rets[sym] = deque(maxlen=10)

    def update_daily(self, sym: str, daily_ret: float, garch_vol: float) -> None:
        self._ensure(sym)
        rets = self._rets[sym]
        rets.append(daily_ret)
        if len(rets) < 6:
            return
        feats = list(rets)[-5:] + [garch_vol]
        label = 1.0 if daily_ret > 0 else 0.0
        self._models[sym].train_one(feats, label)
        self._counts[sym] += 1

    def predict(self, sym: str, recent_rets: list[float], garch_vol: float) -> float:
        self._ensure(sym)
        if self._counts.get(sym, 0) < self._N_WARMUP:
            return 0.0
        feats = list(recent_rets)[-5:] + [garch_vol]
        return self._models[sym].predict(feats)


class RLExitPolicy:
    """
    Exit decision policy. Loads config/rl_exit_qtable.json if available,
    else uses a heuristic approximating the trained policy.
    """

    _TABLE_PATH = _REPO_ROOT / "config" / "rl_exit_qtable.json"
    _N_BINS = 5
    _STOP_LOSS = -0.03

    def __init__(self) -> None:
        self._qtable: dict[str, list[float]] | None = None
        self._load_table()

    def _load_table(self) -> None:
        try:
            if self._TABLE_PATH.exists():
                data = json.loads(self._TABLE_PATH.read_text())
                self._qtable = data
                log.info("RLExitPolicy: loaded Q-table with %d states", len(data))
        except Exception as exc:
            log.debug("RLExitPolicy: Q-table load failed (%s), using heuristic", exc)

    def _discretize(self, v: float, lo: float = -1.0, hi: float = 1.0) -> int:
        clipped = max(lo, min(hi, v))
        idx = int((clipped - lo) / (hi - lo) * self._N_BINS)
        return min(self._N_BINS - 1, max(0, idx))

    def _state_key(self, pnl_pct: float, bars_held: int, bh_mass: float,
                   bh_active: bool, atr_ratio: float) -> str:
        f0 = self._discretize(pnl_pct * 2.0)
        f1 = self._discretize(bars_held / 50.0 - 1.0)
        f2 = self._discretize(bh_mass * 2.0 - 1.0)
        f3 = 1 if bh_active else 0
        f4 = self._discretize(atr_ratio - 1.0)
        return f"{f0},{f1},{f2},{f3},{f4}"

    def should_exit(self, pnl_pct: float, bars_held: int, bh_mass: float,
                    bh_active: bool, atr_ratio: float = 1.0) -> bool:
        if pnl_pct < self._STOP_LOSS:
            return True
        if self._qtable is not None:
            key = self._state_key(pnl_pct, bars_held, bh_mass, bh_active, atr_ratio)
            qs = self._qtable.get(key)
            if qs is not None and len(qs) == 2:
                return float(qs[1]) > float(qs[0])
        # Heuristic fallback
        if bh_active and pnl_pct > 0.005:
            return False
        if not bh_active and bars_held > 16:
            return True
        if pnl_pct < -0.015:
            return True
        return False


# =============================================================================
# PER-SYMBOL BACKTEST STATE
# =============================================================================

@dataclass
class SymbolState:
    """All per-symbol state for the backtester."""
    sym: str
    asset_class: str

    # BH per timeframe
    bh_15m: BHPhysicsEngine = field(default=None)
    bh_1h: BHPhysicsEngine = field(default=None)
    bh_4h: BHPhysicsEngine = field(default=None)

    # CF cross detectors
    cf_15m: CFCrossDetector = field(default=None)
    cf_1h: CFCrossDetector = field(default=None)
    cf_4h: CFCrossDetector = field(default=None)

    # Other indicators
    atr_1h: ATRTracker = field(default=None)
    atr_4h: ATRTracker = field(default=None)
    garch: GARCHTracker = field(default=None)
    ou: OUDetector = field(default=None)
    hurst: HurstEstimator = field(default=None)
    nav: QuatNavState = field(default=None)

    # Buffers for 1h/4h aggregation
    h1_buf: list = field(default_factory=list)
    h4_buf: list = field(default_factory=list)
    last_h1_bucket: Any = field(default=None)
    last_h4_bucket: Any = field(default=None)

    # Position tracking
    last_frac: float = 0.0
    entry_time: Any = None
    entry_px: float | None = None
    bars_held: int = 0
    ou_pos: float = 0.0
    pos_floor: float = 0.0
    prev_close: float | None = None
    bar_count: int = 0
    warmup_done: bool = False

    # Daily tracking
    prev_daily_close: float | None = None
    daily_returns: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        pass  # fields already set by caller


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class LARSAv18Strategy:
    """
    LARSA v18 strategy -- processes bars chronologically and returns
    target portfolio fractions per symbol.
    """

    OU_DISABLED_SYMS = {"AVAX", "DOT", "LINK"}

    def __init__(self, cfg: LARSAv18Config | None = None) -> None:
        self.cfg = cfg or LARSAv18Config()
        self._states: dict[str, SymbolState] = {}
        self._dynamic_corr: float = self.cfg.CORR_NORMAL
        self._btc_e200: float | None = None
        self._daily_returns: dict[str, deque] = {}

        # Wave-4 modules
        syms = list(self.cfg.INSTRUMENTS.keys())
        self._event_cal = EventCalendarFilter() if self.cfg.USE_EVENT_CAL else None
        self._granger = (NetworkSignalTracker(syms)
                         if self.cfg.USE_GRANGER else None)
        self._ml_module = MLSignalModule() if self.cfg.USE_ML else None
        self._rl_policy = (RLExitPolicy()
                           if (self.cfg.USE_RL and self.cfg.RL_EXIT_ACTIVE) else None)

        self._init_states()

    def _init_states(self) -> None:
        for sym, inst_cfg in self.cfg.INSTRUMENTS.items():
            asset_class = inst_cfg.get("asset_class", "crypto")
            st = SymbolState(sym=sym, asset_class=asset_class)
            st.bh_15m = BHPhysicsEngine(inst_cfg["cf_15m"], self.cfg)
            st.bh_1h = BHPhysicsEngine(inst_cfg["cf_1h"], self.cfg)
            st.bh_4h = BHPhysicsEngine(inst_cfg["cf_4h"], self.cfg)
            st.cf_15m = CFCrossDetector(inst_cfg["cf_15m"])
            st.cf_1h = CFCrossDetector(inst_cfg["cf_1h"])
            st.cf_4h = CFCrossDetector(inst_cfg["cf_4h"])
            st.atr_1h = ATRTracker()
            st.atr_4h = ATRTracker()
            st.garch = GARCHTracker(
                omega=self.cfg.GARCH_OMEGA,
                alpha=self.cfg.GARCH_ALPHA,
                beta=self.cfg.GARCH_BETA,
                warmup=self.cfg.GARCH_WARMUP,
                target_vol=self.cfg.GARCH_TARGET_VOL,
            )
            st.ou = OUDetector()
            st.hurst = HurstEstimator(
                window=self.cfg.HURST_WINDOW,
                min_bars=self.cfg.HURST_MIN_BARS,
            )
            st.nav = QuatNavState(ema_alpha=self.cfg.NAV_EMA_ALPHA)
            self._states[sym] = st
            self._daily_returns[sym] = deque(maxlen=30)

    def reset(self) -> None:
        """Reset all state -- used between backtest runs."""
        self._dynamic_corr = self.cfg.CORR_NORMAL
        self._btc_e200 = None
        for sym in self._states:
            self._daily_returns[sym].clear()
        self._init_states()

    def on_bar(self, bar: dict, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Process one 15m bar for all symbols present in bar.
        bar: {sym: {"open", "high", "low", "close", "volume", "timestamp"}}
        history: {sym: pd.DataFrame of recent bars}
        Returns {sym: target_fraction}
        """
        ts: datetime = bar.get("timestamp", datetime.now(timezone.utc))
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Process each symbol's bar
        for sym, st in self._states.items():
            if sym not in bar:
                continue
            b = bar[sym]
            o = float(b.get("open", b.get("close", 0)))
            h = float(b.get("high", b.get("close", 0)))
            l = float(b.get("low", b.get("close", 0)))
            c = float(b["close"])
            v = float(b.get("volume", 1.0))

            st.bar_count += 1

            # 15m BH + CF cross
            st.bh_15m.update(c, v)
            st.cf_15m.update(c)

            # QuatNav
            if self.cfg.USE_QUATNAV:
                st.nav.update(c)

            # Hurst
            if self.cfg.USE_HURST:
                st.hurst.update(c)

            # GARCH
            if st.prev_close and st.prev_close > 0:
                st.garch.update(math.log(c / st.prev_close))
            st.prev_close = c

            # OU
            st.ou.update(c)

            # Aggregate 1h
            h1_bucket = ts.replace(minute=0, second=0, microsecond=0)
            st.h1_buf.append({"o": o, "h": h, "l": l, "c": c, "ts": ts})
            if st.last_h1_bucket != h1_bucket:
                if st.last_h1_bucket is not None and len(st.h1_buf) >= 2:
                    self._flush_1h(sym)
                st.last_h1_bucket = h1_bucket
                st.h1_buf = [{"o": o, "h": h, "l": l, "c": c, "ts": ts}]

            # Aggregate 4h
            h4_bucket = ts.replace(
                hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0
            )
            st.h4_buf.append({"o": o, "h": h, "l": l, "c": c, "ts": ts})
            if st.last_h4_bucket != h4_bucket:
                if st.last_h4_bucket is not None and len(st.h4_buf) >= 8:
                    self._flush_4h(sym)
                st.last_h4_bucket = h4_bucket
                st.h4_buf = [{"o": o, "h": h, "l": l, "c": c, "ts": ts}]

            # Warmup gate
            if st.bar_count >= self.cfg.WARMUP_BARS:
                st.warmup_done = True

            # Daily rollover
            if ts.hour == 0 and ts.minute == 0:
                self._on_daily_close(sym, c)

        # Compute and return targets
        return self.compute_targets(ts, bar)

    def _flush_1h(self, sym: str) -> None:
        st = self._states[sym]
        buf = st.h1_buf
        if not buf:
            return
        h1h = max(b["h"] for b in buf)
        h1l = min(b["l"] for b in buf)
        h1c = buf[-1]["c"]
        st.bh_1h.update(h1c)
        st.cf_1h.update(h1c)
        st.atr_1h.update(h1h, h1l, h1c)

    def _flush_4h(self, sym: str) -> None:
        st = self._states[sym]
        buf = st.h4_buf
        if not buf:
            return
        h4h = max(b["h"] for b in buf)
        h4l = min(b["l"] for b in buf)
        h4c = buf[-1]["c"]
        st.bh_4h.update(h4c)
        st.cf_4h.update(h4c)
        st.atr_4h.update(h4h, h4l, h4c)

    def _on_daily_close(self, sym: str, close: float) -> None:
        st = self._states[sym]
        if st.prev_daily_close and st.prev_daily_close > 0:
            daily_ret = math.log(close / st.prev_daily_close)
            self._daily_returns[sym].append(daily_ret)
            if sym == "BTC":
                self._btc_e200 = _ema(self._btc_e200, close, _alpha(200))
            if self._granger:
                self._granger.update_daily({sym: daily_ret})
            if self._ml_module:
                garch_vol = st.garch.vol or 0.5
                self._ml_module.update_daily(sym, daily_ret, garch_vol)
        st.prev_daily_close = close
        self._recompute_dynamic_corr()

    def _recompute_dynamic_corr(self) -> None:
        active = [list(self._daily_returns[s]) for s in self._states
                  if len(self._daily_returns[s]) >= 30]
        if len(active) < 2:
            return
        mat = np.array(active)
        corr = np.corrcoef(mat)
        n = corr.shape[0]
        avg = (np.sum(corr) - n) / (n * (n - 1)) if n > 1 else 0.0
        self._dynamic_corr = (self.cfg.CORR_STRESS
                              if avg > self.cfg.CORR_STRESS_THRESHOLD
                              else self.cfg.CORR_NORMAL)

    def compute_targets(self, bar_time: datetime,
                        current_bar: dict | None = None) -> dict[str, float]:
        """Compute target fractions for all symbols. Full LARSA v18 logic."""
        n_eff = len(self._states)
        corr_factor = math.sqrt(
            n_eff + n_eff * (n_eff - 1) * self._dynamic_corr
        )
        per_inst_risk = self.cfg.DAILY_RISK / corr_factor

        bar_hour = bar_time.hour
        blocked = bar_hour in self.cfg.BLOCKED_ENTRY_HOURS_UTC
        boosted = bar_hour in self.cfg.BOOST_ENTRY_HOURS_UTC

        # BTC lead
        btc = self._states.get("BTC")
        btc_lead = (btc is not None
                    and btc.bh_4h.active
                    and btc.bh_1h.active)
        btc_bh = (btc is not None
                  and (btc.bh_1h.active or btc.bh_4h.active))

        raw: dict[str, float] = {}

        for sym, st in self._states.items():
            if not st.warmup_done:
                raw[sym] = 0.0
                continue

            d_active = st.bh_4h.active
            h_active = st.bh_1h.active
            m_active = st.bh_15m.active
            tf = ((4 if d_active else 0)
                  + (2 if h_active else 0)
                  + (1 if m_active else 0))

            if st.asset_class == "equity":
                ceiling = min(self.cfg.TF_CAP.get(tf, 0.0), self.cfg.EQUITY_CAP_FRAC)
            else:
                ceiling = min(self.cfg.TF_CAP.get(tf, 0.0), self.cfg.CRYPTO_CAP_FRAC)

            if ceiling == 0.0:
                raw[sym] = 0.0
                continue

            # Direction
            direction = 0
            if h_active and st.bh_1h.bh_dir:
                direction = st.bh_1h.bh_dir
            elif d_active and st.bh_4h.bh_dir:
                direction = st.bh_4h.bh_dir

            if st.asset_class == "crypto" and direction <= 0:
                raw[sym] = 0.0
                continue
            if direction == 0:
                raw[sym] = 0.0
                continue

            # Vol-adjusted base size
            atr = st.atr_1h.atr or st.atr_4h.atr
            cp = st.prev_close or 1.0
            vol = (atr / cp * math.sqrt(6.5)) if (atr and cp > 0) else 0.01
            base = min(per_inst_risk / (vol + 1e-9),
                       min(ceiling, self.cfg.DELTA_MAX_FRAC))
            raw[sym] = base * st.garch.vol_scale * direction

            # QuatNav angular velocity sizing
            if self.cfg.USE_QUATNAV and st.nav.omega_ema and st.nav.omega_ema > 1e-9:
                omega_ratio = st.nav._angular_velocity / st.nav.omega_ema
                nav_scale = 1.0 / (1.0 + self.cfg.NAV_OMEGA_SCALE_K
                                   * max(0.0, omega_ratio - 1.0))
                raw[sym] *= nav_scale

            # QuatNav geodesic entry gate
            is_new_entry = math.isclose(st.last_frac, 0.0)
            if (self.cfg.USE_QUATNAV and is_new_entry
                    and st.nav.geo_ema and st.nav.geo_ema > 1e-9):
                geo_ratio = st.nav._geodesic_deviation / st.nav.geo_ema
                if geo_ratio > self.cfg.NAV_GEO_ENTRY_GATE:
                    raw[sym] = 0.0
                    continue

            # Hurst regime modifier
            if (self.cfg.USE_HURST
                    and st.hurst.is_mean_reverting
                    and st.asset_class == "crypto"):
                raw[sym] *= 0.6

        # Mayer Multiple dampener -- crypto only
        mayer_damp = 1.0
        btc_px = (self._states["BTC"].prev_close
                  if "BTC" in self._states else 0.0) or 0.0
        if self._btc_e200 and self._btc_e200 > 0 and btc_px > 0:
            mayer = btc_px / self._btc_e200
            if mayer > 2.4:
                mayer_damp = max(0.5, 1.0 - (mayer - 2.4) / 2.2)
            elif mayer < 1.0:
                mayer_damp = min(1.2, 1.0 + (1.0 - mayer) * 0.3)
        for sym in raw:
            if self.cfg.INSTRUMENTS[sym].get("asset_class", "crypto") == "crypto":
                raw[sym] = raw.get(sym, 0.0) * mayer_damp

        # BTC cross-asset lead: 1.4x boost on other crypto
        for sym in raw:
            if sym == "BTC":
                continue
            if self.cfg.INSTRUMENTS[sym].get("asset_class", "crypto") != "crypto":
                continue
            if btc_lead and raw.get(sym, 0.0) > 0:
                raw[sym] *= 1.4

        # Granger network boost
        if self.cfg.USE_GRANGER and self._granger:
            for sym in raw:
                if self.cfg.INSTRUMENTS[sym].get("asset_class", "crypto") != "crypto":
                    continue
                if raw.get(sym, 0.0) != 0.0:
                    raw[sym] *= self._granger.boost_multiplier(sym, btc_bh)

        # ML signal modifier
        if self.cfg.USE_ML and self._ml_module:
            for sym, st in self._states.items():
                if raw.get(sym, 0.0) == 0.0:
                    continue
                try:
                    garch_vol = st.garch.vol or 0.5
                    ml_sig = self._ml_module.predict(
                        sym, list(self._daily_returns.get(sym, [])), garch_vol
                    )
                    if ml_sig > self.cfg.ML_SIGNAL_BOOST_THRESH:
                        raw[sym] *= self.cfg.ML_SIGNAL_BOOST
                    elif (ml_sig < self.cfg.ML_SIGNAL_SUPPRESS_THRESH
                          and math.isclose(st.last_frac, 0.0)):
                        raw[sym] = 0.0
                except Exception:
                    pass

        # Event calendar filter
        if self.cfg.USE_EVENT_CAL and self._event_cal:
            cal_mult = self._event_cal.position_multiplier(bar_time)
            if cal_mult < 1.0:
                for sym in raw:
                    raw[sym] *= cal_mult

        # Hour boost -- new entries only
        if boosted:
            for sym, st in self._states.items():
                if raw.get(sym, 0.0) > 0 and math.isclose(st.last_frac, 0.0):
                    raw[sym] *= self.cfg.HOUR_BOOST_MULTIPLIER

        # Blocked hours -- suppress new entries
        if blocked:
            for sym, st in self._states.items():
                if math.isclose(st.last_frac, 0.0):
                    raw[sym] = 0.0

        # OU mean reversion overlay
        for sym, st in self._states.items():
            if not st.warmup_done:
                continue
            if st.bh_4h.active or st.bh_1h.active:
                continue
            if sym in self.OU_DISABLED_SYMS:
                continue
            if st.ou.long_signal and st.ou_pos <= 0:
                raw[sym] = raw.get(sym, 0.0) + self.cfg.OU_FRAC
                st.ou_pos = self.cfg.OU_FRAC
            elif st.ou.exit_signal and st.ou_pos > 0:
                st.ou_pos = 0.0
            elif st.ou.short_signal:
                st.ou_pos = 0.0

        # Position floor
        for sym, st in self._states.items():
            tgt = raw.get(sym, 0.0)
            d_active = st.bh_4h.active
            h_active = st.bh_1h.active
            m_active = st.bh_15m.active
            tf = ((4 if d_active else 0)
                  + (2 if h_active else 0)
                  + (1 if m_active else 0))
            if tf >= 6 and abs(tgt) > 0.15 and st.bh_1h.ctl >= 5:
                st.pos_floor = max(st.pos_floor, 0.70 * abs(tgt))
            if st.pos_floor > 0 and tf >= 4 and not math.isclose(st.last_frac, 0.0):
                raw[sym] = math.copysign(
                    max(abs(tgt), st.pos_floor), st.last_frac
                )
                st.pos_floor *= 0.95
            if tf < 4 or math.isclose(tgt, 0.0):
                st.pos_floor = 0.0
            if not d_active and not h_active:
                st.pos_floor = 0.0

        # Normalize if sum > 1
        total = sum(abs(v) for v in raw.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        for sym in raw:
            raw[sym] *= scale

        return raw

    def rl_exit_check(self, sym: str, pnl_pct: float, bars_held: int) -> bool:
        """Check RL exit policy for an open position."""
        if not self.cfg.USE_RL or self._rl_policy is None:
            return False
        st = self._states.get(sym)
        if st is None:
            return False
        bh_mass = max(st.bh_15m.mass, st.bh_1h.mass, st.bh_4h.mass)
        bh_active = st.bh_15m.active or st.bh_1h.active or st.bh_4h.active
        atr = st.atr_1h.atr
        cp = st.prev_close or 1.0
        atr_ratio = (atr / cp) / 0.01 if (atr and cp > 0) else 1.0
        return self._rl_policy.should_exit(
            pnl_pct, bars_held, bh_mass, bh_active, atr_ratio
        )

    def update_last_frac(self, sym: str, frac: float) -> None:
        if sym in self._states:
            self._states[sym].last_frac = frac


# =============================================================================
# TRADE / RESULT DATACLASSES
# =============================================================================

@dataclass
class Trade:
    sym: str
    side: str                 # "buy" | "sell"
    frac: float               # portfolio fraction
    price: float
    timestamp: datetime
    cost_bps: float = 0.0
    pnl: float = 0.0          # realised P&L (populated on exit)
    entry_time: datetime | None = None
    bars_held: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_list: list[Trade]
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    total_return: float
    win_rate: float
    per_symbol_pnl: dict[str, float]
    n_trades: int
    avg_hold_bars: float


def _compute_metrics(equity: pd.Series) -> dict[str, float]:
    """Compute performance metrics from an equity curve."""
    if len(equity) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "total_return": 0.0}

    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # Sharpe (annualised, 15m bars -- ~26280 per year)
    bars_per_year = 26280.0
    ann_factor = math.sqrt(bars_per_year)
    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    sharpe = (mean_ret / (std_ret + 1e-12)) * ann_factor

    # Sortino
    down_rets = rets[rets < 0]
    down_std = float(down_rets.std()) if len(down_rets) > 1 else 1e-12
    sortino = (mean_ret / (down_std + 1e-12)) * ann_factor

    # Max drawdown
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min())

    # Calmar
    n_years = len(equity) / bars_per_year
    ann_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.001)) - 1.0
    calmar = ann_return / (abs(max_dd) + 1e-12)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_return": total_return,
    }


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_data(
    symbols: list[str],
    n_bars: int = 5000,
    start: datetime | None = None,
    seed: int = 42,
    freq_minutes: int = 15,
) -> dict[str, pd.DataFrame]:
    """
    Generate GBM synthetic OHLCV bars for testing.
    Returns {sym: DataFrame[open, high, low, close, volume, timestamp]}.
    """
    rng = np.random.default_rng(seed)
    if start is None:
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)

    params: dict[str, tuple[float, float, float]] = {
        "BTC":  (50000.0, 0.0005, 0.015),
        "ETH":  (3000.0,  0.0004, 0.018),
        "XRP":  (0.50,    0.0003, 0.022),
        "AVAX": (20.0,    0.0003, 0.020),
        "LINK": (15.0,    0.0003, 0.019),
        "SPY":  (450.0,   0.00015, 0.005),
        "QQQ":  (380.0,   0.00015, 0.006),
        "GLD":  (180.0,   0.00010, 0.004),
        "NVDA": (600.0,   0.0003,  0.015),
        "AAPL": (180.0,   0.00015, 0.008),
        "TSLA": (250.0,   0.0003,  0.018),
    }

    data: dict[str, pd.DataFrame] = {}
    freq = timedelta(minutes=freq_minutes)

    for sym in symbols:
        s0, mu, sigma = params.get(sym, (100.0, 0.0002, 0.010))
        logrets = rng.normal(mu, sigma, n_bars)
        closes = s0 * np.exp(np.cumsum(logrets))

        # Generate OHLV from closes
        noise = rng.uniform(0.001, 0.005, n_bars)
        highs = closes * (1.0 + noise)
        lows = closes * (1.0 - noise)
        opens = np.roll(closes, 1)
        opens[0] = s0
        volumes = rng.uniform(1e6, 1e7, n_bars)

        timestamps = [start + freq * i for i in range(n_bars)]

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "timestamp": timestamps,
        })
        df.set_index("timestamp", inplace=True)
        data[sym] = df

    return data


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class LARSAv18Backtest:
    """
    Production backtesting engine for LARSA v18.
    Processes bars chronologically, tracks equity, positions, and P&L.
    """

    def __init__(self) -> None:
        self._db_path = _REPO_ROOT / "execution" / "live_trades.db"

    def load_data(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        source: str = "auto",
        data_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for symbols between start and end.
        source: "sqlite" | "csv" | "synthetic" | "auto"
        Falls back to synthetic if no real data is available.
        """
        if source == "synthetic":
            n_bars = int((end - start).total_seconds() / 900)
            return generate_synthetic_data(symbols, n_bars=n_bars, start=start)

        if source in ("sqlite", "auto"):
            try:
                data = self._load_from_sqlite(symbols, start, end)
                if data and all(len(df) > 0 for df in data.values()):
                    return data
            except Exception as exc:
                log.debug("SQLite load failed: %s", exc)

        if source in ("csv", "auto") and data_dir:
            try:
                data = self._load_from_csv(symbols, start, end, data_dir)
                if data and all(len(df) > 0 for df in data.values()):
                    return data
            except Exception as exc:
                log.debug("CSV load failed: %s", exc)

        log.warning("No real data found -- falling back to synthetic GBM data")
        n_bars = int((end - start).total_seconds() / 900)
        return generate_synthetic_data(symbols, n_bars=n_bars, start=start)

    def _load_from_sqlite(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]:
        if not self._db_path.exists():
            return {}
        conn = sqlite3.connect(str(self._db_path))
        data: dict[str, pd.DataFrame] = {}
        try:
            for sym in symbols:
                query = """
                    SELECT fill_time AS timestamp,
                           price AS close,
                           price AS open, price AS high, price AS low,
                           qty AS volume
                    FROM live_trades
                    WHERE symbol = ?
                      AND fill_time >= ?
                      AND fill_time <= ?
                    ORDER BY fill_time
                """
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(sym, start.isoformat(), end.isoformat()),
                    parse_dates=["timestamp"],
                )
                if len(df) > 0:
                    df.set_index("timestamp", inplace=True)
                    data[sym] = df
        finally:
            conn.close()
        return data

    def _load_from_csv(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        data_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            for pattern in [f"{sym}.csv", f"{sym}_15m.csv",
                            f"{sym.lower()}.csv", f"{sym.lower()}_15m.csv"]:
                fpath = data_dir / pattern
                if fpath.exists():
                    df = pd.read_csv(fpath, parse_dates=["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    df.index = pd.to_datetime(df.index, utc=True)
                    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                    df = df.loc[mask]
                    if len(df) > 0:
                        data[sym] = df
                    break
        return data

    def run(
        self,
        data: dict[str, pd.DataFrame],
        config: LARSAv18Config | None = None,
        initial_equity: float = 100_000.0,
    ) -> BacktestResult:
        """
        Run the backtest.
        data: {sym: DataFrame with columns [open, high, low, close, volume]}
              index must be DatetimeIndex (UTC)
        """
        cfg = config or LARSAv18Config()
        strategy = LARSAv18Strategy(cfg)
        symbols = list(data.keys())

        # Align all bars to a common timeline
        all_timestamps: list[pd.Timestamp] = sorted(
            set().union(*[df.index.tolist() for df in data.values()])
        )

        equity = initial_equity
        equity_peak = equity
        dd_halted = False

        equity_curve: list[tuple[pd.Timestamp, float]] = []
        trade_list: list[Trade] = []
        per_symbol_pnl: dict[str, float] = {s: 0.0 for s in symbols}

        # Position state: sym -> (frac, entry_price, entry_time, bars_held)
        positions: dict[str, tuple[float, float, pd.Timestamp, int]] = {}

        last_bar_ts: pd.Timestamp | None = None

        for ts in all_timestamps:
            # Build bar dict for this timestamp
            bar: dict[str, Any] = {"timestamp": ts.to_pydatetime()}
            for sym in symbols:
                if ts in data[sym].index:
                    row = data[sym].loc[ts]
                    bar[sym] = {
                        "open": float(row.get("open", row["close"])),
                        "high": float(row.get("high", row["close"])),
                        "low": float(row.get("low", row["close"])),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 1.0)),
                    }

            # Check drawdown circuit breaker
            dd = (equity - equity_peak) / equity_peak
            if dd < -cfg.DD_HALT_PCT:
                dd_halted = True
            if dd_halted and equity >= equity_peak * (1.0 - cfg.DD_RESUME_PCT):
                dd_halted = False

            # Get target fractions from strategy
            targets = strategy.on_bar(bar, data)

            # Process position changes
            for sym in symbols:
                if sym not in bar:
                    continue

                close_px = bar[sym]["close"]
                target_frac = targets.get(sym, 0.0)
                current_pos = positions.get(sym)

                # Min-hold enforcement
                if current_pos is not None:
                    frac, entry_px, entry_t, bars_h = current_pos
                    hold_mins = (ts - entry_t).total_seconds() / 60.0
                    pnl_pct = (close_px - entry_px) / (entry_px + 1e-12) * frac

                    # RL exit check
                    rl_exit = strategy.rl_exit_check(sym, pnl_pct, bars_h)

                    if (hold_mins < cfg.MIN_HOLD_MINUTES
                            and not rl_exit
                            and abs(target_frac) >= abs(frac) * 0.5):
                        # Enforce minimum hold -- don't reduce position
                        target_frac = frac

                # Drawdown halt -- no new entries
                if dd_halted and (current_pos is None or current_pos[0] == 0.0):
                    target_frac = 0.0

                current_frac = current_pos[0] if current_pos else 0.0
                delta = target_frac - current_frac

                if abs(delta) < cfg.MIN_TRADE_FRAC:
                    # Update bars held
                    if current_pos:
                        f, ep, et, bh = current_pos
                        positions[sym] = (f, ep, et, bh + 1)
                    continue

                # Determine transaction cost
                asset_class = cfg.INSTRUMENTS.get(sym, {}).get("asset_class", "crypto")
                cost_bps = (cfg.CRYPTO_COST_BPS if asset_class == "crypto"
                            else cfg.EQUITY_COST_BPS)
                cost = abs(delta) * equity * cost_bps / 10000.0

                side = "buy" if delta > 0 else "sell"

                if current_pos and not math.isclose(current_frac, 0.0):
                    # Closing or reducing -- book P&L
                    f, entry_px, entry_t, bars_h = current_pos
                    close_frac = min(abs(delta), abs(f))
                    direction = math.copysign(1.0, f)
                    pnl = direction * close_frac * equity * (close_px - entry_px) / (entry_px + 1e-12)
                    pnl -= cost
                    equity += pnl
                    per_symbol_pnl[sym] = per_symbol_pnl.get(sym, 0.0) + pnl

                    trade = Trade(
                        sym=sym, side=side, frac=abs(delta),
                        price=close_px, timestamp=ts.to_pydatetime(),
                        cost_bps=cost_bps, pnl=pnl,
                        entry_time=entry_t.to_pydatetime() if hasattr(entry_t, "to_pydatetime") else entry_t,
                        bars_held=bars_h,
                    )
                    trade_list.append(trade)

                    if math.isclose(target_frac, 0.0):
                        del positions[sym]
                    else:
                        # Partial close / direction change
                        positions[sym] = (target_frac, close_px, ts, 0)
                else:
                    # New entry
                    equity -= cost
                    per_symbol_pnl[sym] = per_symbol_pnl.get(sym, 0.0) - cost
                    positions[sym] = (target_frac, close_px, ts, 0)
                    trade_list.append(Trade(
                        sym=sym, side=side, frac=abs(target_frac),
                        price=close_px, timestamp=ts.to_pydatetime(),
                        cost_bps=cost_bps, pnl=0.0,
                    ))

                strategy.update_last_frac(sym, target_frac)

                # Update bars held for remaining positions
                for s in list(positions.keys()):
                    if s != sym:
                        f2, ep2, et2, bh2 = positions[s]
                        positions[s] = (f2, ep2, et2, bh2 + 1)

            # Mark-to-market open positions
            mtm = 0.0
            for sym, (frac, entry_px, _, _) in positions.items():
                if sym in bar:
                    px = bar[sym]["close"]
                    mtm += frac * equity * (px - entry_px) / (entry_px + 1e-12)

            equity_curve.append((ts, equity + mtm))
            equity_peak = max(equity_peak, equity + mtm)
            last_bar_ts = ts

        # Close all open positions at end
        if all_timestamps:
            final_ts = all_timestamps[-1]
            for sym, (frac, entry_px, entry_t, bars_h) in list(positions.items()):
                if sym in data and final_ts in data[sym].index:
                    exit_px = float(data[sym].loc[final_ts, "close"])
                    direction = math.copysign(1.0, frac)
                    pnl = direction * abs(frac) * equity * (exit_px - entry_px) / (entry_px + 1e-12)
                    equity += pnl
                    per_symbol_pnl[sym] = per_symbol_pnl.get(sym, 0.0) + pnl
                    trade_list.append(Trade(
                        sym=sym, side="sell", frac=abs(frac),
                        price=exit_px, timestamp=final_ts.to_pydatetime(),
                        cost_bps=0.0, pnl=pnl,
                        entry_time=(entry_t.to_pydatetime()
                                    if hasattr(entry_t, "to_pydatetime") else entry_t),
                        bars_held=bars_h,
                    ))

        # Build equity series
        if equity_curve:
            idx, vals = zip(*equity_curve)
            eq_series = pd.Series(vals, index=pd.DatetimeIndex(idx))
        else:
            eq_series = pd.Series([initial_equity], dtype=float)

        metrics = _compute_metrics(eq_series)

        # Win rate
        closed_trades = [t for t in trade_list if t.pnl != 0.0]
        win_rate = (sum(1 for t in closed_trades if t.pnl > 0) / len(closed_trades)
                    if closed_trades else 0.0)

        avg_hold = (sum(t.bars_held for t in closed_trades) / len(closed_trades)
                    if closed_trades else 0.0)

        return BacktestResult(
            equity_curve=eq_series,
            trade_list=trade_list,
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            max_drawdown=metrics["max_drawdown"],
            calmar=metrics["calmar"],
            total_return=metrics["total_return"],
            win_rate=win_rate,
            per_symbol_pnl=per_symbol_pnl,
            n_trades=len(closed_trades),
            avg_hold_bars=avg_hold,
        )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LARSA v18 Backtest")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTC", "ETH", "XRP", "SPY", "QQQ"])
    parser.add_argument("--source", default="auto",
                        choices=["auto", "sqlite", "csv", "synthetic"])
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    bt = LARSAv18Backtest()
    data = bt.load_data(
        args.symbols, start, end,
        source=args.source,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )

    log.info("Loaded data: %s", {s: len(df) for s, df in data.items()})

    cfg = LARSAv18Config()
    result = bt.run(data, cfg, initial_equity=args.equity)

    print("\n=== LARSA v18 Backtest Results ===")
    print(f"  Total Return : {result.total_return:.2%}")
    print(f"  Sharpe       : {result.sharpe:.3f}")
    print(f"  Sortino      : {result.sortino:.3f}")
    print(f"  Max Drawdown : {result.max_drawdown:.2%}")
    print(f"  Calmar       : {result.calmar:.3f}")
    print(f"  Win Rate     : {result.win_rate:.2%}")
    print(f"  N Trades     : {result.n_trades}")
    print(f"  Avg Hold (bars): {result.avg_hold_bars:.1f}")
    print("\n  Per-Symbol P&L:")
    for sym, pnl in sorted(result.per_symbol_pnl.items(), key=lambda x: -x[1]):
        print(f"    {sym:6s}: ${pnl:,.2f}")


if __name__ == "__main__":
    main()
