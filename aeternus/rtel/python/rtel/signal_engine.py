"""
AETERNUS Real-Time Execution Layer (RTEL)
signal_engine.py — Alpha signal generation and combination

Provides:
- Individual alpha signals (momentum, mean-reversion, LOB imbalance, vol surface)
- Signal combination via IC-weighted ensemble
- Online IC/ICIR tracking
- Signal decay and half-life estimation
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base signal class
# ---------------------------------------------------------------------------

class AlphaSignal(ABC):
    """Abstract base for alpha signals."""

    def __init__(self, name: str, lookback: int = 20):
        self.name     = name
        self.lookback = lookback
        self._history: Dict[int, deque] = {}  # asset_id -> price deque

    def _get_history(self, asset_id: int) -> deque:
        if asset_id not in self._history:
            self._history[asset_id] = deque(maxlen=self.lookback * 3)
        return self._history[asset_id]

    def update(self, asset_id: int, price: float) -> None:
        self._get_history(asset_id).append(price)

    @abstractmethod
    def compute(self, asset_id: int) -> Optional[float]:
        """Return signal in [-1, +1] or None if insufficient data."""
        ...

    def compute_all(self, assets: List[int]) -> Dict[int, float]:
        result = {}
        for aid in assets:
            s = self.compute(aid)
            if s is not None:
                result[aid] = s
        return result

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------

class MomentumSignal(AlphaSignal):
    """Cross-sectional momentum: rank by past N-period return."""

    def __init__(self, lookback: int = 20, skip_period: int = 1):
        super().__init__("momentum", lookback)
        self.skip_period = skip_period

    def compute(self, asset_id: int) -> Optional[float]:
        hist = self._get_history(asset_id)
        n    = len(hist)
        required = self.lookback + self.skip_period + 1
        if n < required:
            return None
        prices = list(hist)
        p_now  = prices[-1 - self.skip_period]
        p_then = prices[-required]
        if p_then <= 0:
            return None
        ret = (p_now - p_then) / p_then
        # Tanh squashing
        return float(math.tanh(ret * 10.0))


# ---------------------------------------------------------------------------
# Mean-reversion signal
# ---------------------------------------------------------------------------

class MeanReversionSignal(AlphaSignal):
    """Z-score based mean-reversion signal."""

    def __init__(self, lookback: int = 20, z_cap: float = 3.0):
        super().__init__("mean_reversion", lookback)
        self.z_cap = z_cap

    def compute(self, asset_id: int) -> Optional[float]:
        hist = list(self._get_history(asset_id))
        if len(hist) < self.lookback:
            return None
        window = np.array(hist[-self.lookback:])
        mean   = window.mean()
        std    = window.std()
        if std < 1e-12:
            return None
        z = (hist[-1] - mean) / std
        z = np.clip(z, -self.z_cap, self.z_cap) / self.z_cap
        return float(-z)  # negative z → buy signal


# ---------------------------------------------------------------------------
# LOB imbalance signal
# ---------------------------------------------------------------------------

class LOBImbalanceSignal(AlphaSignal):
    """Signal from order-book bid/ask quantity imbalance."""

    def __init__(self, ewma_alpha: float = 0.1):
        super().__init__("lob_imbalance", lookback=50)
        self.alpha = ewma_alpha
        self._ewma_imbal: Dict[int, float] = {}
        self._ewma_spread: Dict[int, float] = {}

    def update_lob(self, asset_id: int, bid_qty: float, ask_qty: float,
                   spread: float, mid_price: float) -> None:
        total = bid_qty + ask_qty
        imbal = (bid_qty - ask_qty) / (total + 1e-10)
        a = self.alpha
        self._ewma_imbal[asset_id]  = a*imbal  + (1-a)*self._ewma_imbal.get(asset_id, 0.0)
        self._ewma_spread[asset_id] = a*spread + (1-a)*self._ewma_spread.get(asset_id, spread)
        self.update(asset_id, mid_price)

    def compute(self, asset_id: int) -> Optional[float]:
        if asset_id not in self._ewma_imbal:
            return None
        return float(np.tanh(self._ewma_imbal[asset_id] * 2.0))


# ---------------------------------------------------------------------------
# Volatility surface signal
# ---------------------------------------------------------------------------

class VolSurfaceSignal(AlphaSignal):
    """Signal from implied volatility term structure and skew."""

    def __init__(self, lookback: int = 30):
        super().__init__("vol_surface", lookback)
        self._atm_vol_hist: Dict[int, deque] = {}
        self._skew_hist:    Dict[int, deque] = {}

    def update_vol(self, asset_id: int, atm_vol: float, skew: float,
                   mid_price: float) -> None:
        if asset_id not in self._atm_vol_hist:
            self._atm_vol_hist[asset_id] = deque(maxlen=self.lookback * 2)
            self._skew_hist[asset_id]    = deque(maxlen=self.lookback * 2)
        self._atm_vol_hist[asset_id].append(atm_vol)
        self._skew_hist[asset_id].append(skew)
        self.update(asset_id, mid_price)

    def compute(self, asset_id: int) -> Optional[float]:
        if asset_id not in self._atm_vol_hist:
            return None
        vols = list(self._atm_vol_hist[asset_id])
        if len(vols) < self.lookback:
            return None
        window = np.array(vols[-self.lookback:])
        # Vol contraction → positive signal (expect mean-reversion of price)
        vol_z = (vols[-1] - window.mean()) / (window.std() + 1e-10)
        # Skew: negative skew (puts expensive) → bearish
        skews = list(self._skew_hist[asset_id])
        skew_signal = float(-np.tanh(skews[-1] * 2.0)) if skews else 0.0
        # Combine
        combined = -0.5 * math.tanh(vol_z) + 0.5 * skew_signal
        return float(combined)


# ---------------------------------------------------------------------------
# Trend-following signal (EMA crossover)
# ---------------------------------------------------------------------------

class EMACrossoverSignal(AlphaSignal):
    """Fast/slow EMA crossover."""

    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__(f"ema_{fast}_{slow}", lookback=slow * 2)
        self.fast_alpha = 2.0 / (fast + 1)
        self.slow_alpha = 2.0 / (slow + 1)
        self._fast_ema: Dict[int, float] = {}
        self._slow_ema: Dict[int, float] = {}
        self._n:        Dict[int, int]   = {}

    def update(self, asset_id: int, price: float) -> None:
        super().update(asset_id, price)
        n = self._n.get(asset_id, 0)
        if n == 0:
            self._fast_ema[asset_id] = price
            self._slow_ema[asset_id] = price
        else:
            self._fast_ema[asset_id] = (self.fast_alpha * price +
                                        (1 - self.fast_alpha) * self._fast_ema[asset_id])
            self._slow_ema[asset_id] = (self.slow_alpha * price +
                                        (1 - self.slow_alpha) * self._slow_ema[asset_id])
        self._n[asset_id] = n + 1

    def compute(self, asset_id: int) -> Optional[float]:
        n = self._n.get(asset_id, 0)
        min_n = int(2.0 / self.slow_alpha)
        if n < min_n:
            return None
        fast = self._fast_ema[asset_id]
        slow = self._slow_ema[asset_id]
        if slow < 1e-12:
            return None
        diff = (fast - slow) / slow
        return float(math.tanh(diff * 50.0))


# ---------------------------------------------------------------------------
# Volume price trend signal
# ---------------------------------------------------------------------------

class VPTSignal(AlphaSignal):
    """Volume-Price Trend signal."""

    def __init__(self, lookback: int = 20):
        super().__init__("vpt", lookback)
        self._vpt: Dict[int, float]       = {}
        self._vpt_hist: Dict[int, deque] = {}

    def update_bar(self, asset_id: int, price: float, prev_price: float,
                   volume: float) -> None:
        if prev_price > 0:
            change = (price - prev_price) / prev_price
            vpt_delta = volume * change
        else:
            vpt_delta = 0.0
        self._vpt[asset_id] = self._vpt.get(asset_id, 0.0) + vpt_delta
        if asset_id not in self._vpt_hist:
            self._vpt_hist[asset_id] = deque(maxlen=self.lookback * 2)
        self._vpt_hist[asset_id].append(self._vpt[asset_id])
        self.update(asset_id, price)

    def compute(self, asset_id: int) -> Optional[float]:
        if asset_id not in self._vpt_hist:
            return None
        hist = list(self._vpt_hist[asset_id])
        if len(hist) < self.lookback:
            return None
        window = np.array(hist[-self.lookback:])
        mean, std = window.mean(), window.std()
        if std < 1e-12:
            return None
        z = (hist[-1] - mean) / std
        return float(math.tanh(z))


# ---------------------------------------------------------------------------
# RSI signal
# ---------------------------------------------------------------------------

class RSISignal(AlphaSignal):
    """Relative Strength Index signal."""

    def __init__(self, period: int = 14, overbought: float = 70.0,
                 oversold: float = 30.0):
        super().__init__(f"rsi_{period}", lookback=period * 3)
        self.period      = period
        self.overbought  = overbought
        self.oversold    = oversold
        self._avg_gain:  Dict[int, float] = {}
        self._avg_loss:  Dict[int, float] = {}
        self._prev_price: Dict[int, float] = {}
        self._n:          Dict[int, int]   = {}

    def update(self, asset_id: int, price: float) -> None:
        super().update(asset_id, price)
        n    = self._n.get(asset_id, 0)
        prev = self._prev_price.get(asset_id, price)
        change = price - prev
        gain = max(0.0, change)
        loss = max(0.0, -change)
        alpha = 1.0 / self.period
        self._avg_gain[asset_id] = alpha*gain + (1-alpha)*self._avg_gain.get(asset_id, 0.0)
        self._avg_loss[asset_id] = alpha*loss + (1-alpha)*self._avg_loss.get(asset_id, 0.0)
        self._prev_price[asset_id] = price
        self._n[asset_id] = n + 1

    def _rsi(self, asset_id: int) -> Optional[float]:
        if self._n.get(asset_id, 0) < self.period:
            return None
        ag = self._avg_gain.get(asset_id, 0.0)
        al = self._avg_loss.get(asset_id, 0.0)
        if al < 1e-12:
            return 100.0
        rs  = ag / al
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    def compute(self, asset_id: int) -> Optional[float]:
        rsi = self._rsi(asset_id)
        if rsi is None:
            return None
        # Convert RSI to [-1, 1]: sell at overbought, buy at oversold
        if rsi > self.overbought:
            return -1.0 * (rsi - self.overbought) / (100.0 - self.overbought)
        elif rsi < self.oversold:
            return 1.0 * (self.oversold - rsi) / self.oversold
        return 0.0


# ---------------------------------------------------------------------------
# IC tracker — information coefficient monitoring
# ---------------------------------------------------------------------------

class ICTracker:
    """Online IC (rank correlation) between signals and forward returns."""

    def __init__(self, signal_name: str, window: int = 60):
        self.signal_name  = signal_name
        self.window       = window
        self._ic_history  = deque(maxlen=window)

    def update(self, signals: np.ndarray, forward_rets: np.ndarray) -> float:
        """Compute IC and update history. Returns current IC."""
        if len(signals) < 2 or len(forward_rets) != len(signals):
            return 0.0
        # Rank correlation (Spearman)
        from scipy.stats import spearmanr  # type: ignore
        try:
            ic, _ = spearmanr(signals, forward_rets)
            ic    = float(ic) if not math.isnan(ic) else 0.0
        except Exception:
            ic = 0.0
        self._ic_history.append(ic)
        return ic

    def ic_mean(self) -> float:
        if not self._ic_history:
            return 0.0
        return float(np.mean(list(self._ic_history)))

    def icir(self) -> float:
        if len(self._ic_history) < 2:
            return 0.0
        ics = np.array(list(self._ic_history))
        return float(ics.mean() / (ics.std() + 1e-10))

    def hit_rate(self) -> float:
        if not self._ic_history:
            return 0.0
        ics = list(self._ic_history)
        return float(sum(1 for ic in ics if ic > 0) / len(ics))


class ICWeightedEnsemble:
    """Combines multiple signals using ICIR as weights."""

    def __init__(self, window: int = 60, min_icir: float = 0.1):
        self.window   = window
        self.min_icir = min_icir
        self._trackers: Dict[str, ICTracker] = {}
        self._last_signals: Dict[str, Dict[int, float]] = {}

    def register_signal(self, name: str) -> None:
        self._trackers[name] = ICTracker(name, self.window)
        self._last_signals[name] = {}

    def update_signal(self, name: str, signals: Dict[int, float]) -> None:
        if name not in self._trackers:
            self.register_signal(name)
        self._last_signals[name] = signals

    def update_ic(self, name: str, forward_rets: Dict[int, float]) -> float:
        if name not in self._trackers or name not in self._last_signals:
            return 0.0
        asset_ids  = sorted(set(self._last_signals[name]) & set(forward_rets))
        if len(asset_ids) < 2:
            return 0.0
        sigs = np.array([self._last_signals[name][a] for a in asset_ids])
        rets = np.array([forward_rets[a] for a in asset_ids])
        return self._trackers[name].update(sigs, rets)

    def compute_weights(self) -> Dict[str, float]:
        icirs = {
            name: max(0.0, tracker.icir())
            for name, tracker in self._trackers.items()
        }
        # Zero out below min_icir
        icirs = {k: v for k, v in icirs.items() if v >= self.min_icir}
        total = sum(icirs.values())
        if total < 1e-10:
            # Equal weight fallback
            n = len(self._trackers)
            return {name: 1.0/n for name in self._trackers} if n > 0 else {}
        return {k: v/total for k, v in icirs.items()}

    def combine(self, asset_ids: List[int]) -> Dict[int, float]:
        weights = self.compute_weights()
        combined = {a: 0.0 for a in asset_ids}
        for name, w in weights.items():
            signals = self._last_signals.get(name, {})
            for aid in asset_ids:
                combined[aid] += w * signals.get(aid, 0.0)
        # Clip to [-1, 1]
        return {k: float(np.clip(v, -1.0, 1.0)) for k, v in combined.items()}

    def icir_summary(self) -> Dict[str, float]:
        return {name: t.icir() for name, t in self._trackers.items()}


# ---------------------------------------------------------------------------
# Signal decay / half-life estimator
# ---------------------------------------------------------------------------

class SignalDecayEstimator:
    """Estimates signal half-life via autocorrelation of signal × return product."""

    def __init__(self, max_lags: int = 20):
        self.max_lags    = max_lags
        self._ic_series  = deque(maxlen=200)

    def update(self, ic: float) -> None:
        self._ic_series.append(ic)

    def estimate_halflife(self) -> Optional[float]:
        """Fit AR(1) decay model: IC_t = rho * IC_{t-1} + eps."""
        ics = np.array(list(self._ic_series))
        if len(ics) < self.max_lags + 2:
            return None
        # OLS: regress IC_t on IC_{t-1}
        y = ics[1:]
        x = ics[:-1]
        cov_xy = float(np.cov(x, y)[0, 1])
        var_x  = float(np.var(x))
        if var_x < 1e-12:
            return None
        rho = cov_xy / var_x
        # Half-life: -ln(2) / ln(rho)
        if rho <= 0 or rho >= 1.0:
            return None
        halflife = -math.log(2.0) / math.log(rho)
        return float(halflife)

    def autocorrelations(self) -> List[float]:
        ics = np.array(list(self._ic_series))
        if len(ics) < 4:
            return []
        mean = ics.mean()
        var  = ((ics - mean)**2).mean()
        if var < 1e-12:
            return []
        acfs = []
        for lag in range(1, min(self.max_lags + 1, len(ics))):
            cov = float(((ics[lag:] - mean) * (ics[:-lag] - mean)).mean())
            acfs.append(cov / var)
        return acfs


# ---------------------------------------------------------------------------
# Signal normalization
# ---------------------------------------------------------------------------

class CrossSectionalNormalizer:
    """Z-score normalize signals cross-sectionally."""

    @staticmethod
    def normalize(signals: Dict[int, float]) -> Dict[int, float]:
        if not signals:
            return {}
        vals = np.array(list(signals.values()))
        mean = vals.mean()
        std  = vals.std()
        if std < 1e-12:
            return {k: 0.0 for k in signals}
        return {k: float((v - mean) / std) for k, v in signals.items()}

    @staticmethod
    def rank_normalize(signals: Dict[int, float]) -> Dict[int, float]:
        """Convert to [-1, +1] via rank normalization."""
        if not signals:
            return {}
        sorted_keys = sorted(signals, key=lambda k: signals[k])
        n = len(sorted_keys)
        return {k: float(2.0 * i / (n - 1) - 1.0) if n > 1 else 0.0
                for i, k in enumerate(sorted_keys)}

    @staticmethod
    def winsorize(signals: Dict[int, float], z_cap: float = 3.0) -> Dict[int, float]:
        """Winsorize at z_cap standard deviations."""
        if not signals:
            return {}
        normalized = CrossSectionalNormalizer.normalize(signals)
        return {k: float(np.clip(v, -z_cap, z_cap)) for k, v in normalized.items()}


# ---------------------------------------------------------------------------
# SignalEngine — master signal orchestrator
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Top-level signal engine for AETERNUS.

    Combines all signal sources:
    - Momentum (short and long lookback)
    - Mean reversion
    - LOB imbalance
    - Vol surface
    - EMA crossover
    - RSI

    Uses IC-weighted ensemble with online ICIR tracking.
    """

    def __init__(self, n_assets: int, lookback_short: int = 5,
                 lookback_long: int = 20):
        self.n_assets = n_assets
        self.assets   = list(range(n_assets))

        # Individual signals
        self.momentum_short = MomentumSignal(lookback_short)
        self.momentum_long  = MomentumSignal(lookback_long)
        self.mean_rev       = MeanReversionSignal(lookback_long)
        self.lob_imbal      = LOBImbalanceSignal()
        self.vol_surface    = VolSurfaceSignal(lookback_long)
        self.ema_cross      = EMACrossoverSignal(fast=lookback_short, slow=lookback_long)
        self.rsi            = RSISignal()

        # Ensemble
        self.ensemble = ICWeightedEnsemble()
        for name in ["momentum_short", "momentum_long", "mean_rev",
                     "lob_imbal", "vol_surface", "ema_cross", "rsi"]:
            self.ensemble.register_signal(name)

        # Decay tracking
        self.decay_estimator = SignalDecayEstimator()

        # Per-asset forward return tracking (for IC update)
        self._prev_prices: Dict[int, float] = {}
        self.step = 0

    def update_prices(self, prices: Dict[int, float]) -> None:
        """Update all price-based signals."""
        for aid, price in prices.items():
            self.momentum_short.update(aid, price)
            self.momentum_long.update(aid, price)
            self.mean_rev.update(aid, price)
            self.ema_cross.update(aid, price)
            self.rsi.update(aid, price)

    def update_lob(self, asset_id: int, bid_qty: float, ask_qty: float,
                   spread: float, mid: float) -> None:
        self.lob_imbal.update_lob(asset_id, bid_qty, ask_qty, spread, mid)

    def update_vol_surface(self, asset_id: int, atm_vol: float, skew: float,
                           mid: float) -> None:
        self.vol_surface.update_vol(asset_id, atm_vol, skew, mid)

    def update_forward_returns(self, current_prices: Dict[int, float]) -> None:
        """Called each step to update IC trackers with realized returns."""
        if self._prev_prices:
            fwd_rets = {}
            for aid, price in current_prices.items():
                prev = self._prev_prices.get(aid, price)
                if prev > 1e-12:
                    fwd_rets[aid] = (price - prev) / prev

            if fwd_rets:
                for sig_name, signal_obj in [
                    ("momentum_short", self.momentum_short),
                    ("momentum_long",  self.momentum_long),
                    ("mean_rev",       self.mean_rev),
                    ("lob_imbal",      self.lob_imbal),
                    ("ema_cross",      self.ema_cross),
                    ("rsi",            self.rsi),
                ]:
                    sigs = signal_obj.compute_all(list(fwd_rets.keys()))
                    self.ensemble.update_signal(sig_name, sigs)
                    ic = self.ensemble.update_ic(sig_name, fwd_rets)
                    if sig_name == "momentum_long":
                        self.decay_estimator.update(ic)

        self._prev_prices = dict(current_prices)
        self.step += 1

    def get_combined_signal(self) -> Dict[int, float]:
        """Get IC-weighted ensemble signal for all assets."""
        return self.ensemble.combine(self.assets)

    def get_normalized_signal(self) -> Dict[int, float]:
        """Get cross-sectionally normalized signal."""
        raw = self.get_combined_signal()
        return CrossSectionalNormalizer.rank_normalize(raw)

    def icir_summary(self) -> Dict[str, float]:
        return self.ensemble.icir_summary()

    def signal_halflife(self) -> Optional[float]:
        return self.decay_estimator.estimate_halflife()

    def diagnostics(self) -> dict:
        icirs = self.icir_summary()
        weights = self.ensemble.compute_weights()
        halflife = self.signal_halflife()
        return {
            "step":      self.step,
            "icirs":     icirs,
            "weights":   weights,
            "halflife":  halflife,
            "n_assets":  self.n_assets,
        }
