"""
execution/regime_ensemble.py -- Live regime-switching ensemble for signal combination.

Implements:
  - RegimeState     : dataclass describing the current market regime
  - RegimeClassifier: combines Hurst H, BH mass, GARCH vol, BH alignment
  - SignalWeight    : per-signal weight with regime adjustment multipliers
  - RegimeEnsemble  : Hedge-algorithm online ensemble with regime-specific weights
  - EnsembleLiveAdapter: thread-safe wrapper for bar-by-bar usage
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VolRegime(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class MarketPhase(str, Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"


# ---------------------------------------------------------------------------
# RegimeState
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """
    Snapshot of the current market regime.

    Fields
    ------
    hurst_h       : float in [0,1] -- Hurst exponent (>0.58 trending, <0.42 MR)
    bh_active     : bool           -- is a Bat/Harmonic pattern presently active?
    bh_mass       : float          -- aggregated mass of active BH patterns (>2 extreme)
    vol_regime    : VolRegime      -- low / med / high
    trend_strength: float in [0,1] -- normalised trend strength (e.g. from ADX)
    market_phase  : MarketPhase    -- Wyckoff phase
    confidence    : float in [0,1] -- classifier confidence
    is_trending   : bool           -- derived: H > 0.58
    is_mean_rev   : bool           -- derived: H < 0.42
    timestamp     : float          -- unix time of classification
    """
    hurst_h: float = 0.5
    bh_active: bool = False
    bh_mass: float = 0.0
    vol_regime: VolRegime = VolRegime.MED
    trend_strength: float = 0.0
    market_phase: MarketPhase = MarketPhase.ACCUMULATION
    confidence: float = 0.5
    is_trending: bool = False
    is_mean_rev: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.is_trending = self.hurst_h > 0.58
        self.is_mean_rev = self.hurst_h < 0.42

    def regime_key(self) -> str:
        """Compact string key used as dict key in weight tables."""
        if self.is_trending:
            return f"trending_{self.vol_regime.value}"
        if self.is_mean_rev:
            return f"mean_rev_{self.vol_regime.value}"
        return f"neutral_{self.vol_regime.value}"


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------

class HurstEstimator:
    """
    Rolling Hurst exponent via Rescaled Range (R/S) analysis.
    Requires at least `window` bars of log-returns.
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._log_returns: Deque[float] = deque(maxlen=window)

    def update(self, price: float) -> Optional[float]:
        """Feed one price, return current H estimate or None if not warmed up."""
        if len(self._log_returns) > 0:
            prev = self._prices[-1] if hasattr(self, "_prices") else price
        if not hasattr(self, "_prices"):
            self._prices: Deque[float] = deque(maxlen=self._window + 1)
        self._prices.append(price)
        if len(self._prices) < 2:
            return None
        lr = math.log(self._prices[-1] / self._prices[-2])
        self._log_returns.append(lr)
        if len(self._log_returns) < self._window:
            return None
        return self._compute_hurst(np.array(self._log_returns))

    @staticmethod
    def _compute_hurst(rets: np.ndarray) -> float:
        """
        Multi-scale R/S analysis.  Computes H by regressing log(R/S) on log(n)
        for increasing sub-window sizes.
        """
        n = len(rets)
        min_scale = max(8, n // 16)
        scales = []
        rs_vals = []
        for scale in range(min_scale, n // 2, max(1, (n // 2 - min_scale) // 10)):
            rs = HurstEstimator._rs_stat(rets[:scale])
            if rs is not None and rs > 0:
                scales.append(math.log(scale))
                rs_vals.append(math.log(rs))
        if len(scales) < 3:
            return 0.5
        slope, _, _, _, _ = scipy_stats.linregress(scales, rs_vals)
        return float(np.clip(slope, 0.01, 0.99))

    @staticmethod
    def _rs_stat(rets: np.ndarray) -> Optional[float]:
        n = len(rets)
        if n < 4:
            return None
        mean_r = rets.mean()
        deviations = np.cumsum(rets - mean_r)
        R = deviations.max() - deviations.min()
        S = rets.std(ddof=1)
        if S < 1e-14:
            return None
        return R / S


class GARCHVolEstimator:
    """
    Simple GARCH(1,1) variance tracker.  Tracks rolling vol percentile
    over a 252-day window to produce a vol regime classification.
    """

    OMEGA = 1e-6
    ALPHA = 0.10
    BETA = 0.85

    def __init__(self, percentile_window: int = 252) -> None:
        self._h: float = 0.0001          # variance
        self._vol_history: Deque[float] = deque(maxlen=percentile_window)

    def update(self, ret: float) -> Tuple[float, float]:
        """
        Feed one log-return.
        Returns (current_vol, vol_percentile_0_to_100).
        """
        self._h = self.OMEGA + self.ALPHA * ret ** 2 + self.BETA * self._h
        self._h = max(self._h, 1e-10)
        vol = math.sqrt(self._h)
        self._vol_history.append(vol)
        if len(self._vol_history) < 5:
            return vol, 50.0
        arr = np.array(self._vol_history)
        pct = float(scipy_stats.percentileofscore(arr[:-1], vol))
        return vol, pct

    def current_vol(self) -> float:
        return math.sqrt(self._h)


class RegimeClassifier:
    """
    Combines multiple regime signals into a unified RegimeState.

    Inputs per bar
    --------------
    - price (1h or 4h)
    - optional BH pattern mass and active flag
    - optional ADX for trend strength

    Rolling 20-bar persistence filter prevents rapid regime switching.
    """

    HURST_TREND_THRESH = 0.58
    HURST_MR_THRESH = 0.42
    BH_EXTREME_MASS = 2.0
    PERSISTENCE_BARS = 20

    def __init__(
        self,
        hurst_window: int = 100,
        garch_window: int = 252,
        persistence_bars: int = 20,
    ) -> None:
        self._hurst = HurstEstimator(window=hurst_window)
        self._garch = GARCHVolEstimator(percentile_window=garch_window)
        self._persistence = persistence_bars

        # Regime persistence buffer -- last N regime candidates
        self._candidate_buffer: Deque[str] = deque(maxlen=persistence_bars)
        # Locked regime until buffer majority changes
        self._current_regime_key: str = "neutral_med"
        self._current_state: RegimeState = RegimeState()
        self._bars_since_last_update: int = 0
        self._prev_price: Optional[float] = None

    def update(
        self,
        price: float,
        bh_active: bool = False,
        bh_mass: float = 0.0,
        adx: float = 25.0,
        timeframe_alignment: float = 0.0,  # 4h/1h BH alignment score [-1,1]
    ) -> RegimeState:
        """
        Feed one bar.  Returns current RegimeState (may lag due to persistence filter).
        """
        # --- Hurst ---
        hurst_h = self._hurst.update(price)
        if hurst_h is None:
            hurst_h = 0.5

        # --- GARCH vol ---
        ret = 0.0
        if self._prev_price is not None and self._prev_price > 0:
            ret = math.log(price / self._prev_price)
        self._prev_price = price
        _, vol_pct = self._garch.update(ret)

        vol_regime = self._classify_vol(vol_pct)

        # --- Trend strength ---
        trend_strength = float(np.clip(adx / 50.0, 0.0, 1.0))

        # --- Market phase (simplified Wyckoff via price/trend/vol combos) ---
        market_phase = self._classify_phase(
            hurst_h, vol_regime, trend_strength, bh_mass
        )

        # --- Raw regime key ---
        is_trending = hurst_h > self.HURST_TREND_THRESH
        is_mr = hurst_h < self.HURST_MR_THRESH
        if is_trending:
            raw_key = f"trending_{vol_regime.value}"
        elif is_mr:
            raw_key = f"mean_rev_{vol_regime.value}"
        else:
            raw_key = f"neutral_{vol_regime.value}"

        self._candidate_buffer.append(raw_key)

        # --- Persistence filter: only switch if majority of buffer agrees ---
        filtered_key = self._apply_persistence_filter(raw_key)

        # --- Confidence ---
        confidence = self._compute_confidence(
            hurst_h, vol_pct, bh_mass, timeframe_alignment, filtered_key
        )

        state = RegimeState(
            hurst_h=round(hurst_h, 4),
            bh_active=bh_active,
            bh_mass=round(bh_mass, 4),
            vol_regime=vol_regime,
            trend_strength=round(trend_strength, 4),
            market_phase=market_phase,
            confidence=round(confidence, 4),
            is_trending=hurst_h > self.HURST_TREND_THRESH,
            is_mean_rev=hurst_h < self.HURST_MR_THRESH,
            timestamp=time.time(),
        )
        self._current_state = state
        self._current_regime_key = filtered_key
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_vol(self, vol_pct: float) -> VolRegime:
        if vol_pct >= 75:
            return VolRegime.HIGH
        if vol_pct <= 30:
            return VolRegime.LOW
        return VolRegime.MED

    def _classify_phase(
        self,
        hurst_h: float,
        vol_regime: VolRegime,
        trend_strength: float,
        bh_mass: float,
    ) -> MarketPhase:
        """
        Simplified Wyckoff phase mapping:
          - Low vol + neutral H + low trend -> accumulation
          - Trending + rising vol + strong trend -> markup
          - High vol + trending + declining (bh_mass high) -> distribution
          - Mean-reverting + high vol -> markdown
        """
        if vol_regime == VolRegime.LOW and hurst_h < 0.55 and trend_strength < 0.4:
            return MarketPhase.ACCUMULATION
        if hurst_h > self.HURST_TREND_THRESH and trend_strength > 0.5:
            if vol_regime == VolRegime.HIGH or bh_mass > self.BH_EXTREME_MASS:
                return MarketPhase.DISTRIBUTION
            return MarketPhase.MARKUP
        if hurst_h < self.HURST_MR_THRESH and vol_regime in (VolRegime.MED, VolRegime.HIGH):
            return MarketPhase.MARKDOWN
        return MarketPhase.ACCUMULATION

    def _apply_persistence_filter(self, raw_key: str) -> str:
        """
        Returns the majority regime key from the candidate buffer.
        Requires > 50% agreement to flip from the current key.
        """
        if len(self._candidate_buffer) < self._persistence // 2:
            return self._current_regime_key
        from collections import Counter
        counts = Counter(self._candidate_buffer)
        majority_key, majority_count = counts.most_common(1)[0]
        ratio = majority_count / len(self._candidate_buffer)
        if ratio > 0.5:
            return majority_key
        return self._current_regime_key

    def _compute_confidence(
        self,
        hurst_h: float,
        vol_pct: float,
        bh_mass: float,
        alignment: float,
        regime_key: str,
    ) -> float:
        """
        Confidence is higher when:
        - Hurst strongly deviates from 0.5
        - BH mass is clear (either near 0 or > 2)
        - 4h/1h alignment is high
        """
        hurst_conf = abs(hurst_h - 0.5) * 2.0           # 0 at H=0.5, 1 at H=0/1
        bh_conf = min(1.0, bh_mass / self.BH_EXTREME_MASS) if bh_mass > 0 else 0.3
        align_conf = (abs(alignment) + 1.0) / 2.0        # map [-1,1] -> [0,1]
        raw = 0.50 * hurst_conf + 0.25 * bh_conf + 0.25 * align_conf
        return float(np.clip(raw, 0.1, 0.99))


# ---------------------------------------------------------------------------
# SignalWeight
# ---------------------------------------------------------------------------

@dataclass
class SignalWeight:
    """
    Per-signal weight configuration.

    Fields
    ------
    signal_name        : str
    base_weight        : float -- weight when no regime adjustment applies
    regime_adjustments : dict mapping regime_key -> multiplier
      e.g. {"trending_low": 2.0, "mean_rev_low": 0.3}
    """
    signal_name: str
    base_weight: float
    regime_adjustments: Dict[str, float] = field(default_factory=dict)

    def effective_weight(self, regime: RegimeState) -> float:
        key = regime.regime_key()
        multiplier = self.regime_adjustments.get(key, 1.0)
        return max(0.0, self.base_weight * multiplier)


# ---------------------------------------------------------------------------
# Per-regime IC tracker
# ---------------------------------------------------------------------------

class RollingIC:
    """Tracks rolling Spearman IC between signal values and forward returns."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._signals: Deque[float] = deque(maxlen=window)
        self._returns: Deque[float] = deque(maxlen=window)

    def update(self, signal_val: float, fwd_return: float) -> None:
        self._signals.append(signal_val)
        self._returns.append(fwd_return)

    def ic(self) -> float:
        if len(self._signals) < 5:
            return 0.0
        r, _ = scipy_stats.spearmanr(list(self._signals), list(self._returns))
        return float(r) if not math.isnan(r) else 0.0

    def icir(self) -> float:
        """Information Coefficient Information Ratio."""
        if len(self._signals) < 5:
            return 0.0
        ics = []
        sigs = list(self._signals)
        rets = list(self._returns)
        half = len(sigs) // 2
        if half < 3:
            return 0.0
        for i in range(half, len(sigs)):
            r, _ = scipy_stats.spearmanr(sigs[: i + 1], rets[: i + 1])
            if not math.isnan(r):
                ics.append(r)
        if len(ics) < 3:
            return 0.0
        arr = np.array(ics)
        return float(arr.mean() / (arr.std() + 1e-9))


# ---------------------------------------------------------------------------
# RegimeEnsemble
# ---------------------------------------------------------------------------

class RegimeEnsemble:
    """
    Online regime-switching ensemble using the Hedge algorithm for weight adaptation.

    The Hedge algorithm maintains exponential weights over signals.  After each
    bar the weights are updated proportionally to exp(-eta * loss), where loss
    is the squared error between each signal's prediction and the realized return.

    Additionally, regime-specific weight multipliers are applied before
    normalization so that signals better suited to the current regime receive
    higher effective weight.

    Parameters
    ----------
    eta : float
        Hedge learning rate (default 0.1)
    ic_window : int
        Rolling window for per-regime IC tracking
    """

    ETA_DEFAULT = 0.10
    IC_WINDOW = 30

    def __init__(
        self,
        eta: float = ETA_DEFAULT,
        ic_window: int = IC_WINDOW,
    ) -> None:
        self._eta = eta
        self._ic_window = ic_window

        # Registered signals: list of (name, weight_cfg, predict_fn)
        self._signals: List[Tuple[str, SignalWeight, Callable]] = []

        # Hedge weights (unnormalized, one per signal)
        self._hedge_weights: np.ndarray = np.array([], dtype=float)

        # Per-signal, per-regime IC trackers
        # regime_key -> signal_name -> RollingIC
        self._regime_ics: Dict[str, Dict[str, RollingIC]] = {}

        # Last predictions (for weight update on next bar)
        self._last_preds: Dict[str, float] = {}
        self._last_regime: Optional[RegimeState] = None

        # Performance history per regime
        self._regime_perf: Dict[str, Deque[float]] = {}

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def register_signal(
        self,
        name: str,
        weight_cfg: SignalWeight,
        predict_fn: Callable[..., float],
    ) -> None:
        """
        Register a signal source.

        Parameters
        ----------
        name       : str   -- unique signal name
        weight_cfg : SignalWeight
        predict_fn : callable returning a float prediction in [-1,1]
        """
        self._signals.append((name, weight_cfg, predict_fn))
        # Initialize Hedge weight to base weight
        new_w = np.append(self._hedge_weights, weight_cfg.base_weight)
        self._hedge_weights = new_w
        self._last_preds[name] = 0.0

    # ------------------------------------------------------------------
    # Core combination
    # ------------------------------------------------------------------

    def combine(
        self,
        signal_values: Dict[str, float],
        regime: RegimeState,
    ) -> float:
        """
        Weighted combination of signal values under the current regime.

        Parameters
        ----------
        signal_values : dict mapping signal_name -> prediction in [-1,1]
        regime        : current RegimeState

        Returns
        -------
        float in [-1,1] -- ensemble signal
        """
        if not self._signals:
            return 0.0

        regime_key = regime.regime_key()
        effective_weights = np.zeros(len(self._signals))

        for i, (name, weight_cfg, _) in enumerate(self._signals):
            regime_mult = weight_cfg.regime_adjustments.get(regime_key, 1.0)
            # Apply both Hedge weight and regime multiplier
            effective_weights[i] = self._hedge_weights[i] * regime_mult

        w_sum = effective_weights.sum()
        if w_sum < 1e-12:
            effective_weights = np.ones(len(self._signals))
            w_sum = float(len(self._signals))

        norm_weights = effective_weights / w_sum

        result = 0.0
        for i, (name, _, _) in enumerate(self._signals):
            val = signal_values.get(name, 0.0)
            val = float(np.clip(val, -1.0, 1.0))
            result += norm_weights[i] * val
            self._last_preds[name] = val

        self._last_regime = regime
        return float(np.clip(result, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Weight update (Hedge algorithm)
    # ------------------------------------------------------------------

    def update_weights(
        self,
        realized_return: float,
        regime: RegimeState,
    ) -> None:
        """
        Update Hedge weights and per-regime IC trackers using the realized return.

        Hedge loss = squared prediction error per signal.
        """
        if not self._signals:
            return

        regime_key = regime.regime_key()

        # Ensure IC dict exists for this regime
        if regime_key not in self._regime_ics:
            self._regime_ics[regime_key] = {}
        if regime_key not in self._regime_perf:
            self._regime_perf[regime_key] = deque(maxlen=100)

        self._regime_perf[regime_key].append(realized_return)

        new_weights = np.zeros(len(self._signals))
        for i, (name, _, _) in enumerate(self._signals):
            pred = self._last_preds.get(name, 0.0)
            loss = (pred - realized_return) ** 2
            new_weights[i] = self._hedge_weights[i] * math.exp(-self._eta * loss)

            # Update rolling IC
            ic_tracker = self._regime_ics[regime_key].setdefault(
                name, RollingIC(window=self._ic_window)
            )
            ic_tracker.update(pred, realized_return)

        # Re-normalize to prevent weight collapse
        total = new_weights.sum()
        if total > 1e-12:
            self._hedge_weights = new_weights / total * len(self._signals)
        else:
            self._hedge_weights = np.ones(len(self._signals))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_regime_report(self) -> Dict[str, Any]:
        """
        Returns a dict containing:
          - current_regime_key : str
          - hedge_weights : dict name->weight
          - regime_ics : dict regime_key -> signal_name -> IC float
          - regime_perf : dict regime_key -> recent mean return
        """
        weight_map = {}
        for i, (name, _, _) in enumerate(self._signals):
            if i < len(self._hedge_weights):
                weight_map[name] = round(float(self._hedge_weights[i]), 6)

        regime_key = (
            self._last_regime.regime_key() if self._last_regime else "unknown"
        )

        ic_report: Dict[str, Dict[str, float]] = {}
        for rk, sig_map in self._regime_ics.items():
            ic_report[rk] = {n: round(tracker.ic(), 4) for n, tracker in sig_map.items()}

        perf_report: Dict[str, float] = {}
        for rk, hist in self._regime_perf.items():
            if hist:
                perf_report[rk] = round(float(np.mean(hist)), 6)

        return {
            "active_regime": regime_key,
            "hedge_weights": weight_map,
            "regime_ics": ic_report,
            "regime_performance": perf_report,
            "n_signals": len(self._signals),
        }


# ---------------------------------------------------------------------------
# EnsembleLiveAdapter
# ---------------------------------------------------------------------------

class EnsembleLiveAdapter:
    """
    Thread-safe adapter that wraps RegimeEnsemble and RegimeClassifier for
    bar-by-bar live trading.

    Usage
    -----
    adapter = EnsembleLiveAdapter()
    adapter.register_signal(name, weight_cfg, predict_fn)
    ...
    signal = await adapter.on_bar(symbol, bar_data, regime_state)

    on_bar() returns 0.0 during the warmup period (first `warmup_bars` bars).

    Thread safety
    -------------
    An asyncio.Lock guards internal state so multiple coroutines can safely
    call on_bar() concurrently for different or the same symbol.
    """

    WARMUP_BARS = 60

    def __init__(
        self,
        warmup_bars: int = WARMUP_BARS,
        eta: float = RegimeEnsemble.ETA_DEFAULT,
        ic_window: int = RegimeEnsemble.IC_WINDOW,
        classifier_hurst_window: int = 100,
        classifier_garch_window: int = 252,
    ) -> None:
        self._warmup_bars = warmup_bars
        self._ensemble = RegimeEnsemble(eta=eta, ic_window=ic_window)
        self._classifier = RegimeClassifier(
            hurst_window=classifier_hurst_window,
            garch_window=classifier_garch_window,
        )
        self._lock = asyncio.Lock()

        # Per-symbol bar counters
        self._bar_counts: Dict[str, int] = {}

        # Last bar close prices for return computation
        self._prev_closes: Dict[str, float] = {}

        # Buffer of (signal_val, close_price) for weight update on next bar
        self._pending_update: Dict[str, Tuple[float, float]] = {}

        # Classifier inputs from last bar (for BH / adx injection)
        self._regime_state: Optional[RegimeState] = None

    # ------------------------------------------------------------------
    # Signal registration (delegates to ensemble)
    # ------------------------------------------------------------------

    def register_signal(
        self,
        name: str,
        weight_cfg: SignalWeight,
        predict_fn: Callable[..., float],
    ) -> None:
        self._ensemble.register_signal(name, weight_cfg, predict_fn)

    # ------------------------------------------------------------------
    # Main bar processing
    # ------------------------------------------------------------------

    async def on_bar(
        self,
        symbol: str,
        bar_data: Dict[str, Any],
        regime_state: Optional[RegimeState] = None,
        signal_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Process one bar for `symbol`.

        Parameters
        ----------
        symbol         : str -- ticker
        bar_data       : dict with keys: open, high, low, close, volume
          optional: bh_active, bh_mass, adx, alignment
        regime_state   : RegimeState (if None, classifier is run internally)
        signal_overrides: dict name->float (bypasses predict_fn for testing)

        Returns
        -------
        float in [-1,1] or 0.0 during warmup
        """
        async with self._lock:
            close = float(bar_data["close"])
            self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1
            bar_n = self._bar_counts[symbol]

            # --- Weight update from previous bar's signal ---
            if symbol in self._pending_update and bar_n > 1:
                prev_signal, prev_close = self._pending_update[symbol]
                if prev_close > 0:
                    realized = math.log(close / prev_close)
                    if self._regime_state is not None:
                        self._ensemble.update_weights(realized, self._regime_state)

            # --- Determine regime ---
            if regime_state is None:
                bh_active = bool(bar_data.get("bh_active", False))
                bh_mass = float(bar_data.get("bh_mass", 0.0))
                adx = float(bar_data.get("adx", 25.0))
                alignment = float(bar_data.get("alignment", 0.0))
                regime_state = self._classifier.update(
                    close, bh_active=bh_active, bh_mass=bh_mass,
                    adx=adx, timeframe_alignment=alignment,
                )
            self._regime_state = regime_state

            # --- Warmup gate ---
            if bar_n < self._warmup_bars:
                logger.debug(
                    "symbol=%s warmup bar %d/%d", symbol, bar_n, self._warmup_bars
                )
                self._prev_closes[symbol] = close
                return 0.0

            # --- Gather signal values ---
            if signal_overrides is not None:
                signal_values = signal_overrides
            else:
                signal_values = {}
                for name, _, predict_fn in self._ensemble._signals:
                    try:
                        val = predict_fn(symbol, bar_data, regime_state)
                        signal_values[name] = float(np.clip(val, -1.0, 1.0))
                    except Exception as exc:
                        logger.warning("Signal %s raised: %s", name, exc)
                        signal_values[name] = 0.0

            # --- Combine ---
            result = self._ensemble.combine(signal_values, regime_state)

            # Store for next-bar weight update
            self._pending_update[symbol] = (result, close)
            self._prev_closes[symbol] = close

            return result

    # ------------------------------------------------------------------

    def get_regime_report(self) -> Dict[str, Any]:
        return self._ensemble.get_regime_report()

    @property
    def regime_state(self) -> Optional[RegimeState]:
        return self._regime_state

    @property
    def ensemble(self) -> RegimeEnsemble:
        return self._ensemble

    @property
    def classifier(self) -> RegimeClassifier:
        return self._classifier


# ---------------------------------------------------------------------------
# SignalDecayMonitor -- see signal_decay_monitor.py for the full implementation.
# This module re-exports key symbols for convenience.
# ---------------------------------------------------------------------------

__all__ = [
    "VolRegime",
    "MarketPhase",
    "RegimeState",
    "HurstEstimator",
    "GARCHVolEstimator",
    "RegimeClassifier",
    "SignalWeight",
    "RollingIC",
    "RegimeEnsemble",
    "EnsembleLiveAdapter",
]
