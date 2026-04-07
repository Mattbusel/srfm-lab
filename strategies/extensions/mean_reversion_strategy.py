"""
mean_reversion_strategy.py # Mean-reversion strategies for SRFM.

Activated when the Hurst exponent indicates a mean-reverting regime
(H < 0.42).  Two signal sources:
  1. Ornstein-Uhlenbeck (OU) z-score model
  2. Bollinger Band squeeze + band-touch model

The MeanReversionEnsemble combines both with weights 0.6 / 0.4 and
auto-deactivates when Hurst drifts above 0.50 (trending regime).
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------

HURST_MR_MAX     = 0.42   # Hurst threshold: below = mean-reverting
HURST_DEACT_MIN  = 0.50   # auto-deactivate if Hurst rises above this
Z_ENTRY_THRESH   = 2.0    # |z| > 2.0 triggers entry
Z_EXIT_THRESH    = 0.20   # |z| < 0.20 triggers exit (back near mean)
BB_WIDTH_MULT    = 1.5    # trade only if BB_width < 1.5x historical median
BB_DEFAULT_STD   = 2.0    # standard deviations for Bollinger Bands
OU_WEIGHT        = 0.60
BB_WEIGHT        = 0.40


# ---------------------------------------------------------------------------
# OUParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class OUParams:
    """
    Parameters for an Ornstein-Uhlenbeck process:
      dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Attributes
    #--------
    theta:
        Mean reversion speed (per bar).  Higher = faster reversion.
    mu:
        Long-run mean level.
    sigma:
        Noise term (standard deviation of residuals).
    half_life:
        Half-life of mean reversion in bars = ln(2) / theta.
    sigma_eq:
        Equilibrium standard deviation = sigma / sqrt(2 * theta).
    """
    theta:    float
    mu:       float
    sigma:    float
    half_life: float
    sigma_eq: float


# ---------------------------------------------------------------------------
# MRSignal enum
# ---------------------------------------------------------------------------

class MRSignal(str, Enum):
    """Bollinger Band mean-reversion signal states."""
    LONG_ENTRY  = "LONG_ENTRY"
    SHORT_ENTRY = "SHORT_ENTRY"
    EXIT_LONG   = "EXIT_LONG"
    EXIT_SHORT  = "EXIT_SHORT"
    HOLD        = "HOLD"


# ---------------------------------------------------------------------------
# MeanReversionStrategy # OU-based
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """
    OU-process mean-reversion signal generator.

    Feed prices via update().  Internally fits OU parameters on a rolling
    window and computes z-scores for entry/exit decisions.

    Entry rules:
      z > +Z_ENTRY_THRESH  => SHORT (price above equilibrium)
      z < -Z_ENTRY_THRESH  => LONG  (price below equilibrium)
    Exit rules:
      |z| < Z_EXIT_THRESH  => close position (back to mean)
    """

    def __init__(
        self,
        fit_window:    int   = 60,
        z_entry:       float = Z_ENTRY_THRESH,
        z_exit:        float = Z_EXIT_THRESH,
        min_half_life: float = 2.0,    # ignore if too fast (noisy)
        max_half_life: float = 40.0,   # ignore if too slow (too long to revert)
    ):
        self._fit_window    = fit_window
        self._z_entry       = z_entry
        self._z_exit        = z_exit
        self._min_half_life = min_half_life
        self._max_half_life = max_half_life

        self._prices: Deque[float] = deque(maxlen=fit_window)
        self._params: Optional[OUParams] = None
        self._bar_count: int = 0

    # ------------------------------------------------------------------

    def update(self, price: float) -> Tuple[float, Optional[OUParams]]:
        """
        Feed one price.  Returns (z_score, OUParams).
        z_score is 0.0 if params not yet fitted or invalid.
        """
        self._prices.append(float(price))
        self._bar_count += 1

        if len(self._prices) < self._fit_window:
            return 0.0, None

        # Refit every 10 bars to save compute
        if self._bar_count % 10 == 0 or self._params is None:
            try:
                arr = np.array(self._prices, dtype=float)
                self._params = self.fit_ou(arr)
            except Exception:
                self._params = None

        if self._params is None:
            return 0.0, None

        # Validate half-life is within tradeable range
        hl = self._params.half_life
        if hl < self._min_half_life or hl > self._max_half_life:
            return 0.0, self._params

        z = self.z_score(price, self._params)
        return z, self._params

    @staticmethod
    def fit_ou(prices: np.ndarray) -> OUParams:
        """
        Estimate OU parameters from price series using OLS on the
        discretised OU equation:
          X_{t+1} - X_t = theta*(mu - X_t)*dt + epsilon

        Rewritten as regression:
          delta_X = A + B * X_t  =>  A = theta*mu*dt, B = -theta*dt

        Parameters
        #--------
        prices:
            1-D array of prices (log prices give better fit for assets).

        Returns
        #-----
        OUParams
        """
        if len(prices) < 10:
            raise ValueError("Need at least 10 prices to fit OU params")

        # Use log prices for stationarity
        lp = np.log(np.maximum(prices, 1e-9))
        x  = lp[:-1]
        y  = lp[1:]
        delta = y - x    # increments

        # OLS: delta = A + B*x + eps
        X_mat = np.column_stack([np.ones_like(x), x])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X_mat, delta, rcond=None)
        except np.linalg.LinAlgError:
            raise ValueError("OLS failed in fit_ou")

        A, B = coeffs[0], coeffs[1]

        # B = -theta*dt => theta = -B (dt=1 bar)
        theta = max(1e-6, -B)

        # mu = A / theta (the long-run mean in log-price space, then exponentiated)
        mu_log = A / theta if theta > 1e-6 else float(np.mean(lp))
        mu = math.exp(mu_log)

        # Residual sigma
        pred   = A + B * x
        resids = delta - pred
        sigma  = float(np.std(resids, ddof=2)) if len(resids) > 2 else 1e-4
        sigma  = max(sigma, 1e-8)

        # Derived quantities
        half_life = math.log(2.0) / theta
        sigma_eq  = sigma / math.sqrt(max(2.0 * theta, 1e-9))

        return OUParams(
            theta     = theta,
            mu        = mu,
            sigma     = sigma,
            half_life = half_life,
            sigma_eq  = sigma_eq,
        )

    @staticmethod
    def z_score(price: float, params: OUParams) -> float:
        """
        Compute z-score = (price - mu) / sigma_eq.
        Positive z means price is above equilibrium mean.
        """
        denom = params.sigma_eq if params.sigma_eq > 1e-10 else 1e-10
        return (price - params.mu) / denom

    def get_params(self) -> Optional[OUParams]:
        return self._params

    def signal_from_z(self, z: float) -> MRSignal:
        """Convert z-score to a MRSignal action label."""
        if z > self._z_entry:
            return MRSignal.SHORT_ENTRY
        if z < -self._z_entry:
            return MRSignal.LONG_ENTRY
        if abs(z) < self._z_exit:
            return MRSignal.EXIT_LONG    # generic exit (position handler decides)
        return MRSignal.HOLD


# ---------------------------------------------------------------------------
# BollingerBandMR
# ---------------------------------------------------------------------------

class BollingerBandMR:
    """
    Bollinger Band mean-reversion signal generator.

    Emits LONG_ENTRY when price touches the lower band in a tight-band
    environment, SHORT_ENTRY when it touches the upper band.

    Band-width filter: only trade when BB_width < 1.5x historical median
    (tight bands indicate consolidation, not breakout).
    """

    def __init__(
        self,
        window:           int   = 20,
        num_std:          float = BB_DEFAULT_STD,
        width_mult_limit: float = BB_WIDTH_MULT,
        width_hist_bars:  int   = 100,
    ):
        self._window          = window
        self._num_std         = num_std
        self._width_mult      = width_mult_limit
        self._prices: Deque[float] = deque(maxlen=max(window, width_hist_bars))
        self._width_hist: Deque[float] = deque(maxlen=width_hist_bars)

        # Track open position direction for exit signals
        self._position: int = 0    # +1 long, -1 short, 0 flat
        self._bar_count: int = 0

        # Last computed bands
        self.upper: float  = 0.0
        self.lower: float  = 0.0
        self.middle: float = 0.0
        self.width: float  = 0.0
        self.pct_b: float  = 0.5

    # ------------------------------------------------------------------

    def update(self, price: float) -> MRSignal:
        """
        Feed one price and return the current MRSignal.
        """
        self._prices.append(float(price))
        self._bar_count += 1

        if len(self._prices) < self._window:
            return MRSignal.HOLD

        arr  = np.array(list(self._prices)[-self._window:], dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        self.middle = mean
        self.upper  = mean + self._num_std * std
        self.lower  = mean - self._num_std * std
        self.width  = (self.upper - self.lower) / (mean + 1e-9)

        if mean > 1e-9:
            self.pct_b = (price - self.lower) / (self.upper - self.lower + 1e-9)
        else:
            self.pct_b = 0.5

        # Update width history every bar
        self._width_hist.append(self.width)

        # Check if bands are tight (mean-reverting condition)
        if not self._bands_are_tight():
            # Bands are wide: only allow exits
            return self._check_exit(price)

        return self._check_entry_exit(price)

    def _bands_are_tight(self) -> bool:
        """True if current BB width is below 1.5x historical median."""
        if len(self._width_hist) < 20:
            return True    # assume tight until enough data
        median_width = float(np.median(list(self._width_hist)))
        return self.width < (self._width_mult * median_width)

    def _check_entry_exit(self, price: float) -> MRSignal:
        """Entry and exit logic for tight-band regime."""
        # Exit checks first
        exit_sig = self._check_exit(price)
        if exit_sig != MRSignal.HOLD:
            return exit_sig

        # Entry signals
        if price <= self.lower and self._position <= 0:
            self._position = 1
            return MRSignal.LONG_ENTRY

        if price >= self.upper and self._position >= 0:
            self._position = -1
            return MRSignal.SHORT_ENTRY

        return MRSignal.HOLD

    def _check_exit(self, price: float) -> MRSignal:
        """Check if price has returned to mean (exit zone)."""
        if self._position == 1 and price >= self.middle:
            self._position = 0
            return MRSignal.EXIT_LONG
        if self._position == -1 and price <= self.middle:
            self._position = 0
            return MRSignal.EXIT_SHORT
        return MRSignal.HOLD

    def set_position(self, pos: int) -> None:
        """Externally update position tracker (+1, -1, 0)."""
        self._position = pos

    def get_bands(self) -> Tuple[float, float, float]:
        """Return (lower, middle, upper)."""
        return self.lower, self.middle, self.upper

    def normalized_signal(self) -> float:
        """
        Return normalized signal in [-1, 1] based on pct_b.
          pct_b = 0 (at lower band)  => +1 (strong long)
          pct_b = 1 (at upper band)  => -1 (strong short)
          pct_b = 0.5 (at middle)    => 0
        """
        return float(np.clip(1.0 - 2.0 * self.pct_b, -1.0, 1.0))


# ---------------------------------------------------------------------------
# MeanReversionEnsemble
# ---------------------------------------------------------------------------

class MeanReversionEnsemble:
    """
    Ensemble mean-reversion signal combining OU z-score and Bollinger Bands.

    Weights:
      OU component:  0.60
      BB component:  0.40

    Auto-deactivation:
      If Hurst rises above HURST_DEACT_MIN (0.50), the ensemble returns 0.0
      regardless of OU/BB signals, indicating a regime shift to trending.

    compute_signal() returns float in [-1, 1]:
      +1 = strong long (price far below mean)
      -1 = strong short (price far above mean)
       0 = no signal or deactivated
    """

    def __init__(
        self,
        hurst_mr_max:  float = HURST_MR_MAX,
        hurst_deact:   float = HURST_DEACT_MIN,
        ou_window:     int   = 60,
        bb_window:     int   = 20,
        ou_weight:     float = OU_WEIGHT,
        bb_weight:     float = BB_WEIGHT,
    ):
        self._hurst_mr_max = hurst_mr_max
        self._hurst_deact  = hurst_deact
        self._ou_weight    = ou_weight
        self._bb_weight    = bb_weight

        self._ou = MeanReversionStrategy(fit_window=ou_window)
        self._bb = BollingerBandMR(window=bb_window)

        self._current_hurst: float = 0.5
        self._active:        bool  = False
        self._bar_count:     int   = 0

        # Last ensemble signal for continuity
        self._last_signal: float = 0.0

    # ------------------------------------------------------------------

    def update_hurst(self, hurst: float) -> None:
        """Feed latest Hurst estimate.  Triggers activation/deactivation."""
        prev = self._current_hurst
        self._current_hurst = float(hurst)

        if self._current_hurst < self._hurst_mr_max:
            self._active = True
        elif self._current_hurst > self._hurst_deact:
            self._active = False
        # else: hysteresis zone [0.42, 0.50] -- maintain previous state

    def compute_signal(self, price: float) -> float:
        """
        Feed one price bar and return ensemble signal [-1, 1].

        Must call update_hurst() first (or periodically) for activation logic
        to function correctly.
        """
        self._bar_count += 1

        if not self._active:
            self._last_signal = 0.0
            return 0.0

        # OU signal
        z, params = self._ou.update(price)
        if params is not None and params.sigma_eq > 1e-10:
            # Normalise z to [-1, 1] using tanh squashing
            ou_raw = float(np.tanh(z / Z_ENTRY_THRESH))
            ou_sig = -ou_raw   # positive z => short => negative signal
        else:
            ou_sig = 0.0

        # BB signal
        _ = self._bb.update(price)
        bb_sig = self._bb.normalized_signal()

        # Ensemble combination
        ensemble = self._ou_weight * ou_sig + self._bb_weight * bb_sig
        ensemble = float(np.clip(ensemble, -1.0, 1.0))

        self._last_signal = ensemble
        return ensemble

    def is_active(self) -> bool:
        return self._active

    def get_ou_params(self) -> Optional[OUParams]:
        return self._ou.get_params()

    def get_z_score(self, price: float) -> float:
        """Return current OU z-score without updating the ensemble."""
        params = self._ou.get_params()
        if params is None:
            return 0.0
        return MeanReversionStrategy.z_score(price, params)

    def reset(self) -> None:
        """Full reset of internal state."""
        self._ou          = MeanReversionStrategy(fit_window=self._ou._fit_window)
        self._bb          = BollingerBandMR(window=self._bb._window)
        self._active      = False
        self._last_signal = 0.0
        self._bar_count   = 0
