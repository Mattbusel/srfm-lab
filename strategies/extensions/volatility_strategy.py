"""
volatility_strategy.py # Volatility trading strategies for SRFM.

Three components:
  1. VolatilityBreakoutStrategy  # Trades breakouts from low-vol consolidation
  2. VolatilityArbitrageSignal   # IV rank vs realized vol comparison
  3. GARCHVolForecast            # Full GARCH(1,1) MLE implementation

All computations are pure Python + numpy/scipy (no external finance libs).
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNUALIZE_FACTOR      = 252.0     # trading days per year
BREAKOUT_SIGNAL_MIN   = 2.0       # signal > 2 => breakout confirmed
IV_RANK_SELL_VOL      = 80.0      # percentile above which: sell vol
IV_RANK_BUY_VOL       = 20.0      # percentile below which: buy vol
CONSOL_WINDOW         = 20        # bars for consolidation detection
ATR_PCT_CONSOL        = 50        # ATR percentile threshold for consolidation
BB_PCT_CONSOL         = 25        # BB width percentile for consolidation
GARCH_OMEGA_MIN       = 1e-8
GARCH_ALPHA_MAX       = 0.99
GARCH_BETA_MAX        = 0.99
GARCH_PERSIST_MAX     = 0.9998    # alpha + beta <= this for stationarity


# ---------------------------------------------------------------------------
# GARCHParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class GARCHParams:
    """
    Parameters for a GARCH(1,1) model:
      sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

    Attributes
    #--------
    omega:
        Constant term in variance equation (> 0).
    alpha:
        ARCH coefficient (news impact).  alpha + beta < 1 for stationarity.
    beta:
        GARCH coefficient (variance persistence).
    long_run_vol:
        Unconditional (long-run) annualized volatility =
        sqrt(omega / (1 - alpha - beta)) * sqrt(ANNUALIZE_FACTOR).
    log_likelihood:
        Negative log-likelihood achieved at the fitted parameters.
    """
    omega:          float
    alpha:          float
    beta:           float
    long_run_vol:   float
    log_likelihood: float = 0.0


# ---------------------------------------------------------------------------
# GARCHVolForecast
# ---------------------------------------------------------------------------

class GARCHVolForecast:
    """
    GARCH(1,1) volatility model fitted by Maximum Likelihood Estimation
    using Nelder-Mead simplex optimisation (scipy.optimize.minimize).

    Usage:
      params = GARCHVolForecast.fit(returns)
      fwd_vols = GARCHVolForecast.forecast(params, n_steps=5)
    """

    # ------------------------------------------------------------------
    # Static API
    # ------------------------------------------------------------------

    @staticmethod
    def fit(returns: np.ndarray) -> GARCHParams:
        """
        Fit GARCH(1,1) to a 1-D array of returns using MLE.

        Parameters
        #--------
        returns:
            Daily (or per-bar) returns as fractions, e.g. 0.01 = 1%.
            Requires at least 50 observations.

        Returns
        #-----
        GARCHParams with fitted omega, alpha, beta and derived long_run_vol.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 50:
            raise ValueError("GARCH fit requires at least 50 return observations")

        # Demean
        r = r - r.mean()

        # Initial variance estimate
        var0 = float(np.var(r))
        if var0 < 1e-12:
            raise ValueError("Returns have near-zero variance # cannot fit GARCH")

        # Starting parameters: omega small, alpha=0.10, beta=0.85
        init_omega = var0 * (1.0 - 0.10 - 0.85)
        init_omega = max(init_omega, GARCH_OMEGA_MIN)
        x0 = np.array([
            GARCHVolForecast._to_unconstrained(init_omega, 0.0, None),
            GARCHVolForecast._to_unconstrained(0.10, 0.0, 1.0),
            GARCHVolForecast._to_unconstrained(0.85, 0.0, 1.0),
        ])

        def neg_ll(params_unc: np.ndarray) -> float:
            try:
                omega = GARCHVolForecast._from_unconstrained(params_unc[0], 0.0, None)
                alpha = GARCHVolForecast._from_unconstrained(params_unc[1], 0.0, 1.0)
                beta  = GARCHVolForecast._from_unconstrained(params_unc[2], 0.0, 1.0)

                if alpha + beta >= GARCH_PERSIST_MAX:
                    return 1e10
                if omega <= 0:
                    return 1e10

                ll = GARCHVolForecast._log_likelihood(r, omega, alpha, beta, var0)
                return -ll
            except Exception:
                return 1e10

        result = minimize(
            neg_ll,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )

        omega = GARCHVolForecast._from_unconstrained(result.x[0], 0.0, None)
        alpha = GARCHVolForecast._from_unconstrained(result.x[1], 0.0, 1.0)
        beta  = GARCHVolForecast._from_unconstrained(result.x[2], 0.0, 1.0)

        # Clamp for safety
        omega = max(omega, GARCH_OMEGA_MIN)
        alpha = min(max(alpha, 0.0), GARCH_ALPHA_MAX)
        beta  = min(max(beta,  0.0), GARCH_BETA_MAX)
        if alpha + beta >= 1.0:
            beta = max(0.0, 0.9998 - alpha)

        persist  = alpha + beta
        lrv      = omega / max(1.0 - persist, 1e-8)
        long_run = math.sqrt(max(lrv, 0.0) * ANNUALIZE_FACTOR)

        return GARCHParams(
            omega          = omega,
            alpha          = alpha,
            beta           = beta,
            long_run_vol   = long_run,
            log_likelihood = float(-result.fun),
        )

    @staticmethod
    def forecast(params: GARCHParams, n_steps: int, last_return: float = 0.0) -> np.ndarray:
        """
        Multi-step variance forecast from GARCH(1,1).

        The forecast reverts toward the unconditional variance:
          h_{t+k} = LRV + (alpha + beta)^k * (h_t - LRV)

        Returns annualised volatility forecasts as 1-D array of length n_steps.
        """
        omega   = params.omega
        alpha   = params.alpha
        beta    = params.beta
        persist = alpha + beta

        lrv = omega / max(1.0 - persist, 1e-8)

        # Starting conditional variance using last observed return
        h0 = omega + alpha * last_return ** 2 + beta * lrv

        vols = np.empty(n_steps, dtype=float)
        for k in range(1, n_steps + 1):
            h_k = lrv + (persist ** k) * (h0 - lrv)
            h_k = max(h_k, GARCH_OMEGA_MIN)
            vols[k - 1] = math.sqrt(h_k * ANNUALIZE_FACTOR)

        return vols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_likelihood(
        r: np.ndarray,
        omega: float,
        alpha: float,
        beta:  float,
        h0:    float,
    ) -> float:
        """Gaussian log-likelihood for GARCH(1,1)."""
        n  = len(r)
        h  = h0
        ll = 0.0
        for i in range(n):
            ll += -0.5 * (math.log(2.0 * math.pi) + math.log(h + 1e-15) + r[i] ** 2 / (h + 1e-15))
            h   = omega + alpha * r[i] ** 2 + beta * h
            h   = max(h, GARCH_OMEGA_MIN)
        return ll

    @staticmethod
    def _to_unconstrained(x: float, lo: float, hi: Optional[float]) -> float:
        """Map constrained parameter to unconstrained via log / logit."""
        if hi is None:
            return math.log(max(x - lo, 1e-10))
        eps = 1e-6
        x_c = min(max(x, lo + eps), hi - eps)
        return math.log((x_c - lo) / (hi - x_c))

    @staticmethod
    def _from_unconstrained(u: float, lo: float, hi: Optional[float]) -> float:
        """Inverse of _to_unconstrained."""
        if hi is None:
            return math.exp(u) + lo
        return lo + (hi - lo) / (1.0 + math.exp(-u))


# ---------------------------------------------------------------------------
# VolatilityBreakoutStrategy
# ---------------------------------------------------------------------------

class VolatilityBreakoutStrategy:
    """
    Breakout strategy that activates after confirmed low-volatility consolidation.

    Consolidation is detected when both:
      - Current ATR < 50th percentile of rolling ATR history
      - Current BB width < 25th percentile of rolling BB width history

    Once consolidation is confirmed, a breakout is triggered when price
    moves more than 2 range half-widths from the consolidation midpoint.
    """

    def __init__(
        self,
        window:           int   = CONSOL_WINDOW,
        atr_pct:          float = ATR_PCT_CONSOL,
        bb_pct:           float = BB_PCT_CONSOL,
        breakout_thresh:  float = BREAKOUT_SIGNAL_MIN,
        bb_num_std:       float = 2.0,
        history_bars:     int   = 100,
    ):
        self._window           = window
        self._atr_pct          = atr_pct
        self._bb_pct           = bb_pct
        self._breakout_thresh  = breakout_thresh
        self._bb_num_std       = bb_num_std

        self._prices:    Deque[float] = deque(maxlen=max(window, history_bars))
        self._highs:     Deque[float] = deque(maxlen=max(window, history_bars))
        self._lows:      Deque[float] = deque(maxlen=max(window, history_bars))
        self._atrs:      Deque[float] = deque(maxlen=history_bars)
        self._bb_widths: Deque[float] = deque(maxlen=history_bars)

        # Consolidation tracking
        self._in_consolidation:  bool  = False
        self._consol_high:       float = 0.0
        self._consol_low:        float = float("inf")
        self._consol_bars:       int   = 0
        self._consol_start_bar:  int   = 0

        self._bar_index: int = 0
        self._prev_close: Optional[float] = None

    # ------------------------------------------------------------------

    def update(self, bar: dict) -> Tuple[float, bool]:
        """
        Feed one bar dict (close, high, low required).
        Returns (breakout_signal, in_consolidation).

        breakout_signal:
          0.0  => no breakout / no signal
          > 0  => bullish breakout (magnitude = strength)
          < 0  => bearish breakout
        """
        close = float(bar.get("close", 0.0))
        high  = float(bar.get("high",  close))
        low   = float(bar.get("low",   close))

        self._prices.append(close)
        self._highs.append(high)
        self._lows.append(low)
        self._bar_index += 1

        if len(self._prices) < self._window:
            return 0.0, False

        atr = self._compute_atr()
        bb_width = self._compute_bb_width()

        self._atrs.append(atr)
        self._bb_widths.append(bb_width)

        self._prev_close = close

        if len(self._atrs) < 10:
            return 0.0, False

        in_consol = self.detect_consolidation(
            atr_current  = atr,
            bb_width_current = bb_width,
        )

        if in_consol:
            # Update consolidation range
            self._consol_high = max(self._consol_high, high)
            self._consol_low  = min(self._consol_low, low)
            self._consol_bars += 1
            if not self._in_consolidation:
                self._in_consolidation = True
                self._consol_start_bar = self._bar_index
            return 0.0, True

        else:
            # Check for breakout if we were in consolidation
            if self._in_consolidation and self._consol_bars >= 3:
                consol_range = (self._consol_high, self._consol_low)
                sig = self.compute_breakout_signal(bar, consol_range)
            else:
                sig = 0.0

            # Reset consolidation state
            self._in_consolidation = False
            self._consol_high      = 0.0
            self._consol_low       = float("inf")
            self._consol_bars      = 0

            return sig, False

    def detect_consolidation(
        self,
        atr_current:      Optional[float] = None,
        bb_width_current: Optional[float] = None,
        bars:             Optional[List[dict]] = None,
        window:           int = CONSOL_WINDOW,
    ) -> bool:
        """
        Returns True if current market is in a low-volatility consolidation.
        Can be called standalone with a bars list, or internally with
        pre-computed atr/bb values.
        """
        # Standalone mode
        if bars is not None:
            for b in bars:
                self.update(b)
            return self._in_consolidation

        # Internal mode (atr and bb_width already computed)
        if atr_current is None or bb_width_current is None:
            return False
        if len(self._atrs) < 10:
            return False

        atr_50th = float(np.percentile(list(self._atrs), self._atr_pct))
        bb_50th  = float(np.percentile(list(self._bb_widths), self._bb_pct))

        return (atr_current < atr_50th) and (bb_width_current < bb_50th)

    def compute_breakout_signal(
        self,
        bar: dict,
        consolidation_range: Tuple[float, float],
    ) -> float:
        """
        Compute breakout signal magnitude and direction.

        Parameters
        #--------
        bar:
            Current bar with close price.
        consolidation_range:
            (range_high, range_low) of the consolidation zone.

        Returns
        #-----
        float: |price - range_mid| / range_half_width, signed by direction.
        Positive = bullish breakout, Negative = bearish.
        """
        close = float(bar.get("close", 0.0))
        r_high, r_low = consolidation_range
        range_mid  = (r_high + r_low) / 2.0
        half_width = (r_high - r_low) / 2.0

        if half_width < 1e-8:
            return 0.0

        raw = (close - range_mid) / half_width
        # Apply threshold gate: only emit if beyond breakout threshold
        if abs(raw) < self._breakout_thresh:
            return 0.0

        return float(np.clip(raw, -5.0, 5.0))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_atr(self) -> float:
        """Compute ATR over self._window bars using true range."""
        prices = list(self._prices)
        highs  = list(self._highs)
        lows   = list(self._lows)
        n      = min(self._window, len(prices))
        if n < 2:
            return 0.0

        trs = []
        for i in range(1, n):
            idx  = len(prices) - n + i
            h    = highs[idx]
            l    = lows[idx]
            c_p  = prices[idx - 1]
            tr   = max(h - l, abs(h - c_p), abs(l - c_p))
            trs.append(tr)

        return float(np.mean(trs)) if trs else 0.0

    def _compute_bb_width(self) -> float:
        """BB width = (upper - lower) / middle."""
        arr = np.array(list(self._prices)[-self._window:], dtype=float)
        if len(arr) < 2:
            return 0.0
        m   = arr.mean()
        s   = arr.std(ddof=1)
        mid = m if m > 1e-9 else 1e-9
        return (2.0 * 2.0 * s) / mid    # 2-sigma bands


# ---------------------------------------------------------------------------
# VolatilityArbitrageSignal
# ---------------------------------------------------------------------------

class VolatilityArbitrageSignal:
    """
    Vol-arb signal comparing implied volatility rank to realized volatility.

    When IV rank is elevated (> 80th pct): sell vol (negative signal).
    When IV rank is depressed (< 20th pct): buy vol (positive signal).
    Neutral otherwise.

    Note: implied vol data must be supplied externally (options feed).
    If not available, the signal degrades to realized-vol regime only.
    """

    def __init__(
        self,
        iv_sell_rank:  float = IV_RANK_SELL_VOL,
        iv_buy_rank:   float = IV_RANK_BUY_VOL,
        rv_window:     int   = 21,
        lookback:      int   = 252,
    ):
        self._iv_sell  = iv_sell_rank
        self._iv_buy   = iv_buy_rank
        self._rv_win   = rv_window
        self._lookback = lookback

        self._iv_history: Deque[float] = deque(maxlen=lookback)
        self._returns:    Deque[float] = deque(maxlen=max(rv_window, lookback))

    # ------------------------------------------------------------------

    def update_iv(self, current_iv: float) -> None:
        """Feed current implied volatility (annualized fraction, e.g. 0.25)."""
        self._iv_history.append(float(current_iv))

    def update_price(self, price: float) -> None:
        """Feed price for realized vol computation."""
        if self._returns.maxlen and len(self._returns) >= 1:
            prev = list(self._returns)[-1] if self._returns else price
        else:
            prev = price

        # Store prices in deque, compute return on each update
        if not hasattr(self, "_prices_buf"):
            self._prices_buf: Deque[float] = deque(maxlen=self._lookback + 10)
        self._prices_buf.append(float(price))
        if len(self._prices_buf) >= 2:
            r = math.log(self._prices_buf[-1] / max(self._prices_buf[-2], 1e-9))
            self._returns.append(r)

    def iv_rank(
        self,
        current_iv: float,
        hist_ivs: List[float],
        lookback: int = 252,
    ) -> float:
        """
        Compute IV percentile rank.

        Parameters
        #--------
        current_iv:
            Current implied volatility level.
        hist_ivs:
            Historical IV values over lookback period.
        lookback:
            How many historical IV values to use.

        Returns
        #-----
        float in [0, 100]: percentile rank of current_iv.
        """
        hist = np.array(hist_ivs[-lookback:], dtype=float)
        if len(hist) < 10:
            return 50.0
        rank = float(np.sum(hist < current_iv) / len(hist) * 100.0)
        return round(rank, 2)

    def realized_vol(self, returns: np.ndarray, window: int = 21) -> float:
        """
        Compute annualized realized volatility.

        Parameters
        #--------
        returns:
            Array of log returns.
        window:
            Rolling window (most recent n observations).

        Returns
        #-----
        Annualized vol as a decimal fraction (e.g. 0.20 = 20%).
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 2:
            return 0.0
        subset = r[-window:] if len(r) > window else r
        rv = float(np.std(subset, ddof=1)) * math.sqrt(ANNUALIZE_FACTOR)
        return rv

    def compute_signal(self, current_iv: Optional[float] = None) -> float:
        """
        Compute vol-arb signal.

        Returns +1 (long vol), -1 (short vol), 0 (neutral).
        If current_iv is None, uses last value fed via update_iv().
        """
        if current_iv is not None:
            self.update_iv(current_iv)

        if len(self._iv_history) < 20:
            return 0.0

        iv_now   = float(list(self._iv_history)[-1])
        hist_ivs = list(self._iv_history)
        rank     = self.iv_rank(iv_now, hist_ivs)

        if rank > self._iv_sell:
            return -1.0   # sell vol: IV is elevated, expect mean reversion
        if rank < self._iv_buy:
            return +1.0   # buy vol: IV is suppressed, expect expansion

        # Graduated signal in the middle zone
        if rank > 50.0:
            # Trending toward sell
            return float(-((rank - 50.0) / (self._iv_sell - 50.0)) * 0.5)
        else:
            # Trending toward buy
            return float(((50.0 - rank) / (50.0 - self._iv_buy)) * 0.5)

    def get_realized_vol_now(self) -> float:
        """Return current realized vol using buffered returns."""
        if len(self._returns) < 5:
            return 0.0
        return self.realized_vol(np.array(list(self._returns)), window=self._rv_win)
