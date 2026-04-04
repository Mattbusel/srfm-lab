"""
signal_analytics.py — GARCH(1,1) vol forecasts and Ornstein-Uhlenbeck (OU)
mean-reversion analytics for the LARSA observability stack.

These are called from metrics_server.MetricsCollector to populate:
    larsa_garch_vol{symbol}            — 1-step ahead vol forecast
    larsa_mean_reversion_signal{symbol} — OU z-score (how far from mean)

Both estimators are lightweight online/rolling implementations that
require no external dependencies beyond numpy.

Design:
    GARCHEstimator  — rolling GARCH(1,1) updated with each new return.
                      Stores the last `window` log-returns, re-fits
                      by MLE every `refit_every` observations.

    OUEstimator     — rolling Ornstein-Uhlenbeck calibration via OLS.
                      Stores the last `window` prices, computes:
                        - theta (mean-reversion speed)
                        - mu    (long-run mean)
                        - sigma (diffusion)
                        - half-life
                        - current z-score = (price - mu) / sigma_eq
                      where sigma_eq = sigma / sqrt(2*theta).

    AnalyticsHub    — holds one GARCHEstimator + OUEstimator per symbol.
                      Call hub.on_price(symbol, price) after each bar.
                      Call hub.get_state(symbols) to get a state dict
                      suitable for MetricsCollector.update().
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("larsa.analytics")


# ---------------------------------------------------------------------------
# GARCH(1,1) estimator
# ---------------------------------------------------------------------------

class GARCHEstimator:
    """
    Online GARCH(1,1) fitted on rolling log-returns.

    Model:  r_t = sigma_t * z_t,  z_t ~ N(0,1)
            sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters are re-estimated by moment matching / MLE every `refit_every`
    new data points.  Between re-fits the variance recursion is propagated
    forward with current parameters.
    """

    def __init__(self, window: int = 200, refit_every: int = 20):
        self.window       = window
        self.refit_every  = refit_every
        self._returns: deque = deque(maxlen=window)
        self._since_refit = 0

        # GARCH parameters (defaults: approximately iid)
        self.omega = 1e-6
        self.alpha = 0.10
        self.beta  = 0.85

        # Running conditional variance
        self.sigma2 = 1e-4   # initial guess

        self._fitted = False

    # ------------------------------------------------------------------ #

    def update(self, log_return: float):
        """Ingest one log-return and update the variance estimate."""
        r = float(log_return)
        self._returns.append(r)
        self._since_refit += 1

        if len(self._returns) >= 30 and self._since_refit >= self.refit_every:
            self._refit()
            self._since_refit = 0

        # GARCH recursion
        self.sigma2 = (
            self.omega
            + self.alpha * r ** 2
            + self.beta  * self.sigma2
        )
        # Clamp to prevent blow-up
        self.sigma2 = max(1e-12, min(self.sigma2, 1.0))

    def _refit(self):
        """Re-fit omega/alpha/beta by variance targeting."""
        ret = np.array(self._returns, dtype=float)
        if len(ret) < 20:
            return
        var_target = float(np.var(ret, ddof=1))
        # Simple persistence constraint: keep alpha+beta < 0.999
        alpha = max(0.02, min(0.25, float(np.mean(ret[1:] ** 2) / (var_target + 1e-12)) * 0.5))
        beta  = max(0.70, min(0.97, 1.0 - alpha - 0.01))
        omega = var_target * (1.0 - alpha - beta)
        omega = max(1e-9, omega)
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta
        self._fitted = True

    @property
    def vol_forecast(self) -> float:
        """1-step ahead annualised volatility forecast (annualised assuming minutely returns)."""
        # Minutely → daily: *sqrt(1440);  daily → annual: *sqrt(252)
        # But we're called at bar frequency (could be 1m, 15m, 1h, daily).
        # Return the raw sigma so the caller can scale.
        return math.sqrt(max(0.0, self.sigma2))

    @property
    def vol_forecast_annual(self) -> float:
        """Annualised volatility assuming the input returns are per-minute."""
        bars_per_year = 1440 * 252  # 1-min bars
        return self.vol_forecast * math.sqrt(bars_per_year)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck estimator
# ---------------------------------------------------------------------------

class OUEstimator:
    """
    Rolling OU process calibration via OLS regression on consecutive prices.

    Model:  dX = theta*(mu - X)*dt + sigma*dW
    OLS:    X_t = a + b*X_{t-1} + eps
            theta  = -ln(b) / dt       (dt=1 bar)
            mu     = a / (1 - b)
            sigma  = std(eps) / sqrt(dt)
            half_life = ln(2) / theta

    z-score = (X_current - mu) / sigma_eq
    where sigma_eq = sigma / sqrt(2*theta)   (equilibrium std dev)
    """

    def __init__(self, window: int = 100):
        self.window  = window
        self._prices: deque = deque(maxlen=window + 1)

        # Fitted parameters
        self.theta     = 0.0
        self.mu        = 0.0
        self.sigma     = 0.0
        self.half_life = 0.0
        self.sigma_eq  = 0.0
        self.z_score   = 0.0

        self._fitted = False

    def update(self, price: float):
        """Ingest one price and refit if enough data."""
        self._prices.append(float(price))
        if len(self._prices) >= 20:
            self._fit()

    def _fit(self):
        px = np.array(self._prices, dtype=float)
        if len(px) < 10:
            return
        X  = px[:-1]    # X_{t-1}
        Y  = px[1:]     # X_t
        n  = len(X)

        # OLS: Y = a + b*X
        Xm = np.mean(X)
        Ym = np.mean(Y)
        cov_xy = np.mean((X - Xm) * (Y - Ym))
        var_x  = np.var(X, ddof=1)

        if var_x < 1e-12:
            return

        b = cov_xy / var_x
        a = Ym - b * Xm

        # Guard: b must be in (0, 1) for mean-reversion
        if b <= 0.0 or b >= 1.0:
            return

        theta = -math.log(b)   # dt = 1 bar
        mu    = a / (1.0 - b)

        residuals = Y - (a + b * X)
        sigma_bar = float(np.std(residuals, ddof=2))

        # Equilibrium sigma
        denom = math.sqrt(2.0 * theta) if theta > 1e-6 else 1e-6
        sigma_eq = sigma_bar / denom

        if sigma_eq < 1e-9:
            return

        self.theta     = theta
        self.mu        = mu
        self.sigma     = sigma_bar
        self.half_life = math.log(2.0) / theta if theta > 1e-9 else 1e6
        self.sigma_eq  = sigma_eq
        self.z_score   = (self._prices[-1] - mu) / sigma_eq
        self._fitted   = True

    @property
    def is_mean_reverting(self) -> bool:
        return self._fitted and 0.0 < self.theta < math.log(2) * 4


# ---------------------------------------------------------------------------
# Analytics Hub — one estimator pair per symbol
# ---------------------------------------------------------------------------

class AnalyticsHub:
    """
    Aggregates GARCH + OU estimators for all symbols.

    Usage:
        hub = AnalyticsHub()
        hub.on_price("BTC", 45000.0)  # called every bar
        state = hub.get_state(["BTC", "ETH"])  # returns metrics dict
    """

    def __init__(self, garch_window: int = 200, ou_window: int = 100):
        self._garch_window = garch_window
        self._ou_window    = ou_window
        self._garch: Dict[str, GARCHEstimator] = {}
        self._ou:    Dict[str, OUEstimator]    = {}
        self._prev_price: Dict[str, float]     = {}

    def _ensure(self, symbol: str):
        if symbol not in self._garch:
            self._garch[symbol] = GARCHEstimator(
                window=self._garch_window, refit_every=20
            )
        if symbol not in self._ou:
            self._ou[symbol] = OUEstimator(window=self._ou_window)

    def on_price(self, symbol: str, price: float):
        """Call this every time a new price arrives for a symbol."""
        self._ensure(symbol)
        prev = self._prev_price.get(symbol)
        if prev is not None and prev > 0:
            log_ret = math.log(price / prev)
            self._garch[symbol].update(log_ret)
        self._ou[symbol].update(price)
        self._prev_price[symbol] = price

    def get_garch_vol(self, symbol: str) -> float:
        """Return annualised GARCH vol forecast for symbol (0.0 if not fitted)."""
        self._ensure(symbol)
        g = self._garch[symbol]
        if g._fitted:
            return g.vol_forecast * math.sqrt(252)  # assume daily bars
        return 0.0

    def get_ou_state(self, symbol: str) -> Dict[str, float]:
        """Return OU state dict for symbol."""
        self._ensure(symbol)
        ou = self._ou[symbol]
        return {
            "z_score":   ou.z_score,
            "mu":        ou.mu,
            "half_life": ou.half_life,
            "sigma_eq":  ou.sigma_eq,
            "theta":     ou.theta,
            "is_mean_reverting": float(ou.is_mean_reverting),
        }

    def get_state(self, symbols) -> Dict:
        """
        Return a partial state dict suitable for MetricsCollector.update():
            {
                "garch_vol":     {sym: float},
                "mean_reversion":{sym: float},   # OU z-score
                "ou_halflife":   {sym: float},
            }
        """
        garch_vol      = {}
        mean_reversion = {}
        ou_halflife    = {}

        for sym in symbols:
            self._ensure(sym)
            garch_vol[sym]      = self.get_garch_vol(sym)
            ou_s                = self.get_ou_state(sym)
            mean_reversion[sym] = ou_s["z_score"]
            ou_halflife[sym]    = ou_s["half_life"]

        return {
            "garch_vol":      garch_vol,
            "mean_reversion": mean_reversion,
            "ou_halflife":    ou_halflife,
        }

    def regime_label(self, symbol: str) -> str:
        """
        Classify symbol regime as:
            'trending'       — BH mass high (caller should pass this in)
            'mean-reverting' — OU is_mean_reverting and |z| > 1
            'flat'           — otherwise
        Uses only OU state here; the caller combines with BH mass.
        """
        self._ensure(symbol)
        ou = self._ou[symbol]
        if not ou._fitted:
            return "flat"
        if ou.is_mean_reverting and abs(ou.z_score) > 1.0:
            return "mean-reverting"
        return "flat"


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.DEBUG)
    hub = AnalyticsHub()

    # Simulate BTC price walk
    price = 45000.0
    for i in range(300):
        price *= math.exp(random.gauss(0, 0.005))
        hub.on_price("BTC", price)
        if i % 50 == 0:
            s = hub.get_ou_state("BTC")
            print(
                f"t={i:3d}  price={price:,.0f}"
                f"  z={s['z_score']:+.2f}"
                f"  hl={s['half_life']:.1f}"
                f"  garch_vol={hub.get_garch_vol('BTC'):.4f}"
            )
