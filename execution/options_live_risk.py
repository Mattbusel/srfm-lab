"""
execution/options_live_risk.py -- Real-time options portfolio risk monitoring.

Integrates with lib/options/ (greeks, pricing, vol_surface) to provide:
  - Per-position and portfolio-level Greeks
  - Vol surface cache with SVI model and flat-vol fallback
  - Delta-hedging signals
  - Gamma scalping signals
  - Vega P&L estimation
  - Scenario P&L matrix (+/-5%, +/-10% spot, +/-2 vol points)
  - FastAPI endpoints for live monitoring

Classes
-------
  OptionsPosition    : dataclass for a live options position
  LiveGreeks         : dataclass for aggregated Greeks (position or portfolio)
  VolSurfaceCache    : per-symbol SVI surface with async refresh and fallback
  OptionsRiskMonitor : main risk engine
  OptionsRiskAPI     : FastAPI router (GET/POST/DELETE endpoints)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- FastAPI / lib.options gracefully degraded if missing
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel as _PydanticBase
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed -- OptionsRiskAPI will not be available.")

try:
    from lib.options.greeks import GreeksResult, GreeksEngine
    from lib.options.pricing import BlackScholes
    from lib.options.vol_surface import SVIModel, SVIParams, VolSurface
    _OPTIONS_LIB_AVAILABLE = True
except ImportError:
    _OPTIONS_LIB_AVAILABLE = False
    logger.warning("lib.options not importable -- falling back to analytic stubs.")


# ---------------------------------------------------------------------------
# Fallback stubs when lib.options is unavailable
# ---------------------------------------------------------------------------

if not _OPTIONS_LIB_AVAILABLE:
    @dataclass
    class GreeksResult:  # type: ignore[no-redef]
        delta: float = 0.0
        gamma: float = 0.0
        vega: float = 0.0
        theta: float = 0.0
        rho: float = 0.0
        vanna: float = 0.0
        volga: float = 0.0
        charm: float = 0.0
        speed: float = 0.0
        color: float = 0.0
        ultima: float = 0.0
        zomma: float = 0.0

    class BlackScholes:  # type: ignore[no-redef]
        def __init__(self, S, K, T, r, sigma, q=0.0, option_type="call"):
            self.S = S; self.K = K; self.T = max(T, 1e-10)
            self.r = r; self.sigma = sigma; self.q = q
            self.option_type = option_type.lower()

        def price(self) -> float:
            from scipy.stats import norm
            d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
            d2 = d1 - self.sigma * math.sqrt(self.T)
            if self.option_type == "call":
                return self.S * math.exp(-self.q * self.T) * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
            return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)

        def greeks(self) -> "GreeksResult":
            from scipy.stats import norm
            d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
            d2 = d1 - self.sigma * math.sqrt(self.T)
            nd1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
            if self.option_type == "call":
                delta = math.exp(-self.q * self.T) * norm.cdf(d1)
            else:
                delta = -math.exp(-self.q * self.T) * norm.cdf(-d1)
            gamma = math.exp(-self.q * self.T) * nd1 / (self.S * self.sigma * math.sqrt(self.T))
            vega = self.S * math.exp(-self.q * self.T) * nd1 * math.sqrt(self.T) / 100.0
            theta = -(self.S * math.exp(-self.q * self.T) * nd1 * self.sigma / (2 * math.sqrt(self.T)) +
                      self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2 if self.option_type == "call" else -d2)
                      * (1 if self.option_type == "call" else -1)) / 365.0
            rho = (self.K * self.T * math.exp(-self.r * self.T) *
                   (norm.cdf(d2) if self.option_type == "call" else -norm.cdf(-d2))) / 100.0
            vanna = -math.exp(-self.q * self.T) * nd1 * d2 / self.sigma
            volga = self.S * math.exp(-self.q * self.T) * nd1 * math.sqrt(self.T) * d1 * d2 / self.sigma
            return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta,
                                rho=rho, vanna=vanna, volga=volga)


# ---------------------------------------------------------------------------
# OptionsPosition
# ---------------------------------------------------------------------------

@dataclass
class OptionsPosition:
    """
    A single live options position.

    Fields
    ------
    symbol      : str   -- underlying ticker (e.g. "AAPL")
    expiry      : str   -- expiry date string "YYYY-MM-DD"
    strike      : float -- option strike price
    right       : str   -- "call" or "put"
    qty         : float -- signed quantity (positive = long)
    entry_price : float -- price paid/received per contract
    entry_time  : float -- unix timestamp of position open
    spot        : float -- underlying spot price at time of last update
    sigma       : float -- implied vol used in most recent Greeks calculation
    r           : float -- risk-free rate
    q           : float -- continuous dividend yield
    multiplier  : float -- contract multiplier (default 100 for equity)
    """
    symbol: str
    expiry: str
    strike: float
    right: str         # "call" or "put"
    qty: float
    entry_price: float
    entry_time: float = field(default_factory=time.time)
    spot: float = 100.0
    sigma: float = 0.20
    r: float = 0.05
    q: float = 0.0
    multiplier: float = 100.0

    def time_to_expiry(self, as_of: Optional[float] = None) -> float:
        """
        Compute T (years) from now (or as_of timestamp) to expiry.
        Falls back to 0.0 if expiry is in the past.
        """
        try:
            from datetime import timezone
            exp_dt = datetime.strptime(self.expiry, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            now_ts = as_of if as_of else time.time()
            now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
            delta_secs = (exp_dt - now_dt).total_seconds()
            return max(delta_secs / (365.25 * 86400), 1e-6)
        except Exception:
            return 0.0833  # default 1 month

    @property
    def position_id(self) -> str:
        return f"{self.symbol}_{self.expiry}_{self.strike}_{self.right}"

    def current_greeks(self, spot: Optional[float] = None, sigma: Optional[float] = None) -> GreeksResult:
        """Compute Black-Scholes Greeks for this position at current market state."""
        S = spot if spot is not None else self.spot
        iv = sigma if sigma is not None else self.sigma
        T = self.time_to_expiry()
        try:
            bs = BlackScholes(S, self.strike, T, self.r, iv, self.q, self.right)
            return bs.greeks()
        except Exception as exc:
            logger.debug("Greeks computation failed for %s: %s", self.position_id, exc)
            return GreeksResult()

    def current_price(self, spot: Optional[float] = None, sigma: Optional[float] = None) -> float:
        """Theoretical price at current market state."""
        S = spot if spot is not None else self.spot
        iv = sigma if sigma is not None else self.sigma
        T = self.time_to_expiry()
        try:
            bs = BlackScholes(S, self.strike, T, self.r, iv, self.q, self.right)
            return bs.price()
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# LiveGreeks
# ---------------------------------------------------------------------------

@dataclass
class LiveGreeks:
    """
    Aggregated Greeks for a position or portfolio.

    Signed quantities are taken into account: a short call has negative delta.
    All values are in dollar terms (multiplied by qty * multiplier).
    """
    delta: float = 0.0      # dV/dS -- dollar delta
    gamma: float = 0.0      # d2V/dS2
    vega: float = 0.0       # dV/d_sigma (per 1% vol move)
    theta: float = 0.0      # dV/dt (per calendar day)
    rho: float = 0.0        # dV/dr (per 1% rate move)
    vanna: float = 0.0      # d2V/(dS d_sigma)
    volga: float = 0.0      # d2V/d_sigma2 (vomma)
    n_positions: int = 0
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, float]:
        return {
            "delta": round(self.delta, 6),
            "gamma": round(self.gamma, 6),
            "vega": round(self.vega, 6),
            "theta": round(self.theta, 6),
            "rho": round(self.rho, 6),
            "vanna": round(self.vanna, 6),
            "volga": round(self.volga, 6),
            "n_positions": self.n_positions,
            "timestamp": self.timestamp,
        }

    def __add__(self, other: "LiveGreeks") -> "LiveGreeks":
        return LiveGreeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            n_positions=self.n_positions + other.n_positions,
        )


# ---------------------------------------------------------------------------
# VolSurfaceCache
# ---------------------------------------------------------------------------

class VolSurfaceCache:
    """
    Per-symbol volatility surface cache.

    Stores a SVI-parameterized vol surface per (symbol, expiry) pair.
    Refreshes every `refresh_interval` seconds when `fetch_fn` is provided.
    Falls back to historical realized vol (flat vol) on fetch failure.

    Parameters
    ----------
    refresh_interval : float -- seconds between refreshes (default 300 = 5 min)
    fallback_vol     : float -- flat vol used on failure (default 0.20)
    fetch_fn         : optional async callable(symbol) -> dict of surface params
    """

    DEFAULT_REFRESH = 300.0
    DEFAULT_FALLBACK_VOL = 0.20

    def __init__(
        self,
        refresh_interval: float = DEFAULT_REFRESH,
        fallback_vol: float = DEFAULT_FALLBACK_VOL,
        fetch_fn=None,
    ) -> None:
        self._refresh_interval = refresh_interval
        self._fallback_vol = fallback_vol
        self._fetch_fn = fetch_fn
        self._lock = asyncio.Lock()

        # symbol -> {"last_refresh": float, "surface": dict, "realized_vol": float, "svi_params": dict}
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Historical realized vol buffer per symbol
        self._realized_vol_buf: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=252))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_vol(
        self,
        symbol: str,
        strike: float,
        expiry_t: float,
        spot: float,
    ) -> float:
        """
        Get implied vol for the given strike/expiry.

        Uses SVI surface if available and fresh, else falls back to flat vol.
        Refreshes surface in background if stale.

        Parameters
        ----------
        symbol   : str
        strike   : float
        expiry_t : float -- time to expiry in years
        spot     : float -- current spot price

        Returns
        -------
        float -- implied vol (annualised decimal)
        """
        async with self._lock:
            entry = self._cache.get(symbol)
            now = time.time()
            needs_refresh = (
                entry is None
                or (now - entry.get("last_refresh", 0.0)) > self._refresh_interval
            )

        if needs_refresh and self._fetch_fn is not None:
            await self._refresh_surface(symbol, spot)

        async with self._lock:
            entry = self._cache.get(symbol)
            if entry is None:
                return self._fallback_vol

            # Try SVI surface
            svi_params = entry.get("svi_params")
            if svi_params and expiry_t > 1e-6:
                try:
                    vol = self._svi_vol(svi_params, spot, strike, expiry_t)
                    if 0.01 <= vol <= 5.0:
                        return vol
                except Exception as exc:
                    logger.debug("SVI vol lookup failed for %s: %s", symbol, exc)

            # Fall back to realized vol
            rv = entry.get("realized_vol", self._fallback_vol)
            return float(rv) if rv and rv > 0 else self._fallback_vol

    def update_realized_vol(self, symbol: str, log_return: float) -> None:
        """Feed a daily log-return for realized vol estimation."""
        buf = self._realized_vol_buf[symbol]
        buf.append(log_return)
        if len(buf) >= 5:
            ann_vol = float(np.std(list(buf)) * math.sqrt(252))
            async_entry = self._cache.setdefault(symbol, {})
            async_entry["realized_vol"] = ann_vol

    def set_svi_params(self, symbol: str, svi_params: Dict[str, float]) -> None:
        """Directly inject SVI params (e.g. from broker feed)."""
        entry = self._cache.setdefault(symbol, {})
        entry["svi_params"] = svi_params
        entry["last_refresh"] = time.time()

    def set_fallback_vol(self, symbol: str, vol: float) -> None:
        """Override per-symbol fallback vol."""
        entry = self._cache.setdefault(symbol, {})
        entry["realized_vol"] = vol

    def clear(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()

    def is_cached(self, symbol: str) -> bool:
        return symbol in self._cache

    def cache_age(self, symbol: str) -> float:
        """Seconds since last refresh for symbol, or inf if never cached."""
        entry = self._cache.get(symbol)
        if entry is None:
            return float("inf")
        return time.time() - entry.get("last_refresh", 0.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _refresh_surface(self, symbol: str, spot: float) -> None:
        """Fetch and store updated surface params."""
        try:
            result = await self._fetch_fn(symbol, spot)
            if result and isinstance(result, dict):
                async with self._lock:
                    entry = self._cache.setdefault(symbol, {})
                    entry.update(result)
                    entry["last_refresh"] = time.time()
                logger.debug("Vol surface refreshed for %s.", symbol)
        except Exception as exc:
            logger.warning("Vol surface fetch failed for %s: %s", symbol, exc)
            # Ensure entry exists with fallback
            async with self._lock:
                entry = self._cache.setdefault(symbol, {})
                if "realized_vol" not in entry:
                    entry["realized_vol"] = self._fallback_vol
                entry["last_refresh"] = time.time()  # avoid hammering on failure

    @staticmethod
    def _svi_vol(
        params: Dict[str, float],
        spot: float,
        strike: float,
        T: float,
    ) -> float:
        """
        Compute SVI implied vol from raw SVI parameters.

        SVI total variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        where k = log(K/F) is the log-moneyness.
        """
        a = params.get("a", 0.04)
        b = params.get("b", 0.10)
        rho = params.get("rho", -0.3)
        m = params.get("m", 0.0)
        sigma = params.get("sigma", 0.10)

        F = spot  # approximate forward with spot (ignoring carry for simplicity)
        if F <= 0 or strike <= 0 or T <= 0:
            return 0.20

        k = math.log(strike / F)
        inner = math.sqrt((k - m) ** 2 + sigma ** 2)
        w = a + b * (rho * (k - m) + inner)
        w = max(w, 1e-8)
        return math.sqrt(w / T)


# ---------------------------------------------------------------------------
# OptionsRiskMonitor
# ---------------------------------------------------------------------------

class OptionsRiskMonitor:
    """
    Real-time options portfolio risk monitor.

    Maintains a list of OptionsPosition objects, computes aggregate Greeks,
    and emits hedging/scalping signals when risk thresholds are breached.

    Parameters
    ----------
    delta_limit     : float -- max |portfolio_delta / notional| before hedge signal
    gamma_threshold : float -- max |gamma * spot^2 * 0.01^2| before scalp flag
    vol_surface_cache : VolSurfaceCache -- shared cache (created internally if None)
    notional        : float -- reference notional for delta limit normalisation
    """

    DEFAULT_DELTA_LIMIT = 0.10
    DEFAULT_GAMMA_THRESHOLD = 0.005
    DEFAULT_NOTIONAL = 1_000_000.0

    # Scenario grid: (spot_pct, vol_shift)
    SCENARIO_SPOTS = (-0.10, -0.05, 0.0, +0.05, +0.10)
    SCENARIO_VOLS = (-0.02, 0.0, +0.02)

    def __init__(
        self,
        delta_limit: float = DEFAULT_DELTA_LIMIT,
        gamma_threshold: float = DEFAULT_GAMMA_THRESHOLD,
        notional: float = DEFAULT_NOTIONAL,
        vol_surface_cache: Optional[VolSurfaceCache] = None,
    ) -> None:
        self._delta_limit = delta_limit
        self._gamma_threshold = gamma_threshold
        self._notional = notional
        self._vol_cache = vol_surface_cache or VolSurfaceCache()
        self._lock = asyncio.Lock()

        # Positions keyed by position_id
        self._positions: Dict[str, OptionsPosition] = {}

        # Pending hedge signals: list of dicts
        self._hedge_signals: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Last computed portfolio Greeks
        self._last_greeks: Optional[LiveGreeks] = None

        # Scenario P&L cache
        self._last_scenarios: Optional[Dict[str, Any]] = None

        # Per-symbol spot prices (last known)
        self._spot_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def add_position(self, pos: OptionsPosition) -> None:
        """Thread-safely add a position to the portfolio."""
        async with self._lock:
            self._positions[pos.position_id] = pos
            if pos.spot > 0:
                self._spot_prices[pos.symbol] = pos.spot
            logger.info("Added position: %s qty=%.0f", pos.position_id, pos.qty)

    async def remove_position(
        self, symbol: str, expiry: str, strike: float, right: Optional[str] = None
    ) -> bool:
        """
        Remove a position by symbol/expiry/strike (and optionally right).
        Returns True if found and removed, False otherwise.
        """
        async with self._lock:
            keys_to_remove = []
            for pid, pos in self._positions.items():
                if pos.symbol == symbol and pos.expiry == expiry and pos.strike == strike:
                    if right is None or pos.right.lower() == right.lower():
                        keys_to_remove.append(pid)
            for k in keys_to_remove:
                del self._positions[k]
                logger.info("Removed position: %s", k)
            return len(keys_to_remove) > 0

    async def update_spot(self, symbol: str, spot: float) -> None:
        """Update the spot price for a symbol."""
        async with self._lock:
            self._spot_prices[symbol] = spot
            for pos in self._positions.values():
                if pos.symbol == symbol:
                    pos.spot = spot

    # ------------------------------------------------------------------
    # Greeks computation
    # ------------------------------------------------------------------

    async def compute_portfolio_greeks(
        self, spot_prices: Optional[Dict[str, float]] = None
    ) -> LiveGreeks:
        """
        Aggregate Greeks across all positions.

        Parameters
        ----------
        spot_prices : optional dict symbol->spot (overrides stored prices)

        Returns
        -------
        LiveGreeks -- dollar-weighted portfolio Greeks
        """
        async with self._lock:
            positions = list(self._positions.values())
            spots = dict(self._spot_prices)

        if spot_prices:
            spots.update(spot_prices)

        portfolio = LiveGreeks()

        for pos in positions:
            spot = spots.get(pos.symbol, pos.spot)
            pos.spot = spot

            # Fetch vol from cache
            T = pos.time_to_expiry()
            try:
                iv = await self._vol_cache.get_vol(pos.symbol, pos.strike, T, spot)
            except Exception:
                iv = pos.sigma

            pos.sigma = iv
            g = pos.current_greeks(spot=spot, sigma=iv)
            scale = pos.qty * pos.multiplier

            portfolio.delta += g.delta * scale
            portfolio.gamma += g.gamma * scale
            portfolio.vega += g.vega * scale
            portfolio.theta += g.theta * scale
            portfolio.rho += g.rho * scale
            portfolio.vanna += (g.vanna if hasattr(g, "vanna") else 0.0) * scale
            portfolio.volga += (g.volga if hasattr(g, "volga") else 0.0) * scale
            portfolio.n_positions += 1

        portfolio.timestamp = time.time()
        self._last_greeks = portfolio

        # Check hedge threshold
        await self._check_hedge_signal(portfolio, spots)

        return portfolio

    # ------------------------------------------------------------------
    # Hedge and scalp signals
    # ------------------------------------------------------------------

    async def _check_hedge_signal(
        self, greeks: LiveGreeks, spots: Dict[str, float]
    ) -> None:
        """Emit a hedge signal if |portfolio delta / notional| > delta_limit."""
        norm_delta = abs(greeks.delta) / max(self._notional, 1.0)
        if norm_delta > self._delta_limit:
            signal = {
                "type": "delta_hedge",
                "portfolio_delta": round(greeks.delta, 4),
                "norm_delta": round(norm_delta, 6),
                "delta_limit": self._delta_limit,
                "hedge_qty": round(-greeks.delta, 4),
                "timestamp": time.time(),
            }
            # Avoid duplicate signals within 60s
            if self._hedge_signals:
                last = self._hedge_signals[-1]
                if time.time() - last["timestamp"] < 60:
                    return
            async with self._lock:
                self._hedge_signals.append(signal)
            logger.info(
                "Delta hedge signal: portfolio_delta=%.4f (limit=%.4f)",
                greeks.delta, self._delta_limit * self._notional,
            )

    def check_gamma_scalp(
        self, spot: float, greeks: Optional[LiveGreeks] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if portfolio gamma warrants scalping.

        Dollar gamma risk = |gamma * spot^2 * (0.01)^2|
        (P&L impact of a 1% spot move squared)

        Returns a signal dict or None.
        """
        g = greeks or self._last_greeks
        if g is None:
            return None
        dollar_gamma = abs(g.gamma * spot ** 2 * 0.0001)  # 0.01^2 = 0.0001
        if dollar_gamma > self._gamma_threshold:
            return {
                "type": "gamma_scalp",
                "portfolio_gamma": round(g.gamma, 8),
                "dollar_gamma_risk": round(dollar_gamma, 4),
                "threshold": self._gamma_threshold,
                "spot": spot,
                "timestamp": time.time(),
            }
        return None

    def estimate_vega_pnl(self, vol_change: float, greeks: Optional[LiveGreeks] = None) -> float:
        """
        Estimate P&L from a vol change.

        P&L ~= vega * vol_change + 0.5 * volga * vol_change^2
        vega is per 1% vol move (i.e. divided by 100 in BSM convention).

        Parameters
        ----------
        vol_change : float -- change in implied vol (decimal, e.g. 0.02 for +2 vol pts)

        Returns
        -------
        float -- estimated vega P&L
        """
        g = greeks or self._last_greeks
        if g is None:
            return 0.0
        linear = g.vega * vol_change * 100  # vega stored per 1% = per 0.01 vol
        convexity = 0.5 * g.volga * (vol_change ** 2)
        return float(linear + convexity)

    # ------------------------------------------------------------------
    # Scenario matrix
    # ------------------------------------------------------------------

    async def compute_scenario_matrix(
        self, spot_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute P&L matrix for all positions under spot and vol shocks.

        Spot shocks  : -10%, -5%, 0, +5%, +10%
        Vol shocks   : -2 vol pts, 0, +2 vol pts (-0.02, 0, +0.02)

        Returns
        -------
        dict with keys:
          "scenarios": list of scenario dicts
          "positions": list of per-position scenario P&Ls
          "timestamp": float
        """
        async with self._lock:
            positions = list(self._positions.values())
            base_spots = dict(self._spot_prices)

        if spot_prices:
            base_spots.update(spot_prices)

        scenarios = []
        for spot_pct in self.SCENARIO_SPOTS:
            for vol_shift in self.SCENARIO_VOLS:
                scenario_pnl = 0.0
                pos_pnls = {}

                for pos in positions:
                    base_spot = base_spots.get(pos.symbol, pos.spot)
                    shocked_spot = base_spot * (1.0 + spot_pct)
                    T = pos.time_to_expiry()

                    try:
                        base_iv = await self._vol_cache.get_vol(
                            pos.symbol, pos.strike, T, base_spot
                        )
                    except Exception:
                        base_iv = pos.sigma

                    shocked_iv = max(0.01, base_iv + vol_shift)

                    # Base price
                    try:
                        base_price = BlackScholes(
                            base_spot, pos.strike, T, pos.r, base_iv, pos.q, pos.right
                        ).price()
                    except Exception:
                        base_price = pos.entry_price

                    # Shocked price
                    try:
                        shocked_price = BlackScholes(
                            shocked_spot, pos.strike, T, pos.r, shocked_iv, pos.q, pos.right
                        ).price()
                    except Exception:
                        shocked_price = base_price

                    pnl = (shocked_price - base_price) * pos.qty * pos.multiplier
                    scenario_pnl += pnl
                    pos_pnls[pos.position_id] = round(pnl, 4)

                scenarios.append({
                    "spot_shift_pct": spot_pct,
                    "vol_shift": vol_shift,
                    "portfolio_pnl": round(scenario_pnl, 4),
                    "position_pnls": pos_pnls,
                })

        result = {
            "scenarios": scenarios,
            "n_positions": len(positions),
            "timestamp": time.time(),
        }
        self._last_scenarios = result
        return result

    # ------------------------------------------------------------------
    # Snapshot and reporting
    # ------------------------------------------------------------------

    async def get_risk_snapshot(
        self, spot_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Full risk snapshot: Greeks + scenario P&L matrix.

        Returns
        -------
        dict with keys: greeks, scenarios, hedge_signals, timestamp
        """
        greeks = await self.compute_portfolio_greeks(spot_prices)
        scenarios = await self.compute_scenario_matrix(spot_prices)

        async with self._lock:
            hedge_signals = list(self._hedge_signals)

        return {
            "greeks": greeks.as_dict(),
            "scenarios": scenarios["scenarios"],
            "hedge_signals": hedge_signals[-10:],  # last 10 signals
            "n_positions": greeks.n_positions,
            "timestamp": time.time(),
        }

    def get_pending_hedge_signals(self) -> List[Dict[str, Any]]:
        """Returns pending delta-hedge signals (not yet acted on)."""
        return list(self._hedge_signals)

    def clear_hedge_signals(self) -> None:
        self._hedge_signals.clear()

    @property
    def positions(self) -> Dict[str, OptionsPosition]:
        return dict(self._positions)

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    @property
    def vol_cache(self) -> VolSurfaceCache:
        return self._vol_cache


# ---------------------------------------------------------------------------
# OptionsRiskAPI -- FastAPI integration
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class _PositionRequest(_PydanticBase):
        symbol: str
        expiry: str
        strike: float
        right: str
        qty: float
        entry_price: float
        spot: float = 100.0
        sigma: float = 0.20
        r: float = 0.05
        q: float = 0.0
        multiplier: float = 100.0

    class _RemoveRequest(_PydanticBase):
        symbol: str
        expiry: str
        strike: float
        right: Optional[str] = None

    def build_options_risk_router(monitor: "OptionsRiskMonitor") -> "APIRouter":
        """
        Build a FastAPI APIRouter that exposes the OptionsRiskMonitor.

        Mount in your FastAPI app with:
            app.include_router(build_options_risk_router(monitor), prefix="/options")

        Endpoints
        ---------
        GET  /options/greeks           -- current portfolio Greeks
        GET  /options/scenarios        -- scenario P&L matrix
        POST /options/position/add     -- add a position
        DELETE /options/position/{sym} -- remove positions for a symbol
        GET  /options/delta-hedge      -- pending delta-hedge signals
        """
        router = APIRouter(tags=["options-risk"])

        @router.get("/greeks")
        async def get_greeks():
            greeks = await monitor.compute_portfolio_greeks()
            return greeks.as_dict()

        @router.get("/scenarios")
        async def get_scenarios():
            return await monitor.compute_scenario_matrix()

        @router.post("/position/add")
        async def add_position(req: _PositionRequest):
            pos = OptionsPosition(
                symbol=req.symbol,
                expiry=req.expiry,
                strike=req.strike,
                right=req.right,
                qty=req.qty,
                entry_price=req.entry_price,
                spot=req.spot,
                sigma=req.sigma,
                r=req.r,
                q=req.q,
                multiplier=req.multiplier,
            )
            await monitor.add_position(pos)
            return {"status": "ok", "position_id": pos.position_id}

        @router.delete("/position/{symbol}")
        async def remove_position(symbol: str, expiry: str, strike: float, right: Optional[str] = None):
            removed = await monitor.remove_position(symbol, expiry, strike, right)
            if not removed:
                raise HTTPException(status_code=404, detail=f"Position not found: {symbol}")
            return {"status": "removed", "symbol": symbol}

        @router.get("/delta-hedge")
        async def get_delta_hedge():
            return {"signals": monitor.get_pending_hedge_signals()}

        @router.get("/snapshot")
        async def get_snapshot():
            return await monitor.get_risk_snapshot()

        return router

else:
    def build_options_risk_router(monitor):  # type: ignore[misc]
        raise ImportError("FastAPI is not installed. Cannot build OptionsRiskAPI router.")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_options_risk_monitor(
    delta_limit: float = OptionsRiskMonitor.DEFAULT_DELTA_LIMIT,
    gamma_threshold: float = OptionsRiskMonitor.DEFAULT_GAMMA_THRESHOLD,
    notional: float = OptionsRiskMonitor.DEFAULT_NOTIONAL,
    vol_refresh_interval: float = VolSurfaceCache.DEFAULT_REFRESH,
    fallback_vol: float = VolSurfaceCache.DEFAULT_FALLBACK_VOL,
    vol_fetch_fn=None,
) -> OptionsRiskMonitor:
    """
    Factory that wires together VolSurfaceCache and OptionsRiskMonitor.

    Parameters
    ----------
    delta_limit          : float -- normalised delta threshold (default 0.10)
    gamma_threshold      : float -- dollar gamma scalp threshold (default 0.005)
    notional             : float -- reference notional (default 1M)
    vol_refresh_interval : float -- vol cache refresh in seconds (default 300)
    fallback_vol         : float -- flat vol on surface failure (default 0.20)
    vol_fetch_fn         : async callable(symbol, spot) -> dict -- surface fetcher

    Returns
    -------
    OptionsRiskMonitor
    """
    cache = VolSurfaceCache(
        refresh_interval=vol_refresh_interval,
        fallback_vol=fallback_vol,
        fetch_fn=vol_fetch_fn,
    )
    return OptionsRiskMonitor(
        delta_limit=delta_limit,
        gamma_threshold=gamma_threshold,
        notional=notional,
        vol_surface_cache=cache,
    )


__all__ = [
    "OptionsPosition",
    "LiveGreeks",
    "VolSurfaceCache",
    "OptionsRiskMonitor",
    "build_options_risk_router",
    "create_options_risk_monitor",
]
