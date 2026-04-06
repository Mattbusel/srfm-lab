"""
vol_surface.py — Implied Volatility Surface construction and analytics.

Features
--------
* Fetch full option chains from Alpaca Markets REST API
* Black-Scholes IV solver via scipy.optimize.brentq
* SVI (Stochastic Volatility Inspired) parameterization per expiry slice
* Surface interpolation: cubic spline across strikes, linear across time
* Full Greeks: delta, gamma, theta, vega, vanna, volga, charm
* Arbitrage detection: calendar-spread and butterfly violations
* Interactive 3-D Plotly surface + contour with expiry slices
* Persist surface to vol_surface_{date}.json

CLI
---
    python vol_surface.py --symbol SPY --plot
    python vol_surface.py --symbol NVDA --no-plot --out-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq, minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT"]
ALPACA_BASE_URL = "https://data.alpaca.markets/v2"
RISK_FREE_RATE = 0.0525  # approximate Fed-funds rate proxy
MIN_IV = 1e-4
MAX_IV = 10.0
MIN_OPTION_PRICE = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionContract:
    symbol: str
    expiry: date
    strike: float
    option_type: str          # "call" | "put"
    mid_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    underlying_price: float
    dte: float                # calendar days to expiry
    iv: Optional[float] = None

    @property
    def t(self) -> float:
        """Time to expiry in years (trading-day adjusted)."""
        return max(self.dte / 365.0, 1e-6)

    @property
    def log_moneyness(self) -> float:
        if self.underlying_price <= 0 or self.strike <= 0:
            return 0.0
        fwd = self.underlying_price * math.exp(RISK_FREE_RATE * self.t)
        return math.log(self.strike / fwd)


@dataclass
class SVIParams:
    """SVI raw parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    a: float = 0.04
    b: float = 0.1
    rho: float = -0.3
    m: float = 0.0
    sigma: float = 0.2
    expiry: Optional[date] = None
    t: Optional[float] = None
    fit_error: float = 0.0


@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # per calendar day
    vega: float = 0.0    # per 1-vol-point move
    rho: float = 0.0
    vanna: float = 0.0   # dDelta/dVol
    volga: float = 0.0   # d²Price/dVol²
    charm: float = 0.0   # dDelta/dTime


@dataclass
class SurfaceSlice:
    expiry: date
    t: float
    strikes: List[float]
    ivs: List[float]
    svi: Optional[SVIParams] = None


@dataclass
class VolSurfaceData:
    symbol: str
    as_of: str
    underlying_price: float
    slices: List[SurfaceSlice] = field(default_factory=list)
    arbitrage_violations: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Black-Scholes engine
# ---------------------------------------------------------------------------

class BlackScholes:
    """Closed-form Black-Scholes price, IV solver, and Greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def price(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> float:
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
    ) -> Optional[float]:
        """Solve for IV using Brent's method. Returns None if unsolvable."""
        if market_price < MIN_OPTION_PRICE or T <= 0:
            return None
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if market_price <= intrinsic:
            return None

        def objective(sigma: float) -> float:
            return BlackScholes.price(S, K, T, r, sigma, option_type) - market_price

        try:
            lo_val = objective(MIN_IV)
            hi_val = objective(MAX_IV)
            if lo_val * hi_val > 0:
                return None
            iv = brentq(objective, MIN_IV, MAX_IV, xtol=1e-8, maxiter=200)
            return iv if MIN_IV < iv < MAX_IV else None
        except (ValueError, RuntimeError):
            return None

    @staticmethod
    def greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> Greeks:
        if T <= 0 or sigma <= 0:
            return Greeks()
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        npd1 = norm.pdf(d1)
        sqrt_T = math.sqrt(T)
        disc = math.exp(-r * T)

        if option_type == "call":
            delta = nd1
            theta_num = (
                -S * npd1 * sigma / (2 * sqrt_T)
                - r * K * disc * nd2
            )
        else:
            delta = nd1 - 1.0
            theta_num = (
                -S * npd1 * sigma / (2 * sqrt_T)
                + r * K * disc * norm.cdf(-d2)
            )

        gamma = npd1 / (S * sigma * sqrt_T)
        vega = S * npd1 * sqrt_T / 100.0  # per 1 vol point
        theta = theta_num / 365.0          # per calendar day
        rho_greek = (
            K * T * disc * nd2 / 100.0 if option_type == "call"
            else -K * T * disc * norm.cdf(-d2) / 100.0
        )
        # Second-order
        vanna = -npd1 * d2 / sigma          # dDelta/dVol
        volga = vega * d1 * d2 / sigma      # d²Price/dVol²
        charm = (                           # dDelta/dTime (per day)
            -npd1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            / 365.0
        )

        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho_greek,
            vanna=vanna,
            volga=volga,
            charm=charm,
        )


# ---------------------------------------------------------------------------
# SVI parameterization
# ---------------------------------------------------------------------------

class SVIFitter:
    """
    Fit SVI raw parameterization to a single expiry slice.

    Total implied variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+sigma^2))
    where k = log(K/F) is log-moneyness.
    """

    @staticmethod
    def svi_w(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        inner = rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
        return a + b * inner

    @staticmethod
    def svi_vol(k: np.ndarray, t: float, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        w = SVIFitter.svi_w(k, a, b, rho, m, sigma)
        w = np.maximum(w, 1e-8)
        return np.sqrt(w / t)

    @staticmethod
    def is_arbitrage_free(a: float, b: float, rho: float, m: float, sigma: float) -> bool:
        """Check basic no-arbitrage constraints for SVI parameters."""
        if b <= 0:
            return False
        if abs(rho) >= 1.0:
            return False
        if sigma <= 0:
            return False
        if a + b * sigma * math.sqrt(1 - rho ** 2) < 0:
            return False
        return True

    @classmethod
    def fit(
        cls, log_moneyness: np.ndarray, market_ivs: np.ndarray, t: float, expiry: date
    ) -> SVIParams:
        """Fit SVI to market IVs. Returns best-fit SVIParams."""
        # Convert IV -> total variance
        w_mkt = market_ivs ** 2 * t

        def objective(params: np.ndarray) -> float:
            a, b, rho, m, sigma = params
            if sigma <= 0 or b <= 0 or abs(rho) >= 1:
                return 1e9
            w_fit = cls.svi_w(log_moneyness, a, b, rho, m, sigma)
            if np.any(w_fit < 0):
                return 1e9
            return float(np.sum((w_fit - w_mkt) ** 2))

        # Multiple starting points to avoid local minima
        best_result = None
        best_val = np.inf
        atm_var = float(np.median(w_mkt))

        starts = [
            [atm_var * 0.8, 0.1, -0.3, 0.0, 0.2],
            [atm_var * 0.5, 0.2, -0.5, -0.1, 0.3],
            [atm_var * 0.9, 0.05, -0.1, 0.05, 0.15],
        ]

        bounds = [
            (-0.5, 1.0),   # a
            (1e-4, 2.0),   # b
            (-0.999, 0.999),  # rho
            (-1.0, 1.0),   # m
            (1e-4, 2.0),   # sigma
        ]

        for x0 in starts:
            try:
                res = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            return SVIParams(expiry=expiry, t=t, fit_error=999.0)

        a, b, rho, m, sigma = best_result.x
        rmse = math.sqrt(best_val / max(len(log_moneyness), 1))
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=expiry, t=t, fit_error=rmse)


# ---------------------------------------------------------------------------
# Alpaca data fetcher
# ---------------------------------------------------------------------------

class AlpacaOptionFetcher:
    """Fetch option chains from Alpaca Data API."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        })

    def get_latest_quote(self, symbol: str) -> float:
        """Fetch latest mid-price for underlying."""
        url = f"{ALPACA_BASE_URL}/stocks/{symbol}/quotes/latest"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            q = data.get("quote", {})
            bid = q.get("bp", 0)
            ask = q.get("ap", 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return q.get("lp", 0) or q.get("price", 0)
        except Exception as exc:
            warnings.warn(f"Failed to fetch quote for {symbol}: {exc}")
            return 0.0

    def get_option_chain(self, symbol: str) -> List[OptionContract]:
        """
        Fetch full option chain snapshot from Alpaca.
        Returns a list of OptionContract (bid/ask/volume/OI).
        """
        url = f"{ALPACA_BASE_URL}/options/snapshots/{symbol}"
        params = {
            "feed": "indicative",
            "limit": 1000,
        }
        contracts: List[OptionContract] = []
        underlying_price = self.get_latest_quote(symbol)
        today = date.today()

        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                warnings.warn(f"Option chain fetch failed for {symbol}: {exc}")
                break

            snapshots: dict = data.get("snapshots", {})
            for contract_sym, snap in snapshots.items():
                try:
                    details = snap.get("latestQuote", {})
                    greeks_raw = snap.get("greeks", {})
                    meta = snap.get("impliedVolatility", None)

                    # Parse symbol: e.g. SPY240119C00480000
                    parsed = _parse_option_symbol(contract_sym)
                    if parsed is None:
                        continue
                    otype, expiry_dt, strike = parsed

                    dte = (expiry_dt - today).days
                    if dte < 1 or dte > 730:
                        continue

                    bid = float(details.get("bp", 0) or 0)
                    ask = float(details.get("ap", 0) or 0)
                    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
                    volume = int(snap.get("dailyBar", {}).get("v", 0) or 0)
                    oi = int(snap.get("openInterest", 0) or 0)

                    if mid < MIN_OPTION_PRICE:
                        continue

                    oc = OptionContract(
                        symbol=contract_sym,
                        expiry=expiry_dt,
                        strike=strike,
                        option_type=otype,
                        mid_price=mid,
                        bid=bid,
                        ask=ask,
                        volume=volume,
                        open_interest=oi,
                        underlying_price=underlying_price if underlying_price else strike,
                        dte=float(dte),
                    )
                    contracts.append(oc)
                except Exception:
                    continue

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return contracts


def _parse_option_symbol(sym: str) -> Optional[Tuple[str, date, float]]:
    """
    Parse OCC-style option symbol: ROOT + YYMMDD + C/P + 8-digit strike*1000.
    Returns (option_type, expiry_date, strike) or None.
    """
    try:
        # Find C or P separator (last occurrence after root)
        for i in range(len(sym) - 1, -1, -1):
            if sym[i] in ("C", "P") and i >= 3:
                otype = "call" if sym[i] == "C" else "put"
                date_str = sym[i - 6: i]
                strike_str = sym[i + 1:]
                expiry_dt = datetime.strptime(date_str, "%y%m%d").date()
                strike = float(strike_str) / 1000.0
                return otype, expiry_dt, strike
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Arbitrage detection
# ---------------------------------------------------------------------------

class ArbitrageDetector:
    """Check vol surface for standard arbitrage conditions."""

    @staticmethod
    def calendar_spread_violations(slices: List[SurfaceSlice]) -> List[dict]:
        """
        Calendar: total variance w = IV^2 * T must be non-decreasing in T
        for the same log-moneyness k.
        """
        violations = []
        if len(slices) < 2:
            return violations

        # Build common k grid
        all_strikes = sorted({s for sl in slices for s in sl.strikes})
        slices_sorted = sorted(slices, key=lambda s: s.t)

        for i in range(len(slices_sorted) - 1):
            sl1 = slices_sorted[i]
            sl2 = slices_sorted[i + 1]
            # Interpolate both slices onto common strikes
            for k in all_strikes:
                iv1 = _interpolate_iv(sl1, k)
                iv2 = _interpolate_iv(sl2, k)
                if iv1 is None or iv2 is None:
                    continue
                w1 = iv1 ** 2 * sl1.t
                w2 = iv2 ** 2 * sl2.t
                if w2 < w1 - 1e-6:
                    violations.append({
                        "type": "calendar_spread",
                        "strike": k,
                        "expiry_near": str(sl1.expiry),
                        "expiry_far": str(sl2.expiry),
                        "w_near": round(w1, 6),
                        "w_far": round(w2, 6),
                        "violation_amount": round(w1 - w2, 6),
                    })
        return violations

    @staticmethod
    def butterfly_violations(sl: SurfaceSlice) -> List[dict]:
        """
        Butterfly arbitrage: local vol^2 must be positive everywhere.
        Approximated by checking d²C/dK² >= 0 for call prices.
        We check the second derivative of total variance w.r.t. k.
        """
        violations = []
        if len(sl.strikes) < 3:
            return violations

        strikes = np.array(sl.strikes)
        ivs = np.array(sl.ivs)
        valid = (ivs > 0) & np.isfinite(ivs)
        if valid.sum() < 3:
            return violations
        strikes = strikes[valid]
        ivs = ivs[valid]
        w = ivs ** 2 * sl.t

        for i in range(1, len(strikes) - 1):
            dk1 = strikes[i] - strikes[i - 1]
            dk2 = strikes[i + 1] - strikes[i]
            if dk1 <= 0 or dk2 <= 0:
                continue
            # Second derivative via finite difference on w
            d2w = (w[i + 1] - 2 * w[i] + w[i - 1]) / ((dk1 + dk2) / 2) ** 2
            if d2w < -1e-4:
                violations.append({
                    "type": "butterfly",
                    "expiry": str(sl.expiry),
                    "strike": strikes[i],
                    "d2w": round(float(d2w), 8),
                })
        return violations


def _interpolate_iv(sl: SurfaceSlice, strike: float) -> Optional[float]:
    """Linear interpolation of IV at a given strike within a slice."""
    strikes = sl.strikes
    ivs = sl.ivs
    if strike < strikes[0] or strike > strikes[-1]:
        return None
    for i in range(len(strikes) - 1):
        if strikes[i] <= strike <= strikes[i + 1]:
            t = (strike - strikes[i]) / (strikes[i + 1] - strikes[i])
            return ivs[i] * (1 - t) + ivs[i + 1] * t
    return None


# ---------------------------------------------------------------------------
# Surface interpolator
# ---------------------------------------------------------------------------

class SurfaceInterpolator:
    """
    Bivariate interpolation of the IV surface.
    - Cubic spline across strikes within each expiry slice.
    - Linear interpolation across time dimension.
    """

    def __init__(self, slices: List[SurfaceSlice]):
        self.slices = sorted(slices, key=lambda s: s.t)
        self._splines: List[CubicSpline] = []
        self._build_splines()

    def _build_splines(self) -> None:
        for sl in self.slices:
            x = np.array(sl.strikes)
            y = np.array(sl.ivs)
            valid = (y > 0) & np.isfinite(y)
            if valid.sum() < 2:
                self._splines.append(None)  # type: ignore
                continue
            try:
                cs = CubicSpline(x[valid], y[valid], extrapolate=False)
                self._splines.append(cs)
            except Exception:
                self._splines.append(None)  # type: ignore

    def interpolate(self, strike: float, t: float) -> Optional[float]:
        """Return interpolated IV at (strike, t)."""
        times = [sl.t for sl in self.slices]
        if t < times[0] or t > times[-1]:
            return None

        # Find bracketing slices
        idx = np.searchsorted(times, t)
        idx = int(np.clip(idx, 1, len(times) - 1))
        t1, t2 = times[idx - 1], times[idx]
        sp1, sp2 = self._splines[idx - 1], self._splines[idx]

        iv1 = float(sp1(strike)) if sp1 is not None else None
        iv2 = float(sp2(strike)) if sp2 is not None else None

        if iv1 is None and iv2 is None:
            return None
        if iv1 is None:
            return iv2
        if iv2 is None:
            return iv1

        # Linear interpolation in time
        w = (t - t1) / (t2 - t1)
        return iv1 * (1 - w) + iv2 * w


# ---------------------------------------------------------------------------
# Main VolSurface class
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Orchestrates IV surface construction for a single underlying.

    Usage
    -----
    vs = VolSurface("SPY")
    vs.build()          # fetches chain, solves IV, fits SVI
    vs.plot()           # opens interactive Plotly figure
    vs.save()           # writes vol_surface_{date}.json
    """

    def __init__(
        self,
        symbol: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.symbol = symbol.upper()
        self.r = risk_free_rate
        self.fetcher = AlpacaOptionFetcher(api_key, api_secret)
        self.contracts: List[OptionContract] = []
        self.slices: List[SurfaceSlice] = []
        self.interpolator: Optional[SurfaceInterpolator] = None
        self.underlying_price: float = 0.0
        self.as_of: str = datetime.utcnow().isoformat()
        self.violations: List[dict] = []

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def build(self) -> "VolSurface":
        print(f"[VolSurface] Fetching option chain for {self.symbol}...")
        self.contracts = self.fetcher.get_option_chain(self.symbol)
        if not self.contracts:
            print(f"[VolSurface] WARNING: No contracts returned for {self.symbol}.")
            return self

        # Pick representative underlying price
        prices = [c.underlying_price for c in self.contracts if c.underlying_price > 0]
        self.underlying_price = float(np.median(prices)) if prices else 0.0
        print(f"[VolSurface] Underlying price: {self.underlying_price:.2f}, contracts: {len(self.contracts)}")

        self._solve_ivs()
        self._build_slices()
        self._fit_svi()
        self._detect_arbitrage()
        self.interpolator = SurfaceInterpolator(self.slices)
        return self

    def _solve_ivs(self) -> None:
        """Solve implied volatility for each contract."""
        solved = 0
        for c in self.contracts:
            iv = BlackScholes.implied_vol(
                c.mid_price, c.underlying_price, c.strike, c.t, self.r, c.option_type
            )
            c.iv = iv
            if iv is not None:
                solved += 1
        print(f"[VolSurface] IV solved: {solved}/{len(self.contracts)}")

    def _build_slices(self) -> None:
        """Group contracts by expiry into SurfaceSlice objects."""
        expiry_map: Dict[date, List[OptionContract]] = {}
        for c in self.contracts:
            if c.iv is not None and MIN_IV < c.iv < MAX_IV:
                expiry_map.setdefault(c.expiry, []).append(c)

        self.slices = []
        for expiry, contracts in sorted(expiry_map.items()):
            # Use call IVs; prefer ATM region
            calls = [c for c in contracts if c.option_type == "call" and c.iv is not None]
            if len(calls) < 3:
                # Fall back to puts
                puts = [c for c in contracts if c.option_type == "put" and c.iv is not None]
                calls = puts if len(puts) >= 3 else calls

            if len(calls) < 3:
                continue

            calls_sorted = sorted(calls, key=lambda c: c.strike)
            strikes = [c.strike for c in calls_sorted]
            ivs = [c.iv for c in calls_sorted]
            t = calls_sorted[0].t  # same expiry → same T

            sl = SurfaceSlice(
                expiry=expiry,
                t=t,
                strikes=strikes,
                ivs=ivs,
            )
            self.slices.append(sl)

        print(f"[VolSurface] Expiry slices: {len(self.slices)}")

    def _fit_svi(self) -> None:
        """Fit SVI to each expiry slice."""
        fitted = 0
        for sl in self.slices:
            S = self.underlying_price
            F = S * math.exp(self.r * sl.t)
            k = np.array([math.log(K / F) for K in sl.strikes])
            ivs = np.array(sl.ivs)
            params = SVIFitter.fit(k, ivs, sl.t, sl.expiry)
            sl.svi = params
            if params.fit_error < 0.01:
                fitted += 1
        print(f"[VolSurface] SVI fitted: {fitted}/{len(self.slices)} slices (RMSE < 1%)")

    def _detect_arbitrage(self) -> None:
        cal_viol = ArbitrageDetector.calendar_spread_violations(self.slices)
        bf_viol: List[dict] = []
        for sl in self.slices:
            bf_viol.extend(ArbitrageDetector.butterfly_violations(sl))
        self.violations = cal_viol + bf_viol
        if self.violations:
            print(f"[VolSurface] Arbitrage violations detected: {len(self.violations)}")

    # ------------------------------------------------------------------
    # Greeks at arbitrary (S, K, T)
    # ------------------------------------------------------------------

    def surface_greeks(
        self,
        strike: float,
        t: float,
        option_type: str = "call",
        S: Optional[float] = None,
    ) -> Optional[Greeks]:
        """Compute Greeks using interpolated surface IV."""
        if self.interpolator is None:
            return None
        iv = self.interpolator.interpolate(strike, t)
        if iv is None:
            return None
        S = S or self.underlying_price
        return BlackScholes.greeks(S, strike, t, self.r, iv, option_type)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, save_html: Optional[str] = None) -> None:
        """Interactive 3-D Plotly surface + contour."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("[VolSurface] plotly not installed. Run: pip install plotly")
            return

        if not self.slices:
            print("[VolSurface] No surface data to plot.")
            return

        # Build grid
        all_strikes = np.linspace(
            min(s for sl in self.slices for s in sl.strikes),
            max(s for sl in self.slices for s in sl.strikes),
            80,
        )
        ts = np.array([sl.t for sl in self.slices])
        iv_grid = np.full((len(ts), len(all_strikes)), np.nan)

        for i, sl in enumerate(self.slices):
            if self.interpolator and self.interpolator._splines[i] is not None:
                sp = self.interpolator._splines[i]
                vals = sp(all_strikes)
                # Extrapolation guard
                in_range = (all_strikes >= sl.strikes[0]) & (all_strikes <= sl.strikes[-1])
                iv_row = np.where(in_range, vals, np.nan)
                iv_grid[i] = np.where(iv_row > 0, iv_row * 100, np.nan)  # percent

        expiry_labels = [str(sl.expiry) for sl in self.slices]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "contour"}]],
            subplot_titles=["IV Surface (%)", "IV Contour (%)"],
        )

        fig.add_trace(
            go.Surface(
                z=iv_grid,
                x=all_strikes,
                y=ts * 365,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(x=0.45, title="IV %"),
                name="IV Surface",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Contour(
                z=iv_grid,
                x=all_strikes,
                y=ts * 365,
                colorscale="Viridis",
                showscale=False,
                contours=dict(showlabels=True),
                name="IV Contour",
            ),
            row=1, col=2,
        )

        # Add individual slice lines to 3D plot
        slice_colors = [f"hsl({int(i * 240 / max(len(self.slices), 1))},80%,50%)" for i in range(len(self.slices))]
        for i, sl in enumerate(self.slices):
            fig.add_trace(
                go.Scatter3d(
                    x=sl.strikes,
                    y=[sl.t * 365] * len(sl.strikes),
                    z=[iv * 100 for iv in sl.ivs],
                    mode="lines+markers",
                    line=dict(color=slice_colors[i], width=3),
                    marker=dict(size=3),
                    name=str(sl.expiry),
                    showlegend=True,
                ),
                row=1, col=1,
            )

        fig.update_layout(
            title=f"{self.symbol} Implied Volatility Surface — {self.as_of[:10]}",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="DTE",
                zaxis_title="IV (%)",
            ),
            xaxis2_title="Strike",
            yaxis2_title="DTE",
            height=700,
            template="plotly_dark",
        )

        if save_html:
            fig.write_html(save_html)
            print(f"[VolSurface] Plot saved to {save_html}")
        else:
            fig.show()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        out = {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "underlying_price": self.underlying_price,
            "risk_free_rate": self.r,
            "slices": [],
            "arbitrage_violations": self.violations,
        }
        for sl in self.slices:
            slice_d = {
                "expiry": str(sl.expiry),
                "t_years": sl.t,
                "dte": round(sl.t * 365),
                "strikes": sl.strikes,
                "ivs": [round(iv, 6) for iv in sl.ivs],
                "svi": None,
            }
            if sl.svi:
                slice_d["svi"] = {
                    "a": sl.svi.a,
                    "b": sl.svi.b,
                    "rho": sl.svi.rho,
                    "m": sl.svi.m,
                    "sigma": sl.svi.sigma,
                    "fit_error": sl.svi.fit_error,
                }
            out["slices"].append(slice_d)
        return out

    def save(self, out_dir: str = ".") -> str:
        today_str = date.today().isoformat()
        fname = os.path.join(out_dir, f"vol_surface_{self.symbol}_{today_str}.json")
        with open(fname, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[VolSurface] Saved to {fname}")
        return fname


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build IV surface for an options underlying.")
    parser.add_argument("--symbol", default="SPY", choices=INSTRUMENTS, help="Underlying symbol")
    parser.add_argument("--plot", action="store_true", default=False, help="Show interactive plot")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--out-dir", default=".", help="Directory to save JSON output")
    parser.add_argument("--save-html", default=None, help="Save plot as HTML file instead of showing")
    parser.add_argument("--rate", type=float, default=RISK_FREE_RATE, help="Risk-free rate (default 5.25%)")
    args = parser.parse_args()

    vs = VolSurface(args.symbol, risk_free_rate=args.rate)
    vs.build()

    if vs.slices:
        print(f"\n{'Expiry':<12} {'DTE':>5} {'Strikes':>8} {'ATM IV':>8} {'SVI RMSE':>10}")
        print("-" * 50)
        S = vs.underlying_price
        for sl in vs.slices:
            # Find nearest-ATM IV
            diffs = [abs(k - S) for k in sl.strikes]
            atm_idx = int(np.argmin(diffs))
            atm_iv = sl.ivs[atm_idx] * 100
            rmse = sl.svi.fit_error * 100 if sl.svi else float("nan")
            print(
                f"{str(sl.expiry):<12} {int(sl.t*365):>5} {len(sl.strikes):>8} "
                f"{atm_iv:>7.2f}% {rmse:>9.3f}%"
            )

        if vs.violations:
            print(f"\nArbitrage violations: {len(vs.violations)}")
            for v in vs.violations[:5]:
                print(f"  {v}")

    vs.save(args.out_dir)

    if args.plot or args.save_html:
        vs.plot(save_html=args.save_html)


if __name__ == "__main__":
    _cli()
