"""
Term structure utilities for the srfm-lab trading system.

Implements yield curves, dividend schedules, borrow rates, and forward prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# YieldCurve
# ---------------------------------------------------------------------------

class YieldCurve:
    """
    Risk-free yield curve built from observable treasury rates.

    Supports bootstrapping, discount factors, forward rates, and
    zero coupon rates. Interpolation uses log-linear discount factors
    for consistency with no-arbitrage.

    Parameters
    ----------
    maturities : array-like
        Maturity times in years, sorted ascending.
    rates : array-like
        Continuously compounded zero rates (decimal) for each maturity.
    """

    def __init__(
        self,
        maturities: np.ndarray,
        rates: np.ndarray,
        interpolation: str = "log_linear",
    ) -> None:
        maturities = np.asarray(maturities, dtype=float)
        rates = np.asarray(rates, dtype=float)
        idx = np.argsort(maturities)
        self._maturities = maturities[idx]
        self._rates = rates[idx]
        self._interpolation = interpolation
        self._discount_factors = np.exp(-self._rates * self._maturities)
        self._build_interpolator()

    def _build_interpolator(self) -> None:
        log_dfs = np.log(self._discount_factors)
        if len(self._maturities) >= 2:
            self._log_df_interp = interp1d(
                self._maturities,
                log_dfs,
                kind="linear",
                fill_value=(log_dfs[0], log_dfs[-1]),
                bounds_error=False,
            )
        else:
            # Single point: flat extrapolation
            self._log_df_interp = lambda t: log_dfs[0] * np.ones_like(np.asarray(t, dtype=float))

    def discount_factor(self, T: float) -> float:
        """P(0, T): zero-coupon bond price for maturity T."""
        if T <= 0:
            return 1.0
        log_df = float(self._log_df_interp(T))
        return math.exp(log_df)

    def zero_rate(self, T: float) -> float:
        """Continuously compounded zero rate for maturity T."""
        T = max(T, 1e-10)
        df = self.discount_factor(T)
        return -math.log(max(df, 1e-15)) / T

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate from T1 to T2.

        r_fwd = (ln(P(0,T1)) - ln(P(0,T2))) / (T2 - T1)
        """
        if T2 <= T1:
            raise ValueError("T2 must be greater than T1")
        df1 = self.discount_factor(T1)
        df2 = self.discount_factor(T2)
        return (math.log(df1) - math.log(max(df2, 1e-15))) / (T2 - T1)

    def par_rate(self, T: float, freq: int = 2) -> float:
        """
        Semi-annual (or freq-per-year) par coupon rate for a bullet bond.

        Parameters
        ----------
        T : float
            Maturity in years.
        freq : int
            Coupon frequency per year.
        """
        dt = 1.0 / freq
        n = max(1, round(T * freq))
        coupon_times = np.arange(1, n + 1) * dt
        sum_dfs = sum(self.discount_factor(t) for t in coupon_times)
        df_T = self.discount_factor(T)
        if abs(sum_dfs) < 1e-15:
            return float("nan")
        return freq * (1.0 - df_T) / sum_dfs

    def forward_curve(self, times: np.ndarray, tenor: float = 0.25) -> np.ndarray:
        """
        Instantaneous forward rate curve as a function of start time.

        f(t) = r_fwd(t, t + tenor)
        """
        return np.array([self.forward_rate(t, t + tenor) for t in times])

    @classmethod
    def bootstrap(
        cls,
        bond_maturities: np.ndarray,
        coupon_rates: np.ndarray,
        bond_prices: np.ndarray,
        face: float = 100.0,
        freq: int = 2,
    ) -> "YieldCurve":
        """
        Bootstrap zero rates from par bond prices.

        Parameters
        ----------
        bond_maturities : ndarray
            Maturities sorted ascending (years).
        coupon_rates : ndarray
            Annual coupon rates (decimal).
        bond_prices : ndarray
            Dirty prices of the bonds.
        face : float
            Face value (default 100).
        freq : int
            Coupon frequency per year.
        """
        bond_maturities = np.asarray(bond_maturities, dtype=float)
        coupon_rates = np.asarray(coupon_rates, dtype=float)
        bond_prices = np.asarray(bond_prices, dtype=float)

        known_zeros: Dict[float, float] = {}

        def zero_rate_from_known(t: float) -> float:
            if not known_zeros:
                return 0.03  # default flat
            ts = sorted(known_zeros.keys())
            rs = [known_zeros[t_] for t_ in ts]
            if t <= ts[0]:
                return rs[0]
            if t >= ts[-1]:
                return rs[-1]
            return float(np.interp(t, ts, rs))

        def df_from_known(t: float) -> float:
            r = zero_rate_from_known(t)
            return math.exp(-r * t)

        zero_rates = []
        for i, T in enumerate(bond_maturities):
            dt = 1.0 / freq
            n = max(1, round(T * freq))
            coupon = coupon_rates[i] / freq * face
            price = bond_prices[i]

            # Sum PV of all coupons except the last
            coupon_times = np.arange(1, n + 1) * dt
            pv_coupons = sum(coupon * df_from_known(t) for t in coupon_times[:-1])

            # Solve for discount factor at T
            # price = pv_coupons + (coupon + face) * df(T)
            last_cf = coupon + face
            df_T = (price - pv_coupons) / last_cf
            df_T = max(df_T, 1e-10)
            z = -math.log(df_T) / T
            known_zeros[T] = z
            zero_rates.append(z)

        return cls(bond_maturities, np.array(zero_rates))

    @classmethod
    def flat(cls, rate: float, maturities: Optional[np.ndarray] = None) -> "YieldCurve":
        """Create a flat yield curve at a constant rate."""
        if maturities is None:
            maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        rates = np.full_like(maturities, rate, dtype=float)
        return cls(maturities, rates)


# ---------------------------------------------------------------------------
# DividendSchedule
# ---------------------------------------------------------------------------

@dataclass
class Dividend:
    """A single discrete dividend."""
    ex_date: float  # In years from valuation date
    amount: float   # Cash dividend amount


class DividendSchedule:
    """
    Discrete dividend schedule for a single underlying.

    Supports:
    - List of (ex_date, cash_amount) pairs
    - Conversion to continuous dividend yield equivalent
    - Present value of dividends for option pricing adjustment
    """

    def __init__(self, dividends: Optional[List[Dividend]] = None) -> None:
        self._dividends = sorted(dividends or [], key=lambda d: d.ex_date)

    def add(self, ex_date: float, amount: float) -> None:
        """Add a dividend to the schedule."""
        self._dividends.append(Dividend(ex_date, amount))
        self._dividends.sort(key=lambda d: d.ex_date)

    def dividends_before(self, T: float) -> List[Dividend]:
        """Return all dividends with ex-date in (0, T]."""
        return [d for d in self._dividends if 0 < d.ex_date <= T]

    def pv_dividends(self, T: float, r: float) -> float:
        """
        Present value of all dividends paid before expiry T.

        PV = sum(D_i * exp(-r * t_i)) for t_i in (0, T]
        """
        return sum(d.amount * math.exp(-r * d.ex_date) for d in self.dividends_before(T))

    def adjusted_spot(self, S: float, T: float, r: float) -> float:
        """
        Return the dividend-adjusted spot price for option pricing.

        S_adj = S - PV(dividends)
        """
        return max(S - self.pv_dividends(T, r), 1e-10)

    def continuous_yield_equivalent(self, S: float, T: float, r: float) -> float:
        """
        Estimate the continuous dividend yield q that matches PV of discrete dividends.

        Solves: S * (1 - exp(-q*T)) = PV(dividends)
        Approximation: q ~ PV(dividends) / (S * T)
        """
        pv = self.pv_dividends(T, r)
        if S * T < 1e-10:
            return 0.0
        # More accurate: solve S*exp(-r*T)*(exp(-q*T) - 1) + S*exp(-q*T) ...
        # Use approximation for practical purposes
        return pv / (S * T)

    def next_dividend(self, t: float = 0.0) -> Optional[Dividend]:
        """Return the next dividend after time t, or None."""
        for d in self._dividends:
            if d.ex_date > t:
                return d
        return None

    def annual_yield(self, S: float) -> float:
        """
        Estimate annual dividend yield from all dividends in schedule.

        Assumes the last dividend recurs annually.
        """
        if not self._dividends:
            return 0.0
        max_T = max(d.ex_date for d in self._dividends)
        total = sum(d.amount for d in self._dividends)
        if max_T < 1e-10:
            return 0.0
        return total / (S * max_T)


# ---------------------------------------------------------------------------
# BorrowRate
# ---------------------------------------------------------------------------

class BorrowRate:
    """
    Security-specific borrow rate (stock loan fee) for short selling.

    The borrow rate enters option pricing as a negative carry component.
    For put options on hard-to-borrow stocks, the effective rate is:
        r_eff = r - borrow_rate

    Maintains a dictionary of per-security borrow rates with
    term structure support.
    """

    def __init__(self) -> None:
        self._rates: Dict[str, float] = {}
        self._curves: Dict[str, YieldCurve] = {}

    def set_flat(self, ticker: str, rate: float) -> None:
        """Set a flat borrow rate for a security."""
        self._rates[ticker] = rate
        self._curves.pop(ticker, None)

    def set_curve(self, ticker: str, maturities: np.ndarray, rates: np.ndarray) -> None:
        """Set a term structure of borrow rates for a security."""
        self._curves[ticker] = YieldCurve(maturities, rates)
        self._rates.pop(ticker, None)

    def get(self, ticker: str, T: float = 1.0) -> float:
        """
        Return borrow rate for ticker at maturity T.

        Returns 0 if no rate is set (i.e., GC rate assumption).
        """
        if ticker in self._curves:
            return self._curves[ticker].zero_rate(T)
        return self._rates.get(ticker, 0.0)

    def effective_rate(self, ticker: str, r: float, T: float = 1.0) -> float:
        """
        Effective financing rate for shorting ticker.

        effective_rate = r - borrow_rate
        """
        return r - self.get(ticker, T)

    def borrow_cost(self, ticker: str, notional: float, T: float) -> float:
        """
        Total borrow cost for a short position.

        cost = notional * borrow_rate * T
        """
        return notional * self.get(ticker, T) * T

    def cheapest_to_borrow(self, tickers: List[str]) -> str:
        """Return the ticker with the lowest borrow rate."""
        return min(tickers, key=lambda t: self.get(t))


# ---------------------------------------------------------------------------
# ForwardPrice
# ---------------------------------------------------------------------------

class ForwardPrice:
    """
    Cost-of-carry forward price model.

    Supports:
    - Continuous dividend yield (equity model)
    - Discrete cash dividends
    - Storage costs and convenience yields (commodity model)
    - Borrow rates for short forward positions
    """

    def __init__(self, yield_curve: Optional[YieldCurve] = None) -> None:
        self.yield_curve = yield_curve

    def _r(self, T: float) -> float:
        """Get risk-free rate for maturity T."""
        if self.yield_curve is not None:
            return self.yield_curve.zero_rate(T)
        return 0.0

    def continuous_dividend(
        self,
        S: float,
        T: float,
        q: float = 0.0,
        r: Optional[float] = None,
    ) -> float:
        """
        Forward price with continuous dividend yield.

        F = S * exp((r - q) * T)
        """
        r_ = r if r is not None else self._r(T)
        return S * math.exp((r_ - q) * T)

    def discrete_dividend(
        self,
        S: float,
        T: float,
        schedule: DividendSchedule,
        r: Optional[float] = None,
    ) -> float:
        """
        Forward price adjusted for discrete dividends.

        F = (S - PV(dividends)) * exp(r * T)
        """
        r_ = r if r is not None else self._r(T)
        S_adj = schedule.adjusted_spot(S, T, r_)
        return S_adj * math.exp(r_ * T)

    def futures(
        self,
        S: float,
        T: float,
        r: Optional[float] = None,
        q: float = 0.0,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
    ) -> float:
        """
        Futures price via cost-of-carry model.

        F = S * exp((r - q + storage_cost - convenience_yield) * T)

        Parameters
        ----------
        storage_cost : float
            Annualised storage cost as fraction of spot (for commodities).
        convenience_yield : float
            Annualised convenience yield (for commodities).
        """
        r_ = r if r is not None else self._r(T)
        carry = r_ - q + storage_cost - convenience_yield
        return S * math.exp(carry * T)

    def with_borrow(
        self,
        S: float,
        T: float,
        borrow_rate: float,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> float:
        """
        Forward price for a hard-to-borrow security.

        F = S * exp((r - q - borrow_rate) * T)
        """
        r_ = r if r is not None else self._r(T)
        return S * math.exp((r_ - q - borrow_rate) * T)

    def implied_forward_rate(
        self,
        S: float,
        F: float,
        T: float,
        q: float = 0.0,
    ) -> float:
        """
        Implied forward rate from spot and forward price.

        r = (ln(F/S) + q * T) / T
        """
        if T <= 0 or S <= 0 or F <= 0:
            return 0.0
        return (math.log(F / S) + q * T) / T

    def forward_curve(
        self,
        S: float,
        T_grid: np.ndarray,
        q: float = 0.0,
        r: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute forward prices at each expiry in T_grid.

        Returns array of forward prices.
        """
        return np.array([
            self.continuous_dividend(S, T, q, r if r is not None else self._r(T))
            for T in T_grid
        ])

    def implied_dividend_yield(
        self,
        S: float,
        F: float,
        T: float,
        r: float,
    ) -> float:
        """
        Implied continuous dividend yield from observed forward/futures price.

        q = r - ln(F/S) / T
        """
        if T <= 0 or S <= 0 or F <= 0:
            return 0.0
        return r - math.log(F / S) / T
