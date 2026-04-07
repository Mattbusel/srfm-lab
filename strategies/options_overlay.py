"""
options_overlay.py -- Options overlay strategy for equity positions.

Overlays:
  1. ProtectivePutOverlay   -- buy OTM puts when BH mass > BH_COLLAPSE
  2. CoveredCallOverlay     -- sell ATM calls when Hurst < 0.42 (mean reversion)
  3. StraddleOverlay        -- buy straddle when GARCH forecast / realized_vol > 2.0

Options pricing: Black-Scholes (European) with delta hedging.

Risk budget:
  - Total premium spend capped at 2% of NAV per month (OptionsRiskBudget)
  - Delta hedge: rebalance when |net delta| > 0.10

BH constants:
  BH_MASS_THRESH = 1.92
  BH_DECAY       = 0.924
  BH_COLLAPSE    = 0.992

LARSA v18 compatible.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr    -- fast standard normal CDF

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# BH physics constants
# ---------------------------------------------------------------------------
BH_MASS_THRESH = 1.92
BH_DECAY       = 0.924
BH_COLLAPSE    = 0.992

HURST_MEAN_REVERT = 0.42    -- covered call threshold
HURST_TRENDING    = 0.58


# ---------------------------------------------------------------------------
# Black-Scholes pricing and Greeks
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return float(ndtr(x))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S           : spot price
    K           : strike
    T           : time to expiry in years
    r           : risk-free rate (annual, continuous)
    sigma       : implied/realized vol (annual)
    option_type : "call" or "put"

    Returns
    -------
    Option price (same units as S).
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic
    sqT   = math.sqrt(T)
    d1    = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2    = d1 - sigma * sqT
    if option_type == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:  -- put
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(0.0, price)


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes delta.

    Call delta: N(d1)
    Put  delta: N(d1) - 1
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    if option_type == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    return _norm_pdf(d1) / (S * sigma * sqT)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (per 1-unit vol change)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    return S * _norm_pdf(d1) * sqT


def bs_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes theta (daily, in price units per day)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    if option_type == "call":
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqT)
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqT)
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))
    return theta / 252.0   -- convert to daily


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    """Represents a single options position."""
    option_type: str      = "call"     -- "call" or "put"
    overlay_type: str     = ""         -- "protective_put", "covered_call", "straddle_call", "straddle_put"
    direction: int        = 1          -- +1 long, -1 short
    n_contracts: float    = 0.0        -- number of contracts (fractional allowed)
    strike: float         = 0.0
    expiry_T: float       = 0.0        -- remaining time to expiry in years
    entry_price: float    = 0.0        -- premium paid/received at entry
    current_price: float  = 0.0        -- current mark
    delta: float          = 0.0
    gamma: float          = 0.0
    vega: float           = 0.0
    theta: float          = 0.0
    pnl: float            = 0.0        -- unrealized P&L
    entry_bar: int        = 0
    is_active: bool       = True


@dataclass
class OverlayState:
    """Snapshot of all active options positions and risk metrics."""
    nav: float                  = 1_000_000.0
    total_premium_spent: float  = 0.0    -- cumulative this month
    monthly_budget_used: float  = 0.0    -- fraction of budget used
    net_delta: float            = 0.0    -- options-only delta
    positions: List[OptionPosition] = field(default_factory=list)


@dataclass
class OptionsBacktestResult:
    total_return: float     = 0.0
    cagr: float             = 0.0
    sharpe: float           = 0.0
    sortino: float          = 0.0
    max_drawdown: float     = 0.0
    calmar: float           = 0.0
    win_rate: float         = 0.0
    profit_factor: float    = 0.0
    n_trades: int           = 0
    avg_trade_return: float = 0.0
    total_premium_paid: float    = 0.0
    total_premium_collected: float = 0.0
    delta_hedge_cost: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series      = field(default_factory=pd.Series)
    overlay_pnl: pd.Series  = field(default_factory=pd.Series)
    params: dict            = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
            f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
            f"PremPaid={self.total_premium_paid:,.0f} "
            f"PremColl={self.total_premium_collected:,.0f}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n = len(equity_curve)
    initial = equity_curve[0]
    final   = equity_curve[-1]
    total_return = final / initial - 1.0
    n_years = max(1, n / 252)
    cagr    = (final / initial) ** (1.0 / n_years) - 1.0
    rets    = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    rets    = np.concatenate([[0.0], rets])
    std     = rets.std()
    sharpe  = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down    = rets[rets < 0]
    sortino_d = np.std(down) if len(down) > 0 else 1e-9
    sortino   = rets.mean() / sortino_d * math.sqrt(252)
    pk  = np.maximum.accumulate(equity_curve)
    dd  = (equity_curve - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0.0
    wins    = [r for r in trade_returns if r > 0]
    losses  = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown=mdd, calmar=calmar, win_rate=win_rate, profit_factor=pf,
        n_trades=len(trade_returns),
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        returns=pd.Series(rets),
    )


def _compute_hurst(prices: np.ndarray) -> float:
    n = len(prices)
    if n < 50:
        return 0.5
    log_p = np.log(prices + 1e-9)
    lags  = sorted(set([max(2, n // k) for k in range(2, min(20, n // 10 + 1))]))
    rs_v, lag_v = [], []
    for lag in lags:
        if lag >= n:
            continue
        diff = np.diff(log_p[:lag])
        if len(diff) < 2:
            continue
        cumdev = np.cumsum(diff - diff.mean())
        R = cumdev.max() - cumdev.min()
        S = diff.std()
        if S < 1e-12:
            continue
        rs_v.append(math.log(R / S + 1e-12))
        lag_v.append(math.log(lag))
    if len(rs_v) < 4:
        return 0.5
    h = float(np.polyfit(lag_v, rs_v, 1)[0])
    return max(0.01, min(0.99, h))


def _garch_vol(returns: np.ndarray, alpha: float = 0.09, beta: float = 0.90) -> float:
    """GARCH(1,1) one-step-ahead vol forecast. Returns annualized sigma."""
    n = len(returns)
    if n < 20:
        return float(np.std(returns) * math.sqrt(252))
    long_var = float(np.var(returns))
    omega    = long_var * (1.0 - alpha - beta)
    sigma2   = long_var
    for r in returns:
        sigma2 = omega + alpha * r ** 2 + beta * sigma2
    return math.sqrt(sigma2 * 252)


def _bh_mass(returns: np.ndarray, window: int = 21) -> float:
    """BH mass proxy: recent abs-return intensity vs. long-run 95th percentile."""
    if len(returns) < window:
        return 0.0
    recent = float(np.abs(returns[-window:]).mean())
    q95    = float(np.percentile(np.abs(returns), 95)) if len(returns) > 50 else recent
    return min(2.0, recent / (q95 + 1e-9) * 1.5)


# ---------------------------------------------------------------------------
# OptionsRiskBudget
# ---------------------------------------------------------------------------

class OptionsRiskBudget:
    """
    Tracks total options premium spend and enforces a monthly cap.

    Cap: max_monthly_premium_pct of NAV per calendar month.

    Parameters
    ----------
    max_monthly_premium_pct : max premium as fraction of NAV (default 0.02 = 2%)
    bars_per_month          : trading days per month (default 21)
    """

    def __init__(
        self,
        max_monthly_premium_pct: float = 0.02,
        bars_per_month: int             = 21,
    ):
        self.max_monthly_pct = max_monthly_premium_pct
        self.bars_per_month  = bars_per_month
        self._month_start    = 0
        self._spent_this_month = 0.0

    def reset_month(self, bar_i: int):
        self._month_start      = bar_i
        self._spent_this_month = 0.0

    def can_spend(self, premium: float, nav: float) -> bool:
        """Return True if spending this premium is within the monthly budget."""
        budget_remaining = self.max_monthly_pct * nav - self._spent_this_month
        return premium <= budget_remaining

    def record_spend(self, premium: float):
        """Register a premium outflow."""
        self._spent_this_month += premium

    def budget_used_pct(self, nav: float) -> float:
        """Fraction of monthly budget consumed."""
        if nav < 1e-6:
            return 0.0
        return self._spent_this_month / (self.max_monthly_pct * nav + 1e-9)

    def new_bar(self, bar_i: int):
        """Call each bar -- resets monthly counter every bars_per_month bars."""
        if (bar_i - self._month_start) >= self.bars_per_month:
            self.reset_month(bar_i)


# ---------------------------------------------------------------------------
# 1. ProtectivePutOverlay
# ---------------------------------------------------------------------------

class ProtectivePutOverlay:
    """
    Buys 1-month 5% OTM puts when BH mass > BH_COLLAPSE (default 0.90).

    Size: 1% premium budget of NAV per activation.
    Greeks: tracks put delta to allow delta hedging of equity.

    Parameters
    ----------
    bh_trigger        : BH mass threshold to buy puts (default BH_COLLAPSE = 0.992)
    otm_pct           : OTM percentage for put strike (default 0.05 = 5% OTM)
    expiry_days       : days to expiry when opening put (default 21 -- ~1 month)
    premium_budget_pct : premium per put initiation as fraction of NAV (default 0.01)
    risk_free         : risk-free rate for BS pricing (default 0.04)
    """

    def __init__(
        self,
        bh_trigger: float          = BH_COLLAPSE,
        otm_pct: float             = 0.05,
        expiry_days: int           = 21,
        premium_budget_pct: float  = 0.01,
        risk_free: float           = 0.04,
    ):
        self.bh_trigger         = bh_trigger
        self.otm_pct            = otm_pct
        self.expiry_days        = expiry_days
        self.premium_budget_pct = premium_budget_pct
        self.risk_free          = risk_free
        self._active_puts: List[OptionPosition] = []

    def check_trigger(self, bh_mass: float) -> bool:
        """Return True if BH mass exceeds trigger level."""
        return bh_mass > self.bh_trigger

    def open_put(
        self,
        spot: float,
        sigma: float,
        nav: float,
        bar_i: int,
        risk_budget: OptionsRiskBudget,
    ) -> Optional[OptionPosition]:
        """
        Open a new protective put if budget allows.
        Returns OptionPosition or None if budget exhausted.
        """
        strike = spot * (1.0 - self.otm_pct)    -- 5% OTM
        T      = self.expiry_days / 252.0
        price  = bs_price(spot, strike, T, self.risk_free, sigma, "put")
        if price <= 0:
            return None

        # Size: spend premium_budget_pct of NAV
        premium_budget = self.premium_budget_pct * nav
        if not risk_budget.can_spend(premium_budget, nav):
            return None

        n_contracts = premium_budget / (price * spot + 1e-9)  -- fractional contracts
        total_prem  = n_contracts * price * spot
        risk_budget.record_spend(total_prem)

        delta  = bs_delta(spot, strike, T, self.risk_free, sigma, "put")
        gamma  = bs_gamma(spot, strike, T, self.risk_free, sigma)
        vega   = bs_vega(spot, strike, T, self.risk_free, sigma)
        theta  = bs_theta(spot, strike, T, self.risk_free, sigma, "put")

        pos = OptionPosition(
            option_type="put",
            overlay_type="protective_put",
            direction=1,          -- long put
            n_contracts=n_contracts,
            strike=strike,
            expiry_T=T,
            entry_price=price,
            current_price=price,
            delta=delta * n_contracts,
            gamma=gamma * n_contracts,
            vega=vega * n_contracts,
            theta=theta * n_contracts,
            pnl=0.0,
            entry_bar=bar_i,
        )
        self._active_puts.append(pos)
        return pos

    def update_positions(
        self,
        spot: float,
        sigma: float,
        bar_i: int,
    ) -> Tuple[float, float]:
        """
        Mark active puts to market, expire expired ones.
        Returns (total_pnl_today, net_delta).
        """
        total_pnl  = 0.0
        net_delta  = 0.0
        to_remove  = []

        for pos in self._active_puts:
            elapsed_years = (bar_i - pos.entry_bar) / 252.0
            remaining_T   = max(0.0, pos.expiry_T - elapsed_years)

            if remaining_T <= 0.0:
                # Expiry -- compute intrinsic value
                intrinsic = max(pos.strike - spot, 0.0)
                pnl_today = (intrinsic - pos.current_price) * pos.n_contracts * spot
                pos.pnl  += pnl_today
                pos.current_price = intrinsic
                pos.is_active     = False
                to_remove.append(pos)
                total_pnl += pnl_today
            else:
                new_price = bs_price(spot, pos.strike, remaining_T,
                                     self.risk_free, sigma, "put")
                pnl_today = (new_price - pos.current_price) * pos.n_contracts * spot
                pos.pnl  += pnl_today
                pos.current_price = new_price
                pos.expiry_T      = remaining_T
                pos.delta  = bs_delta(spot, pos.strike, remaining_T,
                                      self.risk_free, sigma, "put") * pos.n_contracts
                pos.gamma  = bs_gamma(spot, pos.strike, remaining_T,
                                      self.risk_free, sigma) * pos.n_contracts
                pos.theta  = bs_theta(spot, pos.strike, remaining_T,
                                      self.risk_free, sigma, "put") * pos.n_contracts
                total_pnl  += pnl_today
                net_delta  += pos.delta

        for pos in to_remove:
            self._active_puts.remove(pos)

        return total_pnl, net_delta

    def active_count(self) -> int:
        return len(self._active_puts)


# ---------------------------------------------------------------------------
# 2. CoveredCallOverlay
# ---------------------------------------------------------------------------

class CoveredCallOverlay:
    """
    Sells 1-month ATM calls on long equity when Hurst < 0.42 (mean reversion).

    Premium collected adds to return, but caps upside at strike.
    Only opens if there is an underlying long equity position.

    Parameters
    ----------
    hurst_trigger     : Hurst threshold to sell calls (default 0.42)
    expiry_days       : days to expiry (default 21 -- 1 month)
    risk_free         : risk-free rate (default 0.04)
    """

    def __init__(
        self,
        hurst_trigger: float = HURST_MEAN_REVERT,
        expiry_days: int     = 21,
        risk_free: float     = 0.04,
    ):
        self.hurst_trigger = hurst_trigger
        self.expiry_days   = expiry_days
        self.risk_free     = risk_free
        self._active_calls: List[OptionPosition] = []

    def check_trigger(self, hurst: float, equity_position: float) -> bool:
        """Return True if we should sell covered calls."""
        return hurst < self.hurst_trigger and equity_position > 0

    def open_call(
        self,
        spot: float,
        sigma: float,
        bar_i: int,
        nav: float,
        equity_exposure: float,  -- fraction of NAV in equity (0..1)
        risk_budget: OptionsRiskBudget,
    ) -> Optional[OptionPosition]:
        """
        Sell ATM call sized proportional to equity exposure.
        Premium is collected (negative premium cost).
        """
        strike = spot    -- ATM
        T      = self.expiry_days / 252.0
        price  = bs_price(spot, strike, T, self.risk_free, sigma, "call")
        if price <= 0:
            return None

        # n_contracts: match equity position size (1 contract = 1 unit)
        n_contracts = equity_exposure   -- sell 1 contract per unit of equity
        premium_collected = n_contracts * price * spot

        # Covered calls generate premium -- record as negative spend
        risk_budget.record_spend(-premium_collected)  -- negative means collected

        delta  = bs_delta(spot, strike, T, self.risk_free, sigma, "call")
        gamma  = bs_gamma(spot, strike, T, self.risk_free, sigma)
        vega   = bs_vega(spot, strike, T, self.risk_free, sigma)
        theta  = bs_theta(spot, strike, T, self.risk_free, sigma, "call")

        pos = OptionPosition(
            option_type="call",
            overlay_type="covered_call",
            direction=-1,         -- short call
            n_contracts=n_contracts,
            strike=strike,
            expiry_T=T,
            entry_price=price,
            current_price=price,
            delta=-delta * n_contracts,   -- short, so negate
            gamma=-gamma * n_contracts,
            vega=-vega * n_contracts,
            theta=-theta * n_contracts,   -- short theta is positive (collect decay)
            pnl=premium_collected,        -- initial P&L = premium received
            entry_bar=bar_i,
        )
        self._active_calls.append(pos)
        return pos

    def update_positions(
        self, spot: float, sigma: float, bar_i: int
    ) -> Tuple[float, float]:
        """
        Mark-to-market short calls. Returns (total_pnl_today, net_delta).
        P&L for short call: entry_price - current_price (short position).
        """
        total_pnl = 0.0
        net_delta = 0.0
        to_remove = []

        for pos in self._active_calls:
            elapsed_years = (bar_i - pos.entry_bar) / 252.0
            remaining_T   = max(0.0, pos.expiry_T - elapsed_years)

            if remaining_T <= 0.0:
                intrinsic  = max(spot - pos.strike, 0.0)
                pnl_today  = (pos.current_price - intrinsic) * pos.n_contracts * spot
                pos.pnl   += pnl_today
                pos.current_price = intrinsic
                pos.is_active     = False
                to_remove.append(pos)
                total_pnl += pnl_today
            else:
                new_price  = bs_price(spot, pos.strike, remaining_T,
                                      self.risk_free, sigma, "call")
                pnl_today  = (pos.current_price - new_price) * pos.n_contracts * spot  -- short: gain when price drops
                pos.pnl   += pnl_today
                pos.current_price = new_price
                pos.expiry_T      = remaining_T
                pos.delta  = -bs_delta(spot, pos.strike, remaining_T,
                                       self.risk_free, sigma, "call") * pos.n_contracts
                total_pnl  += pnl_today
                net_delta  += pos.delta

        for pos in to_remove:
            self._active_calls.remove(pos)

        return total_pnl, net_delta

    def active_count(self) -> int:
        return len(self._active_calls)


# ---------------------------------------------------------------------------
# 3. StraddleOverlay
# ---------------------------------------------------------------------------

class StraddleOverlay:
    """
    Buys ATM straddle (call + put) when GARCH forecast / realized_vol > 2.0.

    Delta-neutral at initiation, rebalance when |net delta| > delta_rebal_threshold.
    The straddle profits from large moves in either direction.

    Parameters
    ----------
    vol_ratio_trigger    : GARCH / realized ratio to trigger straddle (default 2.0)
    expiry_days          : days to expiry (default 21)
    delta_rebal_threshold: |delta| threshold to rebalance (default 0.10)
    premium_budget_pct   : premium budget per straddle as fraction of NAV (default 0.015)
    risk_free            : risk-free rate (default 0.04)
    """

    def __init__(
        self,
        vol_ratio_trigger: float    = 2.0,
        expiry_days: int            = 21,
        delta_rebal_threshold: float = 0.10,
        premium_budget_pct: float   = 0.015,
        risk_free: float            = 0.04,
    ):
        self.vol_ratio_trigger     = vol_ratio_trigger
        self.expiry_days           = expiry_days
        self.delta_rebal_threshold = delta_rebal_threshold
        self.premium_budget_pct    = premium_budget_pct
        self.risk_free             = risk_free
        self._call_legs: List[OptionPosition] = []
        self._put_legs:  List[OptionPosition] = []
        self._delta_hedge_cost     = 0.0  -- cumulative delta hedging P&L

    def check_trigger(self, garch_vol: float, realized_vol: float) -> bool:
        """Return True if GARCH/realized ratio exceeds trigger."""
        if realized_vol < 1e-6:
            return False
        return (garch_vol / realized_vol) > self.vol_ratio_trigger

    def open_straddle(
        self,
        spot: float,
        sigma: float,
        bar_i: int,
        nav: float,
        risk_budget: OptionsRiskBudget,
    ) -> Tuple[Optional[OptionPosition], Optional[OptionPosition]]:
        """
        Open ATM call + put (straddle). Returns (call_pos, put_pos) or (None, None).
        """
        strike = spot
        T      = self.expiry_days / 252.0
        call_p = bs_price(spot, strike, T, self.risk_free, sigma, "call")
        put_p  = bs_price(spot, strike, T, self.risk_free, sigma, "put")
        total_p = call_p + put_p

        if total_p <= 0:
            return None, None

        premium_budget = self.premium_budget_pct * nav
        if not risk_budget.can_spend(premium_budget, nav):
            return None, None

        n_contracts = premium_budget / (total_p * spot + 1e-9)
        risk_budget.record_spend(n_contracts * total_p * spot)

        def _make_leg(opt_type, price):
            delta  = bs_delta(spot, strike, T, self.risk_free, sigma, opt_type)
            gamma  = bs_gamma(spot, strike, T, self.risk_free, sigma)
            vega   = bs_vega(spot, strike, T, self.risk_free, sigma)
            theta  = bs_theta(spot, strike, T, self.risk_free, sigma, opt_type)
            return OptionPosition(
                option_type=opt_type,
                overlay_type=f"straddle_{opt_type}",
                direction=1,
                n_contracts=n_contracts,
                strike=strike,
                expiry_T=T,
                entry_price=price,
                current_price=price,
                delta=delta * n_contracts,
                gamma=gamma * n_contracts,
                vega=vega * n_contracts,
                theta=theta * n_contracts,
                pnl=0.0,
                entry_bar=bar_i,
            )

        call_pos = _make_leg("call", call_p)
        put_pos  = _make_leg("put",  put_p)
        self._call_legs.append(call_pos)
        self._put_legs.append(put_pos)
        return call_pos, put_pos

    def net_delta(self) -> float:
        """Net delta of entire straddle portfolio."""
        d = sum(p.delta for p in self._call_legs if p.is_active)
        d += sum(p.delta for p in self._put_legs  if p.is_active)
        return d

    def rebalance_delta(
        self, spot: float, sigma: float, equity_price: float
    ) -> float:
        """
        If |net_delta| > threshold, delta hedge by trading underlying.
        Returns cost of delta hedge as a fraction of NAV.
        """
        nd = self.net_delta()
        if abs(nd) <= self.delta_rebal_threshold:
            return 0.0
        # Hedge by trading nd units of underlying at spot
        hedge_cost = abs(nd) * spot * 0.001   -- 0.1% execution cost estimate
        self._delta_hedge_cost += hedge_cost
        # Zero out deltas (conceptually hedged)
        for pos in self._call_legs + self._put_legs:
            pos.delta = 0.0
        return hedge_cost

    def update_positions(
        self, spot: float, sigma: float, bar_i: int, nav: float
    ) -> Tuple[float, float]:
        """Mark to market. Returns (total_pnl_today, net_delta_after_rebal)."""
        total_pnl = 0.0

        for pos_list in [self._call_legs, self._put_legs]:
            to_remove = []
            for pos in pos_list:
                elapsed_years = (bar_i - pos.entry_bar) / 252.0
                remaining_T   = max(0.0, pos.expiry_T - elapsed_years)

                if remaining_T <= 0.0:
                    if pos.option_type == "call":
                        intrinsic = max(spot - pos.strike, 0.0)
                    else:
                        intrinsic = max(pos.strike - spot, 0.0)
                    pnl_today = (intrinsic - pos.current_price) * pos.n_contracts * spot
                    pos.pnl  += pnl_today
                    pos.current_price = intrinsic
                    pos.is_active     = False
                    to_remove.append(pos)
                    total_pnl += pnl_today
                else:
                    new_price = bs_price(spot, pos.strike, remaining_T,
                                         self.risk_free, sigma, pos.option_type)
                    pnl_today = (new_price - pos.current_price) * pos.n_contracts * spot
                    pos.pnl  += pnl_today
                    pos.current_price = new_price
                    pos.expiry_T      = remaining_T
                    pos.delta = bs_delta(spot, pos.strike, remaining_T,
                                         self.risk_free, sigma, pos.option_type) * pos.n_contracts
                    total_pnl += pnl_today
            for pos in to_remove:
                pos_list.remove(pos)

        # Delta rebalance
        hedge_cost = self.rebalance_delta(spot, sigma, spot)
        total_pnl -= hedge_cost

        return total_pnl, self.net_delta()


# ---------------------------------------------------------------------------
# 4. OptionsOverlayStrategy
# ---------------------------------------------------------------------------

class OptionsOverlayStrategy:
    """
    Adds options positions as overlays to existing equity exposure:
      - Protective puts when BH mass > BH_COLLAPSE
      - Covered calls when Hurst < 0.42 (mean reversion)
      - Straddles when GARCH vol forecasts expansion > 2x current

    Parameters
    ----------
    config : dict of parameter overrides
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.config = cfg

        self._put_overlay  = ProtectivePutOverlay(
            bh_trigger=cfg.get("bh_trigger", BH_COLLAPSE),
            otm_pct=cfg.get("otm_pct", 0.05),
            expiry_days=cfg.get("expiry_days", 21),
            premium_budget_pct=cfg.get("put_premium_budget_pct", 0.01),
            risk_free=cfg.get("risk_free", 0.04),
        )
        self._call_overlay = CoveredCallOverlay(
            hurst_trigger=cfg.get("hurst_trigger", HURST_MEAN_REVERT),
            expiry_days=cfg.get("expiry_days", 21),
            risk_free=cfg.get("risk_free", 0.04),
        )
        self._straddle     = StraddleOverlay(
            vol_ratio_trigger=cfg.get("vol_ratio_trigger", 2.0),
            expiry_days=cfg.get("expiry_days", 21),
            delta_rebal_threshold=cfg.get("delta_rebal_threshold", 0.10),
            premium_budget_pct=cfg.get("straddle_premium_budget_pct", 0.015),
            risk_free=cfg.get("risk_free", 0.04),
        )
        self._risk_budget  = OptionsRiskBudget(
            max_monthly_premium_pct=cfg.get("max_monthly_premium_pct", 0.02),
            bars_per_month=cfg.get("bars_per_month", 21),
        )
        self._hurst_window  = cfg.get("hurst_window", 120)
        self._vol_window    = cfg.get("vol_window", 21)
        self._garch_alpha   = cfg.get("garch_alpha", 0.09)
        self._garch_beta    = cfg.get("garch_beta", 0.90)

    def run_bar(
        self,
        bar_i: int,
        spot: float,
        returns_history: np.ndarray,
        prices_history: np.ndarray,
        nav: float,
        equity_exposure: float = 1.0,
    ) -> Dict[str, float]:
        """
        Process one bar. Returns dict with keys:
          put_pnl, call_pnl, straddle_pnl, net_options_pnl,
          bh_mass, hurst, garch_vol, realized_vol
        """
        self._risk_budget.new_bar(bar_i)

        # Compute market signals
        bh_mass_val  = _bh_mass(returns_history)
        hurst_val    = _compute_hurst(prices_history[-self._hurst_window:]) if len(prices_history) >= self._hurst_window else 0.5
        realized_vol = float(np.std(returns_history[-self._vol_window:]) * math.sqrt(252)) if len(returns_history) >= self._vol_window else 0.15
        garch_v      = _garch_vol(returns_history[-100:], self._garch_alpha, self._garch_beta) if len(returns_history) >= 20 else realized_vol
        sigma        = max(realized_vol, 0.05)

        # Update existing positions
        put_pnl,   put_delta   = self._put_overlay.update_positions(spot, sigma, bar_i)
        call_pnl,  call_delta  = self._call_overlay.update_positions(spot, sigma, bar_i)
        strad_pnl, strad_delta = self._straddle.update_positions(spot, sigma, bar_i, nav)

        # Check for new position openings
        if self._put_overlay.check_trigger(bh_mass_val):
            self._put_overlay.open_put(spot, sigma, nav, bar_i, self._risk_budget)

        if self._call_overlay.check_trigger(hurst_val, equity_exposure):
            self._call_overlay.open_call(spot, sigma, bar_i, nav, equity_exposure, self._risk_budget)

        if self._straddle.check_trigger(garch_v, realized_vol):
            self._straddle.open_straddle(spot, sigma, bar_i, nav, self._risk_budget)

        net_pnl = put_pnl + call_pnl + strad_pnl

        return {
            "put_pnl":       put_pnl,
            "call_pnl":      call_pnl,
            "straddle_pnl":  strad_pnl,
            "net_options_pnl": net_pnl,
            "bh_mass":       bh_mass_val,
            "hurst":         hurst_val,
            "garch_vol":     garch_v,
            "realized_vol":  realized_vol,
            "budget_used":   self._risk_budget.budget_used_pct(nav),
        }


# ---------------------------------------------------------------------------
# 5. OptionsBacktest
# ---------------------------------------------------------------------------

class OptionsBacktest:
    """
    Simulate options overlay with Black-Scholes pricing and delta hedging P&L.

    The underlying equity position earns (close returns * equity_signal).
    Options overlays add/subtract from the equity P&L.

    Parameters
    ----------
    config         : passed to OptionsOverlayStrategy
    initial_equity : starting NAV (default 1_000_000)
    commission_pct : equity commission per side (default 0.0001)
    equity_signal  : fixed equity position (default 1.0 -- always long)
    """

    def __init__(
        self,
        config: Optional[dict]  = None,
        initial_equity: float   = 1_000_000.0,
        commission_pct: float   = 0.0001,
        equity_signal: float    = 1.0,
    ):
        self.config         = config or {}
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct
        self.equity_signal  = equity_signal

    def run(self, df: pd.DataFrame) -> OptionsBacktestResult:
        """
        Run options overlay backtest on OHLCV DataFrame.

        df must have 'close' column. 'high'/'low' optional.
        """
        n     = len(df)
        close = df["close"].values
        rets  = np.concatenate([[0.0], np.diff(np.log(close + 1e-9))])

        strat      = OptionsOverlayStrategy(config=self.config)
        equity     = self.initial_equity
        eq_curve   = np.full(n, self.initial_equity, dtype=float)
        overlay_pnl_arr = np.zeros(n)
        trade_ret  = []
        total_prem_paid = 0.0
        total_prem_coll = 0.0

        warmup = max(strat._hurst_window, 50)

        for i in range(1, n):
            # Underlying equity P&L
            eq_ret = float(rets[i]) * self.equity_signal
            equity *= (1.0 + eq_ret)

            if i < warmup:
                eq_curve[i] = equity
                continue

            # Options overlay P&L
            bar_result = strat.run_bar(
                bar_i=i,
                spot=close[i],
                returns_history=rets[max(0, i - 252): i + 1],
                prices_history=close[max(0, i - 252): i + 1],
                nav=equity,
                equity_exposure=self.equity_signal,
            )

            overlay = bar_result["net_options_pnl"]
            equity += overlay
            overlay_pnl_arr[i] = overlay

            if overlay != 0:
                trade_ret.append(overlay / (equity + 1e-9))

            eq_curve[i] = equity

        # Crude split between premiums paid vs collected
        total_prem_paid = max(0.0, strat._risk_budget._spent_this_month)
        total_prem_coll = max(0.0, -strat._risk_budget._spent_this_month)

        stats = _compute_stats(eq_curve, trade_ret)
        return OptionsBacktestResult(
            total_premium_paid=total_prem_paid,
            total_premium_collected=total_prem_coll,
            delta_hedge_cost=strat._straddle._delta_hedge_cost,
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            overlay_pnl=pd.Series(overlay_pnl_arr, index=df.index),
            params=self.config,
        )

    def run_scenarios(
        self, df: pd.DataFrame, param_grid: List[dict]
    ) -> pd.DataFrame:
        """
        Run backtest over a grid of configs. Returns summary DataFrame.
        """
        rows = []
        for cfg in param_grid:
            bt  = OptionsBacktest(config=cfg, initial_equity=self.initial_equity,
                                  commission_pct=self.commission_pct,
                                  equity_signal=self.equity_signal)
            res = bt.run(df)
            row = {
                "config":           str(cfg),
                "total_return":     res.total_return,
                "cagr":             res.cagr,
                "sharpe":           res.sharpe,
                "max_drawdown":     res.max_drawdown,
                "prem_paid":        res.total_premium_paid,
                "prem_collected":   res.total_premium_collected,
                "delta_hedge_cost": res.delta_hedge_cost,
            }
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(21)
    n   = 1260    -- ~5 years
    idx = pd.date_range("2019-01-01", periods=n, freq="B")

    # Simulate with regime changes
    seg = n // 4
    vols   = [0.010, 0.020, 0.035, 0.012]
    drifts = [0.0004, 0.0001, -0.0005, 0.0003]
    rets_raw = np.concatenate([
        rng.normal(d, v, seg) for d, v in zip(drifts, vols)
    ])
    close = 100.0 * np.cumprod(1 + rets_raw[:n])
    high  = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol   = rng.integers(500_000, 2_000_000, n).astype(float)

    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)

    # Run options overlay backtest
    bt     = OptionsBacktest(
        config={"bh_trigger": 0.85, "hurst_trigger": 0.42, "vol_ratio_trigger": 1.8},
        initial_equity=1_000_000,
    )
    result = bt.run(df)
    print("Options overlay:", result.summary())

    # BS pricing sanity check
    S, K, T, r, sigma = 100.0, 100.0, 30 / 252, 0.04, 0.20
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    dc = bs_delta(S, K, T, r, sigma, "call")
    dp = bs_delta(S, K, T, r, sigma, "put")
    print(f"BS call={c:.4f} put={p:.4f} call_delta={dc:.4f} put_delta={dp:.4f}")

    # Scenario grid
    grid = [
        {"bh_trigger": 0.80, "vol_ratio_trigger": 2.0},
        {"bh_trigger": 0.90, "vol_ratio_trigger": 1.5},
        {"bh_trigger": 0.95, "vol_ratio_trigger": 1.8},
    ]
    tbl = bt.run_scenarios(df, grid)
    print(tbl.to_string())
