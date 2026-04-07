"""
transaction_costs.py -- Realistic transaction cost modeling for SRFM.

Implements Almgren-Chriss optimal execution, spread models by asset class,
empirical slippage regression, and a unified cost estimation interface.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost estimate dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostEstimate:
    """Breakdown of estimated transaction costs for a single order."""
    spread_cost_bps: float       # half-spread cost in basis points
    market_impact_bps: float     # estimated market impact in basis points
    commission_bps: float        # commission in basis points
    slippage_bps: float          # additional slippage in basis points
    total_bps: float             # total all-in cost in basis points
    dollar_cost: float           # total cost in dollars
    fill_price: float            # estimated average fill price
    execution_time_bars: int     # estimated bars needed for full fill

    def to_dict(self) -> Dict[str, float]:
        return {
            "spread_cost_bps": self.spread_cost_bps,
            "market_impact_bps": self.market_impact_bps,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_bps": self.total_bps,
            "dollar_cost": self.dollar_cost,
            "fill_price": self.fill_price,
        }


# ---------------------------------------------------------------------------
# Spread model
# ---------------------------------------------------------------------------

# Asset class spread lookup in bps (half-spread, one-way)
SPREAD_TABLE: Dict[str, float] = {
    # Equities
    "equity_large_cap": 3.0,
    "equity_mid_cap": 7.0,
    "equity_small_cap": 15.0,
    "equity_micro_cap": 30.0,
    # Crypto
    "crypto_btc": 2.0,
    "crypto_eth": 3.0,
    "crypto_altcoin": 20.0,
    "crypto_defi": 35.0,
    # Futures
    "futures_equity_index": 1.0,
    "futures_commodity": 4.0,
    "futures_fx": 2.0,
    # Forex
    "fx_major": 1.0,
    "fx_minor": 3.0,
    "fx_exotic": 15.0,
}

# Keywords to classify symbols into asset classes
_CLASS_KEYWORDS: List[Tuple[str, str]] = [
    ("BTC", "crypto_btc"),
    ("ETH", "crypto_eth"),
    ("SOL", "crypto_altcoin"),
    ("AVAX", "crypto_altcoin"),
    ("ADA", "crypto_altcoin"),
    ("DOGE", "crypto_altcoin"),
    ("MATIC", "crypto_altcoin"),
    ("LINK", "crypto_altcoin"),
    ("UNI", "crypto_defi"),
    ("AAVE", "crypto_defi"),
    ("ES", "futures_equity_index"),
    ("NQ", "futures_equity_index"),
    ("CL", "futures_commodity"),
    ("GC", "futures_commodity"),
    ("EUR", "fx_major"),
    ("GBP", "fx_major"),
    ("JPY", "fx_major"),
    ("SPY", "equity_large_cap"),
    ("QQQ", "equity_large_cap"),
    ("IWM", "equity_mid_cap"),
]


class SpreadModel:
    """
    Empirical spread estimates by asset class.

    Spread can vary with ADV: illiquid symbols (low ADV) pay higher spread.
    Uses a square-root adjustment: spread_adj = spread_base * (adv_ref / adv)^0.3
    """

    ADV_REF = 1_000_000.0    # reference ADV in shares/contracts for no adjustment
    ADV_POWER = 0.3           # exponent on liquidity adjustment

    def __init__(self, custom_spreads: Optional[Dict[str, float]] = None):
        self._table = dict(SPREAD_TABLE)
        if custom_spreads:
            self._table.update(custom_spreads)
        self._symbol_class_cache: Dict[str, str] = {}

    def classify_symbol(self, symbol: str) -> str:
        """
        Heuristically classify a symbol into an asset class.

        Returns a key into SPREAD_TABLE.
        """
        sym_upper = symbol.upper().replace("-", "").replace("_", "").replace("/", "")
        for kw, cls in _CLASS_KEYWORDS:
            if kw in sym_upper:
                return cls

        # Default to equity large-cap if symbol length <= 4, else small-cap
        if len(symbol) <= 4:
            return "equity_large_cap"
        return "equity_small_cap"

    def get_spread(self, symbol: str, adv: float = 1_000_000.0) -> float:
        """
        Return estimated half-spread in basis points for the given symbol and ADV.

        adv: average daily volume in shares/contracts/USD
        """
        if symbol not in self._symbol_class_cache:
            self._symbol_class_cache[symbol] = self.classify_symbol(symbol)

        asset_class = self._symbol_class_cache[symbol]
        base_spread = self._table.get(asset_class, SPREAD_TABLE["equity_large_cap"])

        # Liquidity adjustment
        if adv > 0:
            adj = (self.ADV_REF / adv) ** self.ADV_POWER
            adj = max(0.5, min(adj, 10.0))  # clamp adjustment to [0.5x, 10x]
        else:
            adj = 5.0  # very illiquid default

        return base_spread * adj

    def full_spread(self, symbol: str, adv: float = 1_000_000.0) -> float:
        """Return full round-trip spread (2x half-spread) in bps."""
        return 2.0 * self.get_spread(symbol, adv)


# ---------------------------------------------------------------------------
# Almgren-Chriss execution model
# ---------------------------------------------------------------------------

class AlmgrenChrissModel:
    """
    Closed-form optimal execution based on Almgren-Chriss (2001).

    Models temporary and permanent market impact to find the trade trajectory
    that minimizes expected cost + variance of execution cost.

    Notation:
      X    -- total quantity to trade (shares)
      T    -- total time horizon (seconds or bars)
      N    -- number of child orders
      eta  -- temporary impact coefficient (price per unit rate)
      gamma -- permanent impact coefficient (price per unit quantity)
      sigma -- price volatility (price / sqrt(time))
      lam  -- risk aversion (lambda)
    """

    def __init__(
        self,
        eta: float = 0.1,        # temporary impact coefficient
        gamma: float = 0.01,     # permanent impact coefficient
        lam: float = 1e-6,       # risk aversion
    ):
        self.eta = eta
        self.gamma = gamma
        self.lam = lam

    def temporary_impact(
        self,
        participation_rate: float,  # fraction of volume traded per unit time
        sigma: float,               # price volatility
        eta: Optional[float] = None,
    ) -> float:
        """
        Estimate temporary price impact in bps.

        Formula: eta * sigma * sqrt(participation_rate)

        Returns impact in bps (relative to current price).
        """
        _eta = eta if eta is not None else self.eta
        impact = _eta * sigma * np.sqrt(max(participation_rate, 0.0))
        return impact * 10_000.0  # convert to bps

    def permanent_impact(
        self,
        qty: float,    # quantity to trade
        adv: float,    # average daily volume
        sigma: float,  # price volatility (fraction per day)
        gamma: Optional[float] = None,
    ) -> float:
        """
        Estimate permanent price impact in bps.

        Formula: gamma * sigma * (qty / adv)

        Returns impact in bps.
        """
        _gamma = gamma if gamma is not None else self.gamma
        if adv <= 0:
            return 50.0  # default for unknown liquidity
        perm = _gamma * sigma * (abs(qty) / adv)
        return perm * 10_000.0  # convert to bps

    def optimal_trajectory(
        self,
        qty: float,           # total quantity to execute (signed)
        T: float,             # total time (bars)
        n_intervals: int,     # number of child orders
        sigma: float,         # volatility per bar
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute the optimal trade schedule (Almgren-Chriss closed form).

        Returns array of shape (n_intervals,) with trade sizes per interval.
        Positive qty = buy, negative = sell; trade sizes follow same sign.

        The closed-form solution uses:
          kappa = sqrt(lam * sigma^2 / eta)
          x_j = X * sinh(kappa*(T-t_j)) / sinh(kappa*T)
        where x_j is the remaining inventory at time t_j.
        Trades are the differences: n_j = x_{j-1} - x_j
        """
        _eta = eta if eta is not None else self.eta
        _gamma = gamma if gamma is not None else self.gamma

        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")

        dt = T / n_intervals

        # Decay coefficient
        kappa_sq = self.lam * sigma ** 2 / max(_eta, 1e-12)
        kappa = np.sqrt(max(kappa_sq, 0.0))

        if kappa * T < 1e-8:
            # Degenerate case: uniform execution
            trades = np.full(n_intervals, qty / n_intervals)
            return trades

        # Time grid: t_0=0, t_1=dt, ..., t_N=T
        t_grid = np.linspace(0.0, T, n_intervals + 1)

        # Remaining inventory at each node
        sinh_kT = np.sinh(kappa * T)
        if abs(sinh_kT) < 1e-12:
            trades = np.full(n_intervals, qty / n_intervals)
            return trades

        inventory = qty * np.sinh(kappa * (T - t_grid)) / sinh_kT

        # Trades at each interval = inventory decrease
        trades = inventory[:-1] - inventory[1:]
        return trades

    def expected_cost(
        self,
        qty: float,
        trajectory: np.ndarray,
        T: float,
        sigma: float,
        price: float,
        adv: float,
    ) -> float:
        """
        Estimate expected dollar cost of executing via given trajectory.

        Includes both temporary and permanent impact components.
        """
        n = len(trajectory)
        dt = T / max(n, 1)

        total_cost = 0.0
        remaining = abs(qty)

        for i, trade in enumerate(trajectory):
            if remaining <= 0:
                break
            part_rate = abs(trade) / (adv * dt + 1e-12)
            temp_bps = self.temporary_impact(part_rate, sigma)
            perm_bps = self.permanent_impact(abs(trade), adv, sigma)
            trade_value = abs(trade) * price
            total_cost += trade_value * (temp_bps + perm_bps) / 10_000.0
            remaining -= abs(trade)

        return total_cost

    def twap_trajectory(self, qty: float, n_intervals: int) -> np.ndarray:
        """Return uniform TWAP schedule as comparison baseline."""
        return np.full(n_intervals, qty / max(n_intervals, 1))

    def vwap_trajectory(
        self, qty: float, volume_profile: np.ndarray
    ) -> np.ndarray:
        """
        Return VWAP-weighted schedule based on a volume profile.

        volume_profile: relative volume per interval (will be normalized).
        """
        profile = np.array(volume_profile, dtype=float)
        total = profile.sum()
        if total <= 0:
            return self.twap_trajectory(qty, len(profile))
        return qty * profile / total


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    Empirical slippage model fitted from historical fill data.

    Default model (unfitted):
      slippage_bps = 0.5 * spread_bps + impact_bps

    Fitted model:
      Uses OLS regression on features [log(qty/adv), volatility, urgency_flag].
    """

    def __init__(self, spread_model: Optional[SpreadModel] = None):
        self._spread = spread_model or SpreadModel()
        self._fitted = False
        self._coefs: np.ndarray = np.array([0.5, 0.3, 0.2, 1.0])
        # coefs: [intercept, log_pov_coef, vol_coef, urgency_coef]
        self._r2: float = 0.0
        self._n_obs: int = 0

    def fit(self, fills_df: pd.DataFrame) -> None:
        """
        Fit regression model on historical fill data.

        Required columns:
          - symbol: str
          - qty: float
          - adv: float (average daily volume)
          - price: float
          - fill_price: float (actual fill)
          - volatility: float (realized vol, annualized)
          - urgency: int (0=passive, 1=urgent)

        Computes realized_slippage_bps = (fill_price - price) / price * 1e4
        for buys, inverted for sells.
        """
        required = {"symbol", "qty", "adv", "price", "fill_price", "volatility"}
        missing = required - set(fills_df.columns)
        if missing:
            raise ValueError(f"fills_df missing columns: {missing}")

        df = fills_df.dropna(subset=list(required)).copy()
        if len(df) < 10:
            logger.warning("Insufficient fill data (%d rows) to fit slippage model", len(df))
            return

        df["side"] = np.sign(df["qty"])  # +1 buy, -1 sell
        df["realized_slip_bps"] = (
            (df["fill_price"] - df["price"]) / df["price"] * 10_000.0 * df["side"]
        )

        # Feature construction
        df["log_pov"] = np.log1p(np.abs(df["qty"]) / (df["adv"].clip(lower=1.0)))
        df["urgency"] = df.get("urgency", 0).fillna(0).astype(float)

        X = np.column_stack([
            np.ones(len(df)),
            df["log_pov"].values,
            df["volatility"].values,
            df["urgency"].values,
        ])
        y = df["realized_slip_bps"].values

        # OLS: (X'X)^-1 X'y
        try:
            xtx = X.T @ X
            xty = X.T @ y
            coefs = np.linalg.solve(xtx + 1e-8 * np.eye(4), xty)
            y_hat = X @ coefs
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            self._r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            self._coefs = coefs
            self._fitted = True
            self._n_obs = len(df)
            logger.info(
                "SlippageModel fit: n=%d, R2=%.3f, coefs=%s",
                self._n_obs, self._r2, self._coefs.round(4),
            )
        except np.linalg.LinAlgError as exc:
            logger.error("SlippageModel fit failed: %s", exc)

    def predict(
        self,
        symbol: str,
        qty: float,
        adv: float,
        volatility: float,
        urgency: int = 0,
    ) -> float:
        """
        Predict expected slippage in basis points.

        Returns positive bps (always a cost).
        """
        if self._fitted:
            log_pov = float(np.log1p(abs(qty) / max(adv, 1.0)))
            x = np.array([1.0, log_pov, volatility, float(urgency)])
            slip = float(self._coefs @ x)
            return max(0.0, slip)
        else:
            # Default: 0.5 * spread + impact proxy
            spread_bps = self._spread.get_spread(symbol, adv)
            impact_bps = 0.1 * abs(qty) / max(adv, 1.0) * 10_000.0
            return 0.5 * spread_bps + impact_bps

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def r_squared(self) -> float:
        return self._r2


# ---------------------------------------------------------------------------
# Unified transaction cost model
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Unified interface for estimating all-in transaction costs.

    Combines:
      - Spread model (half-spread cost)
      - Almgren-Chriss market impact
      - Slippage regression model
      - Fixed commission schedule
    """

    def __init__(
        self,
        commission_bps: float = 5.0,        # flat commission per side
        spread_model: Optional[SpreadModel] = None,
        impact_model: Optional[AlmgrenChrissModel] = None,
        slippage_model: Optional[SlippageModel] = None,
    ):
        self.commission_bps = commission_bps
        self._spread = spread_model or SpreadModel()
        self._impact = impact_model or AlmgrenChrissModel()
        self._slippage = slippage_model or SlippageModel(spread_model=self._spread)

    def estimate_cost(
        self,
        symbol: str,
        qty: float,            # signed quantity (+buy, -sell)
        price: float,          # current market price
        side: str,             # "BUY" or "SELL"
        urgency: float = 0.5,  # 0=passive/TWAP, 1=immediate/market
        adv: float = 1_000_000.0,
        volatility: float = 0.02,  # annualized vol as fraction
        sigma_bar: float = 0.001,  # per-bar volatility for impact
    ) -> CostEstimate:
        """
        Estimate all-in cost for an order.

        Returns CostEstimate with spread, impact, slippage, commission, total.
        """
        if price <= 0 or qty == 0:
            return CostEstimate(
                spread_cost_bps=0.0,
                market_impact_bps=0.0,
                commission_bps=0.0,
                slippage_bps=0.0,
                total_bps=0.0,
                dollar_cost=0.0,
                fill_price=price,
                execution_time_bars=1,
            )

        abs_qty = abs(qty)
        side_sign = 1.0 if side.upper() == "BUY" else -1.0

        # Spread cost (half-spread per side)
        spread_bps = self._spread.get_spread(symbol, adv)

        # Market impact
        participation_rate = abs_qty / max(adv, 1.0)
        temp_impact_bps = self._impact.temporary_impact(participation_rate, sigma_bar)
        perm_impact_bps = self._impact.permanent_impact(abs_qty, adv, volatility)
        impact_bps = temp_impact_bps + perm_impact_bps

        # Slippage
        urgency_flag = 1 if urgency > 0.6 else 0
        slip_bps = self._slippage.predict(symbol, abs_qty, adv, volatility, urgency_flag)

        # Commission
        comm_bps = self.commission_bps

        # Total
        total_bps = spread_bps + impact_bps + slip_bps + comm_bps

        # Dollar cost
        notional = abs_qty * price
        dollar_cost = notional * total_bps / 10_000.0

        # Estimated fill price
        fill_adj = side_sign * total_bps / 10_000.0
        fill_price = price * (1.0 + fill_adj)

        # Estimated execution bars: based on participation rate
        if participation_rate > 0.20:
            exec_bars = max(1, int(np.ceil(participation_rate / 0.05)))
        else:
            exec_bars = 1

        return CostEstimate(
            spread_cost_bps=spread_bps,
            market_impact_bps=impact_bps,
            commission_bps=comm_bps,
            slippage_bps=slip_bps,
            total_bps=total_bps,
            dollar_cost=dollar_cost,
            fill_price=fill_price,
            execution_time_bars=exec_bars,
        )

    def batch_estimate(
        self,
        orders: List[Dict],
    ) -> pd.DataFrame:
        """
        Estimate costs for a list of orders.

        Each order dict must have: symbol, qty, price, side.
        Optional: urgency, adv, volatility.

        Returns DataFrame with one row per order.
        """
        rows = []
        for order in orders:
            est = self.estimate_cost(
                symbol=order["symbol"],
                qty=float(order["qty"]),
                price=float(order["price"]),
                side=str(order.get("side", "BUY")),
                urgency=float(order.get("urgency", 0.5)),
                adv=float(order.get("adv", 1_000_000.0)),
                volatility=float(order.get("volatility", 0.02)),
            )
            row = {
                "symbol": order["symbol"],
                "qty": order["qty"],
                "price": order["price"],
                **est.to_dict(),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def annual_cost_drag(
        self,
        turnover_per_year: float,   # fraction of portfolio turned over per year
        portfolio_value: float,
        avg_cost_bps: float,
    ) -> float:
        """
        Estimate annualized cost drag in dollars.

        drag = portfolio_value * turnover * avg_cost_bps / 10000
        """
        return portfolio_value * turnover_per_year * avg_cost_bps / 10_000.0


# ---------------------------------------------------------------------------
# Commission schedule by broker/venue type
# ---------------------------------------------------------------------------

COMMISSION_SCHEDULES: Dict[str, Dict[str, float]] = {
    "retail_equity": {
        "per_share": 0.0,
        "minimum": 0.0,
        "bps": 5.0,
    },
    "institutional_equity": {
        "per_share": 0.005,
        "minimum": 1.0,
        "bps": 2.0,
    },
    "crypto_exchange": {
        "per_share": 0.0,
        "minimum": 0.0,
        "bps": 10.0,
    },
    "futures_cme": {
        "per_share": 2.25,    # per contract
        "minimum": 2.25,
        "bps": 0.0,
    },
}


def compute_commission(
    qty: float,
    price: float,
    schedule_name: str = "retail_equity",
    custom_schedule: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute commission in dollars using named schedule or custom schedule.

    Returns commission in dollars.
    """
    sched = custom_schedule or COMMISSION_SCHEDULES.get(
        schedule_name, COMMISSION_SCHEDULES["retail_equity"]
    )
    per_share = float(sched.get("per_share", 0.0))
    minimum = float(sched.get("minimum", 0.0))
    bps = float(sched.get("bps", 5.0))

    comm = max(
        abs(qty) * per_share,
        abs(qty) * price * bps / 10_000.0,
    )
    return max(comm, minimum)


# ---------------------------------------------------------------------------
# Market microstructure analytics
# ---------------------------------------------------------------------------

class MicrostructureAnalytics:
    """
    Analyze market microstructure from tick/bar data.

    Estimates effective spread, price impact, and Kyle's lambda.
    """

    @staticmethod
    def roll_spread(prices: pd.Series) -> float:
        """
        Estimate effective spread using Roll (1984) model.

        spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))

        Returns spread in price units, or 0.0 if cov is positive.
        """
        dp = prices.diff().dropna()
        cov = float(dp.cov(dp.shift(1).dropna().reindex(dp.index)))
        if cov >= 0:
            return 0.0
        return 2.0 * np.sqrt(-cov)

    @staticmethod
    def kyle_lambda(
        price_changes: pd.Series,
        signed_order_flow: pd.Series,
    ) -> float:
        """
        Estimate Kyle's lambda (price impact per unit of order flow).

        lambda = cov(dp, q) / var(q)

        Higher lambda = less liquid.
        """
        dp = price_changes.dropna()
        q = signed_order_flow.reindex(dp.index).dropna()
        dp = dp.reindex(q.index)

        var_q = float(q.var())
        if var_q < 1e-12:
            return 0.0
        cov_dq = float(dp.cov(q))
        return cov_dq / var_q

    @staticmethod
    def amihud_illiquidity(
        returns: pd.Series,
        dollar_volume: pd.Series,
    ) -> float:
        """
        Amihud (2002) illiquidity ratio.

        ILLIQ = mean(|r_t| / dvol_t)

        Higher = less liquid (more impact per dollar traded).
        """
        aligned = pd.DataFrame({"ret": returns, "dvol": dollar_volume}).dropna()
        aligned = aligned[aligned["dvol"] > 0]
        if aligned.empty:
            return 0.0
        ratios = aligned["ret"].abs() / aligned["dvol"]
        return float(ratios.mean())
