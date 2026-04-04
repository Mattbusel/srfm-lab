"""
market_making.py — Market making strategies.

References:
  - Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
  - Guéant, Lehalle & Tapia (2012): "Dealing with the inventory risk"
  - Ho & Stoll (1981): Optimal dealer pricing under transactions and return uncertainty
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MMBacktestResult:
    """Result container for market making backtests."""
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    inventory_std: float = 0.0
    avg_spread_captured: float = 0.0
    fill_rate: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    inventory_series: pd.Series = field(default_factory=pd.Series)
    spread_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"PnL={self.total_pnl:.2f} Return={self.total_return:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"N_trades={self.n_trades} Spread={self.avg_spread_captured:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. AvellanedaStoikovMM
# ─────────────────────────────────────────────────────────────────────────────

class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov optimal market making model (2008).

    The model computes optimal bid/ask quotes that account for:
    1. Inventory risk (risk-aversion parameter gamma)
    2. Order arrival rate (k)
    3. Remaining trading time (T - t)
    4. Mid-price volatility (sigma)

    Key formulas:
        Reservation price: r = s - q * gamma * sigma^2 * (T - t)
        Optimal spread:    delta = gamma * sigma^2 * (T - t) + 2/gamma * ln(1 + gamma/k)

        bid = r - delta/2
        ask = r + delta/2

    where s = mid price, q = current inventory.

    Parameters
    ----------
    gamma  : risk aversion (0 = risk neutral, higher = more averse, default 0.1)
    sigma  : mid-price volatility per unit time (default 0.01)
    k      : market order arrival rate per unit time (default 1.5)
    T      : total trading session length in time units (default 1.0)
    dt     : time step size (default 1/252 for daily, use 1/86400 for seconds)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 0.01,
        k: float = 1.5,
        T: float = 1.0,
        dt: float = 1.0 / 252,
    ):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = T
        self.dt = dt

    def reservation_price(self, mid: float, inventory: float, time_remaining: float) -> float:
        """
        Compute the market maker's reservation price (adjusted for inventory).

        r = s - q * gamma * sigma^2 * (T - t)
        """
        return mid - inventory * self.gamma * self.sigma ** 2 * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """
        Compute the optimal bid-ask spread.

        delta* = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        """
        if time_remaining <= 0:
            return 0.0
        term1 = self.gamma * self.sigma ** 2 * time_remaining
        term2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / (self.k + 1e-9))
        return term1 + term2

    def compute_quotes(
        self,
        mid: float,
        inventory: float,
        time_remaining: float,
    ) -> Tuple[float, float]:
        """
        Compute optimal bid and ask prices.

        Returns (bid_price, ask_price).
        """
        r = self.reservation_price(mid, inventory, time_remaining)
        delta = self.optimal_spread(time_remaining)
        bid = r - delta / 2.0
        ask = r + delta / 2.0
        return bid, ask

    def order_arrival_probability(self, offset: float) -> float:
        """
        Probability that a limit order at distance 'offset' from mid gets filled.

        P(fill) = exp(-k * offset)

        This is derived from the Poisson process assumption in the model.
        """
        return math.exp(-self.k * abs(offset))

    def simulate_session(
        self,
        mid_prices: np.ndarray,
        n_steps: int = None,
        initial_inventory: float = 0.0,
        max_inventory: float = 5.0,
        fill_probability_scale: float = 1.0,
        seed: int = 42,
    ) -> Dict:
        """
        Simulate a trading session using the AS model.

        Parameters
        ----------
        mid_prices             : array of mid prices
        n_steps                : number of time steps (default = len(mid_prices))
        initial_inventory      : starting inventory
        max_inventory          : maximum absolute inventory allowed
        fill_probability_scale : scale factor for fill probabilities
        seed                   : random seed

        Returns
        -------
        dict with pnl, inventory, trades, etc.
        """
        rng = np.random.default_rng(seed)
        n = len(mid_prices) if n_steps is None else min(n_steps, len(mid_prices))

        inventory = initial_inventory
        cash = 0.0
        pnl_series = np.zeros(n)
        inventory_series = np.zeros(n)
        spread_series = np.zeros(n)
        trades = []

        for t in range(n):
            mid = float(mid_prices[t])
            time_remaining = max(0.001, self.T - t * self.dt)

            # Compute quotes
            bid, ask = self.compute_quotes(mid, inventory, time_remaining)
            spread = ask - bid
            spread_series[t] = spread

            # Simulate fill arrivals (Poisson)
            lambda_b = self.k * math.exp(-self.k * (mid - bid) * fill_probability_scale)
            lambda_a = self.k * math.exp(-self.k * (ask - mid) * fill_probability_scale)

            # Fill at bid: sell order hits our bid
            if abs(inventory) < max_inventory:
                bid_fill = rng.poisson(lambda_b * self.dt)
                if bid_fill > 0:
                    n_fills = min(bid_fill, int(max_inventory - inventory))
                    if n_fills > 0:
                        cash -= n_fills * bid  # pay bid to buy
                        inventory += n_fills
                        trades.append({"type": "buy", "price": bid, "qty": n_fills, "t": t})

            # Fill at ask: buy order hits our ask
            if abs(inventory) < max_inventory:
                ask_fill = rng.poisson(lambda_a * self.dt)
                if ask_fill > 0:
                    n_fills = min(ask_fill, int(max_inventory + inventory))
                    if n_fills > 0:
                        cash += n_fills * ask  # receive ask to sell
                        inventory -= n_fills
                        trades.append({"type": "sell", "price": ask, "qty": n_fills, "t": t})

            # MTM PnL
            pnl_series[t] = cash + inventory * mid
            inventory_series[t] = inventory

        return {
            "pnl": pnl_series,
            "inventory": inventory_series,
            "spread": spread_series,
            "trades": trades,
            "final_pnl": float(pnl_series[-1]),
            "final_inventory": float(inventory),
            "n_trades": len(trades),
        }

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        max_inventory_pct: float = 0.05,
    ) -> MMBacktestResult:
        """
        Backtest the AS market making strategy on OHLCV data.

        Uses close price as mid price proxy.
        Volatility is estimated from the data.
        """
        mid = df["close"].values
        n = len(mid)

        # Estimate sigma from data
        rets = np.diff(np.log(mid + 1e-9))
        self.sigma = float(np.std(rets) * math.sqrt(252)) if len(rets) > 5 else self.sigma

        max_inventory = max_inventory_pct * initial_equity / mid.mean()

        sim = self.simulate_session(mid, n, initial_inventory=0.0, max_inventory=max_inventory)

        pnl = sim["pnl"]
        initial_capital = initial_equity
        equity_curve = initial_equity + pnl * (initial_equity / (initial_equity + 1))
        equity_curve = np.maximum(equity_curve, 0)

        total_return = (equity_curve[-1] - initial_equity) / initial_equity
        rets_arr = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        rets_arr = np.concatenate([[0], rets_arr])
        std = rets_arr.std()
        sharpe = rets_arr.mean() / std * math.sqrt(252) if std > 0 else 0.0
        pk = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - pk) / (pk + 1e-9)
        max_dd = dd.min()

        trades = sim["trades"]
        spread_captured = [abs(t["price"] - mid[t["t"]]) for t in trades] if trades else [0.0]
        avg_spread = float(np.mean(spread_captured)) if spread_captured else 0.0

        return MMBacktestResult(
            total_pnl=float(sim["final_pnl"]),
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            n_trades=sim["n_trades"],
            inventory_std=float(np.std(sim["inventory"])),
            avg_spread_captured=avg_spread,
            fill_rate=sim["n_trades"] / n if n > 0 else 0.0,
            equity_curve=pd.Series(equity_curve, index=df.index),
            inventory_series=pd.Series(sim["inventory"], index=df.index),
            spread_series=pd.Series(sim["spread"], index=df.index),
            params={
                "gamma": self.gamma, "sigma": self.sigma,
                "k": self.k, "T": self.T,
            },
        )

    def sensitivity_analysis(
        self,
        mid: float = 100.0,
        inventory_range: List[float] = None,
        time_range: List[float] = None,
    ) -> pd.DataFrame:
        """
        Show how quotes change with inventory and time remaining.
        Returns DataFrame with bid/ask/spread for each combination.
        """
        if inventory_range is None:
            inventory_range = [-5, -3, -1, 0, 1, 3, 5]
        if time_range is None:
            time_range = [1.0, 0.75, 0.5, 0.25, 0.1]

        rows = []
        for t_rem in time_range:
            for q in inventory_range:
                bid, ask = self.compute_quotes(mid, q, t_rem)
                r = self.reservation_price(mid, q, t_rem)
                delta = self.optimal_spread(t_rem)
                rows.append({
                    "time_remaining": t_rem,
                    "inventory": q,
                    "reservation_price": round(r, 4),
                    "bid": round(bid, 4),
                    "ask": round(ask, 4),
                    "spread": round(delta, 6),
                    "bid_offset_from_mid": round(mid - bid, 4),
                    "ask_offset_from_mid": round(ask - mid, 4),
                })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GueantTaillardMM
# ─────────────────────────────────────────────────────────────────────────────

class GueantTaillardMM:
    """
    Guéant-Taillard market making variant (2012).

    Extends A-S model with:
    - Symmetric order flow: identical bid/ask arrival intensity A
    - Exponential order decay: intensity ~ A * exp(-k * offset)
    - Closed-form solution for finite horizon

    Parameters
    ----------
    gamma : risk aversion parameter (default 0.1)
    k     : intensity decay factor (default 1.5)
    A     : base arrival rate (default 140.0 — calibrated to typical LOB)
    B     : intensity constant (default 1.5)  — sets base order flow
    sigma : price diffusion (default 0.01)
    T     : session horizon (default 1.0)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        A: float = 140.0,
        B: float = 1.5,
        sigma: float = 0.01,
        T: float = 1.0,
    ):
        self.gamma = gamma
        self.k = k
        self.A = A
        self.B = B
        self.sigma = sigma
        self.T = T

    def compute_quotes(
        self,
        mid: float,
        inventory: int,
        time_remaining: float,
    ) -> Tuple[float, float]:
        """
        Compute optimal quotes using the GT model.

        The GT solution for symmetric order flow:
            delta_bid = delta_ask = delta*(t) + inventory * correction

        where delta*(t) solves the HJB equation:
            delta*(t) = (1/k) * ln(1 + k/gamma) +
                        (gamma * sigma^2 * (T - t)) / 2
        """
        if time_remaining <= 0:
            return mid - 1e-6, mid + 1e-6

        # Optimal symmetric spread component
        delta_opt = ((1.0 / self.k) * math.log(1.0 + self.k / (self.gamma + 1e-9)) +
                     0.5 * self.gamma * self.sigma ** 2 * time_remaining)

        # Inventory skew: market maker adjusts quotes based on position
        # Positive inventory → skew ask down to sell faster
        inventory_adj = inventory * self.gamma * self.sigma ** 2 * time_remaining / 2.0

        bid = mid - delta_opt - inventory_adj
        ask = mid + delta_opt - inventory_adj

        return bid, ask

    def fill_intensity(self, offset: float) -> float:
        """
        Order arrival intensity at distance 'offset' from mid.
        lambda(delta) = A * exp(-k * delta)
        """
        return self.A * math.exp(-self.k * abs(offset))

    def simulate_session(
        self,
        mid_prices: np.ndarray,
        dt: float = 1.0 / 252,
        initial_inventory: int = 0,
        max_inventory: int = 10,
        seed: int = 42,
    ) -> Dict:
        """Simulate a trading session using GT model."""
        rng = np.random.default_rng(seed)
        n = len(mid_prices)

        inventory = initial_inventory
        cash = 0.0
        pnl_series = np.zeros(n)
        inventory_series = np.zeros(n)
        spread_series = np.zeros(n)
        trades = []

        for t in range(n):
            mid = float(mid_prices[t])
            time_remaining = max(0.001, self.T - t * dt)
            bid, ask = self.compute_quotes(mid, inventory, time_remaining)
            spread_series[t] = ask - bid

            # Simulate fills
            if abs(inventory) < max_inventory:
                bid_lambda = self.fill_intensity(mid - bid) * dt
                ask_lambda = self.fill_intensity(ask - mid) * dt
                bid_fills = rng.poisson(max(0, bid_lambda))
                ask_fills = rng.poisson(max(0, ask_lambda))

                if bid_fills > 0:
                    cash -= bid_fills * bid
                    inventory += bid_fills
                    trades.append({"type": "buy", "price": bid, "qty": bid_fills, "t": t})

                if ask_fills > 0:
                    cash += ask_fills * ask
                    inventory -= ask_fills
                    trades.append({"type": "sell", "price": ask, "qty": ask_fills, "t": t})

            pnl_series[t] = cash + inventory * mid
            inventory_series[t] = inventory

        return {
            "pnl": pnl_series,
            "inventory": inventory_series,
            "spread": spread_series,
            "trades": trades,
            "final_pnl": float(pnl_series[-1]),
            "n_trades": len(trades),
        }

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
    ) -> MMBacktestResult:
        mid = df["close"].values
        n = len(mid)
        rets = np.diff(np.log(mid + 1e-9))
        self.sigma = float(np.std(rets) * math.sqrt(252)) if len(rets) > 5 else self.sigma

        sim = self.simulate_session(mid, dt=1.0 / 252, initial_inventory=0, max_inventory=10)

        pnl = sim["pnl"]
        equity_curve = initial_equity + pnl
        rets_arr = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        rets_arr = np.concatenate([[0], rets_arr])
        std = rets_arr.std()
        sharpe = rets_arr.mean() / std * math.sqrt(252) if std > 0 else 0.0
        pk = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - pk) / (pk + 1e-9)

        return MMBacktestResult(
            total_pnl=sim["final_pnl"],
            total_return=(equity_curve[-1] - initial_equity) / initial_equity,
            sharpe=sharpe,
            max_drawdown=dd.min(),
            n_trades=sim["n_trades"],
            inventory_std=float(np.std(sim["inventory"])),
            avg_spread_captured=float(np.mean(sim["spread"])),
            fill_rate=sim["n_trades"] / n,
            equity_curve=pd.Series(equity_curve, index=df.index),
            inventory_series=pd.Series(sim["inventory"], index=df.index),
            spread_series=pd.Series(sim["spread"], index=df.index),
            params={"gamma": self.gamma, "k": self.k, "A": self.A, "B": self.B},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SimpleSpreadCapture
# ─────────────────────────────────────────────────────────────────────────────

class SimpleSpreadCapture:
    """
    Simple market making: quote a spread around mid price.

    This is the baseline MM strategy:
    - Post bid at (mid - spread/2), ask at (mid + spread/2)
    - Capture the spread when both sides fill
    - Limit inventory to avoid directional risk

    Uses a simple rule-based approach to manage inventory:
    - If inventory > limit: stop posting bids
    - If inventory < -limit: stop posting asks

    Parameters
    ----------
    spread_target    : target half-spread as fraction of mid (default 0.001)
    inventory_limit  : max absolute inventory in units (default 10)
    vol_scaling      : scale spread by realized vol (default True)
    vol_window       : vol estimation window for scaling (default 20)
    """

    def __init__(
        self,
        spread_target: float = 0.001,
        inventory_limit: int = 10,
        vol_scaling: bool = True,
        vol_window: int = 20,
    ):
        self.spread_target = spread_target
        self.inventory_limit = inventory_limit
        self.vol_scaling = vol_scaling
        self.vol_window = vol_window

    def compute_quotes(
        self,
        mid: float,
        inventory: float,
        realized_vol: float = None,
    ) -> Tuple[float, float, bool, bool]:
        """
        Compute bid/ask quotes and whether to post on each side.

        Returns (bid, ask, post_bid, post_ask).
        """
        spread = self.spread_target
        if self.vol_scaling and realized_vol is not None and realized_vol > 0:
            # Scale spread with volatility (wider spread in high vol)
            vol_scale = realized_vol / 0.20  # normalize to 20% annualized vol
            spread = self.spread_target * max(0.5, min(3.0, vol_scale))

        # Inventory skew: adjust mid quote toward position closing
        inventory_skew = -inventory * spread * 0.2
        adj_mid = mid + inventory_skew

        bid = adj_mid * (1 - spread)
        ask = adj_mid * (1 + spread)

        # Post both sides unless at inventory limits
        post_bid = inventory < self.inventory_limit
        post_ask = inventory > -self.inventory_limit

        return bid, ask, post_bid, post_ask

    def simulate(
        self,
        prices: pd.Series,
        volumes: pd.Series = None,
        initial_cash: float = 1_000_000,
        seed: int = 42,
    ) -> Dict:
        """
        Simulate market making on a price series.

        Assumptions:
        - Market orders arrive every bar
        - Fill probability proportional to volume and spread distance
        - Price range (H-L) represents the spread available
        """
        rng = np.random.default_rng(seed)
        n = len(prices)
        mid = prices.values

        cash = initial_cash
        inventory = 0.0
        equity_curve = np.full(n, initial_cash, dtype=float)
        inv_series = np.zeros(n)
        spread_series = np.zeros(n)
        trades = []
        pnl_components = {"spread": 0.0, "inventory": 0.0}

        # Rolling vol
        rets = prices.pct_change().fillna(0)
        rolling_vol = rets.rolling(self.vol_window, min_periods=5).std() * math.sqrt(252)

        for i in range(1, n):
            m = float(mid[i])
            vol = float(rolling_vol.iloc[i]) if not np.isnan(rolling_vol.iloc[i]) else 0.20
            v = float(volumes.iloc[i]) if volumes is not None else 1000.0

            bid, ask, post_bid, post_ask = self.compute_quotes(m, inventory, vol)
            spread_series[i] = ask - bid

            # Simulate: volume-weighted fill probability
            fill_prob = min(0.95, v / 1000.0 * 0.5)

            # Buy fill (we post bid, someone sells to us)
            if post_bid and rng.random() < fill_prob:
                qty = float(rng.integers(1, 4))
                qty = min(qty, self.inventory_limit - inventory)
                if qty > 0:
                    cash -= qty * bid
                    inventory += qty
                    pnl_components["spread"] += qty * (m - bid)
                    trades.append({"type": "buy", "price": bid, "qty": qty, "t": i})

            # Sell fill (we post ask, someone buys from us)
            if post_ask and rng.random() < fill_prob:
                qty = float(rng.integers(1, 4))
                qty = min(qty, self.inventory_limit + inventory)
                if qty > 0:
                    cash += qty * ask
                    inventory -= qty
                    pnl_components["spread"] += qty * (ask - m)
                    trades.append({"type": "sell", "price": ask, "qty": qty, "t": i})

            mtm = cash + inventory * m
            inventory_change = mtm - (cash + inventory * float(mid[i - 1]))
            pnl_components["inventory"] += float(mid[i] - mid[i - 1]) * inventory

            equity_curve[i] = mtm
            inv_series[i] = inventory

        total_pnl = equity_curve[-1] - initial_cash
        rets_arr = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        rets_arr = np.concatenate([[0], rets_arr])

        return {
            "equity_curve": equity_curve,
            "inventory": inv_series,
            "spread": spread_series,
            "trades": trades,
            "total_pnl": total_pnl,
            "spread_pnl": pnl_components["spread"],
            "inventory_pnl": pnl_components["inventory"],
            "n_trades": len(trades),
            "returns": rets_arr,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
    ) -> MMBacktestResult:
        prices = df["close"]
        volumes = df.get("volume", pd.Series(1000.0, index=df.index))
        sim = self.simulate(prices, volumes, initial_equity)

        ec = sim["equity_curve"]
        rets = sim["returns"]
        std = rets.std()
        sharpe = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
        pk = np.maximum.accumulate(ec)
        dd = (ec - pk) / (pk + 1e-9)

        trades = sim["trades"]
        spread_cap = [abs(t["price"] - prices.iloc[t["t"]]) for t in trades]
        avg_spread = float(np.mean(spread_cap)) if spread_cap else 0.0

        return MMBacktestResult(
            total_pnl=sim["total_pnl"],
            total_return=sim["total_pnl"] / initial_equity,
            sharpe=sharpe,
            max_drawdown=dd.min(),
            n_trades=sim["n_trades"],
            inventory_std=float(np.std(sim["inventory"])),
            avg_spread_captured=avg_spread,
            fill_rate=sim["n_trades"] / len(prices),
            equity_curve=pd.Series(ec, index=df.index),
            inventory_series=pd.Series(sim["inventory"], index=df.index),
            spread_series=pd.Series(sim["spread"], index=df.index),
            params={"spread_target": self.spread_target, "inventory_limit": self.inventory_limit},
        )

    def optimal_spread(self, vol: float, fill_rate_target: float = 0.5) -> float:
        """
        Compute the spread that achieves a target fill rate.
        fill_rate ~ exp(-k * spread) → spread = -ln(fill_rate) / k
        Using k ~ 1/vol as a rough approximation.
        """
        k = 1.0 / (vol + 1e-9)
        return -math.log(max(1e-6, fill_rate_target)) / k

    def inventory_metrics(self, inventory_series: pd.Series) -> dict:
        """Compute inventory risk metrics."""
        return {
            "mean_inventory": float(inventory_series.mean()),
            "std_inventory": float(inventory_series.std()),
            "max_long": float(inventory_series.max()),
            "max_short": float(inventory_series.min()),
            "time_flat_pct": float((inventory_series.abs() < 0.5).mean()),
            "time_long_pct": float((inventory_series > 0.5).mean()),
            "time_short_pct": float((inventory_series < -0.5).mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": rng.integers(1000, 10000, n).astype(float)
    }, index=idx)

    # AS model
    as_mm = AvellanedaStoikovMM(gamma=0.1, sigma=0.02, k=1.5, T=1.0)
    res_as = as_mm.backtest(df)
    print("AS Model:", res_as.summary())
    print("Sensitivity:")
    print(as_mm.sensitivity_analysis(100.0).head(10).to_string())

    # GT model
    gt_mm = GueantTaillardMM(gamma=0.1, k=1.5)
    res_gt = gt_mm.backtest(df)
    print("\nGT Model:", res_gt.summary())

    # Simple spread capture
    ssc = SimpleSpreadCapture(spread_target=0.001, inventory_limit=10)
    res_ssc = ssc.backtest(df)
    print("\nSimple Spread Capture:", res_ssc.summary())
    print("Inventory metrics:", ssc.inventory_metrics(res_ssc.inventory_series))
