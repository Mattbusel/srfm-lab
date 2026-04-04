"""
carry.py — Carry strategies across FX, futures, and fixed income.

References:
  - Koijen et al. (2018): "Carry" (AFA)
  - Asness et al. (2013): "Value and Momentum Everywhere"
  - Gorton, Hayashi, Rouwenhorst (2012): commodity roll yield
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Shared BacktestResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    carry_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _stats(ec: np.ndarray, trades: list) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    cagr = (ec[-1] / ec[0]) ** (1 / max(1, n / 252)) - 1
    r = np.diff(np.log(ec + 1e-9))
    r = np.concatenate([[0], r])
    std = r.std()
    sharpe = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    wr = len(wins) / len(trades) if trades else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(total_return=tot, cagr=cagr, sharpe=sharpe, sortino=sortino,
                max_drawdown=mdd, calmar=calmar, win_rate=wr, profit_factor=pf,
                n_trades=len(trades), avg_trade_return=float(np.mean(trades)) if trades else 0.0)


def _signal_to_equity(close: np.ndarray, signal: np.ndarray,
                      initial: float = 1_000_000, cost: float = 0.0002) -> Tuple[np.ndarray, list]:
    n = len(close)
    equity = initial
    ec = np.full(n, initial, dtype=float)
    trades = []
    pos = 0.0
    ep = None

    for i in range(1, n):
        s = float(signal[i - 1]) if not np.isnan(signal[i - 1]) else pos
        if s != pos:
            if ep is not None and pos != 0:
                ret = pos * ((close[i] - ep) / ep - cost * 2)
                trades.append(ret)
            pos = s
            ep = close[i] if s != 0 else None
        if pos != 0:
            equity *= (1 + pos * (close[i] - close[i - 1]) / (close[i - 1] + 1e-9))
        ec[i] = equity
    return ec, trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. ForwardRateCarry
# ─────────────────────────────────────────────────────────────────────────────

class ForwardRateCarry:
    """
    FX Carry Strategy: borrow low-yield currency, invest in high-yield currency.

    The carry = forward premium/discount ≈ interest rate differential.
    For a pair: carry = (spot - forward) / forward ≈ (r_domestic - r_foreign) * T

    If carry > 0, currency is expected to appreciate relative to forward.
    Long the high-yield currency, short the low-yield currency.

    Parameters
    ----------
    entry_threshold : minimum carry to enter (default 0.001 = 0.1% pa)
    exit_threshold  : carry below which to exit (default 0.0)
    annualize_days  : days per year for carry annualization (default 360)
    """

    def __init__(
        self,
        entry_threshold: float = 0.001,
        exit_threshold: float = 0.0,
        annualize_days: int = 360,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.annualize_days = annualize_days

    def compute_carry(
        self,
        spot: pd.Series,
        forward: pd.Series,
        days_to_expiry: pd.Series,
    ) -> pd.Series:
        """
        Compute the annualized carry from spot and forward prices.

        carry = (spot - forward) / forward / (days_to_expiry / annualize_days)

        Positive carry = spot > forward = domestic rate > foreign rate.

        Parameters
        ----------
        spot           : spot price series
        forward        : forward price series (same maturity)
        days_to_expiry : number of days until forward delivery
        """
        fraction_year = days_to_expiry / self.annualize_days
        carry = (spot - forward) / (forward.abs() + 1e-9) / (fraction_year + 1e-9)
        return carry

    def generate_signals(
        self,
        spot: pd.Series,
        forward: pd.Series,
        days_to_expiry: pd.Series,
    ) -> pd.Series:
        """
        Returns signal: +1 long spot (positive carry), -1 short, 0 flat.
        """
        carry = self.compute_carry(spot, forward, days_to_expiry)
        signal = pd.Series(0.0, index=spot.index)
        signal[carry > self.entry_threshold] = 1.0
        signal[carry < -self.entry_threshold] = -1.0
        # Exit if carry drops below exit threshold
        in_long = (signal == 0.0) & (signal.shift(1) == 1.0) & (carry > self.exit_threshold)
        signal[in_long] = 1.0  # maintain position
        return signal

    def backtest(
        self,
        spot: pd.Series,
        forward: pd.Series,
        days_to_expiry: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signal = self.generate_signals(spot, forward, days_to_expiry)
        carry = self.compute_carry(spot, forward, days_to_expiry)

        ec, trades = _signal_to_equity(spot.values, signal.values, initial_equity, commission_pct)
        s = _stats(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=spot.index),
            returns=pd.Series(np.diff(np.log(ec + 1e-9)), index=spot.index[1:]),
            signals=signal,
            carry_series=carry,
            params={"entry_threshold": self.entry_threshold},
        )

    def carry_statistics(
        self,
        spot: pd.Series,
        forward: pd.Series,
        days_to_expiry: pd.Series,
    ) -> dict:
        """Summary statistics of the carry series."""
        carry = self.compute_carry(spot, forward, days_to_expiry)
        return {
            "mean_carry": carry.mean(),
            "std_carry": carry.std(),
            "min_carry": carry.min(),
            "max_carry": carry.max(),
            "pct_positive": (carry > 0).mean(),
            "sharpe_of_carry": carry.mean() / (carry.std() + 1e-9) * math.sqrt(252),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RollYieldCarry
# ─────────────────────────────────────────────────────────────────────────────

class RollYieldCarry:
    """
    Commodity futures roll yield carry strategy.

    Roll yield = gain/loss from rolling a futures contract forward.
    In backwardation (spot > futures): positive roll yield → go long.
    In contango (spot < futures): negative roll yield → go short or flat.

    For a portfolio of commodities, go long the most backwardated,
    short the most contangoed.

    Parameters
    ----------
    roll_cost_pct   : estimated roll transaction cost (default 0.001)
    threshold       : minimum roll yield to enter (default 0.0)
    """

    def __init__(
        self,
        roll_cost_pct: float = 0.001,
        threshold: float = 0.0,
    ):
        self.roll_cost_pct = roll_cost_pct
        self.threshold = threshold

    def compute_roll_yield(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
    ) -> pd.Series:
        """
        Annualized roll yield.

        roll_yield = (near_price - far_price) / far_price / (days_to_roll / 365)

        Positive = backwardation (long-friendly).
        Negative = contango (roll drag).

        Parameters
        ----------
        near_price    : price of near-dated contract
        far_price     : price of far-dated (next) contract
        days_to_roll  : days until near contract expires/rolls
        """
        fraction_year = days_to_roll / 365.0
        roll_yield = (near_price - far_price) / (far_price.abs() + 1e-9) / (fraction_year + 1e-9)
        return roll_yield

    def generate_signals(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
    ) -> pd.Series:
        """Signal: +1 in backwardation, -1 in contango (above threshold)."""
        ry = self.compute_roll_yield(near_price, far_price, days_to_roll)
        signal = pd.Series(0.0, index=near_price.index)
        signal[ry > self.threshold] = 1.0
        signal[ry < -self.threshold] = -1.0
        return signal

    def backtest(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(near_price, far_price, days_to_roll)
        ry = self.compute_roll_yield(near_price, far_price, days_to_roll)

        # P&L includes price change + roll yield for holding period
        n = len(near_price)
        ec = np.full(n, initial_equity, dtype=float)
        equity = initial_equity
        trades = []
        pos = 0.0

        close = near_price.values
        ry_vals = ry.values
        sig_vals = signal.values

        for i in range(1, n):
            s = float(sig_vals[i - 1]) if not np.isnan(sig_vals[i - 1]) else 0.0
            price_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)

            if s != pos:
                if pos != 0:
                    trades.append(pos * price_ret - self.roll_cost_pct)
                pos = s

            if pos != 0:
                # Daily return = price return + daily roll yield component
                daily_ry = ry_vals[i] / 252 if not np.isnan(ry_vals[i]) else 0.0
                total_ret = pos * (price_ret + daily_ry) - abs(s != pos) * self.roll_cost_pct
                equity *= (1 + total_ret)

            ec[i] = equity

        s_stats = _stats(ec, trades)
        carry = ry

        return BacktestResult(
            **s_stats,
            equity_curve=pd.Series(ec, index=near_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=near_price.index[1:]),
            signals=signal,
            carry_series=carry,
            params={"roll_cost_pct": self.roll_cost_pct, "threshold": self.threshold},
        )

    def rank_by_roll_yield(
        self,
        near_prices: Dict[str, pd.Series],
        far_prices: Dict[str, pd.Series],
        days_to_roll: Dict[str, pd.Series],
        at_date=None,
    ) -> pd.Series:
        """
        Rank a universe of commodities by their roll yield at a given date.
        Returns pd.Series sorted descending (most backwardated first).
        """
        scores = {}
        for sym in near_prices.keys():
            n = near_prices[sym]
            f = far_prices[sym]
            d = days_to_roll[sym]
            if at_date is not None:
                n = n.loc[:at_date]
                f = f.loc[:at_date]
                d = d.loc[:at_date]
            ry = self.compute_roll_yield(n, f, d)
            scores[sym] = float(ry.iloc[-1]) if len(ry) > 0 else 0.0
        return pd.Series(scores).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TermStructureCarry
# ─────────────────────────────────────────────────────────────────────────────

class TermStructureCarry:
    """
    Futures term structure carry strategy.

    The carry from a futures position comes from:
    1. The roll yield (as the contract approaches expiry)
    2. The change in the term structure slope

    carry = (F_near - F_far) / F_far * (days_to_roll / DTE_diff)

    This is the basis trade: long the steep backwardation, short contango.

    Parameters
    ----------
    signal_smoothing : days to smooth the carry signal (default 5)
    entry_percentile : percentile threshold for entry (default 0.6)
    exit_percentile  : percentile threshold for exit (default 0.4)
    """

    def __init__(
        self,
        signal_smoothing: int = 5,
        entry_percentile: float = 0.6,
        exit_percentile: float = 0.4,
    ):
        self.signal_smoothing = signal_smoothing
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile

    def compute_carry(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
        dte_diff: float = 30.0,
    ) -> pd.Series:
        """
        Compute annualized term structure carry.

        Parameters
        ----------
        near_price    : near-term futures price
        far_price     : far-term futures price
        days_to_roll  : days until near contract rolls
        dte_diff      : days between near and far contract expiries (default 30)
        """
        basis = (near_price - far_price) / (far_price.abs() + 1e-9)
        # Annualized: roll happens every dte_diff days → multiply by 365/dte_diff
        carry = basis * (365.0 / dte_diff)
        # Smooth the carry signal
        if self.signal_smoothing > 1:
            carry = carry.rolling(self.signal_smoothing, min_periods=1).mean()
        return carry

    def generate_signals(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
        dte_diff: float = 30.0,
        lookback: int = 252,
    ) -> pd.Series:
        """
        Generate signals based on percentile rank of carry.
        Enter long when carry is in top entry_percentile.
        Enter short when carry is in bottom (1 - entry_percentile).
        """
        carry = self.compute_carry(near_price, far_price, days_to_roll, dte_diff)

        # Rolling percentile rank
        carry_rank = carry.rolling(lookback, min_periods=20).rank(pct=True)

        signal = pd.Series(0.0, index=near_price.index)
        signal[carry_rank > self.entry_percentile] = 1.0
        signal[carry_rank < (1.0 - self.entry_percentile)] = -1.0
        # Exit positions
        in_long = signal.shift(1) == 1.0
        in_short = signal.shift(1) == -1.0
        signal[(in_long) & (carry_rank < self.exit_percentile)] = 0.0
        signal[(in_short) & (carry_rank > (1.0 - self.exit_percentile))] = 0.0
        return signal

    def backtest(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
        dte_diff: float = 30.0,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signal = self.generate_signals(near_price, far_price, days_to_roll, dte_diff)
        carry = self.compute_carry(near_price, far_price, days_to_roll, dte_diff)

        ec, trades = _signal_to_equity(near_price.values, signal.values, initial_equity, commission_pct)
        s = _stats(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=near_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=near_price.index[1:]),
            signals=signal,
            carry_series=carry,
            params={"signal_smoothing": self.signal_smoothing,
                    "entry_percentile": self.entry_percentile, "dte_diff": dte_diff},
        )

    def term_structure_slope(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
    ) -> pd.Series:
        """
        Simple term structure slope: (near - far) / far.
        Positive = backwardation. Negative = contango.
        """
        return (near_price - far_price) / (far_price.abs() + 1e-9)

    def expected_return_components(
        self,
        near_price: pd.Series,
        far_price: pd.Series,
        days_to_roll: pd.Series,
        price_return: pd.Series,
        dte_diff: float = 30.0,
    ) -> pd.DataFrame:
        """
        Decompose futures return into price change vs roll yield.

        Returns DataFrame with columns:
            price_return, roll_yield, total_return
        """
        carry = self.compute_carry(near_price, far_price, days_to_roll, dte_diff)
        daily_roll = carry / 252
        total = price_return + daily_roll
        return pd.DataFrame({
            "price_return": price_return,
            "daily_roll_yield": daily_roll,
            "carry_annualized": carry,
            "total_return": total,
        }, index=near_price.index)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-asset carry portfolio
# ─────────────────────────────────────────────────────────────────────────────

class CrossAssetCarry:
    """
    Portfolio carry strategy across multiple asset classes.

    Combines FX carry, commodity roll yield, and bond carry (yield spread)
    into a single diversified carry portfolio.

    Parameters
    ----------
    target_vol       : target annualized volatility (default 0.10)
    n_long           : number of assets to go long (default 3)
    n_short          : number of assets to short (default 3)
    rebal_frequency  : rebalancing frequency in bars (default 21)
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        n_long: int = 3,
        n_short: int = 3,
        rebal_frequency: int = 21,
    ):
        self.target_vol = target_vol
        self.n_long = n_long
        self.n_short = n_short
        self.rebal_frequency = rebal_frequency

    def compute_portfolio_weights(
        self,
        carry_scores: pd.DataFrame,
        returns_df: pd.DataFrame,
        vol_window: int = 63,
    ) -> pd.DataFrame:
        """
        Compute portfolio weights from carry scores.

        Parameters
        ----------
        carry_scores : DataFrame, columns = assets, rows = time, values = annualized carry
        returns_df   : DataFrame of returns for vol estimation
        vol_window   : window for vol estimation

        Returns
        -------
        pd.DataFrame of weights
        """
        weights = pd.DataFrame(0.0, index=carry_scores.index, columns=carry_scores.columns)
        vols = returns_df.rolling(vol_window, min_periods=vol_window // 2).std() * math.sqrt(252)

        for i in range(vol_window, len(carry_scores), self.rebal_frequency):
            scores = carry_scores.iloc[i]
            vol_row = vols.iloc[i]

            valid = scores.dropna()
            if len(valid) == 0:
                continue

            ranked = valid.rank(ascending=True)
            n = len(ranked)
            n_long = min(self.n_long, n // 2)
            n_short = min(self.n_short, n // 2)

            w = pd.Series(0.0, index=carry_scores.columns)
            if n_long > 0:
                long_assets = ranked.nlargest(n_long).index
                for a in long_assets:
                    vol_a = vol_row.get(a, self.target_vol)
                    if vol_a > 0:
                        w[a] = (self.target_vol / vol_a) / n_long

            if n_short > 0:
                short_assets = ranked.nsmallest(n_short).index
                for a in short_assets:
                    vol_a = vol_row.get(a, self.target_vol)
                    if vol_a > 0:
                        w[a] = -(self.target_vol / vol_a) / n_short

            end_i = min(i + self.rebal_frequency, len(weights))
            weights.iloc[i:end_i] = w.values

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        carry_scores: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest cross-asset carry portfolio.

        Parameters
        ----------
        prices       : asset prices, columns = assets
        carry_scores : carry scores per asset per period
        """
        returns = prices.pct_change().fillna(0)
        weights = self.compute_portfolio_weights(carry_scores, returns)

        equity = initial_equity
        n = len(prices)
        ec = np.full(n, initial_equity, dtype=float)
        trades = []

        for i in range(1, n):
            w = weights.iloc[i].values
            r = returns.iloc[i].values
            port_ret = float(np.dot(w, r))
            prev_w = weights.iloc[i - 1].values
            turnover = np.abs(w - prev_w).sum() / 2
            port_ret -= turnover * commission_pct
            if port_ret != 0:
                trades.append(port_ret)
            equity *= (1 + port_ret)
            ec[i] = equity

        s = _stats(ec, trades)
        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=prices.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=prices.index[1:]),
            params={"target_vol": self.target_vol, "n_long": self.n_long, "n_short": self.n_short},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    # Simulate spot and forward
    spot = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0001, 0.008, n)), index=idx)
    # Forward slightly below spot (backwardation regime)
    forward = spot * (1 - rng.uniform(0.001, 0.005, n))
    days_exp = pd.Series(rng.integers(30, 90, n).astype(float), index=idx)

    frc = ForwardRateCarry(entry_threshold=0.01)
    result = frc.backtest(spot, forward, days_exp)
    print("FX Carry:", result.summary())

    # Term structure carry
    near = spot.copy()
    far = spot * (1 - rng.normal(0.002, 0.003, n))
    tsc = TermStructureCarry()
    res2 = tsc.backtest(near, far, days_exp)
    print("Term Structure Carry:", res2.summary())
