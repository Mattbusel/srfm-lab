"""
crypto/funding_rate.py — Crypto perpetual futures funding rate strategies.

Funding rates are periodic payments between long and short traders in
perpetual futures markets. They keep perp prices anchored to spot.

High positive funding → longs pay shorts → perp expensive → basis trade opportunity.
High negative funding → shorts pay longs → perp cheap → reverse basis trade.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    total_funding_collected: float = 0.0
    avg_basis: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"FundingCollected={self.total_funding_collected:.4f}")


def _stats(ec: np.ndarray) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    cagr = (ec[-1] / ec[0]) ** (1 / max(1, n / 252)) - 1
    r = np.diff(ec) / (ec[:-1] + 1e-9)
    r = np.concatenate([[0], r])
    std = r.std()
    sh = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FundingRateArbitrage
# ─────────────────────────────────────────────────────────────────────────────

class FundingRateArbitrage:
    """
    Crypto basis trade / cash-and-carry arbitrage.

    Long spot + short perpetual futures when funding rate is positive.
    This captures the funding rate as pure income while being market-neutral.

    P&L = funding_collected - (basis_change) - transaction_costs

    When funding is positive (longs pay shorts):
    - Hold long spot
    - Short perp (receive funding from longs)
    - Net: delta-neutral exposure to price + funding income

    Parameters
    ----------
    min_funding_rate : minimum annualized funding rate to enter (default 0.10 = 10% pa)
    exit_funding_rate: funding rate below which to exit (default 0.02)
    max_basis_pct    : maximum basis (perp - spot) / spot to enter (default 0.01 = 1%)
    funding_periods  : number of funding periods per day (default 3 for 8h funding)
    """

    def __init__(
        self,
        min_funding_rate: float = 0.10,
        exit_funding_rate: float = 0.02,
        max_basis_pct: float = 0.01,
        funding_periods: int = 3,
    ):
        self.min_funding_rate = min_funding_rate
        self.exit_funding_rate = exit_funding_rate
        self.max_basis_pct = max_basis_pct
        self.funding_periods = funding_periods

    def annualize_funding(self, periodic_rate: pd.Series) -> pd.Series:
        """
        Convert periodic funding rate to annualized rate.

        Annualized = periodic_rate * funding_periods * 365
        """
        return periodic_rate * self.funding_periods * 365

    def compute_basis(self, spot_price: pd.Series, perp_price: pd.Series) -> pd.Series:
        """
        Compute the perp-spot basis.
        Positive basis = perp > spot = contango = funding tends positive.
        """
        return (perp_price - spot_price) / (spot_price + 1e-9)

    def generate_signals(
        self,
        spot_price: pd.Series,
        perp_price: pd.Series,
        funding_rate: pd.Series,
    ) -> pd.Series:
        """
        Signal: +1 = enter basis trade (long spot, short perp)
                -1 = reverse basis trade (short spot, long perp) — when funding very negative
                 0 = flat

        funding_rate: periodic funding rate (e.g., 8-hourly)
        """
        ann_funding = self.annualize_funding(funding_rate)
        basis = self.compute_basis(spot_price, perp_price)

        signal = pd.Series(0.0, index=spot_price.index)
        position = 0

        for i in range(1, len(spot_price)):
            f = float(ann_funding.iloc[i])
            b = float(basis.iloc[i])
            if np.isnan(f) or np.isnan(b):
                continue

            if position == 0:
                # Enter long basis: funding high positive AND reasonable basis
                if f > self.min_funding_rate and abs(b) < self.max_basis_pct:
                    position = 1
                # Enter short basis: funding very negative
                elif f < -self.min_funding_rate and abs(b) < self.max_basis_pct:
                    position = -1

            elif position == 1:
                # Exit when funding drops
                if f < self.exit_funding_rate:
                    position = 0

            elif position == -1:
                if f > -self.exit_funding_rate:
                    position = 0

            signal.iloc[i] = float(position)

        return signal

    def backtest(
        self,
        spot_price: pd.Series,
        perp_price: pd.Series,
        funding_rate: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest the basis trade.

        P&L composition:
        1. Long spot P&L
        2. Short perp P&L (opposite of perp price changes)
        3. Funding collected (per funding period when short perp)
        4. Transaction costs on entry/exit
        """
        signal = self.generate_signals(spot_price, perp_price, funding_rate)
        basis = self.compute_basis(spot_price, perp_price)
        ann_funding = self.annualize_funding(funding_rate)

        spot_arr = spot_price.values
        perp_arr = perp_price.values
        fund_arr = (funding_rate * signal.shift(1).fillna(0)).values  # only collect when in trade
        sig_arr = signal.values
        n = len(spot_arr)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        total_funding = 0.0
        pos = 0.0

        for i in range(1, n):
            s = float(sig_arr[i - 1]) if not np.isnan(sig_arr[i - 1]) else 0.0

            # Spot leg P&L
            spot_ret = (spot_arr[i] - spot_arr[i-1]) / (spot_arr[i-1] + 1e-9)
            # Perp leg P&L (we are short perp when signal = +1)
            perp_ret = (perp_arr[i] - perp_arr[i-1]) / (perp_arr[i-1] + 1e-9)

            # Funding income: funding rate * position (per period)
            daily_funding = float(fund_arr[i]) * self.funding_periods  # daily

            if s == 1.0:
                # Long spot + short perp + collect funding
                pnl = 0.5 * spot_ret - 0.5 * perp_ret + daily_funding
            elif s == -1.0:
                # Short spot + long perp + pay funding
                pnl = -0.5 * spot_ret + 0.5 * perp_ret - daily_funding
            else:
                pnl = 0.0

            # Transaction cost on position changes
            if s != pos:
                pnl -= commission_pct * 2  # both legs
                if pos != 0:
                    trades.append(pnl)

            equity *= (1 + pnl)
            if s != 0:
                total_funding += abs(daily_funding * equity)
            ec[i] = equity
            pos = s

        s_stats = _stats(ec)

        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            total_funding_collected=total_funding,
            avg_basis=float(basis.dropna().mean()),
            equity_curve=pd.Series(ec, index=spot_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=spot_price.index[1:]),
            signals=signal,
            params={"min_funding_rate": self.min_funding_rate,
                    "exit_funding_rate": self.exit_funding_rate},
        )

    def expected_annual_return(
        self,
        funding_rate: pd.Series,
        execution_cost: float = 0.001,
    ) -> dict:
        """
        Estimate expected annualized return from the basis trade.
        """
        ann_funding = self.annualize_funding(funding_rate)
        eligible = ann_funding[ann_funding > self.min_funding_rate]
        return {
            "mean_ann_funding": float(ann_funding.mean()),
            "mean_eligible_funding": float(eligible.mean()) if len(eligible) > 0 else 0.0,
            "pct_time_eligible": float(len(eligible) / len(ann_funding)),
            "gross_expected_return": float(eligible.mean() * len(eligible) / len(ann_funding)),
            "net_expected_return": float(eligible.mean() * len(eligible) / len(ann_funding) - execution_cost * 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FundingRateCarry
# ─────────────────────────────────────────────────────────────────────────────

class FundingRateCarry:
    """
    Funding rate carry strategy: rotate into assets with highest positive funding.

    When an asset has high positive funding, being short the perp earns yield.
    Combined with long spot, this is the cash-and-carry basis trade.

    Alternatively, just go long assets with high positive funding (directional):
    high positive funding often signals strong bull momentum.

    Parameters
    ----------
    symbols        : list of asset symbols
    threshold      : minimum annualized funding rate to enter (default 0.05)
    n_top          : number of top-funding assets to hold (default 3)
    rebal_freq     : rebalancing frequency in bars (default 7)
    directional    : if True, go long high-funding assets (default False = basis)
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        threshold: float = 0.05,
        n_top: int = 3,
        rebal_freq: int = 7,
        directional: bool = False,
    ):
        self.symbols = symbols
        self.threshold = threshold
        self.n_top = n_top
        self.rebal_freq = rebal_freq
        self.directional = directional

    def rank_by_funding(
        self,
        funding_rates: pd.DataFrame,
        at_date=None,
    ) -> pd.Series:
        """
        Rank assets by annualized funding rate.

        Parameters
        ----------
        funding_rates : DataFrame, columns = symbols, rows = dates
                       values = periodic funding rates (e.g., 8h rates)
        at_date       : specific date to rank at (default: latest)

        Returns
        -------
        pd.Series sorted by funding rate descending.
        """
        if at_date is not None:
            row = funding_rates.loc[:at_date].iloc[-1]
        else:
            row = funding_rates.iloc[-1]

        # Annualize: 3 periods per day × 365 days
        ann = row * 3 * 365
        return ann.sort_values(ascending=False)

    def generate_weights(
        self,
        funding_rates: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate portfolio weights based on funding rate carry.

        Parameters
        ----------
        funding_rates : funding rates DataFrame (same cols as prices)
        prices        : spot prices DataFrame

        Returns
        -------
        pd.DataFrame of weights
        """
        cols = self.symbols if self.symbols is not None else list(prices.columns)
        weights = pd.DataFrame(0.0, index=prices.index, columns=cols)

        for i in range(0, len(prices), self.rebal_freq):
            if i >= len(funding_rates):
                break

            row_funding = funding_rates.iloc[i] if i < len(funding_rates) else funding_rates.iloc[-1]
            ann_funding = row_funding * 3 * 365  # annualize

            # Filter: only include assets with funding > threshold
            eligible = ann_funding[ann_funding > self.threshold]
            if len(eligible) == 0:
                continue

            # Take top n_top by funding
            top = eligible.nlargest(self.n_top)
            n = len(top)
            w = pd.Series(0.0, index=cols)

            if self.directional:
                w[top.index] = 1.0 / n  # equal weight long
            else:
                # Basis trade: weight proportional to funding yield
                total_fund = top.sum()
                for sym in top.index:
                    if sym in cols:
                        w[sym] = top[sym] / (total_fund + 1e-9)

            end_i = min(i + self.rebal_freq, len(prices))
            weights.iloc[i:end_i] = w.values

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        funding_rates: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest the funding rate carry portfolio.

        The P&L combines:
        1. Price returns (for directional exposure)
        2. Funding income (funding_rate × notional per period)
        """
        cols = self.symbols if self.symbols is not None else list(prices.columns)
        weights = self.generate_weights(funding_rates[cols], prices[cols])
        returns = prices[cols].pct_change().fillna(0)
        # Daily funding income: 3 periods × funding_rate
        daily_funding = funding_rates[cols].reindex(prices.index).fillna(0) * 3

        equity = initial_equity
        n = len(prices)
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        total_funding = 0.0
        prev_w = np.zeros(len(cols))

        for i in range(1, n):
            w = weights.iloc[i].values
            r = returns.iloc[i].values
            f = daily_funding.iloc[i].values if i < len(daily_funding) else np.zeros(len(cols))

            # Price return component
            price_ret = float(np.dot(w, r))
            # Funding income component
            funding_ret = float(np.dot(np.abs(w), f))  # collect funding on notional

            # Transaction costs
            turnover = np.abs(w - prev_w).sum() / 2
            cost = turnover * commission_pct

            total_ret = price_ret + funding_ret - cost
            equity *= (1 + total_ret)
            ec[i] = equity

            if abs(total_ret) > 1e-9:
                trades.append(total_ret)
            total_funding += funding_ret * equity
            prev_w = w

        s_stats = _stats(ec)
        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            total_funding_collected=total_funding,
            avg_basis=float((funding_rates[cols] * 3 * 365).mean().mean()),
            equity_curve=pd.Series(ec, index=prices.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=prices.index[1:]),
            params={"threshold": self.threshold, "n_top": self.n_top, "directional": self.directional},
        )

    def funding_statistics(self, funding_rates: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics for each asset's funding rate."""
        ann = funding_rates * 3 * 365
        stats = []
        for col in ann.columns:
            s = ann[col].dropna()
            stats.append({
                "symbol": col,
                "mean_ann_funding": float(s.mean()),
                "std_ann_funding": float(s.std()),
                "pct_positive": float((s > 0).mean()),
                "pct_above_threshold": float((s > self.threshold).mean()),
                "max_funding": float(s.max()),
                "min_funding": float(s.min()),
            })
        return pd.DataFrame(stats).set_index("symbol")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="D")

    # Simulate BTC spot/perp with positive funding bias
    spot = pd.Series(40000.0 * np.cumprod(1 + rng.normal(0.001, 0.03, n)), index=idx)
    basis = rng.normal(0.003, 0.002, n).clip(-0.01, 0.02)  # mostly positive
    perp = spot * (1 + basis)
    funding_rate = pd.Series(basis * 0.3, index=idx)  # periodic (8h) rate

    frarb = FundingRateArbitrage(min_funding_rate=0.10, exit_funding_rate=0.03)
    res = frarb.backtest(spot, perp, funding_rate)
    print("Funding Rate Arb:", res.summary())
    print("Expected return:", frarb.expected_annual_return(funding_rate))

    # Multi-asset funding carry
    prices = pd.DataFrame({
        "BTC": spot.values,
        "ETH": (2500.0 * np.cumprod(1 + rng.normal(0.001, 0.04, n))),
        "SOL": (100.0 * np.cumprod(1 + rng.normal(0.002, 0.05, n))),
    }, index=idx)

    funding_rates = pd.DataFrame({
        "BTC": rng.normal(0.0002, 0.0001, n).clip(-0.001, 0.002),
        "ETH": rng.normal(0.0003, 0.0002, n).clip(-0.001, 0.003),
        "SOL": rng.normal(0.0005, 0.0003, n).clip(-0.002, 0.005),
    }, index=idx)

    frc = FundingRateCarry(threshold=0.05, n_top=2, directional=True)
    res2 = frc.backtest(prices, funding_rates)
    print("\nFunding Rate Carry (directional):", res2.summary())
    print("Funding stats:")
    print(frc.funding_statistics(funding_rates))
