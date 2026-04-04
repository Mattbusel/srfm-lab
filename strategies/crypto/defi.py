"""
crypto/defi.py — DeFi yield and liquidity provision strategies.

Implements:
1. YieldFarmingRouter — rotate capital to highest-yielding DeFi protocols
2. LiquidityProvision — LP profitability model with impermanent loss calculation
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

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
    n_rebalances: int = 0
    total_fees_earned: float = 0.0
    total_il_cost: float = 0.0
    net_apy: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    allocation_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"NetAPY={self.net_apy:.2%}")


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
# 1. YieldFarmingRouter
# ─────────────────────────────────────────────────────────────────────────────

class YieldFarmingRouter:
    """
    Yield Farming Router: rotate capital to highest-APY DeFi protocol.

    Yield farming involves depositing tokens into DeFi protocols
    (lending, liquidity pools, staking) to earn yield.
    APYs fluctuate based on supply/demand and token prices.

    Strategy: always move capital to the highest net-yield protocol,
    considering gas costs of switching.

    Parameters
    ----------
    protocols       : list of protocol names
    gas_cost_usd    : estimated gas cost per move in USD (default 50)
    rebal_threshold : minimum APY improvement to justify a move (default 0.02 = 2%)
    min_hold_days   : minimum days to hold before rebalancing (default 7)
    """

    def __init__(
        self,
        protocols: Optional[List[str]] = None,
        gas_cost_usd: float = 50.0,
        rebal_threshold: float = 0.02,
        min_hold_days: int = 7,
    ):
        self.protocols = protocols or []
        self.gas_cost_usd = gas_cost_usd
        self.rebal_threshold = rebal_threshold
        self.min_hold_days = min_hold_days

    def select_protocol(
        self,
        apy_row: pd.Series,
        current_protocol: str,
        equity: float,
    ) -> Tuple[str, float]:
        """
        Select the best protocol, accounting for gas costs.

        Returns (best_protocol, effective_apy_after_gas).
        """
        if len(apy_row.dropna()) == 0:
            return current_protocol, 0.0

        # Best available APY
        best_protocol = apy_row.idxmax()
        best_apy = float(apy_row[best_protocol])
        current_apy = float(apy_row.get(current_protocol, 0.0))

        # Check if switching is worth the gas cost
        if best_protocol != current_protocol:
            # Gas cost as fraction of equity (annualized over min_hold_days)
            gas_annualized = self.gas_cost_usd / (equity + 1e-9) * (365 / self.min_hold_days)
            # Net benefit = APY gain - annualized gas cost
            net_benefit = best_apy - current_apy - gas_annualized

            if net_benefit > self.rebal_threshold:
                return best_protocol, best_apy
            else:
                return current_protocol, current_apy
        return current_protocol, current_apy

    def backtest(
        self,
        protocols: List[str],
        apy_series: pd.DataFrame,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        """
        Backtest yield farming routing strategy.

        Parameters
        ----------
        protocols   : list of protocol names
        apy_series  : DataFrame, columns = protocols, rows = dates, values = annualized APY
        """
        if len(protocols) == 0:
            protocols = list(apy_series.columns)

        n = len(apy_series)
        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        allocation_hist = pd.DataFrame("none", index=apy_series.index, columns=["protocol"])
        fees_earned = 0.0
        n_rebalances = 0

        current_protocol = protocols[0]
        days_held = 0

        for i in range(1, n):
            row = apy_series.iloc[i]
            days_held += 1

            # Check if we should rebalance
            if days_held >= self.min_hold_days:
                new_protocol, effective_apy = self.select_protocol(
                    row[protocols], current_protocol, equity
                )
                if new_protocol != current_protocol:
                    # Pay gas cost
                    gas_fraction = self.gas_cost_usd / (equity + 1e-9)
                    equity *= (1 - gas_fraction)
                    current_protocol = new_protocol
                    days_held = 0
                    n_rebalances += 1

            # Apply daily yield from current protocol
            current_apy = float(row.get(current_protocol, 0.0))
            if np.isnan(current_apy):
                current_apy = 0.0

            daily_yield = current_apy / 365
            fees_earned += equity * daily_yield
            equity *= (1 + daily_yield)
            ec[i] = equity
            allocation_hist.iloc[i] = current_protocol

        s = _stats(ec)
        total_days = n
        net_apy = float((ec[-1] / initial_equity) ** (365 / max(1, total_days)) - 1)

        return BacktestResult(
            **s,
            n_rebalances=n_rebalances,
            total_fees_earned=fees_earned,
            net_apy=net_apy,
            equity_curve=pd.Series(ec, index=apy_series.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=apy_series.index[1:]),
            allocation_history=allocation_hist,
            params={"gas_cost_usd": self.gas_cost_usd, "rebal_threshold": self.rebal_threshold},
        )

    def optimal_allocation(
        self,
        apy_series: pd.DataFrame,
        gas_cost: float = 50.0,
        equity: float = 1_000_000,
    ) -> dict:
        """
        Compute the optimal routing strategy over the full history.

        Returns dict with:
        - best_protocol_by_apy: at each date, best protocol
        - avg_apy_if_optimal: what APY would be achieved with perfect routing
        - avg_apy_buy_and_hold: APY of staying in best protocol at start
        """
        best_by_date = apy_series.idxmax(axis=1)
        avg_best_apy = float(apy_series.max(axis=1).mean())
        hold_apy = float(apy_series.iloc[0].max())

        return {
            "avg_apy_optimal_routing": avg_best_apy,
            "avg_apy_buy_hold_best": hold_apy,
            "best_protocol_history": best_by_date,
            "most_common_best": str(best_by_date.mode().iloc[0]) if len(best_by_date) > 0 else "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LiquidityProvision
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityProvision:
    """
    Uniswap v2/v3 Liquidity Provision profitability model.

    LP P&L = fee income - impermanent loss (IL)

    Impermanent Loss arises when the price ratio of a token pair changes:
        IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1

    where price_ratio = current_price / initial_price.

    For Uniswap v3 (concentrated liquidity):
        IL is amplified within the price range,
        but fee income is also amplified.

    Parameters
    ----------
    pool_a          : column name for token A price
    pool_b          : column name for token B price
    fee_tier        : LP fee tier as fraction (e.g., 0.003 = 0.3%)
    il_model        : "v2" (constant product) or "v3" (concentrated)
    price_range_pct : ±% range for v3 concentrated liquidity (default 0.20 = ±20%)
    volume_fraction : estimated fraction of daily volume through this pool (default 0.01)
    """

    def __init__(
        self,
        pool_a: str = "token_a",
        pool_b: str = "token_b",
        fee_tier: float = 0.003,
        il_model: str = "v2",
        price_range_pct: float = 0.20,
        volume_fraction: float = 0.01,
    ):
        self.pool_a = pool_a
        self.pool_b = pool_b
        self.fee_tier = fee_tier
        self.il_model = il_model
        self.price_range_pct = price_range_pct
        self.volume_fraction = volume_fraction

    def impermanent_loss_v2(self, price_ratio: float) -> float:
        """
        Compute impermanent loss for Uniswap v2.

        IL = 2 * sqrt(k) / (1 + k) - 1
        where k = current_price / initial_price

        Returns IL as a negative fraction (always <= 0).
        """
        if price_ratio <= 0:
            return -1.0
        k = price_ratio
        il = 2.0 * math.sqrt(k) / (1.0 + k) - 1.0
        return float(il)  # always <= 0

    def impermanent_loss_v3(
        self,
        price_ratio: float,
        range_lower: float,
        range_upper: float,
    ) -> float:
        """
        Compute impermanent loss for Uniswap v3 (concentrated liquidity).

        IL is higher when price moves outside the range.
        When price is outside range: same as holding all of one token.

        Parameters
        ----------
        price_ratio   : current_price / initial_price
        range_lower   : lower bound of liquidity range (fraction of initial price)
        range_upper   : upper bound of liquidity range (fraction of initial price)
        """
        if price_ratio <= 0:
            return -1.0

        if price_ratio < range_lower:
            # Below range: all in token B (quote), 0 token A
            # Hold value = ratio * initial_ratio_value
            il = price_ratio - 1.0  # relative to holding
            return float(il)
        elif price_ratio > range_upper:
            # Above range: all in token A (base), 0 token B
            il = 1.0 - price_ratio
            return float(max(il, -1.0))
        else:
            # In range: normal v2 IL but amplified
            # Amplification factor
            L = 1.0 / (1.0 - 2.0 * self.price_range_pct / (1 + self.price_range_pct))
            base_il = self.impermanent_loss_v2(price_ratio)
            return float(max(-1.0, base_il * L))

    def fee_income(
        self,
        pool_volume_usd: pd.Series,
        tvl_usd: pd.Series,
        position_value: float,
    ) -> pd.Series:
        """
        Compute daily fee income for an LP position.

        daily_fee = (position_value / tvl) * pool_volume * fee_tier

        Parameters
        ----------
        pool_volume_usd : daily trading volume in USD
        tvl_usd         : total value locked in the pool
        position_value  : LP position value in USD
        """
        lp_share = position_value / (tvl_usd + 1e-9)
        daily_fee = lp_share * pool_volume_usd * self.fee_tier
        return daily_fee

    def simulate_lp_position(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        volume_usd: pd.Series,
        initial_position_usd: float = 100_000,
        tvl_usd: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Simulate LP position P&L over time.

        Returns dict with:
        - value_series: LP position value over time
        - il_series: impermanent loss over time
        - fee_series: cumulative fees earned
        - total_return: total return vs. holding
        """
        n = len(price_a)
        idx = price_a.index

        if tvl_usd is None:
            # Estimate TVL as 100x daily volume
            tvl_usd = volume_usd * 100

        # Initial allocation: 50/50 A and B
        initial_price_ratio = float(price_a.iloc[0]) / (float(price_b.iloc[0]) + 1e-9)
        position_value = initial_position_usd

        # For v3, compute range bounds
        if self.il_model == "v3":
            range_lower = 1.0 - self.price_range_pct
            range_upper = 1.0 + self.price_range_pct
        else:
            range_lower = 0.0
            range_upper = float("inf")

        value_series = np.full(n, initial_position_usd, dtype=float)
        il_series = np.zeros(n)
        fee_series = np.zeros(n)
        cumulative_fees = 0.0

        for i in range(1, n):
            current_price_ratio = (float(price_a.iloc[i]) / (float(price_b.iloc[i]) + 1e-9)) / (initial_price_ratio + 1e-9)

            # Compute IL
            if self.il_model == "v2":
                il = self.impermanent_loss_v2(current_price_ratio)
            else:
                il = self.impermanent_loss_v3(current_price_ratio, range_lower, range_upper)

            # Holding value
            hold_val = initial_position_usd * (
                0.5 * float(price_a.iloc[i]) / (float(price_a.iloc[0]) + 1e-9) +
                0.5 * float(price_b.iloc[i]) / (float(price_b.iloc[0]) + 1e-9)
            )

            # LP value = holding value * (1 + IL)
            lp_val = hold_val * (1 + il)

            # Daily fees
            daily_fee = float(self.fee_income(
                pd.Series([float(volume_usd.iloc[i])]),
                pd.Series([float(tvl_usd.iloc[i])]),
                position_value,
            ).iloc[0])
            cumulative_fees += daily_fee

            position_value = lp_val + cumulative_fees

            value_series[i] = position_value
            il_series[i] = il
            fee_series[i] = cumulative_fees

        total_return = value_series[-1] / initial_position_usd - 1
        hold_final = initial_position_usd * (
            0.5 * float(price_a.iloc[-1]) / (float(price_a.iloc[0]) + 1e-9) +
            0.5 * float(price_b.iloc[-1]) / (float(price_b.iloc[0]) + 1e-9)
        )
        hold_return = hold_final / initial_position_usd - 1

        return {
            "value_series": pd.Series(value_series, index=idx),
            "il_series": pd.Series(il_series, index=idx),
            "fee_series": pd.Series(fee_series, index=idx),
            "total_return": total_return,
            "hold_return": hold_return,
            "il_impact": total_return - hold_return,
            "cumulative_fees": cumulative_fees,
            "cumulative_fees_pct": cumulative_fees / initial_position_usd,
            "break_even_days": float(initial_position_usd * abs(il_series[-1]) / (cumulative_fees / max(1, n) + 1e-9)),
        }

    def backtest(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        volume_usd: pd.Series,
        initial_equity: float = 1_000_000,
        tvl_usd: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Full backtest of LP position."""
        sim = self.simulate_lp_position(price_a, price_b, volume_usd, initial_equity, tvl_usd)

        ec = sim["value_series"].values
        s = _stats(ec)
        n_days = len(ec)
        net_apy = float((ec[-1] / initial_equity) ** (365 / max(1, n_days)) - 1)

        return BacktestResult(
            **s,
            n_rebalances=0,
            total_fees_earned=sim["cumulative_fees"],
            total_il_cost=abs(float(sim["il_series"].iloc[-1]) * initial_equity),
            net_apy=net_apy,
            equity_curve=sim["value_series"],
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price_a.index[1:]),
            params={"fee_tier": self.fee_tier, "il_model": self.il_model,
                    "price_range_pct": self.price_range_pct},
        )

    def il_breakeven_analysis(
        self,
        price_range: Tuple[float, float],
        daily_volume_usd: float,
        tvl_usd: float,
        position_value: float,
    ) -> dict:
        """
        Compute breakeven analysis for an LP position.

        Parameters
        ----------
        price_range       : (min_price_ratio, max_price_ratio) to test
        daily_volume_usd  : average daily volume
        tvl_usd           : total value locked
        position_value    : our LP position size
        """
        daily_fee = position_value / tvl_usd * daily_volume_usd * self.fee_tier
        annual_fee_pct = daily_fee * 365 / position_value

        results = []
        for price_ratio in np.linspace(price_range[0], price_range[1], 20):
            if self.il_model == "v2":
                il = self.impermanent_loss_v2(price_ratio)
            else:
                il = self.impermanent_loss_v3(
                    price_ratio,
                    1.0 - self.price_range_pct,
                    1.0 + self.price_range_pct,
                )
            days_to_breakeven = abs(il) / (daily_fee / position_value + 1e-9) if daily_fee > 0 else float("inf")
            results.append({
                "price_ratio": price_ratio,
                "il_pct": il,
                "annual_fee_pct": annual_fee_pct,
                "net_return_pct": annual_fee_pct + il,
                "days_to_breakeven": days_to_breakeven,
                "profitable": (annual_fee_pct + il) > 0,
            })

        df = pd.DataFrame(results)
        return {
            "analysis": df,
            "daily_fee_income": daily_fee,
            "annual_fee_pct": annual_fee_pct,
            "max_tolerable_il": annual_fee_pct,
            "max_price_move_before_loss": float(
                df[df["profitable"] == True]["price_ratio"].max()
                if (df["profitable"] == True).any() else 1.0
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 365
    idx = pd.date_range("2023-01-01", periods=n, freq="D")

    # YieldFarming
    protocols = ["Aave", "Compound", "Curve", "Yearn"]
    apy_df = pd.DataFrame({
        "Aave": np.abs(rng.normal(0.05, 0.02, n)),
        "Compound": np.abs(rng.normal(0.04, 0.015, n)),
        "Curve": np.abs(rng.normal(0.08, 0.04, n)),
        "Yearn": np.abs(rng.normal(0.12, 0.08, n)),
    }, index=idx)

    router = YieldFarmingRouter(gas_cost_usd=50, rebal_threshold=0.02, min_hold_days=7)
    res1 = router.backtest(protocols, apy_df)
    print("Yield Farming:", res1.summary())
    print("Allocation history (last 5):")
    print(res1.allocation_history.tail())
    print("Optimal routing:", router.optimal_allocation(apy_df))

    # LP simulation
    eth_price = pd.Series(2000.0 * np.cumprod(1 + rng.normal(0.001, 0.03, n)), index=idx)
    usdc_price = pd.Series(np.ones(n), index=idx)  # stable
    volume = pd.Series(np.abs(rng.normal(50e6, 20e6, n)), index=idx)

    lp = LiquidityProvision(fee_tier=0.003, il_model="v2")
    res2 = lp.backtest(eth_price, usdc_price, volume, initial_equity=100_000)
    print("\nLP v2:", res2.summary())

    lp_v3 = LiquidityProvision(fee_tier=0.003, il_model="v3", price_range_pct=0.20)
    res3 = lp_v3.backtest(eth_price, usdc_price, volume, initial_equity=100_000)
    print("LP v3 (±20%):", res3.summary())

    # IL breakeven
    analysis = lp.il_breakeven_analysis(
        price_range=(0.1, 3.0),
        daily_volume_usd=50e6,
        tvl_usd=100e6,
        position_value=100_000,
    )
    print("\nIL Breakeven analysis:")
    print(analysis["analysis"].to_string())
