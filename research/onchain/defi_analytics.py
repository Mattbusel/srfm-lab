"""
defi_analytics.py — DEX liquidity analysis for Uniswap V3.

Covers:
  - Pool data ingestion from The Graph API
  - TVL tracking and historical charting
  - Liquidity concentration analysis (tick-level)
  - Fee APY calculation
  - Impermanent loss calculator (exact formula + approximate)
  - Pool scoring / ranking
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_fetchers import OnChainDataProvider, TheGraphClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uniswap V3 math helpers
# ---------------------------------------------------------------------------

Q96 = 2**96
MIN_TICK = -887272
MAX_TICK = 887272


def tick_to_price(tick: int, decimals0: int = 18, decimals1: int = 18) -> float:
    """Convert a Uniswap V3 tick index to a token price (token1 per token0)."""
    raw = 1.0001**tick
    return raw * (10**decimals0) / (10**decimals1)


def price_to_tick(price: float, decimals0: int = 18, decimals1: int = 18) -> int:
    adjusted = price * (10**decimals1) / (10**decimals0)
    return int(math.log(adjusted) / math.log(1.0001))


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int = 18, decimals1: int = 18) -> float:
    """Convert sqrtPriceX96 to human-readable price."""
    sqrt_price = sqrt_price_x96 / Q96
    raw = sqrt_price**2
    return raw * (10**decimals0) / (10**decimals1)


def compute_liquidity_amounts(
    liquidity: int,
    sqrt_price: float,
    sqrt_lower: float,
    sqrt_upper: float,
) -> Tuple[float, float]:
    """
    Given a position's liquidity and price range, compute token amounts.
    Returns (amount0, amount1).
    """
    if sqrt_price <= sqrt_lower:
        amount0 = liquidity * (1 / sqrt_lower - 1 / sqrt_upper)
        amount1 = 0.0
    elif sqrt_price >= sqrt_upper:
        amount0 = 0.0
        amount1 = liquidity * (sqrt_upper - sqrt_lower)
    else:
        amount0 = liquidity * (1 / sqrt_price - 1 / sqrt_upper)
        amount1 = liquidity * (sqrt_price - sqrt_lower)
    return amount0, amount1


def fee_tier_to_bps(fee_tier: int) -> float:
    """Uniswap V3 fee tiers: 100 = 0.01%, 500 = 0.05%, 3000 = 0.3%, 10000 = 1%."""
    return fee_tier / 1_000_000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PoolDayData:
    date: int
    volume_usd: float
    fees_usd: float
    tvl_usd: float
    high: float
    low: float
    open_price: float
    close_price: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.date, tz=timezone.utc)

    @property
    def fee_to_tvl(self) -> float:
        return self.fees_usd / self.tvl_usd if self.tvl_usd else 0.0


@dataclass
class PoolInfo:
    pool_id: str
    token0_symbol: str
    token1_symbol: str
    token0_address: str
    token1_address: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: int           # in ppm e.g. 3000
    liquidity: int
    sqrt_price_x96: int
    current_tick: int
    token0_price: float
    token1_price: float
    volume_usd: float
    tvl_usd: float
    day_data: List[PoolDayData] = field(default_factory=list)
    ticks: List[Dict] = field(default_factory=list)

    @property
    def pair_name(self) -> str:
        return f"{self.token0_symbol}/{self.token1_symbol}"

    @property
    def fee_pct(self) -> float:
        return self.fee_tier / 1_000_000

    @property
    def current_price(self) -> float:
        try:
            return sqrt_price_x96_to_price(self.sqrt_price_x96, self.token0_decimals, self.token1_decimals)
        except Exception:
            return self.token0_price


@dataclass
class TickLevel:
    tick_idx: int
    liquidity_net: int
    liquidity_gross: int
    price0: float
    price1: float


@dataclass
class LiquidityConcentration:
    pool_id: str
    current_tick: int
    current_price: float
    in_range_liquidity_pct: float    # fraction of total liquidity within ±5% of spot
    in_range_tick_count: int
    total_tick_count: int
    concentration_score: float       # 0-1, higher = more concentrated
    tick_levels: List[TickLevel] = field(default_factory=list)
    liquidity_by_range: Dict[str, float] = field(default_factory=dict)  # "±1%", "±5%", "±10%", "all"


@dataclass
class FeeAPY:
    pool_id: str
    pair_name: str
    fee_tier: int
    tvl_usd: float
    avg_daily_fees_usd: float
    daily_fee_apr: float      # annualized daily fee / TVL
    weekly_fee_apr: float
    monthly_fee_apr: float
    fee_apy_compounded: float  # assuming daily compounding
    vol_to_tvl_ratio: float
    lookback_days: int


@dataclass
class ImpermanentLoss:
    price_ratio: float          # p1/p0 (price change factor)
    il_pct: float               # negative — actual loss percentage
    hodl_value: float
    lp_value: float
    divergence_loss_usd: float  # absolute dollar loss vs hodl
    token0_qty_lp: float
    token1_qty_lp: float
    token0_qty_hodl: float
    token1_qty_hodl: float

    @property
    def is_loss(self) -> bool:
        return self.il_pct < 0


# ---------------------------------------------------------------------------
# Pool data loader
# ---------------------------------------------------------------------------

class PoolLoader:
    """Fetches and parses Uniswap V3 pool data from The Graph."""

    def __init__(self, graph_client: TheGraphClient = None):
        self.graph = graph_client or TheGraphClient("uniswap/uniswap-v3")

    def load_pools(self, n: int = 100) -> List[PoolInfo]:
        raw = self.graph.get_top_pools(n)
        pools = []
        for r in raw:
            try:
                pools.append(self._parse_pool(r))
            except Exception as exc:
                logger.warning("Failed to parse pool %s: %s", r.get("id", "?"), exc)
        return pools

    def _parse_pool(self, r: Dict) -> PoolInfo:
        t0 = r.get("token0", {})
        t1 = r.get("token1", {})
        day_data = [
            PoolDayData(
                date=int(d.get("date", 0)),
                volume_usd=float(d.get("volumeUSD", 0)),
                fees_usd=float(d.get("feesUSD", 0)),
                tvl_usd=float(d.get("tvlUSD", 0)),
                high=float(d.get("high", 0)),
                low=float(d.get("low", 0)),
                open_price=float(d.get("open", 0)),
                close_price=float(d.get("close", 0)),
            )
            for d in r.get("poolDayData", [])
        ]
        return PoolInfo(
            pool_id=r["id"],
            token0_symbol=t0.get("symbol", "?"),
            token1_symbol=t1.get("symbol", "?"),
            token0_address=t0.get("id", ""),
            token1_address=t1.get("id", ""),
            token0_decimals=int(t0.get("decimals", 18)),
            token1_decimals=int(t1.get("decimals", 18)),
            fee_tier=int(r.get("feeTier", 3000)),
            liquidity=int(r.get("liquidity", 0)),
            sqrt_price_x96=int(r.get("sqrtPrice", 0)),
            current_tick=int(r.get("tick", 0)),
            token0_price=float(r.get("token0Price", 0)),
            token1_price=float(r.get("token1Price", 0)),
            volume_usd=float(r.get("volumeUSD", 0)),
            tvl_usd=float(r.get("totalValueLockedUSD", 0)),
            day_data=day_data,
        )

    def load_ticks(self, pool: PoolInfo) -> List[TickLevel]:
        raw = self.graph.get_pool_ticks(pool.pool_id)
        ticks = []
        for r in raw:
            try:
                ticks.append(TickLevel(
                    tick_idx=int(r.get("tickIdx", 0)),
                    liquidity_net=int(r.get("liquidityNet", 0)),
                    liquidity_gross=int(r.get("liquidityGross", 0)),
                    price0=float(r.get("price0", 0)),
                    price1=float(r.get("price1", 0)),
                ))
            except Exception:
                continue
        pool.ticks = [
            {
                "tick_idx": t.tick_idx,
                "liquidity_net": t.liquidity_net,
                "liquidity_gross": t.liquidity_gross,
                "price0": t.price0,
                "price1": t.price1,
            }
            for t in ticks
        ]
        return ticks


# ---------------------------------------------------------------------------
# TVL tracker
# ---------------------------------------------------------------------------

class TVLTracker:
    """Tracks TVL over time for one or many pools."""

    def __init__(self, pools: List[PoolInfo]):
        self.pools = pools

    def total_tvl(self) -> float:
        return sum(p.tvl_usd for p in self.pools)

    def tvl_series(self, pool: PoolInfo) -> List[Tuple[datetime, float]]:
        """Returns sorted (datetime, tvl_usd) from pool day data."""
        return sorted(
            [(d.dt, d.tvl_usd) for d in pool.day_data],
            key=lambda x: x[0],
        )

    def tvl_change(self, pool: PoolInfo, days: int = 7) -> float:
        """Percentage change in TVL over the last `days` days."""
        series = self.tvl_series(pool)
        if len(series) < 2:
            return 0.0
        recent = [v for _, v in series[-days:]]
        if len(recent) < 2 or recent[0] == 0:
            return 0.0
        return (recent[-1] - recent[0]) / recent[0]

    def top_pools_by_tvl(self, n: int = 10) -> List[PoolInfo]:
        return sorted(self.pools, key=lambda p: p.tvl_usd, reverse=True)[:n]

    def aggregate_tvl_by_token(self) -> Dict[str, float]:
        """Sum TVL contribution by token symbol (each pool counted twice)."""
        out: Dict[str, float] = {}
        for p in self.pools:
            half = p.tvl_usd / 2.0
            out[p.token0_symbol] = out.get(p.token0_symbol, 0.0) + half
            out[p.token1_symbol] = out.get(p.token1_symbol, 0.0) + half
        return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))

    def tvl_momentum(self, pool: PoolInfo, short: int = 3, long: int = 14) -> float:
        """Simple momentum: short-term avg TVL minus long-term avg TVL, normalized."""
        series = self.tvl_series(pool)
        if len(series) < long:
            return 0.0
        vals = [v for _, v in series]
        short_avg = float(np.mean(vals[-short:]))
        long_avg = float(np.mean(vals[-long:]))
        if long_avg == 0:
            return 0.0
        return (short_avg - long_avg) / long_avg

    def compute_all_tvl_changes(self, days: int = 7) -> Dict[str, float]:
        return {p.pool_id: self.tvl_change(p, days) for p in self.pools}


# ---------------------------------------------------------------------------
# Liquidity concentration analyzer
# ---------------------------------------------------------------------------

class LiquidityConcentrationAnalyzer:
    """
    Analyzes how concentrated liquidity is around the current price.
    Uses tick-level data to reconstruct active liquidity at each price.
    """

    def __init__(self, graph_client: TheGraphClient = None):
        self.graph = graph_client or TheGraphClient()
        self.loader = PoolLoader(self.graph)

    def analyze(self, pool: PoolInfo, load_ticks: bool = True) -> LiquidityConcentration:
        if load_ticks and not pool.ticks:
            self.loader.load_ticks(pool)

        ticks = pool.ticks
        if not ticks:
            return LiquidityConcentration(
                pool_id=pool.pool_id,
                current_tick=pool.current_tick,
                current_price=pool.current_price,
                in_range_liquidity_pct=0.0,
                in_range_tick_count=0,
                total_tick_count=0,
                concentration_score=0.0,
            )

        current_tick = pool.current_tick
        tick_spacing = self._fee_to_spacing(pool.fee_tier)

        # Reconstruct cumulative liquidity at each tick
        sorted_ticks = sorted(ticks, key=lambda t: t["tick_idx"])
        running_liq = 0
        liq_by_tick: Dict[int, int] = {}
        for t in sorted_ticks:
            running_liq += t["liquidity_net"]
            liq_by_tick[t["tick_idx"]] = max(0, running_liq)

        total_gross = sum(t["liquidity_gross"] for t in ticks)
        if total_gross == 0:
            total_gross = 1

        def _ticks_in_pct_range(pct: float) -> Tuple[int, float]:
            delta_tick = int(abs(math.log(1 + pct) / math.log(1.0001)))
            lo = current_tick - delta_tick
            hi = current_tick + delta_tick
            in_range = sum(
                t["liquidity_gross"]
                for t in ticks
                if lo <= t["tick_idx"] <= hi
            )
            return (
                sum(1 for t in ticks if lo <= t["tick_idx"] <= hi),
                in_range / total_gross,
            )

        cnt_1, pct_1 = _ticks_in_pct_range(0.01)
        cnt_5, pct_5 = _ticks_in_pct_range(0.05)
        cnt_10, pct_10 = _ticks_in_pct_range(0.10)

        # Concentration score: Herfindahl-like index over liquidity gross
        tick_shares = [t["liquidity_gross"] / total_gross for t in ticks if t["liquidity_gross"] > 0]
        hhi = sum(s**2 for s in tick_shares)  # 0=uniform, 1=fully concentrated
        normalized_hhi = min(1.0, hhi * len(tick_shares))

        return LiquidityConcentration(
            pool_id=pool.pool_id,
            current_tick=current_tick,
            current_price=pool.current_price,
            in_range_liquidity_pct=pct_5,
            in_range_tick_count=cnt_5,
            total_tick_count=len(ticks),
            concentration_score=normalized_hhi,
            tick_levels=[
                TickLevel(
                    tick_idx=t["tick_idx"],
                    liquidity_net=t["liquidity_net"],
                    liquidity_gross=t["liquidity_gross"],
                    price0=t.get("price0", 0.0),
                    price1=t.get("price1", 0.0),
                )
                for t in sorted_ticks
            ],
            liquidity_by_range={
                "±1%": pct_1,
                "±5%": pct_5,
                "±10%": pct_10,
                "all": 1.0,
            },
        )

    @staticmethod
    def _fee_to_spacing(fee_tier: int) -> int:
        mapping = {100: 1, 500: 10, 3000: 60, 10000: 200}
        return mapping.get(fee_tier, 60)

    def concentration_ranking(self, pools: List[PoolInfo]) -> List[Tuple[PoolInfo, LiquidityConcentration]]:
        results = []
        for pool in pools:
            try:
                conc = self.analyze(pool, load_ticks=True)
                results.append((pool, conc))
            except Exception as exc:
                logger.warning("Concentration analysis failed for %s: %s", pool.pool_id, exc)
        return sorted(results, key=lambda x: x[1].concentration_score, reverse=True)


# ---------------------------------------------------------------------------
# Fee APY calculator
# ---------------------------------------------------------------------------

class FeeAPYCalculator:
    """Calculates fee APY for Uniswap V3 pools from historical day data."""

    DAYS_PER_YEAR = 365.25

    def calculate(self, pool: PoolInfo, lookback_days: int = 30) -> FeeAPY:
        days = sorted(pool.day_data, key=lambda d: d.date)[-lookback_days:]
        if not days:
            return FeeAPY(
                pool_id=pool.pool_id,
                pair_name=pool.pair_name,
                fee_tier=pool.fee_tier,
                tvl_usd=pool.tvl_usd,
                avg_daily_fees_usd=0.0,
                daily_fee_apr=0.0,
                weekly_fee_apr=0.0,
                monthly_fee_apr=0.0,
                fee_apy_compounded=0.0,
                vol_to_tvl_ratio=0.0,
                lookback_days=lookback_days,
            )

        fees = [d.fees_usd for d in days]
        tvls = [d.tvl_usd for d in days if d.tvl_usd > 0]
        vols = [d.volume_usd for d in days]

        avg_fees = float(np.mean(fees)) if fees else 0.0
        avg_tvl = float(np.mean(tvls)) if tvls else pool.tvl_usd
        avg_vol = float(np.mean(vols)) if vols else 0.0

        if avg_tvl == 0:
            return FeeAPY(
                pool_id=pool.pool_id,
                pair_name=pool.pair_name,
                fee_tier=pool.fee_tier,
                tvl_usd=pool.tvl_usd,
                avg_daily_fees_usd=avg_fees,
                daily_fee_apr=0.0,
                weekly_fee_apr=0.0,
                monthly_fee_apr=0.0,
                fee_apy_compounded=0.0,
                vol_to_tvl_ratio=0.0,
                lookback_days=lookback_days,
            )

        daily_rate = avg_fees / avg_tvl
        weekly_apr = daily_rate * 7 * self.DAYS_PER_YEAR / 7   # annualized
        monthly_apr = daily_rate * 30 * self.DAYS_PER_YEAR / 30
        daily_apr = daily_rate * self.DAYS_PER_YEAR

        # Compound APY: (1 + daily_rate)^365 - 1
        apy = (1 + daily_rate) ** self.DAYS_PER_YEAR - 1

        return FeeAPY(
            pool_id=pool.pool_id,
            pair_name=pool.pair_name,
            fee_tier=pool.fee_tier,
            tvl_usd=avg_tvl,
            avg_daily_fees_usd=avg_fees,
            daily_fee_apr=daily_apr,
            weekly_fee_apr=weekly_apr,
            monthly_fee_apr=monthly_apr,
            fee_apy_compounded=apy,
            vol_to_tvl_ratio=avg_vol / avg_tvl,
            lookback_days=len(days),
        )

    def rank_by_apy(self, pools: List[PoolInfo], lookback_days: int = 30) -> List[Tuple[PoolInfo, FeeAPY]]:
        results = [(p, self.calculate(p, lookback_days)) for p in pools]
        return sorted(results, key=lambda x: x[1].fee_apy_compounded, reverse=True)

    def fee_apy_summary(self, pools: List[PoolInfo]) -> Dict[str, float]:
        apys = [self.calculate(p).fee_apy_compounded for p in pools if p.tvl_usd > 100_000]
        if not apys:
            return {}
        return {
            "mean": float(np.mean(apys)),
            "median": float(np.median(apys)),
            "p25": float(np.percentile(apys, 25)),
            "p75": float(np.percentile(apys, 75)),
            "max": float(np.max(apys)),
            "min": float(np.min(apys)),
        }


# ---------------------------------------------------------------------------
# Impermanent loss calculator
# ---------------------------------------------------------------------------

class ImpermanentLossCalculator:
    """
    Exact IL formula for Uniswap V2/V3 full-range positions.
    For concentrated V3 positions, uses adjusted formula.
    """

    @staticmethod
    def il_full_range(price_ratio: float) -> float:
        """
        Standard IL formula for full-range positions.
        price_ratio = p1 / p0 (how many times the price has moved)
        Returns IL as a fraction (e.g., -0.057 means -5.7% vs hodl).
        """
        if price_ratio <= 0:
            raise ValueError("price_ratio must be positive")
        # IL = 2*sqrt(r)/(1+r) - 1  where r = price_ratio
        r = price_ratio
        return 2 * math.sqrt(r) / (1 + r) - 1.0

    @staticmethod
    def il_concentrated(
        price_ratio: float,
        p_lower: float,
        p_upper: float,
        p_entry: float = 1.0,
    ) -> float:
        """
        Impermanent loss for a concentrated liquidity position [p_lower, p_upper].
        Prices are in the same units (e.g., USDC per ETH).
        price_ratio = p_exit / p_entry
        """
        p_exit = p_entry * price_ratio
        p_lower_n = p_lower / p_entry
        p_upper_n = p_upper / p_entry

        # Clamp exit price to range boundaries
        p_exit_n = p_exit / p_entry

        sqrt_lo = math.sqrt(p_lower_n)
        sqrt_hi = math.sqrt(p_upper_n)
        sqrt_entry = 1.0  # normalized

        # Amount of each token at entry (equal USD value assumed)
        # Using normalized prices
        def amounts_at_price(sqrt_p: float) -> Tuple[float, float]:
            sqrt_p = max(sqrt_lo, min(sqrt_hi, sqrt_p))
            x = (sqrt_hi - sqrt_p) / (sqrt_p * sqrt_hi)
            y = sqrt_p - sqrt_lo
            return x, y

        x_entry, y_entry = amounts_at_price(sqrt_entry)
        x_exit, y_exit = amounts_at_price(math.sqrt(p_exit_n))

        # Value at exit of LP position (in terms of token1)
        lp_value = x_exit * p_exit_n + y_exit

        # Value at exit if we had held
        hodl_value = x_entry * p_exit_n + y_entry

        if hodl_value == 0:
            return 0.0

        return lp_value / hodl_value - 1.0

    def compute_full(
        self,
        initial_price: float,
        final_price: float,
        initial_value_usd: float,
        position_type: str = "full",
        p_lower: float = None,
        p_upper: float = None,
    ) -> ImpermanentLoss:
        """
        Compute full IL breakdown.
        Returns ImpermanentLoss dataclass with all fields populated.
        """
        ratio = final_price / initial_price
        if position_type == "concentrated" and p_lower and p_upper:
            il_frac = self.il_concentrated(ratio, p_lower, p_upper, initial_price)
        else:
            il_frac = self.il_full_range(ratio)

        # Token quantities at entry (equal weight, price=initial_price in USD)
        # Assume 50/50 split: value_usd/2 in token0, value_usd/2 in token1
        token0_qty_hodl = (initial_value_usd / 2.0) / initial_price
        token1_qty_hodl = initial_value_usd / 2.0

        # At final price, LP position rebalances
        sqrt_r = math.sqrt(ratio)
        token0_qty_lp = token0_qty_hodl / sqrt_r
        token1_qty_lp = token1_qty_hodl * sqrt_r

        hodl_value = token0_qty_hodl * final_price + token1_qty_hodl
        lp_value = token0_qty_lp * final_price + token1_qty_lp

        return ImpermanentLoss(
            price_ratio=ratio,
            il_pct=il_frac,
            hodl_value=hodl_value,
            lp_value=lp_value,
            divergence_loss_usd=lp_value - hodl_value,
            token0_qty_lp=token0_qty_lp,
            token1_qty_lp=token1_qty_lp,
            token0_qty_hodl=token0_qty_hodl,
            token1_qty_hodl=token1_qty_hodl,
        )

    def il_table(self, price_moves: List[float] = None) -> List[Dict[str, float]]:
        """Generate IL table for a set of price move percentages."""
        if price_moves is None:
            price_moves = [-90, -75, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 200, 500]
        rows = []
        for pct in price_moves:
            ratio = 1.0 + pct / 100.0
            if ratio <= 0:
                continue
            il = self.il_full_range(ratio)
            rows.append({
                "price_change_pct": pct,
                "price_ratio": ratio,
                "il_pct": il * 100,
                "breakeven_fee_pct": abs(il) * 100,
            })
        return rows

    def breakeven_volume(
        self,
        pool: PoolInfo,
        initial_value: float,
        final_price: float,
    ) -> float:
        """Volume required for fees to offset IL."""
        il = self.compute_full(pool.current_price, final_price, initial_value)
        dollar_loss = abs(il.divergence_loss_usd)
        fee_rate = pool.fee_pct
        if fee_rate == 0:
            return float("inf")
        # Fees earned = (position_share) * volume * fee_rate
        # Approximation: position earns fee proportional to its TVL share
        tvl_share = initial_value / max(pool.tvl_usd, initial_value)
        return dollar_loss / (fee_rate * tvl_share)


# ---------------------------------------------------------------------------
# Pool scorer / ranker
# ---------------------------------------------------------------------------

class PoolScorer:
    """
    Multi-factor pool ranking.
    Score = weighted combination of: fee APY, TVL stability, volume/TVL, concentration.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "fee_apy": 0.35,
            "vol_tvl": 0.25,
            "tvl_stability": 0.20,
            "concentration": 0.20,
        }
        self.apy_calc = FeeAPYCalculator()
        self.conc_analyzer = LiquidityConcentrationAnalyzer()

    def score_pool(self, pool: PoolInfo) -> Dict[str, float]:
        apy_data = self.apy_calc.calculate(pool)
        conc_data = self.conc_analyzer.analyze(pool, load_ticks=False)

        fee_apy_score = min(1.0, apy_data.fee_apy_compounded / 2.0)   # normalize to 200% APY max
        vol_tvl_score = min(1.0, apy_data.vol_to_tvl_ratio / 5.0)     # normalize to 5x daily turnover

        tvl_changes = []
        days = sorted(pool.day_data, key=lambda d: d.date)
        for i in range(1, len(days)):
            prev = days[i - 1].tvl_usd
            if prev > 0:
                tvl_changes.append(abs((days[i].tvl_usd - prev) / prev))
        tvl_stability = 1.0 - min(1.0, float(np.mean(tvl_changes)) * 10) if tvl_changes else 0.5

        composite = (
            self.weights["fee_apy"] * fee_apy_score +
            self.weights["vol_tvl"] * vol_tvl_score +
            self.weights["tvl_stability"] * tvl_stability +
            self.weights["concentration"] * conc_data.concentration_score
        )

        return {
            "pool_id": pool.pool_id,
            "pair": pool.pair_name,
            "fee_tier": pool.fee_tier,
            "tvl_usd": pool.tvl_usd,
            "fee_apy": apy_data.fee_apy_compounded,
            "vol_tvl_ratio": apy_data.vol_to_tvl_ratio,
            "tvl_stability": tvl_stability,
            "concentration_score": conc_data.concentration_score,
            "in_range_liq_pct": conc_data.in_range_liquidity_pct,
            "composite_score": composite,
        }

    def rank_pools(self, pools: List[PoolInfo]) -> List[Dict[str, float]]:
        scores = []
        for pool in pools:
            try:
                scores.append(self.score_pool(pool))
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", pool.pool_id, exc)
        return sorted(scores, key=lambda x: x["composite_score"], reverse=True)


# ---------------------------------------------------------------------------
# Main analytics facade
# ---------------------------------------------------------------------------

class DeFiAnalytics:
    """
    Top-level entry point combining pool loading, TVL tracking, fee APY,
    liquidity concentration and IL into one unified API.
    """

    def __init__(self, provider: OnChainDataProvider = None):
        self.provider = provider or OnChainDataProvider()
        self.loader = PoolLoader(self.provider.graph)
        self.tvl_tracker = None
        self.fee_calc = FeeAPYCalculator()
        self.conc_analyzer = LiquidityConcentrationAnalyzer(self.provider.graph)
        self.il_calc = ImpermanentLossCalculator()
        self.scorer = PoolScorer()
        self._pools: List[PoolInfo] = []

    def load_pools(self, n: int = 100) -> List[PoolInfo]:
        self._pools = self.loader.load_pools(n)
        self.tvl_tracker = TVLTracker(self._pools)
        return self._pools

    @property
    def pools(self) -> List[PoolInfo]:
        if not self._pools:
            self.load_pools()
        return self._pools

    def top_apy_pools(self, n: int = 10, min_tvl: float = 500_000) -> List[Tuple[PoolInfo, FeeAPY]]:
        eligible = [p for p in self.pools if p.tvl_usd >= min_tvl]
        ranked = self.fee_calc.rank_by_apy(eligible)
        return ranked[:n]

    def top_tvl_pools(self, n: int = 10) -> List[PoolInfo]:
        return sorted(self.pools, key=lambda p: p.tvl_usd, reverse=True)[:n]

    def analyze_pool(self, pool_id: str) -> Dict:
        pool = next((p for p in self.pools if p.pool_id == pool_id), None)
        if pool is None:
            raise ValueError(f"Pool {pool_id} not found in loaded data")

        apy = self.fee_calc.calculate(pool)
        conc = self.conc_analyzer.analyze(pool)
        tvl_change_7d = self.tvl_tracker.tvl_change(pool, 7) if self.tvl_tracker else 0.0

        return {
            "pool": pool,
            "fee_apy": apy,
            "concentration": conc,
            "tvl_change_7d": tvl_change_7d,
        }

    def il_scenario(
        self,
        pool_id: str,
        initial_value_usd: float,
        price_changes: List[float],
    ) -> List[ImpermanentLoss]:
        pool = next((p for p in self.pools if p.pool_id == pool_id), None)
        if pool is None:
            raise ValueError(f"Pool {pool_id} not found")

        results = []
        for pct in price_changes:
            final_price = pool.current_price * (1 + pct / 100)
            il = self.il_calc.compute_full(pool.current_price, final_price, initial_value_usd)
            results.append(il)
        return results

    def full_report(self, n_pools: int = 50) -> Dict:
        pools = self.load_pools(n_pools)
        top_apy = self.top_apy_pools(10)
        top_tvl = self.top_tvl_pools(10)
        ranked = self.scorer.rank_pools(pools[:20])  # top 20 by TVL for speed

        return {
            "total_pools_loaded": len(pools),
            "total_tvl_usd": self.tvl_tracker.total_tvl() if self.tvl_tracker else 0,
            "tvl_by_token": self.tvl_tracker.aggregate_tvl_by_token() if self.tvl_tracker else {},
            "top_apy": [
                {"pair": p.pair_name, "apy": a.fee_apy_compounded, "tvl": a.tvl_usd}
                for p, a in top_apy
            ],
            "top_tvl": [
                {"pair": p.pair_name, "tvl": p.tvl_usd, "volume": p.volume_usd}
                for p in top_tvl
            ],
            "pool_scores": ranked,
            "fee_apy_stats": self.fee_calc.fee_apy_summary(pools),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="DeFi analytics CLI")
    parser.add_argument("--action", choices=["report", "il_table", "top_apy"], default="report")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    analytics = DeFiAnalytics()

    if args.action == "report":
        report = analytics.full_report(args.n)
        print(_json.dumps({
            "total_tvl": report["total_tvl_usd"],
            "top_tvl": report["top_tvl"][:5],
            "top_apy": report["top_apy"][:5],
        }, indent=2))

    elif args.action == "il_table":
        calc = ImpermanentLossCalculator()
        table = calc.il_table()
        print(f"{'Price Change':>15} {'Price Ratio':>12} {'IL %':>10} {'Breakeven Fee':>15}")
        print("-" * 55)
        for row in table:
            print(f"{row['price_change_pct']:>14.0f}% {row['price_ratio']:>12.3f} "
                  f"{row['il_pct']:>9.2f}% {row['breakeven_fee_pct']:>14.2f}%")

    elif args.action == "top_apy":
        pools = analytics.load_pools(args.n)
        ranked = analytics.fee_calc.rank_by_apy(pools)
        print(f"{'Pair':>20} {'Fee Tier':>10} {'TVL':>14} {'Compounded APY':>15}")
        print("-" * 65)
        for pool, apy in ranked[:10]:
            print(f"{pool.pair_name:>20} {pool.fee_tier:>10} "
                  f"${apy.tvl_usd:>13,.0f} {apy.fee_apy_compounded*100:>14.1f}%")
