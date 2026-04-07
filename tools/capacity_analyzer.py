"""
capacity_analyzer.py -- Strategy capacity and scalability analysis for LARSA v18.

Uses the Almgren-Chriss square-root market impact model to estimate how
increasing AUM degrades strategy Sharpe, and finds per-symbol optimal hold
times that minimize round-trip cost per unit of alpha signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market microstructure constants
# ---------------------------------------------------------------------------

# Almgren-Chriss permanent impact coefficient (basis points per sqrt(participation_rate))
DEFAULT_PERM_IMPACT_BPS = 10.0

# Almgren-Chriss temporary impact coefficient (basis points per sqrt(participation_rate))
DEFAULT_TEMP_IMPACT_BPS = 15.0

# Typical daily volume in USD for asset classes (rough defaults)
DEFAULT_DAILY_VOLUME_USD: dict[str, float] = {
    "BTC": 20_000_000_000.0,
    "ETH": 10_000_000_000.0,
    "SOL": 2_000_000_000.0,
    "BNB": 1_500_000_000.0,
    "XRP": 3_000_000_000.0,
    "DEFAULT": 500_000_000.0,
}

# Annualized Sharpe degradation slope per 1% participation rate
# (empirically estimated: ~0.05 Sharpe units per 1% participation)
SHARPE_DEGRADATION_PER_PCT_PARTICIPATION = 0.05


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SymbolCapacityProfile:
    """Capacity profile for a single trading symbol."""
    symbol: str
    daily_volume_usd: float
    avg_spread_bps: float = 5.0
    perm_impact_coeff: float = DEFAULT_PERM_IMPACT_BPS
    temp_impact_coeff: float = DEFAULT_TEMP_IMPACT_BPS
    avg_hold_bars: float = 5.0      # average bars held (used for hold-time optimization)
    bar_duration_minutes: float = 60.0  # bar size in minutes
    base_sharpe: float = 1.0        # strategy Sharpe at negligible size
    signal_decay_halflife_bars: float = 8.0  # how fast the alpha signal decays


@dataclass
class ImpactEstimate:
    """Result of a single market impact calculation."""
    symbol: str
    position_usd: float
    participation_rate: float       # fraction of daily volume
    perm_impact_bps: float
    temp_impact_bps: float
    total_impact_bps: float
    round_trip_cost_usd: float
    sharpe_after_impact: float


# ---------------------------------------------------------------------------
# Almgren-Chriss impact model
# ---------------------------------------------------------------------------

def _ac_impact_bps(
    position_usd: float,
    daily_volume_usd: float,
    perm_coeff: float,
    temp_coeff: float,
) -> tuple[float, float, float]:
    """
    Compute Almgren-Chriss permanent + temporary market impact in basis points.

    impact = coeff * sqrt(participation_rate)

    participation_rate = position_size / daily_volume

    Returns (perm_bps, temp_bps, total_bps).
    """
    if daily_volume_usd <= 0 or position_usd <= 0:
        return 0.0, 0.0, 0.0
    participation = min(position_usd / daily_volume_usd, 1.0)
    sqrt_p = math.sqrt(participation)
    perm = perm_coeff * sqrt_p
    temp = temp_coeff * sqrt_p
    total = perm + temp
    return perm, temp, total


# ---------------------------------------------------------------------------
# CapacityAnalyzer
# ---------------------------------------------------------------------------

class CapacityAnalyzer:
    """
    Estimates strategy capacity limits and scalability for LARSA v18.

    The analysis uses:
    - Almgren-Chriss sqrt(participation) impact model
    - Sharpe degradation as a function of impact cost relative to signal strength
    - Per-symbol hold-time optimization to minimize cost per unit of alpha

    Usage::

        ca = CapacityAnalyzer(profiles=[...], strategy_sharpe=1.2)
        print(ca.compute_capacity_at_sharpe(0.5))
        print(ca.position_size_sensitivity("BTC", [1e5, 5e5, 1e6, 5e6]))
    """

    def __init__(
        self,
        profiles: list[SymbolCapacityProfile] | None = None,
        strategy_sharpe: float = 1.0,
        annual_turnover_pct: float = 500.0,  # 500% annual turnover = ~2x daily
        avg_position_usd: float = 100_000.0,
        num_symbols: int = 10,
        bps_per_sharpe_unit: float = 2.0,  # alpha strength in bps per Sharpe unit
    ):
        self.profiles = {p.symbol: p for p in (profiles or [])}
        self.strategy_sharpe = strategy_sharpe
        self.annual_turnover_pct = annual_turnover_pct
        self.avg_position_usd = avg_position_usd
        self.num_symbols = num_symbols
        self.bps_per_sharpe_unit = bps_per_sharpe_unit

        # Populate defaults for symbols not in profiles
        for sym, daily_vol in DEFAULT_DAILY_VOLUME_USD.items():
            if sym not in self.profiles:
                self.profiles[sym] = SymbolCapacityProfile(
                    symbol=sym,
                    daily_volume_usd=daily_vol,
                    base_sharpe=strategy_sharpe,
                )

    # ------------------------------------------------------------------
    # Core capacity estimate
    # ------------------------------------------------------------------

    def compute_capacity_at_sharpe(
        self,
        target_sharpe: float = 0.5,
        symbol: str = "DEFAULT",
        n_steps: int = 100,
    ) -> float:
        """
        Binary-search AUM at which portfolio Sharpe degrades to ``target_sharpe``.

        Model:
            Sharpe(AUM) = base_sharpe - impact_cost_bps / bps_per_sharpe_unit

        where impact_cost_bps is the Almgren-Chriss cost at AUM / num_symbols
        position size.

        Returns estimated AUM capacity in USD.
        """
        profile = self.profiles.get(symbol) or self.profiles.get("DEFAULT")
        if profile is None:
            profile = SymbolCapacityProfile(
                symbol=symbol,
                daily_volume_usd=DEFAULT_DAILY_VOLUME_USD["DEFAULT"],
            )

        if target_sharpe >= self.strategy_sharpe:
            logger.warning(
                "target_sharpe %.2f >= base Sharpe %.2f -- capacity is 0",
                target_sharpe, self.strategy_sharpe,
            )
            return 0.0

        # Binary search over AUM
        lo, hi = 0.0, profile.daily_volume_usd * 365.0  # max = 1 year daily volume
        for _ in range(60):  # 60 iterations = sub-dollar precision
            mid = (lo + hi) / 2.0
            sharpe = self._sharpe_at_aum(mid, profile)
            if sharpe > target_sharpe:
                lo = mid
            else:
                hi = mid

        capacity = (lo + hi) / 2.0
        logger.info(
            "Capacity at Sharpe=%.2f for %s: $%.0f",
            target_sharpe, symbol, capacity,
        )
        return capacity

    def _sharpe_at_aum(
        self,
        aum: float,
        profile: SymbolCapacityProfile,
    ) -> float:
        """Estimate strategy Sharpe at a given AUM for one symbol's profile."""
        position_per_symbol = aum / self.num_symbols
        _, _, impact_bps = _ac_impact_bps(
            position_per_symbol,
            profile.daily_volume_usd,
            profile.perm_impact_coeff,
            profile.temp_impact_coeff,
        )
        # Round-trip impact: 2x for buy and sell
        rt_impact = impact_bps * 2.0
        sharpe_penalty = rt_impact / self.bps_per_sharpe_unit
        return max(0.0, profile.base_sharpe - sharpe_penalty)

    # ------------------------------------------------------------------
    # Position size sensitivity
    # ------------------------------------------------------------------

    def position_size_sensitivity(
        self,
        symbol: str,
        sizes_usd: list[float],
    ) -> dict:
        """
        For a range of position sizes, compute impact cost and Sharpe degradation.

        Parameters
        ----------
        symbol : asset ticker
        sizes_usd : list of position sizes in USD to evaluate

        Returns dict keyed by position_usd with impact estimates.
        """
        profile = self.profiles.get(symbol) or self.profiles.get("DEFAULT")
        if profile is None:
            raise ValueError(f"No profile for symbol: {symbol}")

        results = {}
        for pos_usd in sizes_usd:
            perm, temp, total = _ac_impact_bps(
                pos_usd,
                profile.daily_volume_usd,
                profile.perm_impact_coeff,
                profile.temp_impact_coeff,
            )
            participation = pos_usd / profile.daily_volume_usd
            rt_cost_bps = total * 2.0
            rt_cost_usd = pos_usd * rt_cost_bps / 10_000.0
            sharpe_after = max(
                0.0,
                profile.base_sharpe - rt_cost_bps / self.bps_per_sharpe_unit,
            )
            results[pos_usd] = ImpactEstimate(
                symbol=symbol,
                position_usd=pos_usd,
                participation_rate=participation,
                perm_impact_bps=perm,
                temp_impact_bps=temp,
                total_impact_bps=total,
                round_trip_cost_usd=rt_cost_usd,
                sharpe_after_impact=sharpe_after,
            )

        return results

    def position_size_sensitivity_df(
        self,
        symbol: str,
        sizes_usd: list[float],
    ) -> pd.DataFrame:
        """Same as position_size_sensitivity but returns a DataFrame."""
        raw = self.position_size_sensitivity(symbol, sizes_usd)
        rows = []
        for pos, est in raw.items():
            rows.append({
                "position_usd": pos,
                "participation_rate_pct": est.participation_rate * 100.0,
                "perm_impact_bps": est.perm_impact_bps,
                "temp_impact_bps": est.temp_impact_bps,
                "total_impact_bps": est.total_impact_bps,
                "rt_cost_usd": est.round_trip_cost_usd,
                "rt_cost_bps": est.total_impact_bps * 2.0,
                "sharpe_after_impact": est.sharpe_after_impact,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Turnover cost
    # ------------------------------------------------------------------

    def turnover_cost_estimate(
        self,
        turnover_pct_daily: float,
        aum: float | None = None,
        spread_bps: float = 5.0,
        commission_bps: float = 1.0,
    ) -> float:
        """
        Estimate annual cost (in USD) of trading at a given daily turnover rate.

        Cost components:
        - Bid/ask spread: spread_bps per round-trip
        - Commission: commission_bps per side
        - Market impact: Almgren-Chriss at avg position size

        Parameters
        ----------
        turnover_pct_daily : fraction of AUM traded per day (e.g. 0.02 = 2% daily)
        aum : total AUM; defaults to num_symbols * avg_position_usd
        spread_bps : half-spread in basis points
        commission_bps : commission per side in basis points

        Returns annual cost in USD.
        """
        if aum is None:
            aum = self.num_symbols * self.avg_position_usd

        daily_turnover_usd = aum * turnover_pct_daily

        # Cost per dollar traded (one side)
        spread_cost_bps = spread_bps / 2.0  # half spread per trade side
        total_bps_per_side = spread_cost_bps + commission_bps

        # Impact on daily turnover
        avg_impact_bps = 0.0
        symbol_list = list(self.profiles.keys())[:self.num_symbols]
        for sym in symbol_list:
            prof = self.profiles[sym]
            pos = daily_turnover_usd / max(1, len(symbol_list))
            _, _, imp = _ac_impact_bps(
                pos, prof.daily_volume_usd,
                prof.perm_impact_coeff, prof.temp_impact_coeff,
            )
            avg_impact_bps += imp / max(1, len(symbol_list))

        total_cost_bps_per_dollar = total_bps_per_side * 2 + avg_impact_bps * 2
        daily_cost_usd = daily_turnover_usd * total_cost_bps_per_dollar / 10_000.0
        annual_cost_usd = daily_cost_usd * 252.0

        logger.info(
            "Annual trading cost at %.1f%% daily turnover: $%.0f (%.1f bps/RT)",
            turnover_pct_daily * 100, annual_cost_usd, total_cost_bps_per_dollar,
        )
        return annual_cost_usd

    def turnover_cost_as_pct_aum(
        self,
        turnover_pct_daily: float,
        aum: float | None = None,
        spread_bps: float = 5.0,
        commission_bps: float = 1.0,
    ) -> float:
        """Same as turnover_cost_estimate but expressed as fraction of AUM."""
        if aum is None:
            aum = self.num_symbols * self.avg_position_usd
        cost = self.turnover_cost_estimate(
            turnover_pct_daily, aum, spread_bps, commission_bps
        )
        return cost / aum if aum > 0 else 0.0

    # ------------------------------------------------------------------
    # Optimal hold time by symbol
    # ------------------------------------------------------------------

    def optimal_hold_time_by_symbol(
        self,
        hold_bar_candidates: list[int] | None = None,
    ) -> dict:
        """
        For each symbol, find the hold time (in bars) that minimizes cost
        per unit of alpha signal.

        Model:
        - Signal decays exponentially: signal(t) = signal_0 * exp(-t / halflife)
        - Impact amortized over hold time: cost_per_bar = rt_impact / hold_bars
        - Net signal at time t: signal(t) - cost_per_bar * t
        - Optimal T* = halflife * ln(signal_0 * halflife / rt_impact_bps)
          (approximate closed form from calculus)

        Returns dict symbol -> {optimal_bars, net_signal_at_optimal, ...}
        """
        if hold_bar_candidates is None:
            hold_bar_candidates = [1, 2, 3, 5, 8, 13, 21, 34, 55]

        results = {}
        for symbol, profile in self.profiles.items():
            if symbol == "DEFAULT":
                continue

            _, _, rt_half_bps = _ac_impact_bps(
                self.avg_position_usd,
                profile.daily_volume_usd,
                profile.perm_impact_coeff,
                profile.temp_impact_coeff,
            )
            rt_bps = rt_half_bps * 2.0
            alpha_bps = profile.base_sharpe * self.bps_per_sharpe_unit
            halflife = profile.signal_decay_halflife_bars

            # Closed-form optimal (continuous time): T* = halflife when cost ~= alpha / e
            if alpha_bps <= rt_bps:
                optimal_bars = 1
            else:
                # Continuous-time approximation
                t_star_continuous = halflife * math.log(alpha_bps / rt_bps) if rt_bps > 0 else halflife
                # Discretize to nearest candidate
                optimal_bars = min(
                    hold_bar_candidates,
                    key=lambda b: abs(b - t_star_continuous),
                )

            # Evaluate all candidates
            candidate_nets = {}
            for bars in hold_bar_candidates:
                signal_t = alpha_bps * math.exp(-bars / halflife)
                cost_t = rt_bps / bars if bars > 0 else rt_bps
                net = signal_t - cost_t
                candidate_nets[bars] = round(net, 4)

            discrete_optimal = max(candidate_nets, key=lambda b: candidate_nets[b])
            results[symbol] = {
                "optimal_bars_discrete": discrete_optimal,
                "optimal_bars_continuous": round(
                    halflife * math.log(max(alpha_bps / rt_bps, 1.0)), 2
                ) if rt_bps > 0 else halflife,
                "net_signal_at_optimal_bps": candidate_nets[discrete_optimal],
                "rt_impact_bps": round(rt_bps, 3),
                "alpha_bps": round(alpha_bps, 3),
                "signal_halflife_bars": halflife,
                "candidate_nets_bps": candidate_nets,
            }

        return results

    # ------------------------------------------------------------------
    # Scaling scenario table
    # ------------------------------------------------------------------

    def scaling_scenario_table(
        self,
        aum_values: list[float] | None = None,
        symbol: str = "BTC",
    ) -> pd.DataFrame:
        """
        Return a DataFrame showing Sharpe, impact cost, and capacity metrics
        at multiple AUM levels.
        """
        if aum_values is None:
            aum_values = [
                100_000, 500_000, 1_000_000, 5_000_000,
                10_000_000, 50_000_000, 100_000_000,
            ]

        profile = self.profiles.get(symbol) or self.profiles.get("DEFAULT")
        if profile is None:
            raise ValueError(f"No profile for {symbol}")

        rows = []
        for aum in aum_values:
            pos = aum / self.num_symbols
            perm, temp, total = _ac_impact_bps(
                pos,
                profile.daily_volume_usd,
                profile.perm_impact_coeff,
                profile.temp_impact_coeff,
            )
            participation = pos / profile.daily_volume_usd
            rt_bps = total * 2.0
            sharpe = max(0.0, profile.base_sharpe - rt_bps / self.bps_per_sharpe_unit)
            annual_impact_usd = (
                self.turnover_cost_estimate(
                    turnover_pct_daily=self.annual_turnover_pct / 100.0 / 252.0,
                    aum=aum,
                )
            )
            rows.append({
                "aum_usd": aum,
                "position_per_symbol_usd": pos,
                "participation_rate_pct": participation * 100.0,
                "impact_bps_one_way": total,
                "rt_impact_bps": rt_bps,
                "estimated_sharpe": sharpe,
                "annual_impact_cost_usd": annual_impact_usd,
                "annual_impact_pct_aum": annual_impact_usd / aum * 100.0,
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Capacity analyzer for LARSA v18"
    )
    parser.add_argument(
        "--sharpe", type=float, default=0.5,
        help="Target Sharpe for capacity estimate",
    )
    parser.add_argument(
        "--symbol", default="BTC", help="Symbol for sensitivity analysis"
    )
    parser.add_argument(
        "--aum", type=float, default=1_000_000.0, help="Base AUM for cost estimates"
    )
    parser.add_argument(
        "--turnover", type=float, default=0.02,
        help="Daily turnover as fraction of AUM (e.g. 0.02 = 2%%)"
    )
    args = parser.parse_args()

    ca = CapacityAnalyzer(avg_position_usd=args.aum / 10)

    capacity = ca.compute_capacity_at_sharpe(args.sharpe, symbol=args.symbol)
    print(f"\nCapacity at Sharpe={args.sharpe}: ${capacity:,.0f}")

    cost = ca.turnover_cost_estimate(args.turnover, aum=args.aum)
    print(f"Annual turnover cost at {args.turnover*100:.1f}% daily: ${cost:,.0f}")

    sizes = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
    print(f"\nPosition size sensitivity for {args.symbol}:")
    print(ca.position_size_sensitivity_df(args.symbol, sizes).to_string(index=False))

    print("\nOptimal hold times by symbol:")
    print(json.dumps(ca.optimal_hold_time_by_symbol(), indent=2))

    print("\nScaling scenarios:")
    print(ca.scaling_scenario_table(symbol=args.symbol).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
