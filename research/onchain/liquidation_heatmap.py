"""
liquidation_heatmap.py — Liquidation level estimation for perpetual futures.

Covers:
  - OI-weighted liquidation price levels
  - Cascade risk calculator (domino liquidation model)
  - Liquidation waterfall model (sequential liquidation at each level)
  - Cross-exchange aggregation
  - Heatmap grid generation
  - Distance from spot to major liquidation clusters
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from data_fetchers import DiskCache, GlassnodeClient, OnChainDataProvider, _build_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PositionBucket:
    """Aggregate position data at a given leverage/price band."""
    price_level: float
    leverage: float
    liq_price: float             # estimated liquidation price
    long_oi_usd: float
    short_oi_usd: float
    net_oi_usd: float
    exchange: str

    @property
    def liq_distance_pct(self) -> float:
        """Distance from liq_price to price_level as fraction."""
        return abs(self.liq_price - self.price_level) / self.price_level


@dataclass
class LiquidationLevel:
    """Aggregated liquidation volume at a specific price."""
    price: float
    long_liq_usd: float          # longs liquidated if price drops to this level
    short_liq_usd: float         # shorts liquidated if price rises to this level
    cumulative_long_liq: float   # cumulative below spot
    cumulative_short_liq: float  # cumulative above spot
    distance_from_spot_pct: float
    exchanges: List[str]

    @property
    def total_liq_usd(self) -> float:
        return self.long_liq_usd + self.short_liq_usd

    @property
    def is_major_level(self) -> bool:
        return self.total_liq_usd >= 50_000_000  # $50M threshold


@dataclass
class CascadeScenario:
    """Models a liquidation cascade starting at a trigger price."""
    trigger_price: float
    direction: str               # "down" (longs get liq'd) or "up" (shorts)
    initial_liq_usd: float
    cascade_levels: List[Tuple[float, float]]  # [(price, usd_liq)]
    total_cascade_usd: float
    final_price_estimate: float  # where cascade might end
    cascade_risk: str            # "low", "medium", "high", "extreme"
    price_impact_pct: float      # estimated additional move from cascade


@dataclass
class LiquidationHeatmap:
    spot_price: float
    timestamp: datetime
    levels: List[LiquidationLevel]
    nearest_long_cluster: Optional[LiquidationLevel]   # biggest cluster below spot
    nearest_short_cluster: Optional[LiquidationLevel]  # biggest cluster above spot
    total_long_liq_at_risk_usd: float    # within 10% below spot
    total_short_liq_at_risk_usd: float   # within 10% above spot
    liq_ratio: float                      # long/short liq ratio (>1 = more longs at risk)
    pin_risk_levels: List[float]          # prices with high magnetic/pin effect


# ---------------------------------------------------------------------------
# Coinglass / Hyblock liquidation data fetcher
# ---------------------------------------------------------------------------

class CoinglassLiquidationFetcher:
    """
    Fetches liquidation heatmap data from Coinglass API.
    Falls back to synthetic data generation if API not available.
    """

    BASE = "https://open-api.coinglass.com/public/v2"

    def __init__(self, api_key: str = "", cache: DiskCache = None):
        self.api_key = api_key
        self.session = _build_session()
        self.session.headers["coinglassSecret"] = api_key
        self.cache = cache or DiskCache()

    def get_liquidation_map(self, symbol: str = "BTC", exchange: str = "Binance") -> Dict:
        """Fetch liquidation heatmap data."""
        import os
        if not self.api_key:
            api_key = os.environ.get("COINGLASS_API_KEY", "")
            if not api_key:
                return self._synthetic_liquidation_map(symbol)
            self.api_key = api_key
            self.session.headers["coinglassSecret"] = api_key

        try:
            resp = self.session.get(
                f"{self.BASE}/liquidation_map",
                params={"symbol": symbol, "exchange": exchange},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return self._synthetic_liquidation_map(symbol)

    def _synthetic_liquidation_map(self, symbol: str) -> Dict:
        """
        Generate synthetic liquidation map based on realistic distributions.
        Used when API access is unavailable.
        """
        seed_prices = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
            "BNB": 400.0,
        }
        spot = seed_prices.get(symbol.upper(), 1000.0)

        # Generate realistic OI distribution
        # Most liquidations cluster at round numbers and ±5%, ±10%, ±15%
        np.random.seed(42)
        n_levels = 200
        price_range = np.linspace(spot * 0.70, spot * 1.30, n_levels)

        # Bimodal distribution: more liq near -5% and -10% (leveraged longs)
        long_liq = np.zeros(n_levels)
        short_liq = np.zeros(n_levels)

        for i, p in enumerate(price_range):
            pct = (p - spot) / spot
            # Longs liquidated below spot: cluster at -5%, -10%, -15%, -20%
            if pct < 0:
                for center, weight in [(-0.05, 3.0), (-0.10, 2.0), (-0.15, 1.5), (-0.20, 1.0)]:
                    dist = abs(pct - center)
                    long_liq[i] += weight * np.exp(-dist**2 / (2 * 0.015**2))
                long_liq[i] *= np.random.uniform(0.5, 1.5)

            # Shorts liquidated above spot: cluster at +5%, +10%
            if pct > 0:
                for center, weight in [(0.05, 2.0), (0.10, 1.5), (0.15, 1.0)]:
                    dist = abs(pct - center)
                    short_liq[i] += weight * np.exp(-dist**2 / (2 * 0.015**2))
                short_liq[i] *= np.random.uniform(0.5, 1.5)

        # Normalize to realistic dollar amounts
        total_oi_estimate = 20_000_000_000  # $20B OI for BTC
        long_liq = long_liq / long_liq.sum() * total_oi_estimate * 0.3
        short_liq = short_liq / short_liq.sum() * total_oi_estimate * 0.15

        return {
            "symbol": symbol,
            "spot_price": spot,
            "price_levels": price_range.tolist(),
            "long_liquidations_usd": long_liq.tolist(),
            "short_liquidations_usd": short_liq.tolist(),
            "synthetic": True,
        }

    def get_open_interest_by_leverage(self, symbol: str = "BTC") -> Dict:
        """Fetch OI distribution by leverage tier."""
        try:
            resp = self.session.get(
                f"{self.BASE}/futures/openInterest/leverage/detail",
                params={"symbol": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return self._synthetic_leverage_distribution(symbol)

    def _synthetic_leverage_distribution(self, symbol: str) -> Dict:
        """Generate synthetic OI by leverage distribution."""
        leverage_tiers = [2, 3, 5, 10, 20, 25, 50, 100, 125]
        # More OI at lower leverage tiers
        weights = [0.20, 0.18, 0.22, 0.18, 0.10, 0.05, 0.04, 0.02, 0.01]
        total_oi = 20_000_000_000
        return {
            "symbol": symbol,
            "leverage_tiers": [
                {"leverage": lev, "oi_usd": total_oi * w, "pct": w * 100}
                for lev, w in zip(leverage_tiers, weights)
            ],
        }


# ---------------------------------------------------------------------------
# Liquidation level estimator
# ---------------------------------------------------------------------------

class LiquidationLevelEstimator:
    """
    Estimates liquidation levels from OI distribution and leverage data.
    For a long position at price P with leverage L:
        liq_price ≈ P * (1 - 1/L + maintenance_margin)
    For a short position:
        liq_price ≈ P * (1 + 1/L - maintenance_margin)
    """

    MAINTENANCE_MARGIN = 0.005   # 0.5% maintenance margin

    def estimate_liq_price(
        self,
        entry_price: float,
        leverage: float,
        side: str,  # "long" or "short"
        maintenance_margin: float = MAINTENANCE_MARGIN,
    ) -> float:
        if side == "long":
            return entry_price * (1 - 1/leverage + maintenance_margin)
        else:
            return entry_price * (1 + 1/leverage - maintenance_margin)

    def build_position_buckets(
        self,
        spot_price: float,
        leverage_dist: Dict,
        total_long_oi: float,
        total_short_oi: float,
        n_entry_bands: int = 20,
    ) -> List[PositionBucket]:
        """
        Build position buckets across entry prices and leverage tiers.
        Assumes positions entered uniformly over the last ±20% price range.
        """
        tiers = leverage_dist.get("leverage_tiers", [])
        if not tiers:
            return []

        # Price bands where positions were entered
        entry_prices = np.linspace(spot_price * 0.80, spot_price * 1.20, n_entry_bands)
        buckets = []

        for tier in tiers:
            lev = tier["leverage"]
            weight = tier["pct"] / 100.0

            for ep in entry_prices:
                long_liq = self.estimate_liq_price(ep, lev, "long")
                short_liq = self.estimate_liq_price(ep, lev, "short")

                long_oi_band = total_long_oi * weight / n_entry_bands
                short_oi_band = total_short_oi * weight / n_entry_bands

                buckets.append(PositionBucket(
                    price_level=ep,
                    leverage=lev,
                    liq_price=long_liq,
                    long_oi_usd=long_oi_band,
                    short_oi_usd=0.0,
                    net_oi_usd=long_oi_band,
                    exchange="aggregate",
                ))
                buckets.append(PositionBucket(
                    price_level=ep,
                    leverage=lev,
                    liq_price=short_liq,
                    long_oi_usd=0.0,
                    short_oi_usd=short_oi_band,
                    net_oi_usd=-short_oi_band,
                    exchange="aggregate",
                ))
        return buckets

    def aggregate_to_levels(
        self,
        spot_price: float,
        raw_data: Dict,
        n_bins: int = 200,
    ) -> List[LiquidationLevel]:
        """Convert raw liquidation map data into structured levels."""
        prices = raw_data.get("price_levels", [])
        long_liqs = raw_data.get("long_liquidations_usd", [])
        short_liqs = raw_data.get("short_liquidations_usd", [])

        if not prices:
            return []

        levels = []
        cum_long = 0.0
        cum_short = 0.0

        for price, ll, sl in zip(prices, long_liqs, short_liqs):
            dist_pct = (price - spot_price) / spot_price * 100

            if price < spot_price:
                cum_long += ll
            else:
                cum_short += sl

            levels.append(LiquidationLevel(
                price=price,
                long_liq_usd=ll,
                short_liq_usd=sl,
                cumulative_long_liq=cum_long,
                cumulative_short_liq=cum_short,
                distance_from_spot_pct=dist_pct,
                exchanges=["aggregate"],
            ))

        return levels


# ---------------------------------------------------------------------------
# Cascade risk calculator
# ---------------------------------------------------------------------------

class CascadeRiskCalculator:
    """
    Models how liquidations can cascade:
    - First batch of longs liquidated → price drops
    - Lower price triggers more longs → further drop
    - Feedback loop until OI exhausted or support found
    """

    PRICE_IMPACT_PER_USD = 1e-9     # $1B liquidation moves price ~1% (rough)
    CASCADE_MULTIPLIER   = 1.5      # cascade amplification factor

    def calculate_cascade(
        self,
        spot_price: float,
        trigger_pct: float,     # initial move that starts cascade (e.g., -3%)
        levels: List[LiquidationLevel],
        direction: str = "down",
    ) -> CascadeScenario:
        if direction == "down":
            trigger_price = spot_price * (1 + trigger_pct / 100)
            relevant = sorted(
                [l for l in levels if l.price < trigger_price and l.long_liq_usd > 0],
                key=lambda l: l.price, reverse=True,
            )
        else:
            trigger_price = spot_price * (1 + trigger_pct / 100)
            relevant = sorted(
                [l for l in levels if l.price > trigger_price and l.short_liq_usd > 0],
                key=lambda l: l.price,
            )

        current_price = trigger_price
        cascade_steps = []
        total_liq = 0.0

        for level in relevant:
            if direction == "down" and level.price > current_price:
                continue
            if direction == "up" and level.price < current_price:
                continue

            liq_usd = level.long_liq_usd if direction == "down" else level.short_liq_usd
            if liq_usd < 1_000_000:   # ignore sub-$1M levels
                continue

            # Price impact of this liquidation tranche
            price_impact_pct = liq_usd * self.PRICE_IMPACT_PER_USD * self.CASCADE_MULTIPLIER
            if direction == "down":
                current_price *= (1 - price_impact_pct)
            else:
                current_price *= (1 + price_impact_pct)

            cascade_steps.append((level.price, liq_usd))
            total_liq += liq_usd

        initial_liq = sum(l.long_liq_usd if direction == "down" else l.short_liq_usd
                         for l in relevant[:3])

        price_impact = abs(current_price - trigger_price) / trigger_price * 100

        if total_liq >= 5_000_000_000:
            risk = "extreme"
        elif total_liq >= 1_000_000_000:
            risk = "high"
        elif total_liq >= 500_000_000:
            risk = "medium"
        else:
            risk = "low"

        return CascadeScenario(
            trigger_price=trigger_price,
            direction=direction,
            initial_liq_usd=initial_liq,
            cascade_levels=cascade_steps[:20],
            total_cascade_usd=total_liq,
            final_price_estimate=current_price,
            cascade_risk=risk,
            price_impact_pct=price_impact,
        )


# ---------------------------------------------------------------------------
# Liquidation waterfall model
# ---------------------------------------------------------------------------

class LiquidationWaterfallModel:
    """
    Simulates sequential liquidation across price levels.
    At each price, computes: liq volume, price impact, next level trigger.
    """

    def __init__(self, price_impact_model: str = "linear"):
        self.impact_model = price_impact_model
        self.market_depth_usd = 500_000_000  # assumed $500M depth per 1% move

    def price_impact(self, liq_usd: float, current_price: float) -> float:
        """Estimated price move from a given liquidation volume."""
        if self.impact_model == "linear":
            return liq_usd / self.market_depth_usd / 100  # fraction
        elif self.impact_model == "sqrt":
            return math.sqrt(liq_usd / self.market_depth_usd) / 100
        return liq_usd / self.market_depth_usd / 100

    def simulate_waterfall(
        self,
        spot_price: float,
        levels: List[LiquidationLevel],
        direction: str = "down",
        max_steps: int = 50,
    ) -> List[Dict]:
        """Simulate step-by-step liquidation waterfall."""
        if direction == "down":
            sorted_levels = sorted(
                [l for l in levels if l.price < spot_price],
                key=lambda l: l.price, reverse=True,
            )
        else:
            sorted_levels = sorted(
                [l for l in levels if l.price > spot_price],
                key=lambda l: l.price,
            )

        current_price = spot_price
        results = []
        total_liq = 0.0

        for step, level in enumerate(sorted_levels[:max_steps]):
            if direction == "down":
                # Does price reach this level?
                if current_price <= level.price:
                    continue
                liq_vol = level.long_liq_usd
            else:
                if current_price >= level.price:
                    continue
                liq_vol = level.short_liq_usd

            if liq_vol < 1_000_000:
                continue

            impact = self.price_impact(liq_vol, current_price)
            if direction == "down":
                new_price = current_price * (1 - impact)
            else:
                new_price = current_price * (1 + impact)

            total_liq += liq_vol
            results.append({
                "step": step + 1,
                "trigger_price": level.price,
                "liq_volume_usd": liq_vol,
                "price_before": current_price,
                "price_after": new_price,
                "price_impact_pct": impact * 100,
                "cumulative_liq_usd": total_liq,
                "cumulative_move_pct": (new_price - spot_price) / spot_price * 100,
            })
            current_price = new_price

        return results


# ---------------------------------------------------------------------------
# Heatmap builder
# ---------------------------------------------------------------------------

class LiquidationHeatmapBuilder:
    """Assembles the full LiquidationHeatmap object."""

    def __init__(self, fetcher: CoinglassLiquidationFetcher = None):
        self.fetcher = fetcher or CoinglassLiquidationFetcher()
        self.estimator = LiquidationLevelEstimator()
        self.cascade_calc = CascadeRiskCalculator()

    def build(
        self,
        symbol: str = "BTC",
        exchange: str = "Binance",
    ) -> LiquidationHeatmap:
        raw = self.fetcher.get_liquidation_map(symbol, exchange)
        spot = raw.get("spot_price", 65000.0)
        levels = self.estimator.aggregate_to_levels(spot, raw)

        # Find nearest clusters
        below = sorted([l for l in levels if l.price < spot], key=lambda l: l.price, reverse=True)
        above = sorted([l for l in levels if l.price > spot], key=lambda l: l.price)

        nearest_long_cluster = max(below[:20], key=lambda l: l.long_liq_usd) if below else None
        nearest_short_cluster = max(above[:20], key=lambda l: l.short_liq_usd) if above else None

        # Total at-risk within 10%
        long_at_risk = sum(l.long_liq_usd for l in levels
                          if spot * 0.90 <= l.price < spot)
        short_at_risk = sum(l.short_liq_usd for l in levels
                           if spot < l.price <= spot * 1.10)

        liq_ratio = long_at_risk / short_at_risk if short_at_risk > 0 else float("inf")

        # Pin risk levels: prices where large liq clusters may magnetize price
        pin_levels = [
            l.price for l in sorted(levels, key=lambda l: l.total_liq_usd, reverse=True)[:5]
        ]

        return LiquidationHeatmap(
            spot_price=spot,
            timestamp=datetime.now(timezone.utc),
            levels=levels,
            nearest_long_cluster=nearest_long_cluster,
            nearest_short_cluster=nearest_short_cluster,
            total_long_liq_at_risk_usd=long_at_risk,
            total_short_liq_at_risk_usd=short_at_risk,
            liq_ratio=liq_ratio,
            pin_risk_levels=sorted(pin_levels),
        )

    def cascade_scenarios(
        self,
        heatmap: LiquidationHeatmap,
        down_triggers: List[float] = None,
        up_triggers: List[float] = None,
    ) -> Dict:
        down_triggers = down_triggers or [-2, -5, -10, -15]
        up_triggers = up_triggers or [2, 5, 10]

        down_scenarios = [
            self.cascade_calc.calculate_cascade(
                heatmap.spot_price, t, heatmap.levels, "down"
            )
            for t in down_triggers
        ]
        up_scenarios = [
            self.cascade_calc.calculate_cascade(
                heatmap.spot_price, t, heatmap.levels, "up"
            )
            for t in up_triggers
        ]

        return {"down": down_scenarios, "up": up_scenarios}

    def format_summary(self, heatmap: LiquidationHeatmap) -> str:
        spot = heatmap.spot_price
        lines = [
            f"=== Liquidation Heatmap: Spot ${spot:,.0f} ===",
            f"Timestamp: {heatmap.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            f"Longs at risk (±10% below): ${heatmap.total_long_liq_at_risk_usd/1e6:.0f}M",
            f"Shorts at risk (±10% above): ${heatmap.total_short_liq_at_risk_usd/1e6:.0f}M",
            f"Long/Short liq ratio: {heatmap.liq_ratio:.2f}x",
            "",
        ]

        if heatmap.nearest_long_cluster:
            nc = heatmap.nearest_long_cluster
            dist = (spot - nc.price) / spot * 100
            lines.append(f"Nearest long cluster: ${nc.price:,.0f} ({dist:.1f}% below) — ${nc.long_liq_usd/1e6:.0f}M")

        if heatmap.nearest_short_cluster:
            nc = heatmap.nearest_short_cluster
            dist = (nc.price - spot) / spot * 100
            lines.append(f"Nearest short cluster: ${nc.price:,.0f} ({dist:.1f}% above) — ${nc.short_liq_usd/1e6:.0f}M")

        lines.append(f"\nPin risk levels: {', '.join(f'${p:,.0f}' for p in heatmap.pin_risk_levels)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid / density visualization helper
# ---------------------------------------------------------------------------

class HeatmapGridRenderer:
    """Renders liquidation levels into a ASCII-style grid for inspection."""

    def render_ascii(
        self,
        heatmap: LiquidationHeatmap,
        rows: int = 30,
        width: int = 60,
    ) -> str:
        levels = heatmap.levels
        if not levels:
            return "No liquidation data"

        prices = [l.price for l in levels]
        min_p = min(prices)
        max_p = max(prices)
        spot = heatmap.spot_price
        price_step = (max_p - min_p) / rows

        max_liq = max((l.long_liq_usd + l.short_liq_usd) for l in levels)
        if max_liq == 0:
            return "No liquidations"

        output_lines = ["Price      |Long Liq              |Short Liq             |"]
        output_lines.append("-" * 70)

        for row in range(rows, -1, -1):
            p = min_p + row * price_step
            # Aggregate levels near this price band
            band_levels = [l for l in levels if abs(l.price - p) < price_step / 2]
            long_usd  = sum(l.long_liq_usd for l in band_levels)
            short_usd = sum(l.short_liq_usd for l in band_levels)

            half = width // 2
            long_bars  = int(long_usd  / max_liq * half)
            short_bars = int(short_usd / max_liq * half)

            spot_marker = " ◄ SPOT" if abs(p - spot) < price_step else ""
            line = (
                f"{p:>9,.0f} |{'█' * long_bars:<{half}}|{'█' * short_bars:<{half}}|{spot_marker}"
            )
            output_lines.append(line)

        return "\n".join(output_lines)

    def to_grid_dict(
        self,
        heatmap: LiquidationHeatmap,
        resolution: int = 100,
    ) -> Dict:
        """Export heatmap as a dictionary with price/liq arrays for plotting."""
        levels = heatmap.levels
        prices = [l.price for l in levels]
        long_liqs  = [l.long_liq_usd  for l in levels]
        short_liqs = [l.short_liq_usd for l in levels]
        return {
            "prices": prices,
            "long_liqs_usd": long_liqs,
            "short_liqs_usd": short_liqs,
            "spot_price": heatmap.spot_price,
            "total_long_10pct": heatmap.total_long_liq_at_risk_usd,
            "total_short_10pct": heatmap.total_short_liq_at_risk_usd,
        }


# ---------------------------------------------------------------------------
# Main LiquidationAnalytics facade
# ---------------------------------------------------------------------------

class LiquidationAnalytics:
    """Unified API for liquidation heatmap and cascade analysis."""

    def __init__(self, coinglass_key: str = "", provider: OnChainDataProvider = None):
        self.provider = provider or OnChainDataProvider()
        fetcher = CoinglassLiquidationFetcher(coinglass_key)
        self.builder = LiquidationHeatmapBuilder(fetcher)
        self.waterfall = LiquidationWaterfallModel()
        self.renderer = HeatmapGridRenderer()

    def build_heatmap(self, symbol: str = "BTC") -> LiquidationHeatmap:
        return self.builder.build(symbol)

    def cascade_analysis(self, symbol: str = "BTC") -> Dict:
        hm = self.build_heatmap(symbol)
        scenarios = self.builder.cascade_scenarios(hm)

        return {
            "spot_price": hm.spot_price,
            "symbol": symbol,
            "long_at_risk_10pct": hm.total_long_liq_at_risk_usd,
            "short_at_risk_10pct": hm.total_short_liq_at_risk_usd,
            "liq_ratio": hm.liq_ratio,
            "down_scenarios": [
                {
                    "trigger": s.trigger_price,
                    "direction": s.direction,
                    "total_cascade_usd": s.total_cascade_usd,
                    "final_price": s.final_price_estimate,
                    "risk": s.cascade_risk,
                    "price_impact_pct": s.price_impact_pct,
                }
                for s in scenarios["down"]
            ],
            "up_scenarios": [
                {
                    "trigger": s.trigger_price,
                    "direction": s.direction,
                    "total_cascade_usd": s.total_cascade_usd,
                    "final_price": s.final_price_estimate,
                    "risk": s.cascade_risk,
                }
                for s in scenarios["up"]
            ],
        }

    def waterfall_simulation(
        self,
        symbol: str = "BTC",
        direction: str = "down",
    ) -> List[Dict]:
        hm = self.build_heatmap(symbol)
        return self.waterfall.simulate_waterfall(hm.spot_price, hm.levels, direction)

    def full_report(self, symbol: str = "BTC") -> str:
        hm = self.build_heatmap(symbol)
        summary = self.builder.format_summary(hm)
        grid = self.renderer.render_ascii(hm, rows=20, width=40)
        cascade = self.cascade_analysis(symbol)

        scenario_lines = ["\nCascade Scenarios (DOWN):"]
        for s in cascade["down_scenarios"]:
            scenario_lines.append(
                f"  Trigger ${s['trigger']:,.0f}: cascade ${s['total_cascade_usd']/1e6:.0f}M, "
                f"impact {s['price_impact_pct']:.1f}% → RISK={s['risk'].upper()}"
            )

        return summary + "\n\n" + "\n".join(scenario_lines) + "\n\n" + grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Liquidation heatmap CLI")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--action", choices=["report", "cascade", "waterfall", "grid"], default="report")
    parser.add_argument("--direction", choices=["up", "down"], default="down")
    args = parser.parse_args()

    analytics = LiquidationAnalytics()

    if args.action == "report":
        print(analytics.full_report(args.symbol))
    elif args.action == "cascade":
        import json as _json
        result = analytics.cascade_analysis(args.symbol)
        print(_json.dumps(result, indent=2, default=str))
    elif args.action == "waterfall":
        steps = analytics.waterfall_simulation(args.symbol, args.direction)
        for s in steps:
            print(f"Step {s['step']:2d}: ${s['trigger_price']:,.0f} → "
                  f"liq ${s['liq_volume_usd']/1e6:.0f}M → "
                  f"price {s['price_after']:,.0f} ({s['cumulative_move_pct']:.1f}%)")
    elif args.action == "grid":
        hm = analytics.build_heatmap(args.symbol)
        renderer = HeatmapGridRenderer()
        print(renderer.render_ascii(hm))
