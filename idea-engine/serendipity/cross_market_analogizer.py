"""
Cross-market analogizer — generates hypotheses by finding structural
analogies between different markets, assets, and historical periods.

Examples:
  - 2020 DeFi summer → current altcoin pattern?
  - Bitcoin post-halving cycles
  - Traditional equity momentum patterns in crypto
  - FX carry trade dynamics in funding rates
  - Commodity super-cycle patterns
  - Credit cycle analogs (IG spreads vs crypto vol)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Optional

from ..hypothesis.types import Hypothesis, HypothesisType, HypothesisStatus


@dataclass
class CrossMarketAnalogizer:
    """Find structural analogs across markets and historical periods."""

    seed: int = 123

    ANALOGY_TEMPLATES = [
        "btc_halving_cycle",
        "defi_summer_pattern",
        "equity_momentum_in_crypto",
        "fx_carry_to_funding",
        "commodity_supercycle",
        "post_crash_recovery",
        "altcoin_dominance_cycle",
        "institutional_adoption_phase",
        "regulatory_shock_pattern",
        "liquidity_squeeze_analog",
    ]

    def generate(self, context: dict[str, Any], n: int = 5) -> list[Hypothesis]:
        rng = random.Random(self.seed + hash(str(sorted(context.items()))))
        selected = rng.sample(self.ANALOGY_TEMPLATES, min(n, len(self.ANALOGY_TEMPLATES)))
        return [h for t in selected if (h := self._from_template(t, context)) is not None]

    def _from_template(self, template: str, ctx: dict) -> Optional[Hypothesis]:
        symbol = ctx.get("symbol", "BTC")
        tf = ctx.get("timeframe", "1h")

        if template == "btc_halving_cycle":
            return self._btc_halving_cycle(symbol, ctx)
        elif template == "defi_summer_pattern":
            return self._defi_summer(symbol, ctx)
        elif template == "equity_momentum_in_crypto":
            return self._equity_momentum(symbol, tf)
        elif template == "fx_carry_to_funding":
            return self._fx_carry_funding(symbol, ctx)
        elif template == "commodity_supercycle":
            return self._commodity_supercycle(symbol, ctx)
        elif template == "post_crash_recovery":
            return self._post_crash_recovery(symbol, ctx)
        elif template == "altcoin_dominance_cycle":
            return self._altcoin_cycle(symbol, ctx)
        elif template == "institutional_adoption_phase":
            return self._institutional_phase(symbol, ctx)
        elif template == "regulatory_shock_pattern":
            return self._regulatory_shock(symbol)
        elif template == "liquidity_squeeze_analog":
            return self._liquidity_squeeze(symbol, ctx)
        return None

    def _btc_halving_cycle(self, symbol: str, ctx: dict) -> Hypothesis:
        days_since_halving = ctx.get("days_since_halving", 180)
        return Hypothesis(
            id=f"halving_cycle_{symbol}",
            name="BTC Halving Cycle Phase Allocation",
            description=(
                f"BTC historically follows a 4-year halving cycle with predictable phases: "
                f"post-halving accumulation (0-6m), bull run (6-18m), blow-off top (18-24m), bear market. "
                f"Currently {days_since_halving}d post-halving. "
                f"Adjust position sizing according to cycle phase risk/reward profile."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": "BTC",
                "days_since_halving": int(days_since_halving),
                "accumulation_size_mult": 1.2,
                "bull_run_size_mult": 1.0,
                "blow_off_size_mult": 0.5,
                "bear_market_size_mult": 0.3,
                "halving_dates": ["2024-04-20"],
                "cycle_phase_boundaries_days": [0, 180, 540, 720, 1460],
            },
            expected_impact=0.05,
            confidence=0.62,
            source_pattern=None,
            tags=["halving_cycle", "btc", "macro_cycle", "cross_market"],
        )

    def _defi_summer_pattern(self, symbol: str, ctx: dict) -> Hypothesis:
        tvl_growth = ctx.get("defi_tvl_growth_30d", 0.0)
        return Hypothesis(
            id=f"defi_summer_{symbol}",
            name="DeFi TVL Growth → DeFi Token Momentum",
            description=(
                f"Analogy to 2020 DeFi Summer: rapid TVL growth (>${tvl_growth:.0f}B 30d increase) "
                f"historically leads DeFi token price action by 2-3 weeks. "
                f"For DeFi tokens (AAVE, UNI, CRV, SUSHI): enter on TVL acceleration signal."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "defi_symbols": ["AAVE", "UNI", "CRV", "SUSHI"],
                "tvl_growth_threshold_pct": 20,
                "tvl_growth_window_days": 30,
                "entry_lag_days": 7,
                "target_hold_days": 21,
                "require_bh_confirmation": True,
            },
            expected_impact=0.08,
            confidence=0.57,
            source_pattern=None,
            tags=["defi", "tvl", "cross_market", "analogy"],
        )

    def _equity_momentum_in_crypto(self, symbol: str, tf: str) -> Hypothesis:
        return Hypothesis(
            id=f"equity_mom_crypto_{symbol}",
            name="Equity 12-1 Momentum Applied to Crypto",
            description=(
                "Classic equity momentum (12-month return minus 1-month, Jegadeesh-Titman) "
                "applied to crypto top-20 monthly. Analogous to Carhart momentum factor. "
                "Long top-3 by 11M-skip-1M return, short bottom-3. "
                "Rebalance monthly. Historically +20-40% annualized in equity before crowding."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "universe": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT", "MATIC"],
                "lookback_months": 12,
                "skip_months": 1,
                "n_long": 3,
                "n_short": 3,
                "rebalance_frequency": "monthly",
                "require_liquidity_filter": True,
                "min_daily_volume_usd": 10_000_000,
            },
            expected_impact=0.15,
            confidence=0.60,
            source_pattern=None,
            tags=["equity_analogy", "momentum", "cross_sectional", "carhart"],
        )

    def _fx_carry_funding(self, symbol: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"fx_carry_funding_{symbol}",
            name="FX Carry Trade Analog: Crypto Funding Carry",
            description=(
                "FX carry trade: borrow in low-rate currency, invest in high-rate. "
                "Crypto analog: on perpetuals, when funding is negative (shorts pay longs), "
                "hold long spot + short perpetual = earn funding + market-neutral exposure. "
                "Implement when funding < -0.02% per 8h on top-5 assets."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "assets": ["BTC", "ETH", "SOL", "BNB", "XRP"],
                "funding_carry_threshold": -0.0002,
                "spot_long_size": 1.0,
                "perp_short_size": 1.0,
                "max_basis_risk": 0.005,
                "unwind_if_funding_positive": 0.0001,
                "expected_annualized_carry": 0.15,
            },
            expected_impact=0.12,
            confidence=0.68,
            source_pattern=None,
            tags=["fx_analogy", "carry_trade", "funding_rate", "market_neutral"],
        )

    def _commodity_supercycle(self, symbol: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"commodity_supercycle_{symbol}",
            name="Commodity Supercycle Analog: Hash Rate as Supply Signal",
            description=(
                "Commodity supercycles driven by supply constraints and demand surges. "
                "BTC analog: hash rate growth rate as supply signal (miner cost basis rising). "
                "When hash rate growth decelerates after rapid expansion → miners near breakeven "
                "→ selling pressure peaks → price historically bottoms 2-4 weeks later."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": "BTC",
                "hashrate_growth_decel_threshold": -0.1,
                "hashrate_lookback_days": 90,
                "entry_weeks_after_decel": 3,
                "miner_revenue_ratio_threshold": 1.2,
            },
            expected_impact=0.07,
            confidence=0.58,
            source_pattern=None,
            tags=["commodity_analogy", "mining", "supply", "hash_rate"],
        )

    def _post_crash_recovery(self, symbol: str, ctx: dict) -> Hypothesis:
        drawdown = ctx.get("current_drawdown", 0.0)
        return Hypothesis(
            id=f"post_crash_{symbol}",
            name="Post-Crash Recovery Pattern Entry",
            description=(
                f"Historical analysis of equity market recoveries (2003, 2009, 2020) shows "
                f"similar V-shape recovery in crypto after -40%+ drawdowns. "
                f"After {drawdown:.0%} drawdown, 80%+ of recovery happens in first 90 days. "
                f"Enter aggressively when price stabilizes 2 weeks after capitulation."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "min_drawdown_trigger": 0.40,
                "stabilization_bars": 20,
                "stabilization_max_vol": 0.05,
                "entry_size_multiplier": 1.5,
                "hold_days": 90,
                "require_volume_confirmation": True,
            },
            expected_impact=0.12,
            confidence=0.63,
            source_pattern=None,
            tags=["crash_recovery", "contrarian", "historical_analog"],
        )

    def _altcoin_cycle(self, symbol: str, ctx: dict) -> Hypothesis:
        btc_dom = ctx.get("btc_dominance", 0.5)
        return Hypothesis(
            id=f"altcoin_cycle_{symbol}",
            name="Altcoin Season Cycle Allocation",
            description=(
                f"Crypto market follows BTC→ETH→Large caps→Mid caps→Small caps rotation. "
                f"Current BTC dominance {btc_dom:.0%}. "
                f"At <45% dominance, rotate to mid/small caps; >55% rotate back to BTC. "
                f"Signal: BTC dominance trend direction + volume flow."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "btc_dominance_altseason_threshold": 0.45,
                "btc_dominance_btcseason_threshold": 0.55,
                "large_cap_symbols": ["ETH", "BNB", "SOL"],
                "mid_cap_symbols": ["AVAX", "LINK", "DOT", "XRP"],
                "small_cap_symbols": ["AAVE", "UNI", "CRV"],
                "rotation_window_days": 7,
            },
            expected_impact=0.04,
            confidence=0.62,
            source_pattern=None,
            tags=["altcoin_cycle", "btc_dominance", "rotation", "sector"],
        )

    def _institutional_phase(self, symbol: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"institutional_{symbol}",
            name="Institutional Accumulation Phase Detection",
            description=(
                "Institutional accumulation patterns (Wyckoff): low-vol sideways after dump, "
                "followed by spring (fake breakdown), then markup. "
                "Signs: declining volume on dips, rising volume on advances, "
                "price range compression for 30+ days after -30% move."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "post_dump_min_pct": 0.30,
                "consolidation_bars": 60,
                "vol_ratio_threshold": 0.7,  # dip volume / rally volume
                "spring_false_break_pct": 0.03,
                "spring_recovery_bars": 5,
                "markup_entry_confirmation": True,
            },
            expected_impact=0.08,
            confidence=0.59,
            source_pattern=None,
            tags=["wyckoff", "institutional", "accumulation", "pattern"],
        )

    def _regulatory_shock(self, symbol: str) -> Hypothesis:
        return Hypothesis(
            id=f"regulatory_shock_{symbol}",
            name="Regulatory Shock Overreaction — Fade the Dump",
            description=(
                "Historical regulatory shocks (China bans, SEC actions) cause initial -20-40% "
                "price dump followed by full recovery within 60-90 days in 75% of cases. "
                "The initial dump is driven by fear and forced selling, not fundamental value change. "
                "Enter on regulatory shock dumps when BTC dominance doesn't simultaneously rise (less systemic)."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "regulatory_dump_threshold": -0.20,
                "dump_window_days": 3,
                "btc_dominance_must_not_rise": True,
                "entry_days_after_dump": 3,
                "target_recovery_pct": 0.15,
                "stop_loss_pct": 0.10,
            },
            expected_impact=0.06,
            confidence=0.60,
            source_pattern=None,
            tags=["regulatory", "overreaction", "contrarian", "event_driven"],
        )

    def _liquidity_squeeze(self, symbol: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"liquidity_squeeze_{symbol}",
            name="Liquidity Squeeze Analog: Reduce Before Year-End",
            description=(
                "Credit market analog: liquidity typically dries up in late December (year-end "
                "balance sheet window dressing) and early September (risk-off seasonality). "
                "In crypto: reduce exposure 2 weeks before Dec 25 and Sep 1. "
                "Historically these windows see elevated realized vol and fat-tail events."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "seasonal_reduce_windows": [
                    {"start_month": 12, "start_day": 10, "end_month": 12, "end_day": 31},
                    {"start_month": 8, "start_day": 25, "end_month": 9, "end_day": 10},
                ],
                "size_multiplier_during_window": 0.5,
                "apply_to_new_entries": True,
                "apply_to_existing": False,
            },
            expected_impact=0.02,
            confidence=0.60,
            source_pattern=None,
            tags=["seasonality", "liquidity", "risk_management", "calendar"],
        )
