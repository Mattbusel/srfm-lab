"""
scenario_library.py - Library of simulation scenarios for the synthetic exchange.

15 market scenarios, ScenarioRunner, and StressTestSuite for comprehensive
strategy evaluation under diverse conditions.
"""

import time
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

class ScenarioCategory(Enum):
    NORMAL = "normal"
    TREND = "trend"
    VOLATILITY = "volatility"
    CRISIS = "crisis"
    MICROSTRUCTURE = "microstructure"
    ADVERSARIAL = "adversarial"


@dataclass
class ExpectedBehavior:
    """What the strategy SHOULD do during this scenario."""
    description: str
    ideal_action: str  # "hold", "buy", "sell", "reduce_exposure", "no_trade"
    ideal_pnl_sign: int  # 1 = positive, -1 = negative OK, 0 = flat acceptable
    max_acceptable_drawdown: float = 0.15
    min_acceptable_sharpe: float = -1.0
    notes: str = ""


@dataclass
class PassCriteria:
    """Quantitative criteria for passing the scenario."""
    max_drawdown: float = 0.20
    min_pnl: Optional[float] = None
    max_loss: Optional[float] = None
    min_sharpe: Optional[float] = None
    max_trades: Optional[int] = None
    must_survive: bool = True  # position must not blow up
    custom_check: Optional[Callable] = None
    custom_check_desc: str = ""

    def evaluate(self, result: "ScenarioResult") -> Tuple[bool, List[str]]:
        """Evaluate whether the result passes all criteria."""
        failures: List[str] = []

        if result.max_dd > self.max_drawdown:
            failures.append(
                f"Drawdown {result.max_dd:.4f} > max {self.max_drawdown:.4f}"
            )

        if self.min_pnl is not None and result.pnl < self.min_pnl:
            failures.append(f"PnL {result.pnl:.2f} < min {self.min_pnl:.2f}")

        if self.max_loss is not None and result.pnl < -abs(self.max_loss):
            failures.append(
                f"Loss {result.pnl:.2f} exceeds max allowed loss {self.max_loss:.2f}"
            )

        if self.min_sharpe is not None and result.sharpe < self.min_sharpe:
            failures.append(
                f"Sharpe {result.sharpe:.3f} < min {self.min_sharpe:.3f}"
            )

        if self.max_trades is not None and result.num_trades > self.max_trades:
            failures.append(
                f"Trades {result.num_trades} > max {self.max_trades}"
            )

        if self.must_survive and result.blew_up:
            failures.append("Strategy blew up (position exceeded limits)")

        if self.custom_check is not None:
            try:
                ok = self.custom_check(result)
                if not ok:
                    failures.append(f"Custom check failed: {self.custom_check_desc}")
            except Exception as exc:
                failures.append(f"Custom check error: {exc}")

        passed = len(failures) == 0
        return passed, failures


@dataclass
class Scenario:
    """A complete simulation scenario definition."""
    name: str
    description: str
    category: ScenarioCategory
    duration_bars: int
    param_overrides: Dict[str, Any]
    event_sequence: List[Dict[str, Any]]  # list of {bar: int, event_type: str, params: dict}
    expected_behavior: ExpectedBehavior
    pass_criteria: PassCriteria
    probability_weight: float = 1.0  # how likely is this scenario in real life
    difficulty: float = 0.5  # 0=easy, 1=extremely hard
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tags:
            self.tags = [self.category.value]


@dataclass
class ScenarioResult:
    """Result of running a strategy through a scenario."""
    scenario_name: str
    strategy_name: str
    pnl: float
    sharpe: float
    max_dd: float
    num_trades: int
    win_rate: float
    profit_factor: float
    blew_up: bool
    passed: bool
    failure_reasons: List[str]
    narrative: str
    equity_curve: List[float]
    bars_processed: int
    execution_time_sec: float
    final_position: float
    total_fees: float
    sortino: float = 0.0
    calmar: float = 0.0
    avg_trade_pnl: float = 0.0
    max_consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "strategy": self.strategy_name,
            "pnl": self.pnl,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_dd": self.max_dd,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "blew_up": self.blew_up,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "narrative": self.narrative,
            "bars_processed": self.bars_processed,
            "execution_time_sec": self.execution_time_sec,
            "final_position": self.final_position,
            "total_fees": self.total_fees,
            "avg_trade_pnl": self.avg_trade_pnl,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


# ---------------------------------------------------------------------------
# 15 Scenario Definitions
# ---------------------------------------------------------------------------

def build_normal_market() -> Scenario:
    """1. Normal trading day: moderate vol, balanced flow, reasonable spreads."""
    return Scenario(
        name="NormalMarket",
        description=(
            "A typical trading day with moderate volatility (~15% annualized), "
            "balanced order flow, and stable spreads. This is the baseline scenario "
            "that every strategy should handle competently. Volume follows a U-shaped "
            "intraday pattern with higher activity at open and close."
        ),
        category=ScenarioCategory.NORMAL,
        duration_bars=500,
        param_overrides={
            "volatility": 0.015,
            "drift": 0.0001,
            "mean_spread_bps": 8.0,
            "num_market_makers": 6,
            "num_noise_traders": 25,
            "num_trend_followers": 5,
            "num_mean_reversion": 5,
            "initial_price": 100.0,
            "volume_profile": "u_shaped",
            "intraday_vol_pattern": [1.3, 1.1, 0.9, 0.8, 0.8, 0.9, 1.0, 1.2],
        },
        event_sequence=[],
        expected_behavior=ExpectedBehavior(
            description="Strategy should trade normally and generate modest returns.",
            ideal_action="trade",
            ideal_pnl_sign=1,
            max_acceptable_drawdown=0.05,
            min_acceptable_sharpe=0.0,
            notes="Baseline scenario. A good strategy should be slightly profitable.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.08,
            max_loss=2000.0,
            min_sharpe=-0.5,
        ),
        probability_weight=5.0,
        difficulty=0.2,
        tags=["normal", "baseline"],
    )


def build_trending_bull() -> Scenario:
    """2. Trending bull market with positive drift and FOMO dynamics."""
    return Scenario(
        name="TrendingBull",
        description=(
            "A strong uptrend with positive drift (+20% annualized), increasing volume "
            "as momentum attracts trend followers, occasional dips that get bought. "
            "Spread narrows during the trend as market makers compete. Late-stage FOMO "
            "dynamics cause acceleration."
        ),
        category=ScenarioCategory.TREND,
        duration_bars=600,
        param_overrides={
            "volatility": 0.018,
            "drift": 0.0008,
            "mean_spread_bps": 6.0,
            "num_trend_followers": 12,
            "num_noise_traders": 30,
            "initial_price": 100.0,
            "fomo_threshold": 0.05,
            "fomo_multiplier": 1.5,
            "dip_buy_probability": 0.7,
            "volume_growth_rate": 0.002,
        },
        event_sequence=[
            {"bar": 200, "event_type": "volume_surge", "params": {"multiplier": 1.5, "duration": 50}},
            {"bar": 400, "event_type": "fomo_wave", "params": {"intensity": 2.0, "duration": 100}},
        ],
        expected_behavior=ExpectedBehavior(
            description="Trend-following strategies should capture upside. Mean-reversion should be cautious.",
            ideal_action="buy",
            ideal_pnl_sign=1,
            max_acceptable_drawdown=0.08,
            min_acceptable_sharpe=0.5,
            notes="A strategy that fails to go long in this scenario is missing obvious alpha.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.12,
            min_pnl=-500.0,  # should not lose money in a bull market
            min_sharpe=-0.3,
        ),
        probability_weight=2.0,
        difficulty=0.3,
        tags=["trend", "bull", "fomo"],
    )


def build_trending_bear() -> Scenario:
    """3. Trending bear with panic selling and spread widening."""
    return Scenario(
        name="TrendingBear",
        description=(
            "A sustained downtrend with negative drift (-25% annualized), panic selling "
            "episodes, widening spreads as market makers pull back, declining liquidity. "
            "Volume spikes during selloffs. Brief relief rallies get sold into."
        ),
        category=ScenarioCategory.TREND,
        duration_bars=600,
        param_overrides={
            "volatility": 0.025,
            "drift": -0.001,
            "mean_spread_bps": 15.0,
            "num_market_makers": 3,
            "num_noise_traders": 20,
            "num_trend_followers": 10,
            "initial_price": 100.0,
            "panic_threshold": -0.03,
            "panic_volume_mult": 3.0,
            "spread_widen_on_sell": 2.0,
            "relief_rally_prob": 0.15,
        },
        event_sequence=[
            {"bar": 150, "event_type": "panic_sell", "params": {"intensity": 2.0, "duration": 30}},
            {"bar": 300, "event_type": "spread_widening", "params": {"factor": 2.5, "duration": 60}},
            {"bar": 450, "event_type": "panic_sell", "params": {"intensity": 3.0, "duration": 20}},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should either go short or stay flat. Buying dips is dangerous.",
            ideal_action="sell",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.10,
            notes="Key test: does the strategy fight the trend or adapt?",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.15,
            max_loss=5000.0,
        ),
        probability_weight=2.0,
        difficulty=0.5,
        tags=["trend", "bear", "panic"],
    )


def build_high_volatility() -> Scenario:
    """4. High volatility: doubled vol, frequent gaps, thin books."""
    return Scenario(
        name="HighVolatility",
        description=(
            "Extremely volatile market with 2-3x normal volatility, frequent price gaps "
            "between bars, thin order books with wide spreads. Market makers quote wider "
            "and in smaller size. Stops get run frequently."
        ),
        category=ScenarioCategory.VOLATILITY,
        duration_bars=400,
        param_overrides={
            "volatility": 0.045,
            "drift": 0.0,
            "mean_spread_bps": 25.0,
            "num_market_makers": 3,
            "book_depth_levels": 10,
            "initial_price": 100.0,
            "gap_probability": 0.05,
            "gap_size_mean": 0.02,
            "gap_size_std": 0.01,
            "mm_quote_size_mult": 0.4,
        },
        event_sequence=[
            {"bar": 100, "event_type": "volatility_spike", "params": {"multiplier": 2.0, "duration": 50}},
            {"bar": 250, "event_type": "book_thinning", "params": {"remove_pct": 60, "duration": 30}},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should reduce position size or stand aside. Survival matters most.",
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.12,
            notes="Wide spreads eat into P&L. Aggressive strategies will suffer.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.18,
            max_loss=8000.0,
            must_survive=True,
        ),
        probability_weight=1.5,
        difficulty=0.6,
        tags=["volatility", "gaps", "thin_book"],
    )


def build_flash_crash() -> Scenario:
    """5. Flash crash: sudden 10%+ drop in <1 min, recovery in 5 min."""
    return Scenario(
        name="FlashCrash",
        description=(
            "Market operating normally then a sudden cascade: a large sell order triggers "
            "stop losses, market makers pull quotes, price drops 10-15% in under 60 bars. "
            "Then buy-side liquidity returns and price recovers 70-90% of the move in 300 bars. "
            "Spreads blow out to 100+ bps during the crash."
        ),
        category=ScenarioCategory.CRISIS,
        duration_bars=500,
        param_overrides={
            "volatility": 0.015,
            "drift": 0.0,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
            "circuit_breaker_pct": 12.0,
        },
        event_sequence=[
            {"bar": 150, "event_type": "flash_crash", "params": {
                "magnitude_pct": 12.0,
                "crash_duration_bars": 30,
                "recovery_duration_bars": 200,
                "recovery_pct": 0.80,
                "spread_blowout_bps": 150.0,
                "mm_withdrawal_pct": 90.0,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description=(
                "Strategy should either (a) not be caught long, (b) cut losses quickly, "
                "or (c) recognize the crash and buy the recovery."
            ),
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.12,
            notes="The defining test: how does the strategy handle a sudden crash?",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.20,
            max_loss=10000.0,
            must_survive=True,
        ),
        probability_weight=0.5,
        difficulty=0.8,
        tags=["crisis", "flash_crash", "tail_risk"],
    )


def build_liquidity_drain() -> Scenario:
    """6. Gradual liquidity drain: market makers withdraw, spreads explode."""
    return Scenario(
        name="LiquidityDrain",
        description=(
            "Market makers gradually reduce quoting over 200 bars. Spread widens from "
            "8bps to 100+bps. Depth shrinks to near zero. Small orders cause large price "
            "impact. The price drifts down as sell pressure has outsized impact."
        ),
        category=ScenarioCategory.MICROSTRUCTURE,
        duration_bars=500,
        param_overrides={
            "volatility": 0.018,
            "drift": -0.0002,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
        },
        event_sequence=[
            {"bar": 50, "event_type": "liquidity_drain", "params": {
                "drain_rate_per_bar": 0.005,
                "duration_bars": 200,
                "min_remaining_mm": 1,
                "spread_growth_factor": 1.02,
            }},
            {"bar": 300, "event_type": "liquidity_return", "params": {
                "return_rate_per_bar": 0.01,
                "duration_bars": 150,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should detect widening spreads and reduce trading frequency.",
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.10,
            notes="Strategies that ignore spreads will bleed from crossing wide markets.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.15,
            max_loss=6000.0,
            max_trades=100,  # should not be trading aggressively
        ),
        probability_weight=1.0,
        difficulty=0.6,
        tags=["microstructure", "liquidity", "spread"],
    )


def build_short_squeeze() -> Scenario:
    """7. Short squeeze: cascading short covering, exponential price rise."""
    return Scenario(
        name="ShortSqueeze",
        description=(
            "Heavily shorted asset gets a positive catalyst. Short covering starts slowly "
            "then accelerates as margin calls trigger forced buys. Price rises 30-50% in "
            "200 bars with parabolic acceleration. Volume explodes to 10x normal. "
            "Then violent reversal as longs take profit."
        ),
        category=ScenarioCategory.CRISIS,
        duration_bars=500,
        param_overrides={
            "volatility": 0.020,
            "drift": 0.0,
            "mean_spread_bps": 10.0,
            "initial_price": 50.0,
            "short_interest_pct": 40.0,
            "margin_call_threshold": 0.15,
        },
        event_sequence=[
            {"bar": 100, "event_type": "short_squeeze", "params": {
                "initial_catalyst_pct": 5.0,
                "covering_acceleration": 1.05,
                "duration_bars": 200,
                "volume_multiplier": 10.0,
                "peak_price_mult": 1.4,
            }},
            {"bar": 350, "event_type": "profit_taking", "params": {
                "sell_pressure_mult": 3.0,
                "duration_bars": 100,
                "retracement_pct": 0.5,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should either ride the squeeze or stay flat. Being short is fatal.",
            ideal_action="buy",
            ideal_pnl_sign=1,
            max_acceptable_drawdown=0.15,
            notes="A strategy caught short can lose 40%+ very quickly.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.25,
            max_loss=15000.0,
            must_survive=True,
        ),
        probability_weight=0.3,
        difficulty=0.7,
        tags=["crisis", "short_squeeze", "gamma"],
    )


def build_circuit_breaker_test() -> Scenario:
    """8. Price hits circuit breaker, halt, resume, second halt."""
    return Scenario(
        name="CircuitBreakerTest",
        description=(
            "Price drops rapidly, triggering a 5-minute halt at -7%. Reopens with a gap, "
            "continues falling, triggers a second halt at -13%. Reopens again. The strategy "
            "must handle being unable to trade during halts and the gap risk on reopening."
        ),
        category=ScenarioCategory.CRISIS,
        duration_bars=500,
        param_overrides={
            "volatility": 0.030,
            "drift": -0.0005,
            "mean_spread_bps": 15.0,
            "initial_price": 100.0,
            "circuit_breaker_levels": [-7.0, -13.0, -20.0],
            "halt_duration_bars": 30,
            "reopen_gap_bps": 200.0,
        },
        event_sequence=[
            {"bar": 100, "event_type": "rapid_selloff", "params": {
                "target_pct": -7.5,
                "duration_bars": 40,
            }},
            {"bar": 140, "event_type": "circuit_breaker_halt", "params": {
                "halt_duration_bars": 30,
                "level": 1,
            }},
            {"bar": 170, "event_type": "reopen_with_gap", "params": {
                "gap_pct": -2.0,
            }},
            {"bar": 200, "event_type": "rapid_selloff", "params": {
                "target_pct": -6.0,
                "duration_bars": 30,
            }},
            {"bar": 230, "event_type": "circuit_breaker_halt", "params": {
                "halt_duration_bars": 30,
                "level": 2,
            }},
            {"bar": 260, "event_type": "reopen_with_gap", "params": {
                "gap_pct": -1.5,
            }},
            {"bar": 300, "event_type": "recovery", "params": {
                "recovery_pct": 0.4,
                "duration_bars": 200,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy must handle halts gracefully and manage gap risk on reopening.",
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.15,
            notes="Key: does the strategy panic on reopen or remain disciplined?",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.22,
            max_loss=12000.0,
            must_survive=True,
        ),
        probability_weight=0.5,
        difficulty=0.7,
        tags=["crisis", "circuit_breaker", "halt"],
    )


def build_fomc_meeting() -> Scenario:
    """9. FOMC meeting: vol compression before, spike after, directional move."""
    return Scenario(
        name="FOMCMeeting",
        description=(
            "Volatility compresses as market waits for FOMC decision. Spreads narrow. "
            "At bar 200 (the announcement), vol spikes 3x, a directional move of +/-2% "
            "occurs in 10 bars. Then vol remains elevated for 100 bars as market digests. "
            "The direction is randomly up or down."
        ),
        category=ScenarioCategory.VOLATILITY,
        duration_bars=500,
        param_overrides={
            "volatility": 0.010,
            "drift": 0.0,
            "mean_spread_bps": 5.0,
            "initial_price": 100.0,
            "pre_event_vol_compression": 0.5,
            "post_event_vol_expansion": 3.0,
        },
        event_sequence=[
            {"bar": 100, "event_type": "vol_compression", "params": {
                "target_vol_mult": 0.4,
                "duration_bars": 100,
            }},
            {"bar": 200, "event_type": "fomc_announcement", "params": {
                "direction": "random",
                "magnitude_pct": 2.0,
                "duration_bars": 10,
                "vol_spike_mult": 3.5,
            }},
            {"bar": 210, "event_type": "post_event_vol", "params": {
                "vol_mult": 2.0,
                "decay_rate": 0.005,
                "duration_bars": 200,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description=(
                "Strategy should reduce exposure pre-FOMC due to low vol/edge, "
                "then potentially trade the post-announcement move."
            ),
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.08,
            notes="The worst outcome is being max long/short right before the announcement.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.12,
            max_loss=5000.0,
        ),
        probability_weight=1.5,
        difficulty=0.5,
        tags=["event", "fomc", "volatility"],
    )


def build_earnings_announcement() -> Scenario:
    """10. Earnings: gap open, increased vol for 30 min."""
    return Scenario(
        name="EarningsAnnouncement",
        description=(
            "Stock gaps up or down 5-8% at the open (bar 50) on earnings. "
            "Volatility is 3x normal for the first 100 bars after the gap. "
            "Volume is 5x normal. The gap may continue or reverse depending "
            "on whether the move was 'warranted'."
        ),
        category=ScenarioCategory.VOLATILITY,
        duration_bars=400,
        param_overrides={
            "volatility": 0.012,
            "drift": 0.0,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
        },
        event_sequence=[
            {"bar": 50, "event_type": "earnings_gap", "params": {
                "gap_pct": 6.0,
                "direction": "random",
                "continuation_probability": 0.6,
                "vol_mult": 3.0,
                "volume_mult": 5.0,
                "elevated_vol_duration": 100,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should handle the gap without blowing up. Fading the gap is risky.",
            ideal_action="hold",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.10,
            notes="Gap risk is the main concern. Overnight positions get gapped.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.15,
            max_loss=7000.0,
        ),
        probability_weight=1.5,
        difficulty=0.5,
        tags=["event", "earnings", "gap"],
    )


def build_crypto_liquidation_cascade() -> Scenario:
    """11. Crypto liquidation cascade: cascading long liquidations."""
    return Scenario(
        name="CryptoLiquidationCascade",
        description=(
            "Price drops 3%, triggering leveraged long liquidations. Forced selling "
            "pushes price down further, triggering more liquidations. A waterfall of "
            "forced selling drives price down 20-30% in 100 bars. Spreads blow out. "
            "Then a violent bounce as bargain hunters step in."
        ),
        category=ScenarioCategory.CRISIS,
        duration_bars=500,
        param_overrides={
            "volatility": 0.035,
            "drift": -0.0002,
            "mean_spread_bps": 20.0,
            "initial_price": 40000.0,
            "leverage_ratio_long": 10.0,
            "liquidation_threshold": 0.10,
            "cascade_acceleration": 1.08,
        },
        event_sequence=[
            {"bar": 100, "event_type": "initial_drop", "params": {
                "magnitude_pct": 3.0,
                "duration_bars": 10,
            }},
            {"bar": 110, "event_type": "liquidation_cascade", "params": {
                "waves": 5,
                "wave_interval_bars": 15,
                "wave_magnitude_pct": [4.0, 5.0, 6.0, 4.0, 3.0],
                "spread_blowout_mult": [2.0, 3.0, 5.0, 4.0, 3.0],
            }},
            {"bar": 250, "event_type": "dead_cat_bounce", "params": {
                "bounce_pct": 12.0,
                "duration_bars": 50,
            }},
            {"bar": 350, "event_type": "gradual_recovery", "params": {
                "target_pct": 8.0,
                "duration_bars": 150,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy must survive the cascade. Catching the knife is extremely dangerous.",
            ideal_action="reduce_exposure",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.20,
            notes="This is a survival test. Many crypto funds blow up in cascades.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.30,
            max_loss=20000.0,
            must_survive=True,
        ),
        probability_weight=0.8,
        difficulty=0.85,
        tags=["crisis", "crypto", "liquidation", "leverage"],
    )


def build_stablecoin_depeg() -> Scenario:
    """12. Stablecoin depeg: one asset loses peg, contagion."""
    return Scenario(
        name="StablecoinDepeg",
        description=(
            "A stablecoin begins trading at 0.98 instead of 1.00. Over 200 bars it drops "
            "to 0.85 as confidence erodes. Other assets react: flight to quality, "
            "increased volatility across the board, correlation spikes."
        ),
        category=ScenarioCategory.CRISIS,
        duration_bars=500,
        param_overrides={
            "volatility": 0.025,
            "drift": -0.0003,
            "mean_spread_bps": 30.0,
            "initial_price": 1.0,
            "peg_target": 1.0,
            "depeg_start_bar": 50,
            "contagion_correlation": 0.8,
        },
        event_sequence=[
            {"bar": 50, "event_type": "depeg_start", "params": {
                "initial_deviation": 0.02,
            }},
            {"bar": 100, "event_type": "depeg_accelerate", "params": {
                "target_price": 0.90,
                "duration_bars": 100,
                "panic_selling_mult": 5.0,
            }},
            {"bar": 200, "event_type": "depeg_floor", "params": {
                "floor_price": 0.85,
                "volatility_at_floor": 0.05,
            }},
            {"bar": 350, "event_type": "partial_repeg", "params": {
                "recovery_target": 0.95,
                "duration_bars": 150,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description="Strategy should short or avoid. Buying the depeg is extremely risky.",
            ideal_action="sell",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.15,
            notes="The key question: can the strategy detect the depeg early?",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.20,
            max_loss=10000.0,
        ),
        probability_weight=0.3,
        difficulty=0.75,
        tags=["crisis", "stablecoin", "depeg", "contagion"],
    )


def build_fat_finger_trade() -> Scenario:
    """13. Fat finger: single massive market order, immediate reversal."""
    return Scenario(
        name="FatFingerTrade",
        description=(
            "Market is operating normally. At bar 200, a single massive market sell order "
            "(100x normal size) blows through the book, crashing price 8%. Within 20 bars, "
            "price recovers fully as the error is recognized and reversed."
        ),
        category=ScenarioCategory.MICROSTRUCTURE,
        duration_bars=400,
        param_overrides={
            "volatility": 0.012,
            "drift": 0.0,
            "mean_spread_bps": 6.0,
            "initial_price": 100.0,
        },
        event_sequence=[
            {"bar": 200, "event_type": "fat_finger", "params": {
                "side": "sell",
                "size_multiple": 100,
                "impact_pct": 8.0,
                "recovery_bars": 20,
                "recovery_pct": 1.0,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description=(
                "The ideal response is to buy the fat-finger dip, recognizing it as an "
                "anomaly rather than real selling pressure."
            ),
            ideal_action="buy",
            ideal_pnl_sign=1,
            max_acceptable_drawdown=0.08,
            notes="Sophisticated strategies can profit from this. Naive stop-losses get stopped out.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.12,
            max_loss=5000.0,
        ),
        probability_weight=0.2,
        difficulty=0.5,
        tags=["microstructure", "fat_finger", "anomaly"],
    )


def build_wash_trading() -> Scenario:
    """14. Wash trading: fake volume from coordinated accounts."""
    return Scenario(
        name="WashTrading",
        description=(
            "An adversary is wash trading: placing matching buy and sell orders to inflate "
            "volume. The real volume is 1/10th of reported. Price barely moves despite "
            "apparently high volume. Strategies relying on volume signals get confused."
        ),
        category=ScenarioCategory.ADVERSARIAL,
        duration_bars=400,
        param_overrides={
            "volatility": 0.010,
            "drift": 0.0,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
            "wash_trading_volume_mult": 10.0,
            "real_volume_fraction": 0.1,
        },
        event_sequence=[
            {"bar": 50, "event_type": "wash_trading_start", "params": {
                "fake_volume_mult": 10.0,
                "fake_trade_size_mean": 5.0,
                "fake_trade_interval_bars": 1,
                "duration_bars": 300,
            }},
            {"bar": 250, "event_type": "wash_trading_reveal", "params": {
                "volume_drop_to": 0.1,
                "duration_bars": 10,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description=(
                "Strategy should not be tricked by fake volume. Volume-based signals "
                "should be robust to manipulation."
            ),
            ideal_action="no_trade",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.05,
            notes="An adversarial test: strategies relying purely on volume will be fooled.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.08,
            max_loss=3000.0,
            max_trades=50,  # should not be trading heavily on fake signals
        ),
        probability_weight=0.5,
        difficulty=0.6,
        tags=["adversarial", "wash_trading", "manipulation"],
    )


def build_spoofing_attack() -> Scenario:
    """15. Spoofing: large limit orders placed and cancelled to manipulate price."""
    return Scenario(
        name="SpoofingAttack",
        description=(
            "An adversary places large limit buy orders below the market, creating the "
            "illusion of strong support. When other participants buy based on this 'support', "
            "the spoofer cancels and sells into the buying pressure. The pattern repeats "
            "with sell-side spoofing too."
        ),
        category=ScenarioCategory.ADVERSARIAL,
        duration_bars=400,
        param_overrides={
            "volatility": 0.014,
            "drift": 0.0,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
            "spoof_order_size_mult": 50,
            "spoof_cancel_delay_bars": 3,
        },
        event_sequence=[
            {"bar": 50, "event_type": "spoof_buy_side", "params": {
                "spoof_levels": 5,
                "spoof_size_per_level": 500.0,
                "distance_from_mid_bps": 10.0,
                "duration_bars": 30,
                "cancel_and_sell_bar": 80,
            }},
            {"bar": 120, "event_type": "spoof_sell_side", "params": {
                "spoof_levels": 5,
                "spoof_size_per_level": 500.0,
                "distance_from_mid_bps": 10.0,
                "duration_bars": 30,
                "cancel_and_buy_bar": 150,
            }},
            {"bar": 200, "event_type": "spoof_both_sides", "params": {
                "alternating_period_bars": 20,
                "duration_bars": 150,
            }},
        ],
        expected_behavior=ExpectedBehavior(
            description=(
                "Strategy should be robust to order book manipulation. Signals based "
                "on book depth should discount large orders that might be spoofs."
            ),
            ideal_action="no_trade",
            ideal_pnl_sign=0,
            max_acceptable_drawdown=0.06,
            notes="Strategies relying on order book imbalance will be manipulated.",
        ),
        pass_criteria=PassCriteria(
            max_drawdown=0.10,
            max_loss=4000.0,
        ),
        probability_weight=0.5,
        difficulty=0.65,
        tags=["adversarial", "spoofing", "manipulation", "order_book"],
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

ALL_SCENARIOS = {
    "NormalMarket": build_normal_market,
    "TrendingBull": build_trending_bull,
    "TrendingBear": build_trending_bear,
    "HighVolatility": build_high_volatility,
    "FlashCrash": build_flash_crash,
    "LiquidityDrain": build_liquidity_drain,
    "ShortSqueeze": build_short_squeeze,
    "CircuitBreakerTest": build_circuit_breaker_test,
    "FOMCMeeting": build_fomc_meeting,
    "EarningsAnnouncement": build_earnings_announcement,
    "CryptoLiquidationCascade": build_crypto_liquidation_cascade,
    "StablecoinDepeg": build_stablecoin_depeg,
    "FatFingerTrade": build_fat_finger_trade,
    "WashTrading": build_wash_trading,
    "SpoofingAttack": build_spoofing_attack,
}


def get_scenario(name: str) -> Scenario:
    builder = ALL_SCENARIOS.get(name)
    if builder is None:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(ALL_SCENARIOS.keys())}")
    return builder()


def get_all_scenarios() -> List[Scenario]:
    return [builder() for builder in ALL_SCENARIOS.values()]


def get_scenarios_by_category(category: ScenarioCategory) -> List[Scenario]:
    return [s for s in get_all_scenarios() if s.category == category]


def get_scenarios_by_tag(tag: str) -> List[Scenario]:
    return [s for s in get_all_scenarios() if tag in s.tags]


# ---------------------------------------------------------------------------
# Simulated bar generator (for offline testing without the Go API)
# ---------------------------------------------------------------------------

class OfflineBarGenerator:
    """
    Generate synthetic OHLCV bars locally, applying scenario event sequences.
    Used when the Go exchange API is not available.
    """

    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.params = {
            "volatility": 0.015,
            "drift": 0.0,
            "mean_spread_bps": 8.0,
            "initial_price": 100.0,
        }
        self.params.update(scenario.param_overrides)
        self._price = self.params["initial_price"]
        self._bar_idx = 0
        self._active_events: List[Dict[str, Any]] = []
        self._vol_multiplier = 1.0
        self._drift_override: Optional[float] = None
        self._spread_multiplier = 1.0
        self._volume_multiplier = 1.0
        self._halted = False
        self._halt_end_bar = 0
        self._gap_pending: Optional[float] = None

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all bars for the scenario."""
        bars = []
        for i in range(self.scenario.duration_bars):
            bar = self.next_bar()
            bars.append(bar)
        return bars

    def next_bar(self) -> Dict[str, Any]:
        """Generate the next bar, applying any active events."""
        self._process_events()

        if self._halted:
            # during a halt, return the last price with zero volume
            bar = self._make_bar(self._price, self._price, self._price, self._price, 0.0)
            self._bar_idx += 1
            if self._bar_idx >= self._halt_end_bar:
                self._halted = False
            return bar

        # apply pending gap
        if self._gap_pending is not None:
            self._price *= (1.0 + self._gap_pending)
            self._gap_pending = None

        # generate return
        base_vol = self.params["volatility"]
        vol = base_vol * self._vol_multiplier
        drift = self._drift_override if self._drift_override is not None else self.params["drift"]
        ret = drift + vol * self.rng.standard_normal()
        new_price = self._price * np.exp(ret)

        # generate OHLCV
        intra_vol = vol * 0.5
        intra_moves = self.rng.standard_normal(4) * intra_vol
        prices_intra = self._price * np.exp(np.cumsum(intra_moves))
        o = self._price
        c = new_price
        h = max(o, c, *prices_intra)
        l = min(o, c, *prices_intra)

        base_volume = 1000.0 * self._volume_multiplier
        volume = max(1.0, base_volume + self.rng.standard_normal() * base_volume * 0.3)

        spread = self.params["mean_spread_bps"] * self._spread_multiplier / 10000.0 * self._price
        bid_depth = max(10.0, 100.0 / self._spread_multiplier + self.rng.standard_normal() * 20)
        ask_depth = max(10.0, 100.0 / self._spread_multiplier + self.rng.standard_normal() * 20)

        bar = self._make_bar(o, h, l, c, volume, spread, bid_depth, ask_depth)
        self._price = new_price
        self._bar_idx += 1
        return bar

    def _make_bar(
        self,
        o: float, h: float, l: float, c: float,
        volume: float,
        spread: float = 0.0,
        bid_depth: float = 100.0,
        ask_depth: float = 100.0,
    ) -> Dict[str, Any]:
        return {
            "bar_idx": self._bar_idx,
            "timestamp": float(self._bar_idx),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(volume),
            "vwap": float((o + h + l + c) / 4.0),
            "num_trades": max(1, int(volume / 10)),
            "spread_avg": float(spread),
            "bid_depth_avg": float(bid_depth),
            "ask_depth_avg": float(ask_depth),
        }

    def _process_events(self) -> None:
        """Check if any events should fire at the current bar."""
        for event_def in self.scenario.event_sequence:
            trigger_bar = event_def["bar"]
            if self._bar_idx == trigger_bar:
                self._apply_event(event_def)

        # decay active event effects
        new_active = []
        for ev in self._active_events:
            ev["remaining"] -= 1
            if ev["remaining"] > 0:
                new_active.append(ev)
            else:
                self._remove_event_effect(ev)
        self._active_events = new_active

    def _apply_event(self, event_def: Dict[str, Any]) -> None:
        etype = event_def["event_type"]
        params = event_def.get("params", {})

        if etype == "flash_crash":
            mag = params.get("magnitude_pct", 10.0)
            dur = params.get("crash_duration_bars", 30)
            self._drift_override = -mag / 100.0 / dur
            self._vol_multiplier *= 3.0
            self._spread_multiplier *= 5.0
            self._active_events.append({
                "type": "flash_crash",
                "remaining": dur,
                "phase": "crash",
                "recovery_dur": params.get("recovery_duration_bars", 200),
                "recovery_pct": params.get("recovery_pct", 0.8),
                "original_price": self._price,
            })

        elif etype == "volatility_spike":
            mult = params.get("multiplier", 2.0)
            dur = params.get("duration_bars", 50)
            self._vol_multiplier *= mult
            self._active_events.append({"type": "vol_spike", "remaining": dur, "mult": mult})

        elif etype == "volume_surge":
            mult = params.get("multiplier", 2.0)
            dur = params.get("duration", 50)
            self._volume_multiplier *= mult
            self._active_events.append({"type": "vol_surge", "remaining": dur, "mult": mult})

        elif etype == "liquidity_drain":
            dur = params.get("duration_bars", 200)
            self._active_events.append({
                "type": "liquidity_drain",
                "remaining": dur,
                "drain_rate": params.get("drain_rate_per_bar", 0.005),
                "spread_growth": params.get("spread_growth_factor", 1.02),
            })

        elif etype == "circuit_breaker_halt":
            dur = params.get("halt_duration_bars", 30)
            self._halted = True
            self._halt_end_bar = self._bar_idx + dur

        elif etype == "reopen_with_gap":
            gap = params.get("gap_pct", -2.0) / 100.0
            self._gap_pending = gap

        elif etype in ("rapid_selloff", "initial_drop", "panic_sell"):
            mag = params.get("magnitude_pct", params.get("target_pct", 5.0))
            dur = params.get("duration_bars", 30)
            self._drift_override = -abs(mag) / 100.0 / max(dur, 1)
            self._vol_multiplier *= params.get("intensity", 1.5)
            self._active_events.append({"type": "selloff", "remaining": dur})

        elif etype == "recovery":
            pct = params.get("recovery_pct", 0.4)
            dur = params.get("duration_bars", 200)
            self._drift_override = abs(pct) * 0.01 / max(dur, 1)
            self._active_events.append({"type": "recovery", "remaining": dur})

        elif etype == "short_squeeze":
            dur = params.get("duration_bars", 200)
            peak = params.get("peak_price_mult", 1.4)
            drift_needed = np.log(peak) / max(dur, 1)
            self._drift_override = drift_needed
            self._vol_multiplier *= 2.5
            self._volume_multiplier *= params.get("volume_multiplier", 10.0)
            self._active_events.append({"type": "squeeze", "remaining": dur})

        elif etype == "profit_taking":
            dur = params.get("duration_bars", 100)
            retrace = params.get("retracement_pct", 0.5)
            self._drift_override = -retrace * 0.01 / max(dur, 1)
            self._active_events.append({"type": "profit_take", "remaining": dur})

        elif etype == "vol_compression":
            mult = params.get("target_vol_mult", 0.4)
            dur = params.get("duration_bars", 100)
            self._vol_multiplier *= mult
            self._active_events.append({"type": "vol_compress", "remaining": dur, "mult": mult})

        elif etype == "fomc_announcement":
            mag = params.get("magnitude_pct", 2.0) / 100.0
            direction = params.get("direction", "random")
            if direction == "random":
                direction = "up" if self.rng.random() > 0.5 else "down"
            sign = 1.0 if direction == "up" else -1.0
            dur = params.get("duration_bars", 10)
            self._drift_override = sign * mag / max(dur, 1)
            self._vol_multiplier *= params.get("vol_spike_mult", 3.5)
            self._active_events.append({"type": "fomc", "remaining": dur})

        elif etype == "post_event_vol":
            mult = params.get("vol_mult", 2.0)
            dur = params.get("duration_bars", 200)
            self._vol_multiplier *= mult
            self._active_events.append({"type": "post_event", "remaining": dur, "mult": mult})

        elif etype == "earnings_gap":
            gap = params.get("gap_pct", 6.0) / 100.0
            direction = params.get("direction", "random")
            if direction == "random":
                direction = "up" if self.rng.random() > 0.5 else "down"
            sign = 1.0 if direction == "up" else -1.0
            self._gap_pending = sign * gap
            dur = params.get("elevated_vol_duration", 100)
            self._vol_multiplier *= params.get("vol_mult", 3.0)
            self._volume_multiplier *= params.get("volume_mult", 5.0)
            cont = params.get("continuation_probability", 0.6)
            if self.rng.random() < cont:
                self._drift_override = sign * 0.001
            self._active_events.append({"type": "earnings", "remaining": dur})

        elif etype == "liquidation_cascade":
            waves = params.get("waves", 5)
            magnitudes = params.get("wave_magnitude_pct", [4.0] * waves)
            total_dur = waves * params.get("wave_interval_bars", 15)
            avg_mag = np.mean(magnitudes)
            self._drift_override = -avg_mag / 100.0 / max(total_dur / waves, 1)
            self._vol_multiplier *= 3.0
            self._spread_multiplier *= 3.0
            self._active_events.append({"type": "cascade", "remaining": total_dur})

        elif etype == "dead_cat_bounce":
            pct = params.get("bounce_pct", 12.0) / 100.0
            dur = params.get("duration_bars", 50)
            self._drift_override = pct / max(dur, 1)
            self._active_events.append({"type": "bounce", "remaining": dur})

        elif etype in ("depeg_start", "depeg_accelerate"):
            target = params.get("target_price", 0.9)
            dur = params.get("duration_bars", 100)
            current = self._price
            if current > 0:
                self._drift_override = np.log(target / current) / max(dur, 1)
            self._vol_multiplier *= 2.0
            self._active_events.append({"type": "depeg", "remaining": dur})

        elif etype == "fat_finger":
            impact = params.get("impact_pct", 8.0) / 100.0
            side = params.get("side", "sell")
            sign = -1.0 if side == "sell" else 1.0
            self._gap_pending = sign * impact
            recovery_bars = params.get("recovery_bars", 20)
            self._active_events.append({
                "type": "fat_finger_recovery",
                "remaining": recovery_bars,
                "recovery_drift": -sign * impact / max(recovery_bars, 1),
            })
            self._drift_override = -sign * impact / max(recovery_bars, 1)

        elif etype in ("wash_trading_start", "spoof_buy_side", "spoof_sell_side", "spoof_both_sides",
                       "fomo_wave", "spread_widening", "book_thinning", "liquidity_return",
                       "gradual_recovery", "partial_repeg", "depeg_floor", "wash_trading_reveal"):
            dur = params.get("duration_bars", params.get("duration", 100))
            self._active_events.append({"type": etype, "remaining": dur})

        else:
            logger.warning("Unknown event type: %s", etype)

    def _remove_event_effect(self, ev: Dict[str, Any]) -> None:
        """Reset effects when an event expires."""
        etype = ev.get("type", "")
        if etype == "vol_spike":
            self._vol_multiplier /= ev.get("mult", 2.0)
        elif etype == "vol_surge":
            self._volume_multiplier /= ev.get("mult", 2.0)
        elif etype == "vol_compress":
            self._vol_multiplier /= ev.get("mult", 0.4)
        elif etype in ("selloff", "recovery", "squeeze", "profit_take", "fomc",
                        "earnings", "cascade", "bounce", "depeg", "fat_finger_recovery"):
            self._drift_override = None
            if self._vol_multiplier > 1.5:
                self._vol_multiplier = max(1.0, self._vol_multiplier * 0.5)
            if self._spread_multiplier > 1.5:
                self._spread_multiplier = max(1.0, self._spread_multiplier * 0.5)
        elif etype == "post_event":
            self._vol_multiplier /= ev.get("mult", 2.0)
        elif etype == "flash_crash":
            # start recovery phase
            rec_dur = ev.get("recovery_dur", 200)
            rec_pct = ev.get("recovery_pct", 0.8)
            orig = ev.get("original_price", self._price)
            target = orig * (1.0 - (1.0 - rec_pct) * (orig - self._price) / (orig + 1e-10))
            if self._price > 0:
                self._drift_override = np.log(target / self._price) / max(rec_dur, 1)
            self._vol_multiplier = max(1.0, self._vol_multiplier * 0.5)
            self._spread_multiplier = max(1.0, self._spread_multiplier * 0.3)
            self._active_events.append({"type": "recovery", "remaining": rec_dur})
        elif etype == "liquidity_drain":
            self._spread_multiplier = max(1.0, self._spread_multiplier * 0.5)


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------

class ScenarioRunner:
    """Execute any scenario and evaluate strategy performance."""

    def __init__(self, exchange_client=None):
        """
        Args:
            exchange_client: An ExchangeClient instance (or None for offline mode).
        """
        self.client = exchange_client

    def run(
        self,
        scenario: Scenario,
        strategy_fn: Callable,
        strategy_name: str = "strategy",
        initial_capital: float = 100000.0,
        use_offline: bool = False,
    ) -> ScenarioResult:
        """
        Run a strategy through a scenario and return the result.

        Args:
            scenario: The scenario to simulate.
            strategy_fn: Signal function with signature
                (bars, position_qty, bar_idx) -> Optional[Dict]
            strategy_name: Name for reporting.
            initial_capital: Starting capital.
            use_offline: If True, use OfflineBarGenerator instead of the exchange API.
        """
        start_time = time.time()
        logger.info("Running scenario '%s' with strategy '%s'", scenario.name, strategy_name)

        if use_offline or self.client is None:
            return self._run_offline(scenario, strategy_fn, strategy_name, initial_capital, start_time)
        else:
            return self._run_online(scenario, strategy_fn, strategy_name, initial_capital, start_time)

    def _run_offline(
        self,
        scenario: Scenario,
        strategy_fn: Callable,
        strategy_name: str,
        initial_capital: float,
        start_time: float,
    ) -> ScenarioResult:
        """Run using offline bar generation."""
        gen = OfflineBarGenerator(scenario)
        all_bars = gen.generate_all()

        position_qty = 0.0
        avg_entry = 0.0
        realized_pnl = 0.0
        total_fees = 0.0
        equity = initial_capital
        equity_curve = [equity]
        trade_pnls: List[float] = []
        returns: List[float] = []
        num_trades = 0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        max_consec_loss = 0
        cur_consec_loss = 0
        blew_up = False
        bar_objects: List[Any] = []

        from strategy_interface import MarketBar, Position as PosClass

        for bar_data in all_bars:
            bar = MarketBar(
                symbol="SIM-USD",
                timestamp=bar_data["timestamp"],
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                vwap=bar_data["vwap"],
                num_trades=bar_data["num_trades"],
                spread_avg=bar_data["spread_avg"],
                bid_depth_avg=bar_data["bid_depth_avg"],
                ask_depth_avg=bar_data["ask_depth_avg"],
            )
            bar_objects.append(bar)
            current_price = bar.close

            # get signal
            try:
                signal = strategy_fn(bar_objects, position_qty, bar_data["bar_idx"])
            except Exception:
                signal = None

            if signal is not None:
                side = signal.get("side", "buy")
                qty = signal.get("qty", 1.0)
                fill_price = current_price
                spread_cost = bar_data["spread_avg"] * 0.5
                if side == "buy":
                    fill_price += spread_cost
                else:
                    fill_price -= spread_cost
                fill_price = max(fill_price, 0.01)

                fee = abs(qty) * fill_price * 0.001
                total_fees += fee

                sign = 1.0 if side == "buy" else -1.0
                fill_qty = qty * sign

                if position_qty == 0:
                    avg_entry = fill_price
                    position_qty = fill_qty
                elif (position_qty > 0 and sign > 0) or (position_qty < 0 and sign < 0):
                    total_cost = avg_entry * abs(position_qty) + fill_price * qty
                    position_qty += fill_qty
                    if abs(position_qty) > 0:
                        avg_entry = total_cost / abs(position_qty)
                else:
                    close_qty = min(abs(position_qty), qty)
                    rpnl = close_qty * (fill_price - avg_entry) * (1.0 if position_qty > 0 else -1.0)
                    realized_pnl += rpnl
                    trade_pnls.append(rpnl)
                    num_trades += 1
                    if rpnl >= 0:
                        wins += 1
                        gross_profit += rpnl
                        cur_consec_loss = 0
                    else:
                        losses += 1
                        gross_loss += abs(rpnl)
                        cur_consec_loss += 1
                        max_consec_loss = max(max_consec_loss, cur_consec_loss)

                    remaining = qty - close_qty
                    position_qty += fill_qty
                    if remaining > 0 and abs(position_qty) > 0:
                        avg_entry = fill_price

            unrealized = position_qty * (current_price - avg_entry) if position_qty != 0 else 0
            equity = initial_capital + realized_pnl + unrealized - total_fees
            equity_curve.append(equity)

            if len(equity_curve) >= 2 and equity_curve[-2] != 0:
                returns.append((equity - equity_curve[-2]) / abs(equity_curve[-2]))

            if equity < initial_capital * 0.5:
                blew_up = True

        # compute metrics
        hwm = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            if e > hwm:
                hwm = e
            dd = (hwm - e) / (hwm + 1e-10)
            if dd > max_dd:
                max_dd = dd

        pnl = equity - initial_capital
        returns_arr = np.array(returns) if returns else np.array([0.0])
        sharpe = 0.0
        if len(returns_arr) > 1:
            mu = np.mean(returns_arr)
            sigma = np.std(returns_arr, ddof=1)
            if sigma > 0:
                sharpe = mu / sigma * np.sqrt(252)

        sortino = 0.0
        if len(returns_arr) > 1:
            downside = returns_arr[returns_arr < 0]
            if len(downside) > 1:
                ds = np.std(downside, ddof=1)
                if ds > 0:
                    sortino = np.mean(returns_arr) / ds * np.sqrt(252)

        calmar = 0.0
        if max_dd > 0 and len(returns_arr) > 1:
            calmar = np.mean(returns_arr) * 252 / max_dd

        win_rate = wins / num_trades if num_trades > 0 else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_trade = pnl / num_trades if num_trades > 0 else 0.0

        # evaluate pass criteria
        result = ScenarioResult(
            scenario_name=scenario.name,
            strategy_name=strategy_name,
            pnl=pnl,
            sharpe=sharpe,
            max_dd=max_dd,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=pf,
            blew_up=blew_up,
            passed=False,
            failure_reasons=[],
            narrative="",
            equity_curve=equity_curve,
            bars_processed=len(all_bars),
            execution_time_sec=time.time() - start_time,
            final_position=position_qty,
            total_fees=total_fees,
            sortino=sortino,
            calmar=calmar,
            avg_trade_pnl=avg_trade,
            max_consecutive_losses=max_consec_loss,
        )

        passed, failures = scenario.pass_criteria.evaluate(result)
        result.passed = passed
        result.failure_reasons = failures
        result.narrative = self._generate_narrative(scenario, result)

        return result

    def _run_online(
        self,
        scenario: Scenario,
        strategy_fn: Callable,
        strategy_name: str,
        initial_capital: float,
        start_time: float,
    ) -> ScenarioResult:
        """Run using the live exchange API."""
        from strategy_interface import (
            SimulationConfig, StrategyRunner, StrategyRunConfig, MarketBar,
        )

        sim_config = SimulationConfig()
        for k, v in scenario.param_overrides.items():
            if hasattr(sim_config, k):
                setattr(sim_config, k, v)
        sim_config.duration_bars = scenario.duration_bars

        self.client.start_simulation(sim_config)

        # schedule event injections
        event_timers: List[threading.Timer] = []
        for ev in scenario.event_sequence:
            bar_trigger = ev["bar"]
            delay = bar_trigger * sim_config.bar_interval_ms / 1000.0

            def _inject(et=ev["event_type"], p=ev.get("params", {})):
                try:
                    self.client.inject_event(et, p)
                except Exception as exc:
                    logger.error("Event injection failed: %s", exc)

            t = threading.Timer(delay, _inject)
            t.daemon = True
            t.start()
            event_timers.append(t)

        # wrap strategy_fn to match StrategyRunner interface
        def _adapted_signal(bars, position, bar_idx):
            return strategy_fn(bars, position.quantity, bar_idx)

        run_config = StrategyRunConfig(
            symbol=sim_config.symbols[0] if sim_config.symbols else "SIM-USD",
            initial_capital=initial_capital,
            max_position=100.0,
            max_drawdown_pct=0.25,
            warmup_bars=10,
        )
        runner = StrategyRunner(
            signal_fn=_adapted_signal,
            client=self.client,
            config=run_config,
            name=strategy_name,
        )
        raw_results = runner.run(scenario.duration_bars)

        for t in event_timers:
            t.cancel()

        try:
            self.client.stop_simulation()
        except Exception:
            pass

        perf = raw_results.get("performance", {})
        pnl = raw_results.get("total_pnl", 0)
        max_dd = perf.get("max_drawdown", 0)
        sharpe = perf.get("sharpe_ratio", 0)
        blew_up = max_dd > 0.5

        result = ScenarioResult(
            scenario_name=scenario.name,
            strategy_name=strategy_name,
            pnl=pnl,
            sharpe=sharpe,
            max_dd=max_dd,
            num_trades=perf.get("trade_count", 0),
            win_rate=perf.get("win_rate", 0),
            profit_factor=perf.get("profit_factor", 0),
            blew_up=blew_up,
            passed=False,
            failure_reasons=[],
            narrative="",
            equity_curve=runner.tracker.equity_curve,
            bars_processed=raw_results.get("bars_processed", 0),
            execution_time_sec=time.time() - start_time,
            final_position=raw_results.get("position", 0),
            total_fees=raw_results.get("total_fees", 0),
            sortino=perf.get("sortino_ratio", 0),
            calmar=perf.get("calmar_ratio", 0),
            avg_trade_pnl=perf.get("expectancy", 0),
            max_consecutive_losses=perf.get("max_consecutive_losses", 0),
        )

        passed, failures = scenario.pass_criteria.evaluate(result)
        result.passed = passed
        result.failure_reasons = failures
        result.narrative = self._generate_narrative(scenario, result)
        return result

    def _generate_narrative(self, scenario: Scenario, result: ScenarioResult) -> str:
        """Generate a human-readable narrative of the scenario result."""
        lines: List[str] = []
        lines.append(f"=== {scenario.name} | {result.strategy_name} ===")
        lines.append(f"Scenario: {scenario.description[:120]}...")
        lines.append(f"Duration: {result.bars_processed} bars, Exec time: {result.execution_time_sec:.2f}s")
        lines.append("")

        if result.passed:
            lines.append("RESULT: PASSED")
        else:
            lines.append("RESULT: FAILED")
            for reason in result.failure_reasons:
                lines.append(f"  - {reason}")

        lines.append("")
        lines.append(f"P&L: ${result.pnl:,.2f}")
        lines.append(f"Sharpe: {result.sharpe:.3f} | Sortino: {result.sortino:.3f} | Calmar: {result.calmar:.3f}")
        lines.append(f"Max DD: {result.max_dd:.4f} ({result.max_dd * 100:.2f}%)")
        lines.append(f"Trades: {result.num_trades} | Win rate: {result.win_rate:.2%}")
        lines.append(f"Profit factor: {result.profit_factor:.2f}")
        lines.append(f"Final position: {result.final_position:.2f}")
        lines.append(f"Total fees: ${result.total_fees:.2f}")

        if result.blew_up:
            lines.append("")
            lines.append("*** STRATEGY BLEW UP ***")

        expected = scenario.expected_behavior
        lines.append("")
        lines.append(f"Expected: {expected.description}")
        lines.append(f"Ideal action: {expected.ideal_action}")
        if expected.notes:
            lines.append(f"Note: {expected.notes}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StressTestSuite
# ---------------------------------------------------------------------------

@dataclass
class StressTestReport:
    """Aggregate report from running all scenarios."""
    strategy_name: str
    results: List[ScenarioResult]
    survival_rate: float
    pass_rate: float
    worst_scenario: str
    worst_pnl: float
    best_scenario: str
    best_pnl: float
    aggregate_pnl: float
    aggregate_sharpe: float
    weighted_sharpe: float
    total_trades: int
    total_fees: float
    total_time_sec: float
    scenario_scores: Dict[str, float]
    category_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "survival_rate": self.survival_rate,
            "pass_rate": self.pass_rate,
            "worst_scenario": self.worst_scenario,
            "worst_pnl": self.worst_pnl,
            "best_scenario": self.best_scenario,
            "best_pnl": self.best_pnl,
            "aggregate_pnl": self.aggregate_pnl,
            "aggregate_sharpe": self.aggregate_sharpe,
            "weighted_sharpe": self.weighted_sharpe,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
            "total_time_sec": self.total_time_sec,
            "scenario_scores": self.scenario_scores,
            "category_scores": self.category_scores,
            "results": [r.to_dict() for r in self.results],
        }

    def summary_text(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"STRESS TEST REPORT: {self.strategy_name}",
            f"{'=' * 60}",
            f"Scenarios run: {len(self.results)}",
            f"Survival rate: {self.survival_rate:.1%}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"",
            f"Aggregate P&L: ${self.aggregate_pnl:,.2f}",
            f"Weighted Sharpe: {self.weighted_sharpe:.3f}",
            f"Total trades: {self.total_trades}",
            f"Total fees: ${self.total_fees:,.2f}",
            f"Total time: {self.total_time_sec:.1f}s",
            f"",
            f"Best scenario:  {self.best_scenario} (${self.best_pnl:,.2f})",
            f"Worst scenario: {self.worst_scenario} (${self.worst_pnl:,.2f})",
            f"",
            "Category scores:",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat}: {score:.2f}")
        lines.append("")
        lines.append("Per-scenario:")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            blow = " [BLEW UP]" if r.blew_up else ""
            lines.append(
                f"  {r.scenario_name:30s} {status:4s} PnL=${r.pnl:>10,.2f} "
                f"Sharpe={r.sharpe:>7.3f} DD={r.max_dd:>6.2%}{blow}"
            )
        return "\n".join(lines)


class StressTestSuite:
    """Run ALL scenarios and produce an aggregate report."""

    def __init__(
        self,
        exchange_client=None,
        scenarios: Optional[List[Scenario]] = None,
        use_offline: bool = False,
    ):
        self.client = exchange_client
        self.scenarios = scenarios if scenarios is not None else get_all_scenarios()
        self.use_offline = use_offline
        self._runner = ScenarioRunner(exchange_client)

    def run(
        self,
        strategy_fn: Callable,
        strategy_name: str = "strategy",
        initial_capital: float = 100000.0,
        parallel: bool = False,
    ) -> StressTestReport:
        """Run the strategy through all scenarios."""
        results: List[ScenarioResult] = []

        if parallel:
            results = self._run_parallel(strategy_fn, strategy_name, initial_capital)
        else:
            for scenario in self.scenarios:
                try:
                    result = self._runner.run(
                        scenario=scenario,
                        strategy_fn=strategy_fn,
                        strategy_name=strategy_name,
                        initial_capital=initial_capital,
                        use_offline=self.use_offline or self.client is None,
                    )
                    results.append(result)
                    logger.info(
                        "Scenario '%s': %s (PnL=%.2f, Sharpe=%.3f)",
                        scenario.name,
                        "PASS" if result.passed else "FAIL",
                        result.pnl,
                        result.sharpe,
                    )
                except Exception as exc:
                    logger.error("Scenario '%s' crashed: %s", scenario.name, exc)
                    results.append(ScenarioResult(
                        scenario_name=scenario.name,
                        strategy_name=strategy_name,
                        pnl=0, sharpe=0, max_dd=1.0, num_trades=0,
                        win_rate=0, profit_factor=0, blew_up=True,
                        passed=False,
                        failure_reasons=[f"Scenario crashed: {exc}"],
                        narrative=f"Scenario crashed with error: {exc}",
                        equity_curve=[initial_capital],
                        bars_processed=0,
                        execution_time_sec=0,
                        final_position=0,
                        total_fees=0,
                    ))

        return self._compile_report(strategy_name, results)

    def _run_parallel(
        self,
        strategy_fn: Callable,
        strategy_name: str,
        initial_capital: float,
    ) -> List[ScenarioResult]:
        """Run scenarios in parallel threads (offline mode only)."""
        results: List[ScenarioResult] = [None] * len(self.scenarios)  # type: ignore
        lock = threading.Lock()

        def _run_one(idx: int, scenario: Scenario):
            runner = ScenarioRunner(None)
            try:
                res = runner.run(
                    scenario=scenario,
                    strategy_fn=strategy_fn,
                    strategy_name=strategy_name,
                    initial_capital=initial_capital,
                    use_offline=True,
                )
            except Exception as exc:
                res = ScenarioResult(
                    scenario_name=scenario.name,
                    strategy_name=strategy_name,
                    pnl=0, sharpe=0, max_dd=1.0, num_trades=0,
                    win_rate=0, profit_factor=0, blew_up=True,
                    passed=False,
                    failure_reasons=[f"Crashed: {exc}"],
                    narrative=f"Crashed: {exc}",
                    equity_curve=[initial_capital],
                    bars_processed=0,
                    execution_time_sec=0,
                    final_position=0,
                    total_fees=0,
                )
            with lock:
                results[idx] = res

        threads = []
        for i, sc in enumerate(self.scenarios):
            t = threading.Thread(target=_run_one, args=(i, sc))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return [r for r in results if r is not None]

    def _compile_report(
        self,
        strategy_name: str,
        results: List[ScenarioResult],
    ) -> StressTestReport:
        """Compile all results into an aggregate report."""
        if not results:
            return StressTestReport(
                strategy_name=strategy_name,
                results=[],
                survival_rate=0, pass_rate=0,
                worst_scenario="", worst_pnl=0,
                best_scenario="", best_pnl=0,
                aggregate_pnl=0, aggregate_sharpe=0, weighted_sharpe=0,
                total_trades=0, total_fees=0, total_time_sec=0,
                scenario_scores={}, category_scores={},
            )

        survived = sum(1 for r in results if not r.blew_up)
        passed = sum(1 for r in results if r.passed)
        survival_rate = survived / len(results)
        pass_rate = passed / len(results)

        pnls = [(r.scenario_name, r.pnl) for r in results]
        pnls_sorted = sorted(pnls, key=lambda x: x[1])
        worst = pnls_sorted[0]
        best = pnls_sorted[-1]

        aggregate_pnl = sum(r.pnl for r in results)
        total_trades = sum(r.num_trades for r in results)
        total_fees = sum(r.total_fees for r in results)
        total_time = sum(r.execution_time_sec for r in results)

        sharpes = [r.sharpe for r in results if not np.isnan(r.sharpe) and not np.isinf(r.sharpe)]
        aggregate_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

        # weighted sharpe by scenario probability
        scenario_map = {s.name: s for s in self.scenarios}
        weight_sum = 0.0
        weighted_sharpe_sum = 0.0
        for r in results:
            sc = scenario_map.get(r.scenario_name)
            w = sc.probability_weight if sc else 1.0
            if not np.isnan(r.sharpe) and not np.isinf(r.sharpe):
                weighted_sharpe_sum += r.sharpe * w
                weight_sum += w
        weighted_sharpe = weighted_sharpe_sum / weight_sum if weight_sum > 0 else 0.0

        # per-scenario scores (normalized 0-1)
        scenario_scores: Dict[str, float] = {}
        for r in results:
            score = 0.0
            if r.passed:
                score += 0.5
            if not r.blew_up:
                score += 0.2
            if r.sharpe > 0:
                score += min(0.15, r.sharpe * 0.05)
            if r.max_dd < 0.10:
                score += 0.15
            elif r.max_dd < 0.20:
                score += 0.07
            scenario_scores[r.scenario_name] = min(1.0, score)

        # category scores
        category_results: Dict[str, List[float]] = {}
        for r in results:
            sc = scenario_map.get(r.scenario_name)
            if sc:
                cat = sc.category.value
                category_results.setdefault(cat, []).append(scenario_scores.get(r.scenario_name, 0))
        category_scores = {cat: float(np.mean(scores)) for cat, scores in category_results.items()}

        return StressTestReport(
            strategy_name=strategy_name,
            results=results,
            survival_rate=survival_rate,
            pass_rate=pass_rate,
            worst_scenario=worst[0],
            worst_pnl=worst[1],
            best_scenario=best[0],
            best_pnl=best[1],
            aggregate_pnl=aggregate_pnl,
            aggregate_sharpe=aggregate_sharpe,
            weighted_sharpe=weighted_sharpe,
            total_trades=total_trades,
            total_fees=total_fees,
            total_time_sec=total_time,
            scenario_scores=scenario_scores,
            category_scores=category_scores,
        )


# ---------------------------------------------------------------------------
# Convenience: run a quick stress test
# ---------------------------------------------------------------------------

def quick_stress_test(
    strategy_fn: Callable,
    strategy_name: str = "strategy",
    scenarios: Optional[List[str]] = None,
    initial_capital: float = 100000.0,
) -> StressTestReport:
    """
    Run a quick offline stress test with selected or all scenarios.

    Args:
        strategy_fn: Signal function (bars, position_qty, bar_idx) -> Optional[Dict]
        strategy_name: Name for reporting
        scenarios: List of scenario names (or None for all)
        initial_capital: Starting capital

    Returns:
        StressTestReport with aggregate results
    """
    if scenarios is not None:
        scenario_list = [get_scenario(name) for name in scenarios]
    else:
        scenario_list = get_all_scenarios()

    suite = StressTestSuite(
        exchange_client=None,
        scenarios=scenario_list,
        use_offline=True,
    )
    return suite.run(
        strategy_fn=strategy_fn,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        parallel=True,
    )
