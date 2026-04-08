"""
macro_micro_fusion.py
Hypothesis templates that fuse macro regime signals with micro/equity signals.
Each template is a structured dataclass capturing logic, entry/exit conditions,
signal sources, and sizing guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    LONG_SHORT = "long_short"


class AssetClass(str, Enum):
    EQUITY = "equity"
    RATES = "rates"
    CREDIT = "credit"
    FX = "fx"
    COMMODITY = "commodity"
    VOL = "vol"


class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class EntryCondition:
    signal: str
    operator: str          # "gt", "lt", "crosses_below", "crosses_above", "between"
    threshold: Any
    description: str


@dataclass
class ExitCondition:
    trigger: str
    description: str
    is_hard_stop: bool = False


@dataclass
class HypothesisTemplate:
    name: str
    description: str
    thesis: str
    primary_asset_class: AssetClass
    secondary_asset_class: AssetClass | None
    direction: Direction
    macro_signals: list[str]
    micro_signals: list[str]
    entry_conditions: list[EntryCondition]
    exit_conditions: list[ExitCondition]
    sizing_guidance: str
    expected_holding_period: str
    historical_analogues: list[str]
    risk_factors: list[str]
    signal_strength: SignalStrength
    notes: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template 1 – Yield Curve Inversion Trade
# ---------------------------------------------------------------------------
YIELD_CURVE_INVERSION_TRADE = HypothesisTemplate(
    name="yield_curve_inversion_trade",
    description=(
        "Short risk assets when the 2s10s Treasury yield spread inverts; "
        "reverse to long once the spread un-inverts and macro data confirm a trough."
    ),
    thesis=(
        "An inverted 2s10s curve reliably predicts recessions with a 6–18 month lag. "
        "Markets tend to reprice risk premia during and after inversion. "
        "The un-inversion, particularly when driven by the short end falling, "
        "historically marks the onset of a credit/equity stress period — "
        "a counter-intuitive bearish signal. The long leg is re-entered once "
        "the curve steepens aggressively and leading indicators bottom."
    ),
    primary_asset_class=AssetClass.EQUITY,
    secondary_asset_class=AssetClass.RATES,
    direction=Direction.LONG_SHORT,
    macro_signals=[
        "2s10s_treasury_spread",
        "3m10y_treasury_spread",
        "fed_funds_rate_delta_90d",
        "ism_manufacturing_pmi",
        "us_unemployment_claims_4wk_ma",
        "conference_board_leading_index_yoy",
    ],
    micro_signals=[
        "spx_200d_ma_distance",
        "high_yield_oas",
        "investment_grade_oas",
        "vix_level",
        "equity_put_call_ratio_10d_ma",
        "spx_earnings_revision_breadth",
    ],
    entry_conditions=[
        EntryCondition(
            signal="2s10s_treasury_spread",
            operator="crosses_below",
            threshold=0.0,
            description="2s10s spread crosses below zero (inversion confirmed)",
        ),
        EntryCondition(
            signal="ism_manufacturing_pmi",
            operator="lt",
            threshold=50.0,
            description="Manufacturing PMI below expansion threshold",
        ),
        EntryCondition(
            signal="high_yield_oas",
            operator="gt",
            threshold=400,
            description="HY OAS above 400 bps, credit stress building",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="2s10s_treasury_spread > 50bps for 20 consecutive days",
            description="Curve steepening signals macro recovery ahead",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="spx_drawdown_from_entry > 15%",
            description="Hard stop on position if short-squeeze threatens",
            is_hard_stop=True,
        ),
        ExitCondition(
            trigger="ism_manufacturing_pmi > 52 and hy_oas < 350",
            description="Both macro and credit confirm recovery",
            is_hard_stop=False,
        ),
    ],
    sizing_guidance=(
        "Start at 50% of max position on inversion confirmation. "
        "Add to 100% when PMI falls below 48 AND HY spreads exceed 450 bps. "
        "Reduce by half on any single-day SPX gap >3% against position."
    ),
    expected_holding_period="3–18 months",
    historical_analogues=[
        "2000-03 dot-com bear (inversion 1998-2000)",
        "2006-08 GFC setup (inversion 2005-2007)",
        "2019-20 inversion and COVID amplification",
    ],
    risk_factors=[
        "Fed pivot surprises can un-invert the curve without recession",
        "Fiscal stimulus can override recession signal",
        "Non-US demand for Treasuries distorts the spread signal",
        "Short squeezes in high-beta equity can be violent",
    ],
    signal_strength=SignalStrength.STRONG,
    tags=["macro", "rates", "recession", "yield_curve", "risk_off"],
)


# ---------------------------------------------------------------------------
# Template 2 – Credit-Equity Dislocation Fade
# ---------------------------------------------------------------------------
CREDIT_EQUITY_DISLOCATION = HypothesisTemplate(
    name="credit_equity_dislocations",
    description=(
        "When equity markets rally but credit spreads are simultaneously widening, "
        "fade the equity rally — credit leads equity at cycle turns."
    ),
    thesis=(
        "Credit markets are dominated by institutional participants with superior "
        "fundamental analysis and less retail noise. When HY or IG spreads widen "
        "while equities float higher, it typically reflects credit markets pricing "
        "in rising default risk that equity markets have not yet digested. "
        "This dislocation is mean-reverting: either equity catches down or credit "
        "tightens. Historically, equity catches down ~70% of the time."
    ),
    primary_asset_class=AssetClass.EQUITY,
    secondary_asset_class=AssetClass.CREDIT,
    direction=Direction.SHORT,
    macro_signals=[
        "high_yield_oas_5d_change",
        "investment_grade_oas_5d_change",
        "credit_default_swap_index_cdx_hy",
        "leveraged_loan_index_spread",
        "bank_lending_standards_net_tightening",
    ],
    micro_signals=[
        "spx_5d_return",
        "spx_rsi_14d",
        "equity_earnings_revisions_net",
        "financial_sector_relative_strength",
        "high_yield_etf_hyg_vs_spx_ratio",
    ],
    entry_conditions=[
        EntryCondition(
            signal="spx_5d_return",
            operator="gt",
            threshold=0.02,
            description="S&P 500 up >2% over trailing 5 days",
        ),
        EntryCondition(
            signal="high_yield_oas_5d_change",
            operator="gt",
            threshold=30,
            description="HY OAS widened >30 bps over same 5-day period",
        ),
        EntryCondition(
            signal="spx_rsi_14d",
            operator="gt",
            threshold=65,
            description="Equity momentum elevated — momentum divergence confirmed",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="high_yield_oas_5d_change < -20",
            description="Credit spreads mean-reverting — dislocation resolved bullishly",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="spx_gain_from_entry > 5%",
            description="Hard stop — equity continuing higher, thesis invalidated",
            is_hard_stop=True,
        ),
        ExitCondition(
            trigger="spx_drawdown_from_entry > 8%",
            description="Take profit — equity catching down to credit signal",
            is_hard_stop=False,
        ),
    ],
    sizing_guidance=(
        "Use 30–50% of normal equity short sizing. Signal is timing-sensitive; "
        "scale out 50% of position within 10 trading days regardless of P&L."
    ),
    expected_holding_period="5–20 trading days",
    historical_analogues=[
        "Q4 2018 credit-equity divergence",
        "Early 2020 credit warning before COVID crash",
        "Q3 2022 credit-equity dislocation",
    ],
    risk_factors=[
        "Central bank intervention can compress spreads rapidly",
        "Equity buybacks can levitate prices despite deteriorating credit",
        "Short-term flows can sustain dislocation longer than expected",
    ],
    signal_strength=SignalStrength.MODERATE,
    tags=["credit", "equity", "divergence", "fade", "cross_asset"],
)


# ---------------------------------------------------------------------------
# Template 3 – Dollar Carry Unwind
# ---------------------------------------------------------------------------
DOLLAR_CARRY_UNWIND = HypothesisTemplate(
    name="dollar_carry_unwind",
    description=(
        "When the USD strengthens rapidly alongside risk-off signals, "
        "unwind carry trades funded in low-yield currencies (JPY, CHF) "
        "and exit EM long positions."
    ),
    thesis=(
        "Carry trades funded in JPY/CHF are structurally long risk: long EM bonds, "
        "equities, or high-yield FX. When the USD surges in a risk-off environment, "
        "carry traders face margin calls and must unwind simultaneously, "
        "creating a reflexive amplification loop. The signal is the co-occurrence "
        "of USD strength with VIX spike and EM equity/FX weakness."
    ),
    primary_asset_class=AssetClass.FX,
    secondary_asset_class=AssetClass.EQUITY,
    direction=Direction.SHORT,
    macro_signals=[
        "dxy_index_5d_return",
        "vix_5d_change",
        "usdjpy_5d_return",
        "usdchf_5d_return",
        "em_fx_index_5d_return",
        "global_risk_appetite_index",
    ],
    micro_signals=[
        "eem_etf_5d_return",
        "carry_trade_index_return",
        "jpy_implied_vol_1m",
        "em_bond_spread_change",
        "cftc_jpy_net_speculative_position",
    ],
    entry_conditions=[
        EntryCondition(
            signal="dxy_index_5d_return",
            operator="gt",
            threshold=0.015,
            description="DXY up >1.5% in 5 days",
        ),
        EntryCondition(
            signal="vix_5d_change",
            operator="gt",
            threshold=5.0,
            description="VIX has risen >5 points in 5 days",
        ),
        EntryCondition(
            signal="em_fx_index_5d_return",
            operator="lt",
            threshold=-0.02,
            description="EM FX basket down >2% confirming carry stress",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="dxy_index reverses more than 1% from peak",
            description="Dollar peak potentially in, carry rebuilding",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="vix drops below entry level",
            description="Risk-off impulse exhausted",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="position loss > 2% of NAV",
            description="Hard stop — carry unwind not materializing",
            is_hard_stop=True,
        ),
    ],
    sizing_guidance=(
        "Size to 1.5x normal FX position. Express via long USD/JPY put spreads "
        "or short EM ETF. Reduce on any Fed dovish pivot signal."
    ),
    expected_holding_period="3–15 trading days",
    historical_analogues=[
        "August 2015 China devaluation carry unwind",
        "March 2020 dollar squeeze and EM selloff",
        "Q3 2022 DXY surge and EM stress",
    ],
    risk_factors=[
        "BOJ intervention can rapidly reverse USD/JPY",
        "Fed pivot can collapse USD carry unwind thesis",
        "Geopolitical shock can complicate safe-haven flows",
    ],
    signal_strength=SignalStrength.STRONG,
    tags=["fx", "carry", "dollar", "em", "risk_off", "unwind"],
)


# ---------------------------------------------------------------------------
# Template 4 – Macro-Momentum Alignment
# ---------------------------------------------------------------------------
MACRO_MOMENTUM_ALIGNMENT = HypothesisTemplate(
    name="macro_momentum_alignment",
    description=(
        "Go long risk assets when macro trend (improving PMI, earnings revisions up) "
        "and equity momentum signals are simultaneously aligned bullishly."
    ),
    thesis=(
        "The most persistent equity rallies occur when macro fundamentals and "
        "price momentum reinforce each other. Macro trend confirms that the "
        "fundamental backdrop is improving; momentum confirms that capital flows "
        "are rotating in. When both align, the probability of a sustained trend "
        "is materially higher than when only one signal is present."
    ),
    primary_asset_class=AssetClass.EQUITY,
    secondary_asset_class=None,
    direction=Direction.LONG,
    macro_signals=[
        "global_composite_pmi_3m_trend",
        "earnings_revision_ratio_3m",
        "global_trade_volume_yoy",
        "credit_impulse_g4",
        "oecd_leading_indicator_momentum",
    ],
    micro_signals=[
        "spx_200d_ma_crossover",
        "spx_momentum_12m_minus_1m",
        "advance_decline_line_trend",
        "new_52w_highs_vs_lows_ratio",
        "sector_rotation_breadth_score",
    ],
    entry_conditions=[
        EntryCondition(
            signal="global_composite_pmi_3m_trend",
            operator="gt",
            threshold=0.5,
            description="Global composite PMI trending up (3m slope > 0.5)",
        ),
        EntryCondition(
            signal="earnings_revision_ratio_3m",
            operator="gt",
            threshold=1.2,
            description="Upgrades outnumber downgrades by >20% over 3 months",
        ),
        EntryCondition(
            signal="spx_200d_ma_crossover",
            operator="gt",
            threshold=0,
            description="SPX trading above its 200-day moving average",
        ),
        EntryCondition(
            signal="spx_momentum_12m_minus_1m",
            operator="gt",
            threshold=0.05,
            description="12m-1m momentum factor positive and >5%",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="global_composite_pmi_3m_trend < 0 for 2 consecutive months",
            description="Macro momentum rolling over",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="spx breaks below 200d MA with >1% close",
            description="Technical trend broken",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="drawdown from peak > 8%",
            description="Trailing stop to protect trend gains",
            is_hard_stop=True,
        ),
    ],
    sizing_guidance=(
        "Full risk-on sizing. Can leverage to 1.2x in strong alignment. "
        "Favor high-beta cyclicals and small caps in the equity sleeve."
    ),
    expected_holding_period="1–6 months",
    historical_analogues=[
        "2009 recovery rally (PMI + momentum aligning from trough)",
        "2016 reflation trade post-election",
        "2020-21 post-COVID macro + momentum surge",
    ],
    risk_factors=[
        "Macro data can be revised lower after the fact",
        "Geopolitical shocks can override macro momentum",
        "Valuation risk when entering late in the alignment",
    ],
    signal_strength=SignalStrength.VERY_STRONG,
    tags=["macro", "momentum", "equity", "trend", "risk_on", "alignment"],
)


# ---------------------------------------------------------------------------
# Template 5 – Vol Regime Entry
# ---------------------------------------------------------------------------
VOL_REGIME_ENTRY = HypothesisTemplate(
    name="vol_regime_entry",
    description=(
        "Enter long risk positions when implied volatility is in the bottom "
        "quintile of its 1-year distribution AND macro conditions are stable — "
        "buying cheap optionality and riding low-vol regimes."
    ),
    thesis=(
        "Low IV percentile indicates the market is pricing in low uncertainty. "
        "When macro is also stable (no recession risk, stable growth), "
        "this low-vol environment tends to persist and equity carry is attractive. "
        "Additionally, low IV makes options cheap — an opportunity to express "
        "directional views with defined risk via long calls or call spreads."
    ),
    primary_asset_class=AssetClass.VOL,
    secondary_asset_class=AssetClass.EQUITY,
    direction=Direction.LONG,
    macro_signals=[
        "vix_1y_percentile",
        "macro_stability_composite_score",
        "fed_policy_uncertainty_index",
        "global_epu_index",
        "yield_curve_slope_3m10y",
    ],
    micro_signals=[
        "vix_level",
        "vix_term_structure_slope_1m3m",
        "skew_index",
        "realized_vol_20d",
        "vol_of_vol_vvix",
    ],
    entry_conditions=[
        EntryCondition(
            signal="vix_1y_percentile",
            operator="lt",
            threshold=20,
            description="VIX in bottom 20th percentile of trailing 1-year distribution",
        ),
        EntryCondition(
            signal="macro_stability_composite_score",
            operator="gt",
            threshold=0.6,
            description="Composite macro stability score above 0.6 (scale 0-1)",
        ),
        EntryCondition(
            signal="vix_term_structure_slope_1m3m",
            operator="gt",
            threshold=0,
            description="VIX term structure in contango (normal, not inverted)",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="vix_1y_percentile > 50",
            description="Vol regime has shifted, low-vol thesis no longer valid",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="macro_stability_composite_score < 0.4",
            description="Macro deteriorating, reducing long risk exposure",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="vix single-day spike > 5 points",
            description="Hard stop on vol spike — possible regime break",
            is_hard_stop=True,
        ),
    ],
    sizing_guidance=(
        "Express via long equity + long cheap index puts for tail hedge. "
        "Or long gamma via straddles in low-vol environment (mean-reversion play). "
        "Keep option premium spend < 0.5% of NAV per month."
    ),
    expected_holding_period="2–8 weeks",
    historical_analogues=[
        "2017 persistent low-vol regime",
        "Late 2019 vol compression before COVID",
        "2021 suppressed vol post-stimulus",
    ],
    risk_factors=[
        "Vol can stay suppressed longer than expected (vol seller crowding)",
        "Sudden regime breaks are not forecastable from vol alone",
        "Correlation between vol and macro can break down",
    ],
    signal_strength=SignalStrength.MODERATE,
    tags=["volatility", "regime", "options", "low_vol", "carry"],
)


# ---------------------------------------------------------------------------
# Template 6 – Cross-Asset Breakout
# ---------------------------------------------------------------------------
CROSS_ASSET_BREAKOUT = HypothesisTemplate(
    name="cross_asset_breakout",
    description=(
        "Enter a directional position when multiple asset classes simultaneously "
        "break out of technical consolidation ranges — confirming a macro regime shift."
    ),
    thesis=(
        "Simultaneous breakouts across uncorrelated asset classes (equities, bonds, "
        "commodities, FX) are rare and signal a genuine macro regime shift rather "
        "than noise. When gold, equities, and breakeven inflation all break out "
        "simultaneously to the upside, for example, the reflation trade is confirmed. "
        "The simultaneity reduces the probability of a false breakout in any single asset."
    ),
    primary_asset_class=AssetClass.EQUITY,
    secondary_asset_class=AssetClass.COMMODITY,
    direction=Direction.LONG_SHORT,
    macro_signals=[
        "global_breakout_score_composite",
        "asset_class_correlation_shift_30d",
        "macro_regime_transition_probability",
        "cross_asset_momentum_dispersion",
    ],
    micro_signals=[
        "spx_52w_breakout_signal",
        "gold_52w_breakout_signal",
        "tlt_52w_breakout_signal",
        "oil_52w_breakout_signal",
        "dxy_52w_breakout_signal",
        "breakout_asset_count",
    ],
    entry_conditions=[
        EntryCondition(
            signal="breakout_asset_count",
            operator="gt",
            threshold=3,
            description="At least 4 of 6 tracked asset classes breaking out simultaneously",
        ),
        EntryCondition(
            signal="asset_class_correlation_shift_30d",
            operator="gt",
            threshold=0.15,
            description="30-day correlation shift > 0.15 indicating new regime",
        ),
        EntryCondition(
            signal="macro_regime_transition_probability",
            operator="gt",
            threshold=0.65,
            description="Regime model assigns >65% probability to transition",
        ),
    ],
    exit_conditions=[
        ExitCondition(
            trigger="breakout_asset_count drops below 2",
            description="Breakout not confirmed — assets reverting to range",
            is_hard_stop=False,
        ),
        ExitCondition(
            trigger="asset reverses below breakout level by >2%",
            description="Failed breakout — exit immediately",
            is_hard_stop=True,
        ),
        ExitCondition(
            trigger="position age > 30 trading days",
            description="Regime trades must reassess at 30 days",
            is_hard_stop=False,
        ),
    ],
    sizing_guidance=(
        "Equal-weight across breaking asset classes, rebalanced weekly. "
        "Cap single-asset exposure at 25% of the cross-asset sleeve. "
        "Reduce by 50% if any two assets show failed breakouts."
    ),
    expected_holding_period="2–6 weeks",
    historical_analogues=[
        "2022 simultaneous breakouts in USD, rates, oil (inflation regime)",
        "2020 post-COVID simultaneous equity and gold breakout (stimulus regime)",
        "2016 Trump election cross-asset reflation breakout",
    ],
    risk_factors=[
        "False breakouts during low-liquidity periods (holiday, thin markets)",
        "Correlation convergence can reverse rapidly on policy surprise",
        "Defining 'breakout' consistently across asset classes is non-trivial",
    ],
    signal_strength=SignalStrength.STRONG,
    tags=["breakout", "cross_asset", "regime_shift", "macro", "multi_asset"],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MACRO_MICRO_FUSION_TEMPLATES: dict[str, HypothesisTemplate] = {
    "yield_curve_inversion_trade": YIELD_CURVE_INVERSION_TRADE,
    "credit_equity_dislocations": CREDIT_EQUITY_DISLOCATION,
    "dollar_carry_unwind": DOLLAR_CARRY_UNWIND,
    "macro_momentum_alignment": MACRO_MOMENTUM_ALIGNMENT,
    "vol_regime_entry": VOL_REGIME_ENTRY,
    "cross_asset_breakout": CROSS_ASSET_BREAKOUT,
}


def get_template(name: str) -> HypothesisTemplate:
    """Retrieve a hypothesis template by name."""
    if name not in MACRO_MICRO_FUSION_TEMPLATES:
        available = list(MACRO_MICRO_FUSION_TEMPLATES.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return MACRO_MICRO_FUSION_TEMPLATES[name]


def list_templates() -> list[str]:
    """Return all registered template names."""
    return list(MACRO_MICRO_FUSION_TEMPLATES.keys())


def filter_by_tag(tag: str) -> list[HypothesisTemplate]:
    """Return templates containing the given tag."""
    return [t for t in MACRO_MICRO_FUSION_TEMPLATES.values() if tag in t.tags]


def filter_by_asset_class(asset_class: AssetClass) -> list[HypothesisTemplate]:
    """Return templates whose primary asset class matches."""
    return [
        t for t in MACRO_MICRO_FUSION_TEMPLATES.values()
        if t.primary_asset_class == asset_class
    ]
