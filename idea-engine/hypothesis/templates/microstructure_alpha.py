"""
microstructure_alpha.py
Hypothesis templates derived from market microstructure signals.
Each template captures an edge rooted in order flow, liquidity dynamics,
informed trading detection, or market impact models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MicroEdgeType(str, Enum):
    INFORMED_FLOW = "informed_flow"
    LIQUIDITY_PROVISION = "liquidity_provision"
    ORDER_IMBALANCE = "order_imbalance"
    ADVERSE_SELECTION = "adverse_selection"
    MARKET_IMPACT = "market_impact"
    DARK_POOL = "dark_pool"
    PIN_BASED = "pin_based"


class TimeHorizon(str, Enum):
    INTRADAY = "intraday"          # minutes to hours
    SHORT_TERM = "short_term"      # 1–5 days
    MEDIUM_TERM = "medium_term"    # 1–4 weeks


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FADE = "fade"
    FOLLOW = "follow"
    NEUTRAL = "neutral"


@dataclass
class MicrostructureSignal:
    name: str
    description: str
    typical_range: tuple[float, float]
    interpretation: str


@dataclass
class MicroEntryRule:
    signal: str
    condition: str          # human-readable condition
    threshold: Any
    logic_operator: str     # "AND", "OR" (relative to next rule)


@dataclass
class MicroExitRule:
    trigger: str
    rationale: str
    is_stop: bool = False


@dataclass
class MicrostructureTemplate:
    name: str
    edge_type: MicroEdgeType
    description: str
    theoretical_basis: str
    primary_signals: list[MicrostructureSignal]
    entry_rules: list[MicroEntryRule]
    exit_rules: list[MicroExitRule]
    side: PositionSide
    time_horizon: TimeHorizon
    implementation_notes: str
    data_requirements: list[str]
    known_limitations: list[str]
    signal_decay_halflife: str
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template 1 – Toxic Flow Fade
# ---------------------------------------------------------------------------
TOXIC_FLOW_FADE = MicrostructureTemplate(
    name="toxic_flow_fade",
    edge_type=MicroEdgeType.ADVERSE_SELECTION,
    description=(
        "After a period of high VPIN (Volume-synchronized Probability of Informed Trading), "
        "fade the next large directional move — informed flow exhausts itself and "
        "market makers widen spreads, causing price reversion."
    ),
    theoretical_basis=(
        "VPIN (Easley, de Prado & O'Hara 2012) estimates the probability of informed "
        "trading by comparing buy- and sell-initiated volume imbalances bucketed "
        "by total volume rather than time. When VPIN is elevated (>0.7), informed "
        "traders have been dominant. After their information is incorporated, "
        "subsequent moves are more likely noise-driven and mean-reverting. "
        "Market makers temporarily withdraw liquidity post-high-VPIN, creating "
        "spread widening and subsequent reversion as they return."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="vpin",
            description="Volume-synchronized Probability of Informed Trading",
            typical_range=(0.2, 0.9),
            interpretation="Values > 0.65 indicate elevated informed trading probability",
        ),
        MicrostructureSignal(
            name="bid_ask_spread_zscore",
            description="Z-score of current bid-ask spread vs 20-bucket rolling mean",
            typical_range=(-3.0, 3.0),
            interpretation="High z-score = MMs withdrawing liquidity = post-informed-flow",
        ),
        MicrostructureSignal(
            name="order_flow_toxicity_index",
            description="Composite of VPIN, spread, and depth imbalance",
            typical_range=(0.0, 1.0),
            interpretation="Above 0.7 = toxic flow regime",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="vpin",
            condition="VPIN crosses above 0.70 then falls back below 0.65",
            threshold=0.65,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="bid_ask_spread_zscore",
            condition="Spread z-score > 1.5 (market makers have widened)",
            threshold=1.5,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="price_direction_last_10_buckets",
            condition="Last 10 volume buckets show directional price move > 0.5%",
            threshold=0.005,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="VPIN drops below 0.40",
            rationale="Informed flow completely exhausted, edge dissolved",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Price retraces 50% of the initial informed move",
            rationale="Target reached",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="VPIN re-spikes above 0.75 while position is open",
            rationale="New informed flow contradicts fade thesis — hard stop",
            is_stop=True,
        ),
    ],
    side=PositionSide.FADE,
    time_horizon=TimeHorizon.INTRADAY,
    implementation_notes=(
        "Compute VPIN using 50-bucket rolling window, 1/50th of daily volume per bucket. "
        "Use tick data with trade direction classified via Lee-Ready algorithm. "
        "Fade implemented as limit orders at current mid ± 0.1 ATR. "
        "Do NOT use market orders post-high-VPIN — spreads are wide."
    ),
    data_requirements=[
        "Level 1 tick data with bid/ask at trade time",
        "Trade volume and direction (buy/sell) classification",
        "Intraday volume profile for bucket sizing",
        "Real-time bid-ask spread",
    ],
    known_limitations=[
        "VPIN calculation is sensitive to bucket size choice",
        "Lee-Ready misclassification rate ~15% can distort VPIN",
        "Not suitable for illiquid stocks where noise dominates",
        "HFT activity can artificially inflate VPIN",
    ],
    signal_decay_halflife="30–120 minutes",
    tags=["vpin", "informed_flow", "fade", "microstructure", "intraday"],
)


# ---------------------------------------------------------------------------
# Template 2 – Order Book Imbalance Momentum
# ---------------------------------------------------------------------------
ORDER_BOOK_IMBALANCE_MOMENTUM = MicrostructureTemplate(
    name="order_book_imbalance_momentum",
    edge_type=MicroEdgeType.ORDER_IMBALANCE,
    description=(
        "When persistent order book imbalance (more bid size than ask size) is "
        "observed across multiple levels, follow the direction — latent demand "
        "predicts short-term price appreciation."
    ),
    theoretical_basis=(
        "Order book imbalance (OBI) measures the asymmetry between available "
        "bid and ask liquidity. A persistent imbalance toward bids implies "
        "latent buying pressure that will absorb sells and push price higher "
        "once sufficient market orders arrive. Cont, Kukanov & Stoikov (2014) "
        "showed OBI is a strong short-term predictor of mid-price changes "
        "at the 1-second to 1-minute horizon."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="obi_l1",
            description="Level 1 order book imbalance: (bid_size - ask_size) / (bid_size + ask_size)",
            typical_range=(-1.0, 1.0),
            interpretation="Positive = more bid pressure; negative = ask pressure",
        ),
        MicrostructureSignal(
            name="obi_l5",
            description="Weighted OBI across top 5 levels",
            typical_range=(-1.0, 1.0),
            interpretation="Deeper imbalance signal, slower but more robust",
        ),
        MicrostructureSignal(
            name="obi_persistence_score",
            description="Fraction of last 60 snapshots with same-sign OBI > 0.3",
            typical_range=(0.0, 1.0),
            interpretation="High persistence = systematic pressure, not noise",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="obi_l1",
            condition="OBI_L1 > 0.40 for at least 20 consecutive 1-second snapshots",
            threshold=0.40,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="obi_persistence_score",
            condition="OBI persistence score > 0.70",
            threshold=0.70,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="spread_normalized",
            condition="Spread < 2x median spread (not artificially wide)",
            threshold=2.0,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="OBI_L1 crosses zero (imbalance reverses)",
            rationale="Pressure exhausted or reversed",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Holding period exceeds 5 minutes",
            rationale="OBI signal decays rapidly beyond 5 minutes",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Price moves against position by 0.5x ATR(5min)",
            rationale="Stop out — imbalance not converting to price impact",
            is_stop=True,
        ),
    ],
    side=PositionSide.FOLLOW,
    time_horizon=TimeHorizon.INTRADAY,
    implementation_notes=(
        "Requires sub-second order book snapshots. Implement OBI as a streaming "
        "EWMA to reduce noise. Best applied to large-cap liquid stocks with "
        "narrow spreads. Avoid first and last 30 minutes of session."
    ),
    data_requirements=[
        "Level 2 order book data (top 5 levels minimum)",
        "1-second or faster snapshots",
        "Bid and ask sizes at each level",
    ],
    known_limitations=[
        "Spoofing can create false imbalance signals",
        "Signal degrades significantly after ~3 minutes",
        "Not applicable to opening auction or thin markets",
    ],
    signal_decay_halflife="1–5 minutes",
    tags=["order_book", "imbalance", "momentum", "hft", "intraday", "l2"],
)


# ---------------------------------------------------------------------------
# Template 3 – Spread Compression Entry
# ---------------------------------------------------------------------------
SPREAD_COMPRESSION_ENTRY = MicrostructureTemplate(
    name="spread_compression_entry",
    edge_type=MicroEdgeType.LIQUIDITY_PROVISION,
    description=(
        "When the bid-ask spread narrows after a period of widening, "
        "it signals improved liquidity — enter directional positions "
        "as market impact costs drop and informed traders return."
    ),
    theoretical_basis=(
        "Bid-ask spreads widen when market makers face elevated adverse selection "
        "risk or inventory imbalance. When spreads compress from a widened state, "
        "it indicates MMs are comfortable providing liquidity again — risk appetite "
        "is returning. This post-compression window offers better execution quality "
        "and is often followed by directional price moves as deferred orders execute."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="spread_percentile_10d",
            description="Current spread as percentile of 10-day rolling distribution",
            typical_range=(0.0, 100.0),
            interpretation="Below 25th pctile after being above 75th = compression signal",
        ),
        MicrostructureSignal(
            name="spread_velocity",
            description="Rate of change in spread: (current - 10min_ago) / 10min_ago",
            typical_range=(-0.5, 0.5),
            interpretation="Negative value = spread compressing",
        ),
        MicrostructureSignal(
            name="depth_at_bbo",
            description="Total size at best bid and offer",
            typical_range=(100, 50000),
            interpretation="Increasing depth alongside spread compression = strong liquidity return",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="spread_percentile_10d",
            condition="Spread was above 80th pctile in past 2 hours, now below 40th",
            threshold=40.0,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="spread_velocity",
            condition="Spread velocity < -0.10 (compressing at >10% rate)",
            threshold=-0.10,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="depth_at_bbo",
            condition="BBO depth increased >30% vs 1-hour average",
            threshold=1.30,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="Spread widens back above 60th pctile",
            rationale="Liquidity deteriorating — exit before adverse selection risk increases",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Target price level reached (0.5x ATR from entry)",
            rationale="Profit target",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Spread re-widens to > 90th pctile",
            rationale="Hard stop — liquidity crisis resuming",
            is_stop=True,
        ),
    ],
    side=PositionSide.FOLLOW,
    time_horizon=TimeHorizon.INTRADAY,
    implementation_notes=(
        "Use spread compression as a timing tool within a broader directional view. "
        "Determine direction from OBI or recent order flow before applying this entry trigger. "
        "Works best in ETFs and large-cap stocks with consistent liquidity patterns."
    ),
    data_requirements=[
        "Real-time bid/ask prices and sizes (Level 1 minimum)",
        "10-day historical spread distribution for percentile calculation",
        "BBO depth at sub-minute frequency",
    ],
    known_limitations=[
        "Spread compression can be mechanical (end of auction, algorithmic MM reset)",
        "Low-float stocks have erratic spread patterns",
        "After-hours and pre-market spreads are structurally wider — exclude",
    ],
    signal_decay_halflife="10–60 minutes",
    tags=["spread", "liquidity", "market_making", "entry_timing", "intraday"],
)


# ---------------------------------------------------------------------------
# Template 4 – Dark Pool Print Follow
# ---------------------------------------------------------------------------
DARK_POOL_PRINT_FOLLOW = MicrostructureTemplate(
    name="dark_pool_print_follow",
    edge_type=MicroEdgeType.DARK_POOL,
    description=(
        "When a large dark pool print occurs at a technically significant price level "
        "(support/resistance, round number, 52-week high/low), follow the direction "
        "of the print — it signals institutional accumulation or distribution."
    ),
    theoretical_basis=(
        "Dark pool prints represent institutional-sized trades executed off-exchange "
        "to minimize market impact. When these prints cluster at key technical levels, "
        "they reveal institutional intent. A large buy print at a long-term resistance "
        "level suggests institutions are breaking through deliberately; a sell print "
        "at support suggests distribution. The 'follow' edge arises because institutions "
        "typically have multi-day execution programs — the print is the first visible "
        "tranche of a larger order."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="dark_pool_print_size_percentile",
            description="Print size as percentile of 30-day dark pool print distribution for that stock",
            typical_range=(0.0, 100.0),
            interpretation="Above 90th pctile = institutional-sized, actionable",
        ),
        MicrostructureSignal(
            name="key_level_proximity",
            description="Distance of print price from nearest technical level (% of price)",
            typical_range=(0.0, 0.05),
            interpretation="Below 0.3% = at a key level",
        ),
        MicrostructureSignal(
            name="dark_pool_volume_ratio",
            description="Dark pool volume / total volume in the session",
            typical_range=(0.1, 0.6),
            interpretation="Above 0.35 = elevated institutional participation",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="dark_pool_print_size_percentile",
            condition="Print size > 90th pctile of 30-day distribution",
            threshold=90.0,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="key_level_proximity",
            condition="Print within 0.3% of a defined technical key level",
            threshold=0.003,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="dark_pool_volume_ratio",
            condition="Dark pool ratio > 0.30 for the session",
            threshold=0.30,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="Price moves 1.5x ATR(20d) in follow direction",
            rationale="Profit target — institutional tranche likely complete",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="No follow-through within 2 trading days",
            rationale="Print was likely isolated hedging, not accumulation",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Price breaks key level on high lit volume in opposite direction",
            rationale="Hard stop — institutional order reversed or fake print",
            is_stop=True,
        ),
    ],
    side=PositionSide.FOLLOW,
    time_horizon=TimeHorizon.SHORT_TERM,
    implementation_notes=(
        "Data source: FINRA ATS transparency data or commercial dark pool feeds. "
        "Key levels must be pre-defined (52w high/low, round numbers, prior pivots). "
        "Filter out prints that are likely block crosses or internalized retail flow "
        "by checking if print exceeds 1% of average daily volume."
    ),
    data_requirements=[
        "FINRA ATS trade reporting (T+1 delay) or real-time dark pool feed",
        "Pre-computed technical key level database",
        "30-day rolling dark pool print size distribution per symbol",
        "Session-level dark pool vs lit volume breakdown",
    ],
    known_limitations=[
        "FINRA data is T+1, limiting real-time application",
        "Cannot distinguish between genuine accumulation and hedging prints",
        "Dark pool venue characteristics vary — not all prints are equal",
        "Regulation changes can reduce dark pool availability",
    ],
    signal_decay_halflife="1–3 trading days",
    tags=["dark_pool", "institutional", "block_trade", "follow", "short_term"],
)


# ---------------------------------------------------------------------------
# Template 5 – Kyle Lambda Regime
# ---------------------------------------------------------------------------
KYLE_LAMBDA_REGIME = MicrostructureTemplate(
    name="kyle_lambda_regime",
    edge_type=MicroEdgeType.MARKET_IMPACT,
    description=(
        "Dynamically size positions based on Kyle's lambda (price impact coefficient): "
        "size up when lambda is low (liquid market) and reduce when lambda is high "
        "(illiquid — every trade moves the market more)."
    ),
    theoretical_basis=(
        "Kyle (1985) showed that price impact is linear in signed order flow: "
        "ΔP = λ * OrderFlow. A low λ means large trades have minimal price impact; "
        "a high λ means the market is thin and orders are market-moving. "
        "Trading with high λ erodes alpha through adverse execution. "
        "By estimating λ in real-time via OLS regression of price changes on "
        "signed volume, we can dynamically adjust position sizing to protect "
        "execution quality and alpha preservation."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="kyle_lambda_estimate",
            description="OLS estimate of price impact per unit signed volume (10-minute rolling)",
            typical_range=(1e-8, 1e-5),
            interpretation="Higher = more price impact = less liquid = reduce size",
        ),
        MicrostructureSignal(
            name="kyle_lambda_percentile_30d",
            description="Current lambda as percentile of 30-day rolling distribution",
            typical_range=(0.0, 100.0),
            interpretation="Below 25th pctile = very liquid; above 75th = illiquid",
        ),
        MicrostructureSignal(
            name="lambda_trend_60min",
            description="1-hour trend in lambda: positive = liquidity deteriorating",
            typical_range=(-0.5, 0.5),
            interpretation="Rising lambda = reduce size; falling lambda = increase size",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="kyle_lambda_percentile_30d",
            condition="Lambda in bottom 25th pctile — maximum liquidity",
            threshold=25.0,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="lambda_trend_60min",
            condition="Lambda trend is flat or declining (not deteriorating)",
            threshold=0.0,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="Kyle lambda crosses above 75th pctile",
            rationale="Liquidity has deteriorated — reduce position to limit impact cost",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Lambda spikes >3x its 30-day median in a single 10-min window",
            rationale="Flash illiquidity event — stop and exit immediately",
            is_stop=True,
        ),
    ],
    side=PositionSide.NEUTRAL,
    time_horizon=TimeHorizon.INTRADAY,
    implementation_notes=(
        "Estimate lambda by running OLS regression: ΔP_t = α + λ * V_t + ε_t "
        "where V_t is signed volume (positive for buys, negative for sells) "
        "over a rolling 50-trade or 10-minute window, whichever is longer. "
        "Use as a position sizing multiplier: size = base_size * (lambda_target / lambda_current)."
        "Cap multiplier between 0.25x and 1.5x base size."
    ),
    data_requirements=[
        "Trade-level data with direction classification",
        "Price at trade time",
        "Signed volume estimates",
        "Rolling OLS capability (numpy/scipy)",
    ],
    known_limitations=[
        "Lambda estimate is noisy in thin trading periods",
        "Assumes linear price impact — not valid for very large orders",
        "Regime shifts (news events) break the lambda estimation",
        "Cannot distinguish between temporary and permanent impact",
    ],
    signal_decay_halflife="10–30 minutes",
    tags=["kyle_lambda", "market_impact", "sizing", "liquidity", "execution"],
)


# ---------------------------------------------------------------------------
# Template 6 – PIN-Informed Follow
# ---------------------------------------------------------------------------
PIN_INFORMED_FOLLOW = MicrostructureTemplate(
    name="pin_informed_follow",
    edge_type=MicroEdgeType.PIN_BASED,
    description=(
        "When the Probability of Informed Trading (PIN) is elevated, "
        "follow the net order flow direction — informed traders with "
        "superior information are driving the imbalance."
    ),
    theoretical_basis=(
        "PIN (Easley, Kiefer, O'Hara & Paperman 1996) estimates the fraction of "
        "order flow arising from informed traders using a structural MLE model "
        "of buy/sell arrivals. High PIN indicates that a significant fraction "
        "of trades are by information-motivated traders. Following their net direction "
        "exploits the information asymmetry before it is fully reflected in prices. "
        "PIN is typically estimated daily via MLE; intraday VPIN serves as a proxy."
    ),
    primary_signals=[
        MicrostructureSignal(
            name="pin_estimate_daily",
            description="Daily MLE-estimated PIN (Easley et al. structural model)",
            typical_range=(0.05, 0.60),
            interpretation="Above 0.35 = high informed trading probability",
        ),
        MicrostructureSignal(
            name="net_order_flow_direction",
            description="Sign of (buy-initiated volume - sell-initiated volume) over estimation window",
            typical_range=(-1.0, 1.0),
            interpretation="+1 = net buying pressure; -1 = net selling pressure",
        ),
        MicrostructureSignal(
            name="pin_trend_5d",
            description="5-day trend in daily PIN estimates",
            typical_range=(-0.2, 0.2),
            interpretation="Rising PIN = information event building",
        ),
    ],
    entry_rules=[
        MicroEntryRule(
            signal="pin_estimate_daily",
            condition="Daily PIN > 0.35 (above typical cross-sectional median by >1 SD)",
            threshold=0.35,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="net_order_flow_direction",
            condition="Net order flow has been consistently positive (or negative) for 3+ sessions",
            threshold=0.60,
            logic_operator="AND",
        ),
        MicroEntryRule(
            signal="pin_trend_5d",
            condition="PIN rising over past 5 days (information event accumulating)",
            threshold=0.02,
            logic_operator="AND",
        ),
    ],
    exit_rules=[
        MicroExitRule(
            trigger="PIN drops below 0.20",
            rationale="Informed trading activity has subsided — edge gone",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Net order flow reverses direction for 2 consecutive sessions",
            rationale="Informed traders switching sides or exiting",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Public news announcement explaining the flow",
            rationale="Information is now public — PIN edge fully captured",
            is_stop=False,
        ),
        MicroExitRule(
            trigger="Position loss > 1.5% of NAV",
            rationale="Hard stop — PIN signal not leading to price impact",
            is_stop=True,
        ),
    ],
    side=PositionSide.FOLLOW,
    time_horizon=TimeHorizon.SHORT_TERM,
    implementation_notes=(
        "PIN estimation requires daily buy/sell volume classification via Lee-Ready. "
        "MLE optimization: maximize L(θ) = Σ log P(B_t, S_t | α, δ, μ, ε_b, ε_s). "
        "Use scipy.optimize.minimize with bounds. "
        "For intraday use, substitute with rolling VPIN as PIN proxy. "
        "Cross-sectional ranking of PIN (z-score within sector) improves signal quality."
    ),
    data_requirements=[
        "Daily buy/sell volume split (Lee-Ready classified)",
        "Historical daily trade count by direction for MLE",
        "scipy for MLE optimization",
        "News event calendar to contextualize PIN spikes",
    ],
    known_limitations=[
        "PIN MLE can have convergence issues — multiple local maxima",
        "PIN model assumes Poisson arrivals which may not hold",
        "Structural breaks (market regime changes) invalidate priors",
        "T+1 daily estimation limits real-time application",
        "Potential confound: high PIN around earnings is mechanical",
    ],
    signal_decay_halflife="1–5 trading days",
    tags=["pin", "informed_trading", "order_flow", "follow", "microstructure"],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MICROSTRUCTURE_ALPHA_TEMPLATES: dict[str, MicrostructureTemplate] = {
    "toxic_flow_fade": TOXIC_FLOW_FADE,
    "order_book_imbalance_momentum": ORDER_BOOK_IMBALANCE_MOMENTUM,
    "spread_compression_entry": SPREAD_COMPRESSION_ENTRY,
    "dark_pool_print_follow": DARK_POOL_PRINT_FOLLOW,
    "kyle_lambda_regime": KYLE_LAMBDA_REGIME,
    "pin_informed_follow": PIN_INFORMED_FOLLOW,
}


def get_template(name: str) -> MicrostructureTemplate:
    if name not in MICROSTRUCTURE_ALPHA_TEMPLATES:
        raise KeyError(f"Template '{name}' not found. Available: {list(MICROSTRUCTURE_ALPHA_TEMPLATES)}")
    return MICROSTRUCTURE_ALPHA_TEMPLATES[name]


def list_templates() -> list[str]:
    return list(MICROSTRUCTURE_ALPHA_TEMPLATES.keys())


def filter_by_edge_type(edge_type: MicroEdgeType) -> list[MicrostructureTemplate]:
    return [t for t in MICROSTRUCTURE_ALPHA_TEMPLATES.values() if t.edge_type == edge_type]


def filter_by_horizon(horizon: TimeHorizon) -> list[MicrostructureTemplate]:
    return [t for t in MICROSTRUCTURE_ALPHA_TEMPLATES.values() if t.time_horizon == horizon]
