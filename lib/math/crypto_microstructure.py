"""
crypto_microstructure.py
Crypto-specific quantitative models for microstructure and on-chain analytics.

Covers:
- Perpetual funding rate premium/discount model
- Basis convergence: spot vs futures convergence trade
- Liquidation cascade model
- On-chain velocity: NVT ratio, MVRV Z-score, realised cap
- Exchange flow signal (whale netflow)
- Hash ribbon: miner capitulation signal
- Open interest weighted average price (OIWAP)
- Funding-basis arbitrage scoring
- Altcoin/BTC beta and residual momentum
- Stablecoin dominance as risk-off indicator
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress, percentileofscore


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FundingRateModel:
    """Perpetual funding rate premium/discount diagnostics."""
    funding_rate: float          # current period rate (e.g. 8h)
    annualised_rate: float       # funding_rate * periods_per_year
    premium_index: float         # (perp_price - spot) / spot
    predicted_funding: float     # model-implied next funding
    signal: str                  # 'long_squeeze' / 'short_squeeze' / 'neutral'
    z_score: float               # funding z-score vs rolling window


@dataclass
class BasisTrade:
    """Spot-futures basis convergence trade."""
    basis_pct: float             # (futures - spot) / spot * 100
    annualised_basis: float      # basis_pct * 365 / days_to_expiry
    days_to_expiry: int
    entry_signal: bool           # True when basis is wide enough to trade
    expected_pnl_pct: float      # expected PnL if basis converges to zero
    carry_cost_pct: float        # financing cost over the period
    net_expected_pnl_pct: float


@dataclass
class LiquidationCascade:
    """Leverage-weighted liquidation map at each price level."""
    price_levels: np.ndarray          # price grid
    long_liquidations: np.ndarray     # USD notional longs liquidated at each level
    short_liquidations: np.ndarray    # USD notional shorts liquidated at each level
    net_liquidation: np.ndarray       # long - short (positive = net selling pressure)
    max_long_liq_price: float         # price with peak long liquidations (support crush)
    max_short_liq_price: float        # price with peak short liquidations (squeeze level)
    cascade_risk_score: float         # [0,1] — how concentrated liquidations are


@dataclass
class OnChainMetrics:
    """On-chain velocity and valuation metrics."""
    nvt_ratio: float             # Network Value to Transactions ratio
    nvt_signal: float            # 90-day MA of NVT
    nvt_z_score: float
    mvrv_ratio: float            # Market Value / Realised Value
    mvrv_z_score: float          # MVRV Z-score (vs historical std dev from ATH-adjusted mean)
    realised_cap: float          # sum of each UTXO * price at last move
    signal: str                  # 'overvalued' / 'undervalued' / 'fair'


@dataclass
class ExchangeFlowSignal:
    """Exchange netflow (whale in/out signal)."""
    net_flow_usd: float          # positive = net inflow to exchanges (bearish)
    net_flow_7d_ma: float        # 7-day moving average
    z_score: float               # normalised against historical distribution
    signal: str                  # 'bullish' / 'bearish' / 'neutral'
    large_tx_count: int          # whale transactions (> threshold)
    cumulative_flow_30d: float   # 30-day cumulative net flow


@dataclass
class HashRibbon:
    """Hash ribbon miner capitulation signal."""
    hash_rate_30d_ma: float
    hash_rate_60d_ma: float
    capitulation: bool           # 30d MA < 60d MA
    recovery: bool               # 30d MA crossed back above 60d MA recently
    days_since_cross: int        # days since last MA crossover
    miner_stress_score: float    # [0,1] — 1 = maximum stress
    signal: str                  # 'buy' / 'sell' / 'hold'


@dataclass
class OIWAPResult:
    """Open interest weighted average price."""
    oiwap: float                 # weighted average price
    total_oi: float              # total open interest in USD
    oi_weighted_basis: float     # OI-weighted basis vs spot
    concentration_score: float   # Herfindahl index of OI across exchanges
    top_exchange: str
    top_exchange_oi_pct: float


@dataclass
class FundingArbSignal:
    """Funding-basis arbitrage: long spot + short perp."""
    funding_rate_8h: float
    annualised_funding: float
    basis_pct: float
    total_annualised_yield: float   # funding + basis convergence
    entry_signal: bool              # True when yield > threshold
    break_even_move_pct: float      # spot move that wipes the yield
    recommended_size: float         # Kelly-adjusted position (0-1 of capital)


@dataclass
class AltcoinBeta:
    """Altcoin/BTC beta and residual momentum."""
    symbol: str
    beta: float                   # systematic BTC beta
    alpha_annualised: float       # annualised Jensen's alpha
    residual_momentum: float      # 12-1M momentum unexplained by BTC
    r_squared: float
    residual_vol: float           # idiosyncratic vol
    signal: str                   # 'strong_alpha' / 'btc_proxy' / 'underperformer'


@dataclass
class StablecoinDominance:
    """Stablecoin market cap dominance as risk-off indicator."""
    stablecoin_mcap: float
    total_crypto_mcap: float
    dominance_pct: float
    dominance_7d_change_pct: float
    dominance_30d_ma: float
    z_score: float
    regime: str                   # 'risk_off' / 'risk_on' / 'neutral'
    signal: str                   # 'defensive' / 'aggressive' / 'hold'


# ---------------------------------------------------------------------------
# 1. Perpetual funding rate premium/discount model
# ---------------------------------------------------------------------------

def funding_rate_model(
    perp_price: float,
    spot_price: float,
    funding_rate: float,
    periods_per_day: float = 3.0,     # e.g. 8h funding → 3 per day
    historical_rates: Optional[np.ndarray] = None,
    clamp_pct: float = 0.03,          # typical exchange clamp at ±3%
) -> FundingRateModel:
    """
    Model perpetual swap funding rate dynamics and extract directional signal.

    Perpetual funding = clamp(premium_index + interest_rate_component, ±clamp).
    When funding is highly positive, longs pay shorts → crowded long → squeeze risk.

    Parameters
    ----------
    perp_price      : mark price of perpetual swap
    spot_price      : index/spot price
    funding_rate    : current 8h (or period) funding rate (e.g. 0.0001 = 0.01%)
    periods_per_day : funding settlements per day
    historical_rates: array of past funding rates for z-score
    clamp_pct       : exchange funding clamp limit

    Returns
    -------
    FundingRateModel dataclass.
    """
    premium_index = (perp_price - spot_price) / spot_price
    annualised = funding_rate * periods_per_day * 365.0

    # Simple predicted funding: dampened toward premium index
    predicted_funding = np.clip(premium_index, -clamp_pct, clamp_pct)

    if historical_rates is not None and len(historical_rates) > 1:
        mu = float(np.mean(historical_rates))
        sigma = float(np.std(historical_rates, ddof=1))
        z = (funding_rate - mu) / sigma if sigma > 0 else 0.0
    else:
        z = funding_rate / 0.0001   # normalise vs ~1 bp baseline

    if z > 2.0:
        signal = "long_squeeze"
    elif z < -2.0:
        signal = "short_squeeze"
    else:
        signal = "neutral"

    return FundingRateModel(
        funding_rate=funding_rate,
        annualised_rate=annualised,
        premium_index=premium_index,
        predicted_funding=predicted_funding,
        signal=signal,
        z_score=float(z),
    )


# ---------------------------------------------------------------------------
# 2. Basis convergence: spot vs futures convergence trade
# ---------------------------------------------------------------------------

def basis_convergence_trade(
    spot: float,
    futures: float,
    days_to_expiry: int,
    borrow_rate_annual: float = 0.02,
    min_basis_threshold_pct: float = 0.5,
) -> BasisTrade:
    """
    Evaluate a cash-and-carry (spot long + futures short) convergence trade.

    At expiry, futures converge to spot. The basis (futures premium) represents
    the maximum gross yield; net of financing costs gives the risk-free return.

    Parameters
    ----------
    spot                    : spot price
    futures                 : futures mark price
    days_to_expiry          : calendar days to futures expiry
    borrow_rate_annual      : annual cost to borrow/hold spot (crypto custody + borrow)
    min_basis_threshold_pct : minimum annualised basis to trigger entry signal

    Returns
    -------
    BasisTrade dataclass.
    """
    basis_pct = (futures - spot) / spot * 100.0
    T = days_to_expiry / 365.0
    ann_basis = basis_pct / (T + 1e-9) if T > 0 else 0.0

    carry_cost_pct = borrow_rate_annual * T * 100.0
    expected_pnl_pct = basis_pct                             # if basis fully converges
    net_expected_pnl_pct = expected_pnl_pct - carry_cost_pct
    entry_signal = ann_basis > min_basis_threshold_pct and net_expected_pnl_pct > 0.0

    return BasisTrade(
        basis_pct=basis_pct,
        annualised_basis=ann_basis,
        days_to_expiry=days_to_expiry,
        entry_signal=entry_signal,
        expected_pnl_pct=expected_pnl_pct,
        carry_cost_pct=carry_cost_pct,
        net_expected_pnl_pct=net_expected_pnl_pct,
    )


# ---------------------------------------------------------------------------
# 3. Liquidation cascade model
# ---------------------------------------------------------------------------

def liquidation_cascade_model(
    spot_price: float,
    position_sizes: np.ndarray,   # USD notional of each position
    entry_prices: np.ndarray,     # entry price for each position
    leverages: np.ndarray,        # leverage for each position
    sides: np.ndarray,            # +1 for long, -1 for short
    n_levels: int = 100,
    price_range_pct: float = 0.20,
) -> LiquidationCascade:
    """
    Build a leverage-weighted liquidation map across a price grid.

    Liquidation price for a long = entry * (1 - 1/leverage + maint_margin_rate).
    Liquidation price for a short = entry * (1 + 1/leverage - maint_margin_rate).

    Parameters
    ----------
    spot_price      : current spot price
    position_sizes  : USD notional of each tracked position
    entry_prices    : entry price for each position
    leverages       : leverage (e.g. 10 = 10x)
    sides           : +1 long / -1 short
    n_levels        : number of price levels in grid
    price_range_pct : ± range around spot to model (0.20 = ±20%)

    Returns
    -------
    LiquidationCascade dataclass.
    """
    maint_margin = 0.005   # 0.5% maintenance margin (exchange typical)

    liq_prices = np.where(
        sides == 1,
        entry_prices * (1.0 - 1.0 / leverages + maint_margin),   # long liq
        entry_prices * (1.0 + 1.0 / leverages - maint_margin),   # short liq
    )

    price_lo = spot_price * (1.0 - price_range_pct)
    price_hi = spot_price * (1.0 + price_range_pct)
    price_levels = np.linspace(price_lo, price_hi, n_levels)
    level_width = price_levels[1] - price_levels[0]

    long_liq = np.zeros(n_levels)
    short_liq = np.zeros(n_levels)

    for i, plvl in enumerate(price_levels):
        lo = plvl - level_width / 2.0
        hi = plvl + level_width / 2.0
        in_band = (liq_prices >= lo) & (liq_prices < hi)
        long_mask = in_band & (sides == 1)
        short_mask = in_band & (sides == -1)
        long_liq[i] = float(np.sum(position_sizes[long_mask]))
        short_liq[i] = float(np.sum(position_sizes[short_mask]))

    net_liq = long_liq - short_liq

    max_long_liq_price = float(price_levels[np.argmax(long_liq)])
    max_short_liq_price = float(price_levels[np.argmax(short_liq)])

    # Concentration: Herfindahl index on long liquidations
    total = np.sum(long_liq) + np.sum(short_liq)
    if total > 0:
        shares = np.concatenate([long_liq, short_liq]) / total
        hhi = float(np.sum(shares ** 2))
    else:
        hhi = 0.0
    # normalise to [0,1] (min HHI = 1/2n for n levels each side)
    hhi_min = 1.0 / (2.0 * n_levels)
    cascade_risk = (hhi - hhi_min) / (1.0 - hhi_min) if (1.0 - hhi_min) > 0 else 0.0

    return LiquidationCascade(
        price_levels=price_levels,
        long_liquidations=long_liq,
        short_liquidations=short_liq,
        net_liquidation=net_liq,
        max_long_liq_price=max_long_liq_price,
        max_short_liq_price=max_short_liq_price,
        cascade_risk_score=float(np.clip(cascade_risk, 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# 4. On-chain velocity: NVT ratio, MVRV Z-score, realised cap
# ---------------------------------------------------------------------------

def onchain_metrics(
    market_cap: float,
    daily_tx_volume_usd: np.ndarray,     # array of daily on-chain transfer volume
    coin_supply: float,
    current_price: float,
    utxo_prices: np.ndarray,             # price at last move for each coin in supply
    utxo_quantities: np.ndarray,         # quantity of coins at each UTXO
    historical_mvrv: Optional[np.ndarray] = None,
    nvt_window: int = 90,
) -> OnChainMetrics:
    """
    Compute NVT ratio, MVRV Z-score, and realised cap from on-chain data.

    NVT (Network Value to Transactions): analogous to P/E ratio.
    - High NVT: network overvalued relative to transaction throughput.
    MVRV (Market Value to Realised Value): mean profit of coin holders.
    - MVRV > 3.5 historically marks cycle tops; < 1.0 marks bottoms.

    Parameters
    ----------
    market_cap            : current market capitalisation USD
    daily_tx_volume_usd   : array of daily on-chain transfer volumes (last N days)
    coin_supply           : circulating supply
    current_price         : current USD price
    utxo_prices           : last-moved price for each UTXO bucket
    utxo_quantities       : coins in each UTXO bucket
    historical_mvrv       : historical MVRV series for Z-score
    nvt_window            : rolling window for NVT Signal

    Returns
    -------
    OnChainMetrics dataclass.
    """
    # Realised cap: sum(qty_i * price_at_last_move_i)
    realised_cap = float(np.dot(utxo_quantities, utxo_prices))

    # NVT ratio
    recent_vol = daily_tx_volume_usd[-1] if len(daily_tx_volume_usd) > 0 else 1.0
    nvt_ratio = market_cap / recent_vol if recent_vol > 0 else float("inf")

    # NVT Signal: market cap / 90-day MA of volume
    window = min(nvt_window, len(daily_tx_volume_usd))
    nvt_ma_vol = float(np.mean(daily_tx_volume_usd[-window:]))
    nvt_signal_val = market_cap / nvt_ma_vol if nvt_ma_vol > 0 else float("inf")

    # NVT Z-score: normalise by typical NVT range (rough: 10-150 historically)
    nvt_z = (nvt_ratio - 50.0) / 30.0

    # MVRV
    mvrv = market_cap / realised_cap if realised_cap > 0 else float("nan")

    if historical_mvrv is not None and len(historical_mvrv) > 1:
        mu = float(np.mean(historical_mvrv))
        sigma = float(np.std(historical_mvrv, ddof=1))
        mvrv_z = (mvrv - mu) / sigma if sigma > 0 else 0.0
    else:
        # Simple z-score anchored to empirical range [0.5, 4.0]
        mvrv_z = (mvrv - 1.5) / 0.75

    # Classify
    if mvrv > 3.5 or nvt_z > 2.0:
        signal = "overvalued"
    elif mvrv < 1.0 or nvt_z < -1.5:
        signal = "undervalued"
    else:
        signal = "fair"

    return OnChainMetrics(
        nvt_ratio=nvt_ratio,
        nvt_signal=nvt_signal_val,
        nvt_z_score=float(nvt_z),
        mvrv_ratio=float(mvrv),
        mvrv_z_score=float(mvrv_z),
        realised_cap=realised_cap,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# 5. Exchange flow signal (whale netflow)
# ---------------------------------------------------------------------------

def exchange_flow_signal(
    daily_inflows: np.ndarray,      # USD inflows to exchanges per day
    daily_outflows: np.ndarray,     # USD outflows from exchanges per day
    whale_threshold_usd: float = 1e6,
    large_tx_inflows: Optional[np.ndarray] = None,   # counts of whale txs in
    large_tx_outflows: Optional[np.ndarray] = None,  # counts of whale txs out
) -> ExchangeFlowSignal:
    """
    Generate directional signal from exchange net flow data.

    Net inflow to exchanges → coins moving to sell → bearish.
    Net outflow from exchanges → coins moving to cold storage → bullish.

    Parameters
    ----------
    daily_inflows        : array of daily USD inflows to exchanges (last N days)
    daily_outflows       : array of daily USD outflows from exchanges
    whale_threshold_usd  : single-transaction size to classify as whale
    large_tx_inflows     : daily count of whale-size inflow transactions
    large_tx_outflows    : daily count of whale-size outflow transactions

    Returns
    -------
    ExchangeFlowSignal dataclass.
    """
    assert len(daily_inflows) == len(daily_outflows)
    net_flows = daily_inflows - daily_outflows

    current_net = float(net_flows[-1])
    ma7 = float(np.mean(net_flows[-7:])) if len(net_flows) >= 7 else current_net

    mu = float(np.mean(net_flows))
    sigma = float(np.std(net_flows, ddof=1))
    z = (current_net - mu) / sigma if sigma > 0 else 0.0

    cumulative_30d = float(np.sum(net_flows[-30:])) if len(net_flows) >= 30 else float(np.sum(net_flows))

    whale_count = 0
    if large_tx_inflows is not None:
        whale_count += int(np.sum(large_tx_inflows[-1:]))
    if large_tx_outflows is not None:
        whale_count += int(np.sum(large_tx_outflows[-1:]))

    if z > 1.5:
        signal = "bearish"
    elif z < -1.5:
        signal = "bullish"
    else:
        signal = "neutral"

    return ExchangeFlowSignal(
        net_flow_usd=current_net,
        net_flow_7d_ma=ma7,
        z_score=float(z),
        signal=signal,
        large_tx_count=whale_count,
        cumulative_flow_30d=cumulative_30d,
    )


# ---------------------------------------------------------------------------
# 6. Hash ribbon: miner capitulation signal
# ---------------------------------------------------------------------------

def hash_ribbon(
    daily_hash_rates: np.ndarray,   # array of daily hash rates (TH/s or EH/s)
    recovery_lag: int = 10,         # days after cross to confirm recovery
) -> HashRibbon:
    """
    Generate miner capitulation signal from hash rate moving average crossover.

    The Hash Ribbon (Charles Edwards, 2019) identifies miner capitulation when
    the 30-day MA falls below the 60-day MA, and a buy signal when it crosses
    back above (miner recovery).

    Parameters
    ----------
    daily_hash_rates : array of daily hash rates, most recent last
    recovery_lag     : minimum days the 30d MA must be above 60d MA for 'recovery'

    Returns
    -------
    HashRibbon dataclass.
    """
    n = len(daily_hash_rates)
    assert n >= 60, "Need at least 60 days of hash rate data"

    ma30 = float(np.mean(daily_hash_rates[-30:]))
    ma60 = float(np.mean(daily_hash_rates[-60:]))

    capitulation = ma30 < ma60

    # Detect crossover: find how long ago 30d MA was last below 60d MA
    if n > 60:
        ma30_series = np.array([np.mean(daily_hash_rates[max(0, i-29):i+1]) for i in range(29, n)])
        ma60_series = np.array([np.mean(daily_hash_rates[max(0, i-59):i+1]) for i in range(59, n)])
        min_len = min(len(ma30_series), len(ma60_series))
        diff_series = ma30_series[-min_len:] - ma60_series[-min_len:]
        crosses = np.where(np.diff(np.sign(diff_series)))[0]
        if len(crosses) > 0:
            days_since_cross = int(min_len - crosses[-1] - 1)
        else:
            days_since_cross = n
    else:
        days_since_cross = 0

    recovery = (not capitulation) and (days_since_cross <= recovery_lag)

    # Miner stress: how far is ma30 below ma60 (normalised)
    if ma60 > 0:
        stress = max(0.0, (ma60 - ma30) / ma60)
    else:
        stress = 0.0

    if recovery:
        signal = "buy"
    elif capitulation and stress > 0.1:
        signal = "sell"
    else:
        signal = "hold"

    return HashRibbon(
        hash_rate_30d_ma=ma30,
        hash_rate_60d_ma=ma60,
        capitulation=capitulation,
        recovery=recovery,
        days_since_cross=days_since_cross,
        miner_stress_score=float(np.clip(stress, 0.0, 1.0)),
        signal=signal,
    )


# ---------------------------------------------------------------------------
# 7. Open interest weighted average price (OIWAP)
# ---------------------------------------------------------------------------

def oiwap(
    exchange_names: List[str],
    mark_prices: np.ndarray,       # mark price at each exchange
    open_interests: np.ndarray,    # open interest in USD at each exchange
    spot_price: float,
) -> OIWAPResult:
    """
    Compute open interest weighted average price across exchanges.

    OIWAP provides a consensus mark price and reveals OI concentration.
    High OI concentration in one exchange increases liquidation cascade risk.

    Parameters
    ----------
    exchange_names  : list of exchange labels
    mark_prices     : array of mark prices per exchange
    open_interests  : array of OI (USD notional) per exchange
    spot_price      : underlying spot reference price

    Returns
    -------
    OIWAPResult dataclass.
    """
    assert len(exchange_names) == len(mark_prices) == len(open_interests)

    total_oi = float(np.sum(open_interests))
    if total_oi == 0:
        weights = np.ones(len(mark_prices)) / len(mark_prices)
    else:
        weights = open_interests / total_oi

    oiwap_price = float(np.dot(weights, mark_prices))

    # OI-weighted basis vs spot
    oi_weighted_basis = (oiwap_price - spot_price) / spot_price * 100.0

    # Herfindahl index for concentration
    hhi = float(np.sum(weights ** 2))
    n = len(exchange_names)
    hhi_min = 1.0 / n if n > 0 else 1.0
    concentration = (hhi - hhi_min) / (1.0 - hhi_min) if (1.0 - hhi_min) > 0 else 0.0

    top_idx = int(np.argmax(open_interests))
    top_exchange = exchange_names[top_idx]
    top_oi_pct = float(weights[top_idx] * 100.0)

    return OIWAPResult(
        oiwap=oiwap_price,
        total_oi=total_oi,
        oi_weighted_basis=oi_weighted_basis,
        concentration_score=float(np.clip(concentration, 0.0, 1.0)),
        top_exchange=top_exchange,
        top_exchange_oi_pct=top_oi_pct,
    )


# ---------------------------------------------------------------------------
# 8. Funding-basis arbitrage
# ---------------------------------------------------------------------------

def funding_basis_arb(
    spot_price: float,
    perp_price: float,
    futures_price: float,
    funding_rate_8h: float,
    days_to_futures_expiry: int,
    borrow_rate_annual: float = 0.02,
    min_yield_threshold: float = 0.10,   # 10% annualised minimum
    max_leverage: float = 1.0,           # position as fraction of capital
    vol_annual: float = 0.80,            # underlying vol for Kelly
) -> FundingArbSignal:
    """
    Score the delta-neutral arb: long spot + short perpetual (collect funding).

    The strategy earns:
      - Positive funding (longs pay shorts) when perp > spot
      - Basis convergence at futures expiry

    Parameters
    ----------
    spot_price              : current spot
    perp_price              : perpetual swap mark price
    futures_price           : fixed-expiry futures price
    funding_rate_8h         : current 8h funding rate (e.g. 0.0003 = 0.03%)
    days_to_futures_expiry  : for basis component
    borrow_rate_annual      : annual cost to hold spot position
    min_yield_threshold     : minimum annualised yield to trigger entry
    max_leverage            : max fraction of capital to deploy
    vol_annual              : annualised vol of underlying for Kelly sizing

    Returns
    -------
    FundingArbSignal dataclass.
    """
    # Annualised funding yield (3 settlements/day)
    funding_periods_per_year = 3.0 * 365.0
    ann_funding = funding_rate_8h * funding_periods_per_year

    # Basis component (futures - spot convergence)
    basis_pct = (futures_price - spot_price) / spot_price
    T = days_to_futures_expiry / 365.0
    ann_basis = basis_pct / T if T > 0 else 0.0

    carry_cost = borrow_rate_annual
    total_ann_yield = ann_funding + ann_basis - carry_cost

    entry_signal = total_ann_yield > min_yield_threshold

    # Break-even spot move: spot must not move more than yield in absolute terms
    break_even_pct = abs(total_ann_yield) * T * 100.0

    # Kelly fraction: f* = edge / variance
    # edge = total_ann_yield (in fraction), variance = vol^2
    kelly = total_ann_yield / (vol_annual ** 2) if vol_annual > 0 else 0.0
    recommended_size = float(np.clip(kelly * 0.5, 0.0, max_leverage))   # half-Kelly

    return FundingArbSignal(
        funding_rate_8h=funding_rate_8h,
        annualised_funding=ann_funding,
        basis_pct=basis_pct * 100.0,
        total_annualised_yield=total_ann_yield,
        entry_signal=entry_signal,
        break_even_move_pct=break_even_pct,
        recommended_size=recommended_size,
    )


# ---------------------------------------------------------------------------
# 9. Altcoin/BTC beta and residual momentum
# ---------------------------------------------------------------------------

def altcoin_btc_beta(
    symbol: str,
    alt_returns: np.ndarray,     # daily log-returns of altcoin
    btc_returns: np.ndarray,     # daily log-returns of BTC (same length)
    lookback_momentum: int = 252,  # trading days for 12M momentum
    skip_months: int = 21,         # 1M skip for momentum
) -> AltcoinBeta:
    """
    Compute altcoin/BTC systematic beta, Jensen's alpha, and residual momentum.

    Residual momentum isolates the component of altcoin returns not explained
    by BTC beta, providing an idiosyncratic momentum signal.

    Parameters
    ----------
    symbol            : altcoin ticker
    alt_returns       : daily log-returns
    btc_returns       : daily BTC log-returns (same length)
    lookback_momentum : trading days for 12M momentum window
    skip_months       : recent days to skip for 1M skip

    Returns
    -------
    AltcoinBeta dataclass.
    """
    assert len(alt_returns) == len(btc_returns), "Return series must be same length"
    n = len(alt_returns)

    slope, intercept, r_value, _, _ = linregress(btc_returns, alt_returns)
    beta = float(slope)
    r_squared = float(r_value ** 2)

    alpha_daily = float(intercept)
    alpha_annual = alpha_daily * 252.0

    # Residual returns
    residuals = alt_returns - (alpha_daily + beta * btc_returns)
    residual_vol = float(np.std(residuals, ddof=1) * math.sqrt(252))

    # Residual momentum: sum of residuals over 12M skip-1M window
    if n >= lookback_momentum:
        window_residuals = residuals[-(lookback_momentum): -skip_months] if skip_months > 0 else residuals[-lookback_momentum:]
        residual_momentum = float(np.sum(window_residuals))
    else:
        residual_momentum = float(np.sum(residuals))

    # Classify
    if alpha_annual > 0.20 and r_squared < 0.6:
        signal = "strong_alpha"
    elif r_squared > 0.85:
        signal = "btc_proxy"
    elif alpha_annual < -0.10:
        signal = "underperformer"
    else:
        signal = "neutral"

    return AltcoinBeta(
        symbol=symbol,
        beta=beta,
        alpha_annualised=alpha_annual,
        residual_momentum=residual_momentum,
        r_squared=r_squared,
        residual_vol=residual_vol,
        signal=signal,
    )


def rank_altcoins_by_residual_momentum(
    alt_results: List[AltcoinBeta],
) -> List[AltcoinBeta]:
    """
    Rank a list of AltcoinBeta results by residual momentum (descending).

    Returns the same list sorted with residual_momentum highest first.
    """
    return sorted(alt_results, key=lambda x: x.residual_momentum, reverse=True)


# ---------------------------------------------------------------------------
# 10. Stablecoin market cap dominance as risk-off indicator
# ---------------------------------------------------------------------------

def stablecoin_dominance_signal(
    stablecoin_mcap: float,
    total_crypto_mcap: float,
    historical_dominance: np.ndarray,    # array of past dominance values (fractions)
    risk_off_threshold: float = 0.10,    # dominance > 10% = risk off
    risk_on_threshold: float = 0.06,     # dominance < 6% = risk on
) -> StablecoinDominance:
    """
    Generate risk regime signal from stablecoin market cap dominance.

    Rising stablecoin dominance indicates capital rotating out of volatile
    crypto assets into safety → bearish for BTC/alts.
    Falling dominance signals capital deployment → bullish.

    Parameters
    ----------
    stablecoin_mcap       : total stablecoin market cap (USD)
    total_crypto_mcap     : total crypto market cap (USD)
    historical_dominance  : array of past dominance fractions (length >= 30)
    risk_off_threshold    : dominance fraction above which regime is 'risk_off'
    risk_on_threshold     : dominance fraction below which regime is 'risk_on'

    Returns
    -------
    StablecoinDominance dataclass.
    """
    dominance = stablecoin_mcap / total_crypto_mcap if total_crypto_mcap > 0 else 0.0

    n = len(historical_dominance)
    ma30 = float(np.mean(historical_dominance[-30:])) if n >= 30 else float(np.mean(historical_dominance))

    # 7-day change
    if n >= 8:
        dom_7d_ago = float(historical_dominance[-8])
    elif n >= 2:
        dom_7d_ago = float(historical_dominance[0])
    else:
        dom_7d_ago = dominance

    change_7d = (dominance - dom_7d_ago) / dom_7d_ago * 100.0 if dom_7d_ago > 0 else 0.0

    mu = float(np.mean(historical_dominance))
    sigma = float(np.std(historical_dominance, ddof=1))
    z = (dominance - mu) / sigma if sigma > 0 else 0.0

    if dominance > risk_off_threshold or (change_7d > 5.0 and z > 1.5):
        regime = "risk_off"
        signal = "defensive"
    elif dominance < risk_on_threshold or (change_7d < -5.0 and z < -1.5):
        regime = "risk_on"
        signal = "aggressive"
    else:
        regime = "neutral"
        signal = "hold"

    return StablecoinDominance(
        stablecoin_mcap=stablecoin_mcap,
        total_crypto_mcap=total_crypto_mcap,
        dominance_pct=dominance * 100.0,
        dominance_7d_change_pct=change_7d,
        dominance_30d_ma=ma30 * 100.0,
        z_score=float(z),
        regime=regime,
        signal=signal,
    )
