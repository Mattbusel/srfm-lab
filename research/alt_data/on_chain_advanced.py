"""
on_chain_advanced.py -- Advanced on-chain analytics for Bitcoin/crypto positioning.

This module provides production-grade on-chain signal generators. All methods
accept pre-fetched data as parameters -- no live API calls are made here.
Callers are responsible for sourcing data from providers such as Glassnode,
CryptoQuant, IntoTheBlock, or on-chain node operators.

Signals covered:
  RealizedPriceBands   -- STH-RP and LTH-RP cost basis bands
  ExchangeFlowAnalyzer -- Net exchange flows as accumulation/distribution signals
  MinerSignal          -- Miner revenue vs cost, capitulation detection
  StablecoinRatio      -- Stablecoin supply ratio as dry-powder gauge
  NVTSignal            -- Network Value to Transactions valuation metric
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    """Return mean of values, or NaN if the list is empty."""
    if not values:
        return float("nan")
    return statistics.mean(values)


def _safe_stdev(values: List[float]) -> float:
    """Return sample stdev, or NaN if fewer than 2 values."""
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def _z_score(value: float, history: List[float]) -> float:
    """Compute z-score of value vs history. Returns 0.0 for insufficient data."""
    if len(history) < 2:
        return 0.0
    mean = statistics.mean(history)
    sd = statistics.stdev(history)
    if sd < 1e-12:
        return 0.0
    return (value - mean) / sd


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _rolling_mean(data: List[float], window: int) -> float:
    """Return the mean of the last `window` elements of data."""
    if not data:
        return float("nan")
    tail = data[-window:] if len(data) >= window else data
    return statistics.mean(tail)


# ---------------------------------------------------------------------------
# RealizedPriceBands
# ---------------------------------------------------------------------------

class RealizedPriceBands:
    """
    Tracks Short-Term Holder and Long-Term Holder realized price bands.

    The realized price of a cohort is the average price at which coins in
    that cohort were last moved -- i.e., the aggregate cost basis.

    Definitions:
      STH (Short-Term Holder): coins moved within the last 155 days
      LTH (Long-Term Holder) : coins unmoved for 155+ days

    These bands act as dynamic support/resistance levels and are widely used
    in on-chain analysis to assess market profitability stress.

    Parameters
    ----------
    history_window : int
        Number of periods to retain for z-score normalization. Default 365.
    """

    STH_DAYS = 155  # threshold separating STH and LTH cohorts

    def __init__(self, history_window: int = 365) -> None:
        self.history_window = history_window
        self._sth_rp_history: Deque[float] = deque(maxlen=history_window)
        self._lth_rp_history: Deque[float] = deque(maxlen=history_window)
        self._price_history:  Deque[float] = deque(maxlen=history_window)
        self._sth_rp: Optional[float] = None
        self._lth_rp: Optional[float] = None

    def update(self, price: float, sth_realized_price: float, lth_realized_price: float) -> None:
        """
        Record a new on-chain snapshot.

        Parameters
        ----------
        price               : Current spot price (USD).
        sth_realized_price  : Aggregate cost basis of coins moved in last 155 days.
        lth_realized_price  : Aggregate cost basis of coins unmoved for 155+ days.
        """
        self._sth_rp = sth_realized_price
        self._lth_rp = lth_realized_price
        self._sth_rp_history.append(sth_realized_price)
        self._lth_rp_history.append(lth_realized_price)
        self._price_history.append(price)

    def is_above_sth_rp(self, price: float) -> bool:
        """
        Return True if the current price is above the STH realized price.

        Interpretation:
          True  -> Short-term holders are in aggregate profit (bullish).
          False -> Short-term holders are underwater (potential capitulation risk).
        """
        if self._sth_rp is None:
            raise ValueError("No STH-RP data available -- call update() first.")
        return price > self._sth_rp

    def is_above_lth_rp(self, price: float) -> bool:
        """
        Return True if price is above the LTH realized price.

        The LTH-RP crossing is a rare event -- prices below LTH-RP indicate
        extreme capitulation where even long-term holders are underwater.
        """
        if self._lth_rp is None:
            raise ValueError("No LTH-RP data available -- call update() first.")
        return price > self._lth_rp

    def sth_profit_loss_ratio(self, price: float) -> float:
        """
        Return the STH profit/loss ratio: price / STH_RP - 1.

        Positive -> STH cohort is in profit (fraction above break-even).
        Negative -> STH cohort is at a loss.

        Examples:
          0.15 -> STH holders are 15% in profit on average
         -0.12 -> STH holders are 12% underwater on average
        """
        if self._sth_rp is None:
            raise ValueError("No STH-RP data available -- call update() first.")
        if self._sth_rp < 1e-6:
            return 0.0
        return price / self._sth_rp - 1.0

    def lth_profit_loss_ratio(self, price: float) -> float:
        """
        Return the LTH profit/loss ratio: price / LTH_RP - 1.

        LTH holders typically have much higher profit ratios due to longer
        holding periods. A very low or negative ratio signals extreme bear market.
        """
        if self._lth_rp is None:
            raise ValueError("No LTH-RP data available -- call update() first.")
        if self._lth_rp < 1e-6:
            return 0.0
        return price / self._lth_rp - 1.0

    def realized_price_ratio(self) -> float:
        """
        Return STH-RP / LTH-RP ratio.

        > 1.0 -> STH cost basis above LTH (newer buyers paid more -- common in bull markets)
        < 1.0 -> STH cost basis below LTH (newer buyers capitulating below long-term basis)
        """
        if self._sth_rp is None or self._lth_rp is None:
            raise ValueError("Realized prices not set -- call update() first.")
        if self._lth_rp < 1e-6:
            return 1.0
        return self._sth_rp / self._lth_rp

    def get_support_band(self) -> Tuple[float, float]:
        """
        Return (LTH-RP, STH-RP) as a support band tuple.

        Prices in this band often find support as holders of both cohorts
        approach break-even and are less likely to sell.
        """
        if self._sth_rp is None or self._lth_rp is None:
            raise ValueError("Realized prices not set -- call update() first.")
        lo = min(self._sth_rp, self._lth_rp)
        hi = max(self._sth_rp, self._lth_rp)
        return lo, hi

    def composite_signal(self, price: float) -> float:
        """
        Compute a composite on-chain band signal in [-1, +1].

        Logic:
          Strongly bullish  (+1) : price well above both bands
          Neutral           ( 0) : price inside the support band
          Strongly bearish  (-1) : price well below both bands

        Uses z-score of price relative to combined realized price level.
        """
        if self._sth_rp is None or self._lth_rp is None:
            return 0.0
        band_mid = (self._sth_rp + self._lth_rp) / 2.0
        band_range = max(abs(self._sth_rp - self._lth_rp), 1.0)
        deviation = (price - band_mid) / band_range
        return _clamp(deviation)

    def sth_rp_z_score(self) -> float:
        """Return z-score of current STH-RP vs its stored history."""
        if self._sth_rp is None:
            return 0.0
        return _z_score(self._sth_rp, list(self._sth_rp_history))

    def lth_rp_z_score(self) -> float:
        """Return z-score of current LTH-RP vs its stored history."""
        if self._lth_rp is None:
            return 0.0
        return _z_score(self._lth_rp, list(self._lth_rp_history))

    def to_dict(self, price: float) -> Dict[str, float]:
        """Return a summary dict of all band metrics for the given price."""
        if self._sth_rp is None or self._lth_rp is None:
            return {}
        return {
            "price":                price,
            "sth_rp":               self._sth_rp,
            "lth_rp":               self._lth_rp,
            "above_sth_rp":         float(self.is_above_sth_rp(price)),
            "above_lth_rp":         float(self.is_above_lth_rp(price)),
            "sth_pl_ratio":         self.sth_profit_loss_ratio(price),
            "lth_pl_ratio":         self.lth_profit_loss_ratio(price),
            "rp_ratio_sth_lth":     self.realized_price_ratio(),
            "composite_signal":     self.composite_signal(price),
        }


# ---------------------------------------------------------------------------
# ExchangeFlowAnalyzer
# ---------------------------------------------------------------------------

class ExchangeFlowAnalyzer:
    """
    Tracks net exchange flows (inflows minus outflows) as a proxy for
    selling pressure (accumulation vs distribution phases).

    Net flow > 0 (inflow) : coins moving TO exchanges -- potential sell pressure
    Net flow < 0 (outflow): coins moving FROM exchanges -- accumulation signal

    Parameters
    ----------
    short_window : int
        Rolling window for short-term flow signal (default 7 days).
    long_window  : int
        Rolling window for long-term flow signal (default 30 days).
    history_window : int
        Total history to retain for z-score computation (default 365).
    """

    def __init__(
        self,
        short_window: int = 7,
        long_window:  int = 30,
        history_window: int = 365,
    ) -> None:
        self.short_window = short_window
        self.long_window  = long_window
        self._net_flows: Deque[float] = deque(maxlen=history_window)
        self._inflows:   Deque[float] = deque(maxlen=history_window)
        self._outflows:  Deque[float] = deque(maxlen=history_window)

    def update(self, inflow_btc: float, outflow_btc: float) -> None:
        """
        Record a single period's exchange flow observation.

        Parameters
        ----------
        inflow_btc  : BTC flowing into exchanges this period.
        outflow_btc : BTC flowing out of exchanges this period.
        """
        net = inflow_btc - outflow_btc
        self._net_flows.append(net)
        self._inflows.append(inflow_btc)
        self._outflows.append(outflow_btc)

    def net_flow(self) -> float:
        """Return the most recent period's net flow (positive = inflow)."""
        if not self._net_flows:
            return 0.0
        return self._net_flows[-1]

    def rolling_net_flow(self, window: Optional[int] = None) -> float:
        """
        Return the sum of net flows over the last `window` periods.
        Uses the instance's long_window if window is None.
        """
        w = window or self.long_window
        flows = list(self._net_flows)
        tail = flows[-w:] if len(flows) >= w else flows
        return sum(tail)

    def short_term_net_flow(self) -> float:
        """Return sum of net flows over the short window."""
        return self.rolling_net_flow(self.short_window)

    def long_term_net_flow(self) -> float:
        """Return sum of net flows over the long window."""
        return self.rolling_net_flow(self.long_window)

    def _flow_scale(self) -> float:
        """
        Return a normalisation scale for flow signals.

        Uses the mean absolute gross flow (average of inflows and outflows) as
        a stable denominator. Falls back to the historical stdev of net flows
        when available. This avoids the degenerate case where constant flows
        produce zero stdev and make z-score based signals ill-defined.
        """
        inflows  = list(self._inflows)
        outflows = list(self._outflows)
        if inflows and outflows:
            mean_gross = _safe_mean([abs(i) + abs(o) for i, o in zip(inflows, outflows)])
            if mean_gross > 1e-6:
                return mean_gross
        flows = list(self._net_flows)
        std = _safe_stdev(flows)
        return max(std, 1.0)

    def accumulation_signal(self) -> float:
        """
        Compute an accumulation signal in [0, 1].

        Methodology:
          1. Compute short and long-term net flows.
          2. Sustained negative net flows (outflows) indicate coins leaving
             exchanges -- consistent with accumulation behavior.
          3. Signal = 1.0 when both short and long flows are strongly negative.
          4. Signal = 0.0 when flows are neutral or positive.

        Returns a score in [0, 1] where 1.0 = strong accumulation evidence.
        """
        flows = list(self._net_flows)
        if len(flows) < 2:
            return 0.0
        short_flow = self.short_term_net_flow()
        long_flow  = self.long_term_net_flow()
        scale = self._flow_scale()
        # Negative net flow (outflows > inflows) -> accumulation (positive signal)
        short_z = -short_flow / (scale * self.short_window ** 0.5)
        long_z  = -long_flow  / (scale * self.long_window  ** 0.5)
        combined = (0.6 * short_z + 0.4 * long_z) / 3.0
        return _clamp(combined, 0.0, 1.0)

    def distribution_signal(self) -> float:
        """
        Compute a distribution/sell-pressure signal in [0, 1].

        Large net inflows to exchanges signal that holders are preparing to sell.
        Returns 1.0 for extreme distribution conditions.
        """
        flows = list(self._net_flows)
        if len(flows) < 2:
            return 0.0
        short_flow = self.short_term_net_flow()
        long_flow  = self.long_term_net_flow()
        scale = self._flow_scale()
        # Positive net flow (inflows > outflows) -> distribution (positive signal)
        short_z = short_flow / (scale * self.short_window ** 0.5)
        long_z  = long_flow  / (scale * self.long_window  ** 0.5)
        combined = (0.6 * short_z + 0.4 * long_z) / 3.0
        return _clamp(combined, 0.0, 1.0)

    def net_flow_z_score(self) -> float:
        """Return z-score of the most recent net flow vs stored history."""
        flows = list(self._net_flows)
        if not flows:
            return 0.0
        return _z_score(flows[-1], flows)

    def composite_flow_signal(self) -> float:
        """
        Unified flow signal in [-1, +1].
        -1.0 = strong distribution (sell pressure)
        +1.0 = strong accumulation (buying pressure / coin withdrawal)
        """
        return self.accumulation_signal() - self.distribution_signal()

    def exchange_balance_trend(self) -> str:
        """
        Classify the recent exchange balance trend.

        Returns one of: ACCUMULATING, DISTRIBUTING, NEUTRAL
        """
        if len(self._net_flows) < self.short_window:
            return "NEUTRAL"
        acc  = self.accumulation_signal()
        dist = self.distribution_signal()
        if acc > 0.6:
            return "ACCUMULATING"
        if dist > 0.6:
            return "DISTRIBUTING"
        return "NEUTRAL"

    def to_dict(self) -> Dict[str, float]:
        """Return a summary dict of current flow metrics."""
        return {
            "net_flow_current":    self.net_flow(),
            "short_term_net_flow": self.short_term_net_flow(),
            "long_term_net_flow":  self.long_term_net_flow(),
            "accumulation_signal": self.accumulation_signal(),
            "distribution_signal": self.distribution_signal(),
            "composite_signal":    self.composite_flow_signal(),
            "net_flow_z_score":    self.net_flow_z_score(),
        }


# ---------------------------------------------------------------------------
# MinerSignal
# ---------------------------------------------------------------------------

class MinerSignal:
    """
    Derives trading signals from miner economics.

    Miners are a fundamental sell-side force -- they must sell coins to cover
    electricity and operational costs. When revenue drops near or below cost,
    miners capitulate (shut off rigs), which historically marks major bottoms.
    Conversely, when miners accumulate (hold rather than sell), it is bullish.

    Key metrics:
      Miner Revenue  : block_reward_btc * price + tx_fees_btc * price
      Miner Cost     : estimated electricity cost per day (USD)
      Hash Ribbon    : 30-day vs 60-day SMA of hashrate -- recovery after
                       capitulation is a powerful buy signal

    Parameters
    ----------
    history_window : int
        Number of periods to retain for historical normalization. Default 365.
    """

    CAPITULATION_THRESHOLD = 0.9   # revenue/cost below this -> capitulation
    ACCUMULATION_THRESHOLD = 1.3   # revenue/cost above this -> miners accumulating

    def __init__(self, history_window: int = 365) -> None:
        self.history_window = history_window
        self._revenue_history:   Deque[float] = deque(maxlen=history_window)
        self._cost_history:      Deque[float] = deque(maxlen=history_window)
        self._hashrate_history:  Deque[float] = deque(maxlen=history_window)
        self._ratio_history:     Deque[float] = deque(maxlen=history_window)
        self._miner_outflow:     Deque[float] = deque(maxlen=history_window)

    def update(
        self,
        miner_revenue_usd: float,
        miner_cost_usd: float,
        hashrate_eh_s: float,
        miner_outflow_btc: float = 0.0,
    ) -> None:
        """
        Record a daily miner economics snapshot.

        Parameters
        ----------
        miner_revenue_usd  : Total daily miner revenue in USD (coinbase + fees).
        miner_cost_usd     : Estimated daily electricity + ops cost in USD.
        hashrate_eh_s      : Network hashrate in exahashes per second.
        miner_outflow_btc  : BTC sent from known miner wallets to exchanges.
        """
        self._revenue_history.append(miner_revenue_usd)
        self._cost_history.append(miner_cost_usd)
        self._hashrate_history.append(hashrate_eh_s)
        self._miner_outflow.append(miner_outflow_btc)
        ratio = miner_revenue_usd / max(miner_cost_usd, 1.0)
        self._ratio_history.append(ratio)

    def revenue_cost_ratio(self) -> float:
        """
        Return the most recent miner revenue / cost ratio.

        < 0.9 -> Capitulation zone
        0.9 - 1.3 -> Marginal / neutral
        > 1.3 -> Profitable / accumulation
        """
        if not self._ratio_history:
            raise ValueError("No miner data -- call update() first.")
        return self._ratio_history[-1]

    def is_capitulating(self) -> bool:
        """
        Return True if current revenue/cost ratio is below the capitulation
        threshold. Historically, sustained capitulation marks major cycle bottoms.
        """
        return self.revenue_cost_ratio() < self.CAPITULATION_THRESHOLD

    def is_accumulating(self) -> bool:
        """
        Return True if miners are in a strongly profitable environment
        AND miner outflows are below historical average (holding rather than selling).
        """
        ratio = self.revenue_cost_ratio()
        if ratio < self.ACCUMULATION_THRESHOLD:
            return False
        if self._miner_outflow:
            current_outflow = self._miner_outflow[-1]
            avg_outflow = _safe_mean(list(self._miner_outflow))
            return current_outflow < avg_outflow * 0.8  # selling less than 80% of normal
        return ratio >= self.ACCUMULATION_THRESHOLD

    def hash_ribbon_signal(self) -> float:
        """
        Compute a Hash Ribbon signal -- a recovery from miner capitulation.

        The Hash Ribbon is defined as the 30-day SMA vs 60-day SMA of hashrate.
        When the 30d SMA crosses above the 60d SMA after a period of decline,
        it signals that miners are returning online (buy signal).

        Returns:
          +1.0 : 30d SMA well above 60d SMA (miners expanding, bullish)
           0.0 : neutral
          -1.0 : 30d SMA below 60d SMA (miners shutting down, bearish)
        """
        hr_list = list(self._hashrate_history)
        if len(hr_list) < 60:
            return 0.0
        sma_30 = _rolling_mean(hr_list, 30)
        sma_60 = _rolling_mean(hr_list, 60)
        if sma_60 < 1e-6:
            return 0.0
        ratio = (sma_30 / sma_60) - 1.0
        return _clamp(ratio * 10.0)  # scale: 10% diff -> max signal

    def get_miner_signal(self) -> float:
        """
        Compute a composite miner signal in [-1, +1].

        Components:
          1. Revenue/cost ratio score     (40% weight)
          2. Hash ribbon score            (35% weight)
          3. Miner outflow score          (25% weight)

        Returns:
          +1.0 -> strongly bullish miner setup (profitable, accumulating, ribbons up)
          -1.0 -> strongly bearish miner setup (capitulating, selling, ribbons down)
        """
        # 1. Revenue/cost component
        if not self._ratio_history:
            return 0.0
        ratio = self.revenue_cost_ratio()
        ratio_history = list(self._ratio_history)
        ratio_z = _z_score(ratio, ratio_history)
        revenue_score = _clamp(ratio_z / 2.0)

        # 2. Hash ribbon component
        ribbon_score = self.hash_ribbon_signal()

        # 3. Miner outflow component (negative outflow z-score = miners holding = bullish)
        if len(self._miner_outflow) >= 5:
            outflows = list(self._miner_outflow)
            outflow_z = _z_score(outflows[-1], outflows)
            outflow_score = _clamp(-outflow_z / 2.0)
        else:
            outflow_score = 0.0

        composite = 0.40 * revenue_score + 0.35 * ribbon_score + 0.25 * outflow_score
        return _clamp(composite)

    def days_in_capitulation(self) -> int:
        """
        Return the number of consecutive trailing days the revenue/cost
        ratio has been below the capitulation threshold.
        """
        ratios = list(self._ratio_history)
        count = 0
        for r in reversed(ratios):
            if r < self.CAPITULATION_THRESHOLD:
                count += 1
            else:
                break
        return count

    def to_dict(self) -> Dict[str, float]:
        """Return a summary dict of miner signal components."""
        if not self._ratio_history:
            return {}
        return {
            "revenue_cost_ratio":  self.revenue_cost_ratio(),
            "is_capitulating":     float(self.is_capitulating()),
            "is_accumulating":     float(self.is_accumulating()),
            "hash_ribbon_signal":  self.hash_ribbon_signal(),
            "miner_signal":        self.get_miner_signal(),
            "days_capitulation":   float(self.days_in_capitulation()),
        }


# ---------------------------------------------------------------------------
# StablecoinRatio
# ---------------------------------------------------------------------------

class StablecoinRatio:
    """
    Monitors the stablecoin supply ratio (SSR) as a dry-powder / liquidity gauge.

    SSR = stablecoin_market_cap / btc_market_cap

    A high SSR indicates a large pool of stablecoins relative to BTC, which
    can be deployed into crypto -- acting as latent buying power (bullish).
    A low SSR indicates most capital is already deployed (risk of pullback).

    Parameters
    ----------
    history_window : int
        Number of periods to retain for percentile and z-score computation.
    """

    def __init__(self, history_window: int = 365) -> None:
        self.history_window = history_window
        self._ssr_history: Deque[float] = deque(maxlen=history_window)
        self._stablecoin_mc: Optional[float] = None
        self._crypto_mc: Optional[float] = None

    def update(self, stablecoin_market_cap_usd: float, total_crypto_market_cap_usd: float) -> None:
        """
        Record a new stablecoin ratio observation.

        Parameters
        ----------
        stablecoin_market_cap_usd : Total market cap of all stablecoins (USD).
        total_crypto_market_cap_usd : Total crypto market cap including stablecoins (USD).
        """
        self._stablecoin_mc = stablecoin_market_cap_usd
        self._crypto_mc     = total_crypto_market_cap_usd
        ssr = self._compute_ssr(stablecoin_market_cap_usd, total_crypto_market_cap_usd)
        self._ssr_history.append(ssr)

    def _compute_ssr(self, stable_mc: float, total_mc: float) -> float:
        """Compute the raw stablecoin supply ratio."""
        if total_mc < 1.0:
            return 0.0
        return stable_mc / total_mc

    def current_ssr(self) -> float:
        """Return the most recently recorded stablecoin supply ratio."""
        if not self._ssr_history:
            raise ValueError("No SSR data -- call update() first.")
        return self._ssr_history[-1]

    def ssr_percentile(self) -> float:
        """
        Return the current SSR as a percentile rank within stored history (0-100).
        Higher percentile -> more stablecoins than usual -> more dry powder.
        """
        hist = list(self._ssr_history)
        if len(hist) < 2:
            return 50.0
        current = hist[-1]
        below = sum(1 for h in hist if h <= current)
        return 100.0 * below / len(hist)

    def ssr_z_score(self) -> float:
        """Return z-score of current SSR vs historical distribution."""
        hist = list(self._ssr_history)
        if len(hist) < 2:
            return 0.0
        return _z_score(hist[-1], hist)

    def get_liquidity_signal(self) -> float:
        """
        Compute a liquidity signal in [-1, +1].

        +1.0 -> Very high stablecoin ratio -- lots of dry powder -- bullish for crypto
        0.0  -> Neutral ratio
        -1.0 -> Very low stablecoin ratio -- most capital deployed -- bearish/caution

        Uses the percentile rank of the current SSR to avoid absolute-level
        dependence (which shifts over market cycles).
        """
        pct = self.ssr_percentile()
        # Map percentile [0, 100] to signal [-1, +1]
        # 50th pct -> 0.0 signal, 90th pct -> +0.8 signal, 10th pct -> -0.8 signal
        normalized = (pct - 50.0) / 50.0  # [-1, +1]
        return _clamp(normalized)

    def dry_powder_usd(self) -> float:
        """
        Return the estimated stablecoin market cap in USD.
        This represents the maximum instantaneous buying power.
        """
        return self._stablecoin_mc or 0.0

    def relative_liquidity_change(self, lookback: int = 30) -> float:
        """
        Return the percentage change in the stablecoin ratio over the
        last `lookback` periods. Positive = growing stablecoin share.
        """
        hist = list(self._ssr_history)
        if len(hist) < lookback + 1:
            return 0.0
        old_ssr = hist[-(lookback + 1)]
        new_ssr = hist[-1]
        if old_ssr < 1e-10:
            return 0.0
        return (new_ssr - old_ssr) / old_ssr

    def to_dict(self) -> Dict[str, float]:
        """Return a summary dict of SSR metrics."""
        if not self._ssr_history:
            return {}
        return {
            "current_ssr":         self.current_ssr(),
            "ssr_percentile":      self.ssr_percentile(),
            "ssr_z_score":         self.ssr_z_score(),
            "liquidity_signal":    self.get_liquidity_signal(),
            "dry_powder_usd":      self.dry_powder_usd(),
            "30d_ssr_change_pct":  self.relative_liquidity_change(30),
        }


# ---------------------------------------------------------------------------
# NVTSignal
# ---------------------------------------------------------------------------

class NVTSignal:
    """
    Network Value to Transactions (NVT) Signal -- a crypto valuation metric.

    Analogous to a P/E ratio for crypto:
      NVT = Network Value (market cap) / Daily On-Chain Transaction Volume (USD)

    High NVT -> network is being valued richly relative to transaction activity
               -> potentially overvalued (bearish signal)
    Low NVT  -> network value supported by high transaction throughput
               -> potentially undervalued (bullish signal)

    The NVT Signal uses a 90-day moving average of the denominator (tx volume)
    to smooth out day-to-day noise in transaction data.

    Parameters
    ----------
    history_window : int
        Number of periods to retain for historical NVT percentile computation.
    """

    # NVT interpretation thresholds (approximate, calibrated on BTC history)
    OVERVALUED_PERCENTILE  = 80.0   # NVT above this pct -> overvalued
    UNDERVALUED_PERCENTILE = 25.0   # NVT below this pct -> undervalued

    def __init__(self, history_window: int = 730) -> None:
        self.history_window = history_window
        self._nvt_history: Deque[float] = deque(maxlen=history_window)
        self._tx_vol_history: Deque[float] = deque(maxlen=history_window)

    def nvt_signal(
        self,
        market_cap: float,
        daily_volume: float,
        window: int = 90,
    ) -> float:
        """
        Compute the NVT Signal.

        Parameters
        ----------
        market_cap   : Network market capitalization in USD.
        daily_volume : Daily on-chain transaction volume in USD (not exchange volume).
        window       : Smoothing window for transaction volume. Default 90.

        Returns
        -------
        float : NVT ratio (market_cap / smoothed_tx_volume).
                Lower values are more bullish; higher values are more bearish.
        """
        self._tx_vol_history.append(daily_volume)
        vol_list = list(self._tx_vol_history)
        smoothed_vol = _rolling_mean(vol_list, window)
        if smoothed_vol < 1.0:
            return float("nan")
        nvt = market_cap / smoothed_vol
        self._nvt_history.append(nvt)
        return nvt

    def nvt_signal_normalized(
        self,
        market_cap: float,
        daily_volume: float,
        window: int = 90,
    ) -> float:
        """
        Compute NVT Signal normalized to historical percentile as a [-1, +1] score.

        +1.0 -> NVT at historical lows (very cheap relative to tx activity)
        -1.0 -> NVT at historical highs (very expensive relative to tx activity)
        """
        nvt = self.nvt_signal(market_cap, daily_volume, window)
        if math.isnan(nvt):
            return 0.0
        hist = list(self._nvt_history)
        if len(hist) < 5:
            return 0.0
        pct = 100.0 * sum(1 for h in hist if h <= nvt) / len(hist)
        # Map to [-1, +1]: low percentile -> positive (cheap), high -> negative (expensive)
        return _clamp((50.0 - pct) / 50.0)

    def current_nvt(self) -> float:
        """Return the most recently computed NVT value."""
        if not self._nvt_history:
            return float("nan")
        return self._nvt_history[-1]

    def nvt_percentile(self) -> float:
        """Return the current NVT's percentile rank within stored history (0-100)."""
        hist = list(self._nvt_history)
        if len(hist) < 2:
            return 50.0
        current = hist[-1]
        return 100.0 * sum(1 for h in hist if h <= current) / len(hist)

    def is_overvalued(self) -> bool:
        """Return True if NVT percentile exceeds the overvalued threshold."""
        return self.nvt_percentile() >= self.OVERVALUED_PERCENTILE

    def is_undervalued(self) -> bool:
        """Return True if NVT percentile is below the undervalued threshold."""
        return self.nvt_percentile() <= self.UNDERVALUED_PERCENTILE

    def nvt_z_score(self) -> float:
        """Return the z-score of the current NVT vs its stored history."""
        hist = list(self._nvt_history)
        if len(hist) < 2:
            return 0.0
        return _z_score(hist[-1], hist)

    def nvt_trend(self, lookback: int = 14) -> float:
        """
        Return the relative change in NVT over `lookback` periods.
        Positive = NVT rising (getting more expensive).
        Negative = NVT falling (getting cheaper).
        """
        hist = list(self._nvt_history)
        if len(hist) < lookback + 1:
            return 0.0
        old = hist[-(lookback + 1)]
        new = hist[-1]
        if old < 1e-6:
            return 0.0
        return (new - old) / old

    def to_dict(self, market_cap: float, daily_volume: float) -> Dict[str, float]:
        """Return a summary dict of NVT metrics for the given inputs."""
        nvt = self.nvt_signal(market_cap, daily_volume)
        norm = self.nvt_signal_normalized(market_cap, daily_volume)
        return {
            "market_cap_usd":    market_cap,
            "daily_tx_vol_usd":  daily_volume,
            "nvt_ratio":         nvt if not math.isnan(nvt) else -1.0,
            "nvt_normalized":    norm,
            "nvt_percentile":    self.nvt_percentile(),
            "is_overvalued":     float(self.is_overvalued()),
            "is_undervalued":    float(self.is_undervalued()),
            "nvt_z_score":       self.nvt_z_score(),
        }
