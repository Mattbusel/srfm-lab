"""
network_value.py -- Network value and on-chain valuation metrics for SRFM.

Covers:
  - NVT Ratio (Network Value to Transactions) -- crypto PE ratio
  - NVT Signal (NVT / 28-day MA)
  - Metcalfe Value ratio (actual price vs n^2 expected value)
  - Stock-to-Flow model (scarcity metric)
  - Realized Cap (UTXOs valued at last-moved price)
  - MVRV (Market Value to Realized Value)
  - Signal outputs normalized to [-1, +1]

All time-series methods return pd.Series with DatetimeIndex.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NVT thresholds (Willy Woo calibration)
NVT_OVERBOUGHT:  float = 150.0
NVT_OVERSOLD:    float = 50.0
NVT_NEUTRAL:     float = 90.0

# MVRV cycle thresholds
MVRV_OVERBOUGHT: float = 3.5
MVRV_OVERSOLD:   float = 1.0

# Metcalfe law exponent (n^alpha, some researchers use 1.5-2.0)
METCALFE_EXPONENT: float = 2.0

# S2F model BTC calibration (log-linear from PlanB's original work)
# log(price) = a * log(S2F) + b
S2F_COEFF_A: float = 3.3
S2F_COEFF_B: float = 14.6

# Z-score clip for all normalized signals
ZSCORE_CLIP: float = 3.0


# ---------------------------------------------------------------------------
# Network value metrics
# ---------------------------------------------------------------------------

class NetworkValueMetrics:
    """
    Compute network valuation metrics and produce directional signals.

    All methods are stateless and accept pd.Series inputs unless the method
    is clearly a scalar computation (e.g., realized_cap).

    Signal conventions:
      - Returns float or pd.Series in [-1, +1]
      - Positive = undervalued / bullish
      - Negative = overvalued / bearish
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_cache (
                    key        TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    cached_at  TEXT NOT NULL
                )
            """)
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("NetworkValueMetrics DB init failed: %s", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # NVT Ratio
    # ------------------------------------------------------------------

    def nvt_ratio(
        self,
        market_cap:          pd.Series,
        transaction_volume:  pd.Series,
        smoothing_window:    int = 14,
    ) -> pd.Series:
        """
        Compute the NVT Ratio (Network Value to Transactions).

        Formula:
            NVT = market_cap / transaction_volume (USD)

        Interpretation (Willy Woo calibration):
          - NVT > 150: network overvalued relative to economic throughput
          - NVT < 50:  network undervalued ("cheap")
          - 50-150:    fair value range

        The transaction_volume is smoothed with a rolling mean to reduce
        daily noise before division.

        Parameters
        ----------
        market_cap:
            Daily market capitalization in USD.
        transaction_volume:
            Daily on-chain transaction volume in USD.
        smoothing_window:
            Rolling mean window applied to transaction_volume.

        Returns
        -------
        pd.Series
            NVT ratio (dimensionless).
        """
        if len(market_cap) == 0 or len(transaction_volume) == 0:
            return pd.Series(dtype=float)

        # Align indexes
        common_idx = market_cap.index.intersection(transaction_volume.index)
        mc  = market_cap.reindex(common_idx)
        vol = transaction_volume.reindex(common_idx)

        # Smooth volume to reduce day-to-day noise
        vol_smooth = vol.rolling(window=smoothing_window, min_periods=1).mean()

        # Avoid division by zero
        nvt = mc / (vol_smooth + 1e-9)
        nvt.name = "nvt_ratio"
        return nvt

    def nvt_signal(
        self,
        nvt:    pd.Series,
        window: int = 28,
    ) -> pd.Series:
        """
        Compute NVT Signal = current NVT / N-day moving average NVT.

        Values > 1 mean NVT is above its recent average (overbought).
        Values < 1 mean NVT is below its recent average (oversold).

        Parameters
        ----------
        nvt:
            NVT ratio time series (output of nvt_ratio()).
        window:
            Rolling mean window in days (default 28).

        Returns
        -------
        pd.Series
            NVT Signal (dimensionless ratio).
        """
        ma  = nvt.rolling(window=window, min_periods=5).mean()
        sig = nvt / (ma + 1e-9)
        sig.name = "nvt_signal"
        return sig

    def nvt_directional_signal(
        self,
        market_cap:         pd.Series,
        transaction_volume: pd.Series,
        smoothing_window:   int = 14,
        ma_window:          int = 28,
    ) -> pd.Series:
        """
        Produce a [-1, +1] directional signal from NVT.

        Maps the NVT value:
          - NVT <= 50  -> +1.0 (cheap)
          - NVT >= 150 -> -1.0 (expensive)
          - Linear interpolation between
        """
        nvt = self.nvt_ratio(market_cap, transaction_volume, smoothing_window)
        # Normalize: 1 at NVT=50, -1 at NVT=150, linear
        norm = (nvt - NVT_NEUTRAL) / (NVT_OVERBOUGHT - NVT_NEUTRAL)
        signal = (-norm).clip(-1.0, 1.0)
        signal.name = "nvt_directional_signal"
        return signal

    # ------------------------------------------------------------------
    # Metcalfe Value
    # ------------------------------------------------------------------

    def metcalfe_value(
        self,
        active_addresses: pd.Series,
        price:            pd.Series,
        exponent:         float = METCALFE_EXPONENT,
    ) -> pd.Series:
        """
        Estimate Metcalfe-law theoretical network value and compare to actual price.

        Metcalfe's law: network value is proportional to n^exponent
        where n = number of active nodes (active addresses).

        The ratio (actual_price / metcalfe_price) indicates over/undervaluation.

        Method:
          1. Fit a log-linear regression: log(price) ~ exponent * log(active_addresses) + const
          2. Predict the "fair" Metcalfe price
          3. Return ratio = actual / predicted

        Parameters
        ----------
        active_addresses:
            Daily count of active blockchain addresses.
        price:
            Daily closing price in USD.
        exponent:
            Metcalfe exponent (default 2.0).

        Returns
        -------
        pd.Series
            Ratio of actual price to Metcalfe-predicted price.
            > 1: overvalued relative to network effect
            < 1: undervalued
        """
        common_idx = price.index.intersection(active_addresses.index)
        p  = price.reindex(common_idx)
        aa = active_addresses.reindex(common_idx)

        valid_mask = (p > 0) & (aa > 0)
        if valid_mask.sum() < 2:
            return pd.Series(dtype=float)

        log_p  = np.log(p[valid_mask])
        log_aa = np.log(aa[valid_mask])

        # Ordinary least squares fit: log_p = alpha * log_aa + beta
        log_aa_arr = log_aa.values
        log_p_arr  = log_p.values
        A = np.column_stack([log_aa_arr, np.ones(len(log_aa_arr))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_p_arr, rcond=None)
            fitted_exponent, intercept = coeffs
        except np.linalg.LinAlgError:
            # Fall back to theoretical exponent
            fitted_exponent = exponent
            # Median-based intercept
            intercept = float(np.median(log_p_arr - exponent * log_aa_arr))

        log_metcalfe = fitted_exponent * np.log(aa) + intercept
        metcalfe_price = np.exp(log_metcalfe)

        ratio = p / (metcalfe_price + 1e-9)
        result = pd.Series(ratio, index=common_idx, name="metcalfe_ratio")
        return result

    def metcalfe_signal(
        self,
        active_addresses: pd.Series,
        price:            pd.Series,
    ) -> pd.Series:
        """
        Convert Metcalfe ratio to a [-1, +1] directional signal.

        Ratio > 1 (overvalued) -> negative signal.
        Ratio < 1 (undervalued) -> positive signal.
        Normalized via log-z-score to handle right-skewed distribution.
        """
        ratio = self.metcalfe_value(active_addresses, price)
        if ratio.empty:
            return pd.Series(dtype=float)

        log_ratio = np.log(ratio + 1e-9)
        mu  = log_ratio.rolling(365, min_periods=30).mean()
        std = log_ratio.rolling(365, min_periods=30).std()
        z   = (log_ratio - mu) / (std + 1e-9)
        signal = (-z / ZSCORE_CLIP).clip(-1.0, 1.0)
        signal.name = "metcalfe_signal"
        return signal

    # ------------------------------------------------------------------
    # Stock-to-Flow
    # ------------------------------------------------------------------

    def stock_to_flow_model(
        self,
        supply:          pd.Series,
        production_rate: pd.Series,
        annualize:       bool = True,
    ) -> pd.Series:
        """
        Compute the Stock-to-Flow (S2F) model price prediction.

        S2F = current_supply / annual_new_supply
        Model price = exp(S2F_COEFF_B) * S2F^S2F_COEFF_A

        Based on PlanB's log-linear regression:
          ln(price) = S2F_COEFF_A * ln(S2F) + S2F_COEFF_B

        Parameters
        ----------
        supply:
            Cumulative supply (coins in circulation) time series.
        production_rate:
            Daily new supply issuance (coins per day).
        annualize:
            If True, multiply daily production_rate by 365 to get annual.

        Returns
        -------
        pd.Series
            S2F model predicted price in USD.
        """
        common_idx = supply.index.intersection(production_rate.index)
        s  = supply.reindex(common_idx)
        pr = production_rate.reindex(common_idx)

        if annualize:
            annual_production = pr * 365.0
        else:
            annual_production = pr

        # Avoid division by zero or negative
        s2f = s / (annual_production.clip(lower=1e-9))
        s2f = s2f.clip(lower=0.01)  # clip unrealistic S2F values

        log_price_predicted = S2F_COEFF_A * np.log(s2f) + S2F_COEFF_B
        model_price = np.exp(log_price_predicted)
        model_price.name = "s2f_model_price"
        return model_price

    def s2f_signal(
        self,
        actual_price:    pd.Series,
        supply:          pd.Series,
        production_rate: pd.Series,
    ) -> pd.Series:
        """
        Produce a [-1, +1] signal based on actual price vs S2F model price.

        Log ratio > 0 (price above model) -> overbought -> negative signal.
        Log ratio < 0 (price below model) -> undervalued -> positive signal.
        """
        model = self.stock_to_flow_model(supply, production_rate)
        common_idx = actual_price.index.intersection(model.index)
        a = actual_price.reindex(common_idx)
        m = model.reindex(common_idx)

        log_ratio = np.log((a + 1e-9) / (m + 1e-9))
        std = log_ratio.rolling(180, min_periods=30).std()
        z   = log_ratio / (std + 1e-9)
        signal = (-z / ZSCORE_CLIP).clip(-1.0, 1.0)
        signal.name = "s2f_signal"
        return signal

    # ------------------------------------------------------------------
    # Realized Cap
    # ------------------------------------------------------------------

    def realized_cap(self, utxo_prices: Dict[str, float]) -> float:
        """
        Compute Realized Capitalization.

        Each UTXO (unspent transaction output) is valued at the price when
        that output was last moved, rather than the current market price.

        Formula:
            realized_cap = sum(amount_i * price_at_last_move_i for all UTXOs)

        Parameters
        ----------
        utxo_prices:
            Mapping of utxo_id -> (amount * price_when_last_moved).
            Each value should already be in USD.

        Returns
        -------
        float
            Total realized capitalization in USD.
        """
        if not utxo_prices:
            return 0.0
        return sum(utxo_prices.values())

    def realized_price(
        self,
        utxo_prices: Dict[str, float],
        total_supply: float,
    ) -> float:
        """
        Compute the Realized Price = realized_cap / total_supply.

        Represents the average cost basis of all coins in circulation.
        Price below realized price = majority of holders at a loss.
        """
        if total_supply <= 0:
            raise ValueError("total_supply must be positive")
        return self.realized_cap(utxo_prices) / total_supply

    # ------------------------------------------------------------------
    # MVRV
    # ------------------------------------------------------------------

    def mvrv_ratio(
        self,
        market_cap:    pd.Series,
        realized_caps: pd.Series,
    ) -> pd.Series:
        """
        Market Value to Realized Value (MVRV) ratio.

        Formula:
            MVRV = market_cap / realized_cap

        Interpretation:
          - MVRV > 3.5: historically near cycle tops (most holders in profit)
          - MVRV < 1.0: historically near cycle bottoms (most holders at loss)
          - MVRV = 1.0: market at "fair value" (cost basis = market)

        Parameters
        ----------
        market_cap:
            Time series of daily market capitalization.
        realized_caps:
            Time series of daily realized capitalization.

        Returns
        -------
        pd.Series
            MVRV ratio (dimensionless).
        """
        common_idx = market_cap.index.intersection(realized_caps.index)
        mc = market_cap.reindex(common_idx)
        rc = realized_caps.reindex(common_idx)
        mvrv = mc / (rc + 1e-9)
        mvrv.name = "mvrv_ratio"
        return mvrv

    def mvrv_signal(
        self,
        market_cap:    pd.Series,
        realized_caps: pd.Series,
    ) -> pd.Series:
        """
        Produce a [-1, +1] signal from MVRV.

        Mapping:
          MVRV >= 3.5 -> -1.0 (overbought)
          MVRV <= 1.0 -> +1.0 (oversold)
          Linear between
        """
        mvrv = self.mvrv_ratio(market_cap, realized_caps)
        norm = (mvrv - MVRV_OVERSOLD) / (MVRV_OVERBOUGHT - MVRV_OVERSOLD)
        # Invert: high MVRV = bearish
        signal = (1.0 - 2.0 * norm).clip(-1.0, 1.0)
        signal.name = "mvrv_signal"
        return signal

    # ------------------------------------------------------------------
    # Composite network value signal
    # ------------------------------------------------------------------

    def composite_signal(
        self,
        market_cap:          pd.Series,
        transaction_volume:  pd.Series,
        active_addresses:    pd.Series,
        price:               pd.Series,
        realized_caps:       Optional[pd.Series] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Blend NVT, Metcalfe, and MVRV signals into a single metric.

        Default weights: nvt=0.4, metcalfe=0.35, mvrv=0.25.
        If realized_caps is None, MVRV is excluded and weights are renormalized.
        """
        w = weights or {"nvt": 0.40, "metcalfe": 0.35, "mvrv": 0.25}

        nvt_sig      = self.nvt_directional_signal(market_cap, transaction_volume)
        metcalfe_sig = self.metcalfe_signal(active_addresses, price)

        components: Dict[str, pd.Series] = {
            "nvt":      nvt_sig,
            "metcalfe": metcalfe_sig,
        }

        if realized_caps is not None:
            components["mvrv"] = self.mvrv_signal(market_cap, realized_caps)
        else:
            # Exclude MVRV, renormalize
            w = {k: v for k, v in w.items() if k != "mvrv"}

        total_w = sum(w.values())
        w = {k: v / total_w for k, v in w.items()}

        # Find common index
        common_idx = None
        for key, series in components.items():
            if common_idx is None:
                common_idx = series.index
            else:
                common_idx = common_idx.intersection(series.index)

        if common_idx is None or len(common_idx) == 0:
            return pd.Series(dtype=float)

        blended = pd.Series(0.0, index=common_idx)
        for key, series in components.items():
            blended += w.get(key, 0.0) * series.reindex(common_idx).fillna(0.0)

        blended = blended.clip(-1.0, 1.0)
        blended.name = "network_value_composite"
        return blended

    # ------------------------------------------------------------------
    # Utility: rolling z-score normalization
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_zscore(
        series: pd.Series,
        window: int = 90,
        clip: float = ZSCORE_CLIP,
    ) -> pd.Series:
        """
        Compute rolling z-score of a series, clipped to [-clip, +clip].

        Useful for normalizing raw metric values before combining signals.
        """
        mu  = series.rolling(window, min_periods=10).mean()
        std = series.rolling(window, min_periods=10).std()
        z   = (series - mu) / (std + 1e-9)
        return z.clip(-clip, clip)

    @staticmethod
    def normalize_to_signal(
        series: pd.Series,
        window: int = 90,
    ) -> pd.Series:
        """
        Normalize a raw metric time series to [-1, +1] via rolling z-score and tanh.
        """
        z = NetworkValueMetrics.rolling_zscore(series, window)
        return np.tanh(z / 2.0)
