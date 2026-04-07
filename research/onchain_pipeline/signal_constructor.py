"""
signal_constructor.py -- On-chain alpha signal construction for crypto assets.

Each signal in OnChainSignalLibrary returns a pd.Series in [-1, +1] where:
  +1 = strong bullish signal
  -1 = strong bearish signal
   0 = neutral

Signals are designed to capture different aspects of on-chain behavior:
  - Valuation (MVRV-Z, NUPL)
  - Sentiment (SOPR)
  - Supply dynamics (LTH accumulation, exchange flows)
  - Derivatives positioning (funding rate, OI-price divergence)

OnChainSignalCombiner aggregates individual signals using IC-weighted blending
with a correlation filter to avoid double-counting collinear signals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clip_signal(s: pd.Series) -> pd.Series:
    """Clip values to [-1, 1] and drop NaN."""
    return s.clip(-1.0, 1.0)


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with min_periods=1."""
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score normalized to the given window."""
    mean = s.rolling(window, min_periods=max(1, window // 2)).mean()
    std = s.rolling(window, min_periods=max(1, window // 2)).std()
    # Avoid division by zero
    std = std.where(std > 1e-10, other=np.nan)
    return (s - mean) / std


def _tanh_scale(s: pd.Series, scale: float = 1.0) -> pd.Series:
    """Map values to [-1, 1] using tanh transformation."""
    return np.tanh(s * scale)


def _rank_normalize(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank normalization mapped to [-1, 1].

    For each date, rank the current value within the trailing window.
    """
    result = pd.Series(np.nan, index=s.index, dtype=float)
    arr = s.values
    n = len(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        window_vals = arr[start : i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 2:
            continue
        pct = float(np.sum(valid <= arr[i])) / len(valid)
        result.iloc[i] = (pct - 0.5) * 2.0  # map [0,1] -> [-1, 1]
    return result


# ---------------------------------------------------------------------------
# OnChainSignalLibrary
# ---------------------------------------------------------------------------

class OnChainSignalLibrary:
    """
    Library of on-chain alpha signals for Bitcoin and select altcoins.

    Each method accepts pre-fetched pd.Series (indexed by UTC date) and returns
    a signal Series in [-1, +1]. Callers are responsible for aligning indices
    before combining signals.

    All signals are smoothed to reduce daily noise and avoid overfitting to
    single-day extremes. Smoothing parameters are tuned for daily data.
    """

    # ------------------------------------------------------------------
    # MVRV-Z Signal
    # ------------------------------------------------------------------

    @staticmethod
    def mvrv_z_signal(mvrv_z: pd.Series, smooth_span: int = 7) -> pd.Series:
        """
        Bearish when MVRV-Z is historically high (overvalued); bullish when low.

        Thresholds:
          Z > 3  -> strongly overvalued -> signal approaches -1
          Z < -1 -> strongly undervalued -> signal approaches +1
          Z in [-1, 3] -> linearly interpolated signal

        Output is smoothed with a 7-day EMA to reduce daily noise.
        """
        if mvrv_z.empty:
            return pd.Series(dtype=float, name="mvrv_z_signal")

        # Linear interpolation: map [-1, 3] -> [+1, -1]
        raw = pd.Series(np.interp(mvrv_z.values, [-1.0, 3.0], [1.0, -1.0]), index=mvrv_z.index)

        # Hard cap: beyond thresholds, pin to -1 or +1
        raw = raw.where(mvrv_z >= -1.0, other=1.0)
        raw = raw.where(mvrv_z <= 3.0, other=-1.0)

        smoothed = _ema(raw, span=smooth_span)
        return _clip_signal(smoothed).rename("mvrv_z_signal")

    # ------------------------------------------------------------------
    # NUPL Contrarian Signal
    # ------------------------------------------------------------------

    @staticmethod
    def nupl_contrarian(nupl: pd.Series, smooth_span: int = 5) -> pd.Series:
        """
        Contrarian signal based on Net Unrealized Profit/Loss.

        NUPL > 0.75 (euphoria/greed) -> signal -> -1 (bearish)
        NUPL < 0 (fear/capitulation)  -> signal -> +1 (bullish)
        NUPL in [0, 0.75]             -> linearly interpolated

        Smoothed with 5-day EMA.
        """
        if nupl.empty:
            return pd.Series(dtype=float, name="nupl_contrarian")

        raw = pd.Series(np.interp(nupl.values, [0.0, 0.75], [0.0, -1.0]), index=nupl.index)
        raw = raw.where(nupl >= 0.0, other=1.0)
        raw = raw.where(nupl <= 0.75, other=-1.0)

        smoothed = _ema(raw, span=smooth_span)
        return _clip_signal(smoothed).rename("nupl_contrarian")

    # ------------------------------------------------------------------
    # Exchange Flow Signal
    # ------------------------------------------------------------------

    @staticmethod
    def exchange_flow_signal(net_flows: pd.Series, window: int = 14) -> pd.Series:
        """
        Signal based on net exchange flow z-score over a rolling window.

        Negative z-score (coins leaving exchanges) -> accumulation -> +1.
        Positive z-score (coins entering exchanges) -> distribution -> -1.

        Transformation: signal = -tanh(z / 2) to map z-scores to [-1, 1].
        """
        if net_flows.empty:
            return pd.Series(dtype=float, name="exchange_flow_signal")

        z = _rolling_zscore(net_flows, window=window)
        # Invert: negative flow z-score is bullish
        signal = _tanh_scale(-z, scale=0.5)
        return _clip_signal(signal).rename("exchange_flow_signal")

    # ------------------------------------------------------------------
    # Funding Rate Signal
    # ------------------------------------------------------------------

    @staticmethod
    def funding_rate_signal(funding: pd.Series, threshold: float = 0.01) -> pd.Series:
        """
        Regime signal based on perpetuals funding rate.

        funding >  threshold  -> longs paying heavily -> bearish: signal = -0.5
        funding < -threshold  -> shorts paying heavily -> bullish: signal = +0.5
        |funding| <= threshold -> neutral: signal = 0

        The signal is intentionally capped at +/-0.5 because funding rate alone
        is a weak signal -- it works best as a confirming factor.
        """
        if funding.empty:
            return pd.Series(dtype=float, name="funding_rate_signal")

        signal = pd.Series(0.0, index=funding.index, dtype=float)
        signal = signal.where(funding <= threshold, other=-0.5)
        signal = signal.where(funding >= -threshold, other=0.5)

        smoothed = _ema(signal, span=3)
        return _clip_signal(smoothed).rename("funding_rate_signal")

    # ------------------------------------------------------------------
    # LTH Accumulation Signal
    # ------------------------------------------------------------------

    @staticmethod
    def lth_accumulation_signal(lth_supply: pd.Series, window: int = 30) -> pd.Series:
        """
        Signal based on Long-Term Holder supply trend.

        LTH supply increasing over the window -> accumulation -> bullish.
        LTH supply decreasing -> distribution -> bearish.

        Method: rolling percent change over `window` days, scaled via tanh.
        """
        if lth_supply.empty:
            return pd.Series(dtype=float, name="lth_accumulation_signal")

        pct_change = lth_supply.pct_change(periods=window)
        # Scale: 1% change over 30 days ~ moderate signal
        signal = _tanh_scale(pct_change, scale=100.0)
        smoothed = _ema(signal, span=7)
        return _clip_signal(smoothed).rename("lth_accumulation_signal")

    # ------------------------------------------------------------------
    # SOPR Signal
    # ------------------------------------------------------------------

    @staticmethod
    def sopr_signal(sopr: pd.Series, smooth_span: int = 5) -> pd.Series:
        """
        Signal based on Spent Output Profit Ratio.

        SOPR < 1 sustained (capitulation): spend at loss -> strong buy.
        SOPR > 1.05 sustained (profit taking): heavy realization -> sell.

        Method: smoothed SOPR centered at 1.0, mapped nonlinearly.
          signal = -tanh((smoothed_sopr - 1.0) * 30)

        This gives strong signals at extremes but near-zero in neutral zones.
        """
        if sopr.empty:
            return pd.Series(dtype=float, name="sopr_signal")

        smoothed = _ema(sopr, span=smooth_span)
        # Center at 1.0 and scale: deviation of 0.05 maps to tanh(1.5) ~ 0.9
        signal = _tanh_scale(-(smoothed - 1.0), scale=30.0)
        return _clip_signal(signal).rename("sopr_signal")

    # ------------------------------------------------------------------
    # Open Interest + Price Divergence Signal
    # ------------------------------------------------------------------

    @staticmethod
    def oi_change_signal(
        oi: pd.Series,
        returns: pd.Series,
        window: int = 5,
    ) -> pd.Series:
        """
        Signal based on OI change relative to price direction.

        OI rising AND price rising -> real buyers entering -> bullish.
        OI rising AND price falling -> shorts adding -> bearish.
        OI falling AND price rising -> shorts covering -> mildly bullish.
        OI falling AND price falling -> longs liquidating -> mildly bearish.

        Method: sign(d_OI) * sign(returns), smoothed.
        """
        if oi.empty or returns.empty:
            return pd.Series(dtype=float, name="oi_change_signal")

        common_idx = oi.index.intersection(returns.index)
        if common_idx.empty:
            return pd.Series(dtype=float, name="oi_change_signal")

        oi_aligned = oi.reindex(common_idx)
        ret_aligned = returns.reindex(common_idx)

        d_oi = oi_aligned.pct_change(periods=window).rolling(window, min_periods=1).mean()
        avg_ret = ret_aligned.rolling(window, min_periods=1).mean()

        # Quadrant signal: product of signs, scaled by magnitude
        oi_z = _rolling_zscore(d_oi, window=max(window * 4, 20))
        ret_z = _rolling_zscore(avg_ret, window=max(window * 4, 20))

        signal = _tanh_scale(oi_z * ret_z, scale=0.5)
        smoothed = _ema(signal, span=window)
        return _clip_signal(smoothed).rename("oi_change_signal")

    # ------------------------------------------------------------------
    # Composite on-chain score
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        mvrv_z: Optional[pd.Series] = None,
        nupl: Optional[pd.Series] = None,
        sopr: Optional[pd.Series] = None,
        exchange_net_flows: Optional[pd.Series] = None,
        lth_supply: Optional[pd.Series] = None,
        funding: Optional[pd.Series] = None,
        oi: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """
        Compute all available signals given input data.

        Returns a dict of signal name -> pd.Series. Only computes signals for
        which the required input data is provided and non-empty.
        """
        lib = cls()
        signals: Dict[str, pd.Series] = {}

        if mvrv_z is not None and not mvrv_z.empty:
            signals["mvrv_z_signal"] = lib.mvrv_z_signal(mvrv_z)

        if nupl is not None and not nupl.empty:
            signals["nupl_contrarian"] = lib.nupl_contrarian(nupl)

        if sopr is not None and not sopr.empty:
            signals["sopr_signal"] = lib.sopr_signal(sopr)

        if exchange_net_flows is not None and not exchange_net_flows.empty:
            signals["exchange_flow_signal"] = lib.exchange_flow_signal(exchange_net_flows)

        if lth_supply is not None and not lth_supply.empty:
            signals["lth_accumulation_signal"] = lib.lth_accumulation_signal(lth_supply)

        if funding is not None and not funding.empty:
            signals["funding_rate_signal"] = lib.funding_rate_signal(funding)

        if oi is not None and returns is not None and not oi.empty and not returns.empty:
            signals["oi_change_signal"] = lib.oi_change_signal(oi, returns)

        return signals


# ---------------------------------------------------------------------------
# OnChainSignalCombiner
# ---------------------------------------------------------------------------

class OnChainSignalCombiner:
    """
    Combines individual on-chain signals into a single composite signal.

    Combination methods:
      - "equal_weight": simple average
      - "ic_weight": Information Coefficient weighted average using trailing 30-day IC
      - "rank_weight": signals ranked by recent IC, top half weighted equally

    Correlation filter: signals with Pearson correlation > `max_corr` to an
    already-selected signal are excluded from the combination to avoid
    double-counting collinear factors.
    """

    def __init__(
        self,
        ic_lookback: int = 30,
        max_corr: float = 0.70,
        min_history: int = 10,
    ) -> None:
        self._ic_lookback = ic_lookback
        self._max_corr = max_corr
        self._min_history = min_history

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    def _trailing_ic(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        lookback: int,
    ) -> float:
        """
        Compute rank IC (Spearman correlation) between signal and forward returns
        over the trailing `lookback` periods.

        Returns 0.0 if insufficient data or correlation cannot be computed.
        """
        common = signal.index.intersection(forward_returns.index)
        if len(common) < self._min_history:
            return 0.0

        sig = signal.reindex(common).tail(lookback).dropna()
        ret = forward_returns.reindex(common).tail(lookback).reindex(sig.index).dropna()
        aligned_sig = sig.reindex(ret.index).dropna()
        aligned_ret = ret.reindex(aligned_sig.index)

        if len(aligned_sig) < self._min_history:
            return 0.0

        # Rank IC via Spearman
        sig_ranks = aligned_sig.rank()
        ret_ranks = aligned_ret.rank()
        corr = sig_ranks.corr(ret_ranks)
        return float(corr) if not np.isnan(corr) else 0.0

    # ------------------------------------------------------------------
    # Correlation filter
    # ------------------------------------------------------------------

    def _apply_corr_filter(
        self,
        signals_df: pd.DataFrame,
        ic_scores: Dict[str, float],
    ) -> List[str]:
        """
        Greedy correlation filter: select signals in descending IC order,
        excluding any signal that is too correlated with an already-selected one.

        Returns the list of selected signal names.
        """
        # Sort by absolute IC descending
        ranked = sorted(ic_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        selected: List[str] = []

        for name, _ in ranked:
            if name not in signals_df.columns:
                continue
            candidate = signals_df[name].dropna()
            too_correlated = False
            for sel_name in selected:
                sel = signals_df[sel_name].dropna()
                common = candidate.index.intersection(sel.index)
                if len(common) < self._min_history:
                    continue
                corr = candidate.reindex(common).corr(sel.reindex(common))
                if not np.isnan(corr) and abs(corr) > self._max_corr:
                    logger.debug(
                        "Excluding '%s' (corr=%.2f with '%s' > %.2f)",
                        name, corr, sel_name, self._max_corr,
                    )
                    too_correlated = True
                    break
            if not too_correlated:
                selected.append(name)

        return selected

    # ------------------------------------------------------------------
    # Combination methods
    # ------------------------------------------------------------------

    def _equal_weight(
        self,
        signals_df: pd.DataFrame,
        selected: List[str],
    ) -> pd.Series:
        """Simple equal-weight average of selected signals."""
        if not selected:
            return pd.Series(dtype=float, name="combined_signal")
        return signals_df[selected].mean(axis=1).rename("combined_signal")

    def _ic_weight(
        self,
        signals_df: pd.DataFrame,
        selected: List[str],
        ic_scores: Dict[str, float],
    ) -> pd.Series:
        """
        IC-weighted combination.

        Weights are proportional to |IC|, normalized to sum to 1.
        Signals with negative IC are sign-flipped before weighting so that
        a negative-IC signal still contributes in the correct direction.
        """
        if not selected:
            return pd.Series(dtype=float, name="combined_signal")

        weights: Dict[str, float] = {}
        sign_flip: Dict[str, float] = {}
        for name in selected:
            ic = ic_scores.get(name, 0.0)
            weights[name] = abs(ic)
            sign_flip[name] = 1.0 if ic >= 0 else -1.0

        total_weight = sum(weights.values())
        if total_weight < 1e-10:
            # Fall back to equal weight if no IC information
            return self._equal_weight(signals_df, selected)

        combined = pd.Series(0.0, index=signals_df.index, dtype=float)
        for name in selected:
            w = weights[name] / total_weight
            combined += w * sign_flip[name] * signals_df[name].fillna(0.0)

        return _clip_signal(combined).rename("combined_signal")

    def _rank_weight(
        self,
        signals_df: pd.DataFrame,
        selected: List[str],
        ic_scores: Dict[str, float],
    ) -> pd.Series:
        """
        Rank-weight: take top half of signals by |IC|, equal-weight those.
        """
        if not selected:
            return pd.Series(dtype=float, name="combined_signal")

        sorted_by_ic = sorted(selected, key=lambda n: abs(ic_scores.get(n, 0.0)), reverse=True)
        top_half = sorted_by_ic[: max(1, len(sorted_by_ic) // 2)]
        return self._equal_weight(signals_df, top_half)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(
        self,
        signals: Dict[str, pd.Series],
        method: str = "ic_weight",
        forward_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Combine multiple signals into a single composite signal in [-1, 1].

        Parameters
        ----------
        signals:
            Dict of signal_name -> pd.Series (each in [-1, 1]).
        method:
            One of "equal_weight", "ic_weight", "rank_weight".
        forward_returns:
            pd.Series of 1-day forward returns used to compute IC weights.
            Required for "ic_weight" and "rank_weight"; ignored for "equal_weight".

        Returns
        -------
        pd.Series named "combined_signal".
        """
        if not signals:
            return pd.Series(dtype=float, name="combined_signal")

        if method not in ("equal_weight", "ic_weight", "rank_weight"):
            raise ValueError(f"Unknown combination method: '{method}'. "
                             f"Choose from 'equal_weight', 'ic_weight', 'rank_weight'.")

        # Align all signals to a common datetime index
        signals_df = pd.DataFrame(signals)

        # Compute IC scores (default to 0 if no forward returns provided)
        ic_scores: Dict[str, float] = {}
        if forward_returns is not None and not forward_returns.empty:
            for name, sig in signals.items():
                ic_scores[name] = self._trailing_ic(sig, forward_returns, self._ic_lookback)
        else:
            ic_scores = {name: 1.0 for name in signals}

        # Apply correlation filter
        selected = self._apply_corr_filter(signals_df, ic_scores)

        if not selected:
            logger.warning("No signals passed correlation filter -- returning equal-weight combination.")
            selected = list(signals.keys())

        if method == "equal_weight":
            return self._equal_weight(signals_df, selected)
        elif method == "ic_weight":
            return self._ic_weight(signals_df, selected, ic_scores)
        else:  # rank_weight
            return self._rank_weight(signals_df, selected, ic_scores)

    def combine_with_diagnostics(
        self,
        signals: Dict[str, pd.Series],
        method: str = "ic_weight",
        forward_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, Dict[str, float], List[str]]:
        """
        Like combine(), but also returns IC scores and the list of selected signals.

        Returns (combined_signal, ic_scores, selected_signals).
        """
        if not signals:
            return pd.Series(dtype=float, name="combined_signal"), {}, []

        signals_df = pd.DataFrame(signals)

        ic_scores: Dict[str, float] = {}
        if forward_returns is not None and not forward_returns.empty:
            for name, sig in signals.items():
                ic_scores[name] = self._trailing_ic(sig, forward_returns, self._ic_lookback)
        else:
            ic_scores = {name: 1.0 for name in signals}

        selected = self._apply_corr_filter(signals_df, ic_scores)
        if not selected:
            selected = list(signals.keys())

        if method == "equal_weight":
            combined = self._equal_weight(signals_df, selected)
        elif method == "ic_weight":
            combined = self._ic_weight(signals_df, selected, ic_scores)
        elif method == "rank_weight":
            combined = self._rank_weight(signals_df, selected, ic_scores)
        else:
            raise ValueError(f"Unknown method: {method}")

        return combined, ic_scores, selected
