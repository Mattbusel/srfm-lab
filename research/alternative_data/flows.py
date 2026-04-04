"""
Flow-based alternative data signals.

Implements:
- ETF flow model (creations/redemptions as contrarian/momentum signals)
- Options flow (put/call open interest, smart money proxy)
- COT (Commitments of Traders) simulation signal
- Dark pool proxy (block trade volume divergence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
    mu = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (series - mu) / sd


def _stats(returns: pd.Series, freq: int = 252) -> Dict:
    r = returns.dropna()
    if len(r) == 0:
        return {k: np.nan for k in ["total_return", "cagr", "sharpe", "max_drawdown"]}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    n_years = len(r) / freq
    cagr = float((1 + total) ** (1 / max(n_years, 1e-6)) - 1)
    sr = float(r.mean() / (r.std() + 1e-12) * np.sqrt(freq))
    mdd = float(((eq - eq.cummax()) / (eq.cummax() + 1e-12)).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sr, "max_drawdown": mdd}


# ---------------------------------------------------------------------------
# ETF Flow Model
# ---------------------------------------------------------------------------

class ETFFlowSignal:
    """
    ETF creations/redemptions as a flow signal.

    Large inflows → strong demand (momentum) or overextension (contrarian).
    Large outflows → weak demand (bearish) or washout (contrarian).

    Research suggests:
    - Short-term (< 5 days): momentum in flows
    - Medium-term (1-4 weeks): contrarian

    Parameters
    ----------
    lookback : int
        Window for z-score normalization.
    short_window : int
        Short-term momentum window (days).
    long_window : int
        Long-term reversion window (days).
    blend : float
        Weight on contrarian component (0 = pure momentum, 1 = pure contrarian).
    holding_period : int
        Signal holding period.
    """

    def __init__(
        self,
        lookback: int = 63,
        short_window: int = 3,
        long_window: int = 21,
        blend: float = 0.5,
        holding_period: int = 5,
    ) -> None:
        self.lookback = lookback
        self.short_window = short_window
        self.long_window = long_window
        self.blend = blend
        self.holding_period = holding_period

    def compute_flow_signal(self, flows: pd.Series) -> pd.Series:
        """
        Compute combined momentum-contrarian signal from ETF flows.

        Parameters
        ----------
        flows : pd.Series
            Daily net flows (positive = inflows, negative = outflows).
            Units can be $ millions or shares; signal is z-scored.

        Returns
        -------
        pd.Series
            Combined signal in normalized units.
        """
        flow_z = _rolling_zscore(flows, self.lookback)

        # Short-term momentum component
        momentum = flow_z.rolling(self.short_window, min_periods=1).mean()

        # Long-term contrarian component
        contrarian = -flow_z.rolling(self.long_window, min_periods=1).mean()

        combined = (1 - self.blend) * momentum + self.blend * contrarian
        return combined.rename("etf_flow_signal")

    def cumulative_flow_z(self, flows: pd.Series, window: int = 21) -> pd.Series:
        """Cumulative flow over window, z-scored."""
        cum_flow = flows.rolling(window).sum()
        return _rolling_zscore(cum_flow, self.lookback).rename("cumulative_flow_z")

    def flow_acceleration(self, flows: pd.Series) -> pd.Series:
        """Rate of change of flows (second derivative)."""
        flow_ma = flows.rolling(self.short_window, min_periods=1).mean()
        acceleration = flow_ma.diff(self.short_window)
        return _rolling_zscore(acceleration, self.lookback).rename("flow_acceleration")

    def backtest(self, flows: pd.Series, price: pd.Series) -> Dict:
        """Simple backtest of ETF flow signal."""
        signal = self.compute_flow_signal(flows)
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}

    def flow_statistics(self, flows: pd.Series) -> Dict:
        """Descriptive statistics of the flow series."""
        clean = flows.dropna()
        return {
            "mean_daily_flow": round(float(clean.mean()), 4),
            "std_daily_flow": round(float(clean.std()), 4),
            "max_inflow": round(float(clean.max()), 4),
            "max_outflow": round(float(clean.min()), 4),
            "pct_inflow_days": round(float((clean > 0).mean()), 4),
            "autocorr_1d": round(float(clean.autocorr(1)), 4),
        }


# ---------------------------------------------------------------------------
# Options Flow Signal
# ---------------------------------------------------------------------------

class OptionsFlowSignal:
    """
    Options market flow signals.

    Signals derived from:
    - Put/call open interest ratio
    - Put/call volume ratio (short-term)
    - Large block trades (smart money proxy)
    - Unusual options activity (volume / OI ratio)

    Parameters
    ----------
    lookback : int
        Window for z-score normalization.
    pcr_ma_short : int
        Short MA for put-call ratio (days).
    pcr_ma_long : int
        Long MA for put-call ratio (days).
    unusual_activity_threshold : float
        Volume/OI ratio threshold for unusual activity.
    """

    def __init__(
        self,
        lookback: int = 21,
        pcr_ma_short: int = 5,
        pcr_ma_long: int = 21,
        unusual_activity_threshold: float = 2.0,
    ) -> None:
        self.lookback = lookback
        self.pcr_ma_short = pcr_ma_short
        self.pcr_ma_long = pcr_ma_long
        self.unusual_activity_threshold = unusual_activity_threshold

    def put_call_ratio_signal(
        self,
        put_volume: pd.Series,
        call_volume: pd.Series,
    ) -> pd.Series:
        """
        Contrarian signal from put/call volume ratio.

        High PCR (fearful) → bullish contrarian signal.
        Low PCR (complacent) → bearish contrarian signal.

        Returns
        -------
        pd.Series
            Signal z-score (high = contrarian bullish).
        """
        pcr = put_volume / (call_volume.replace(0, np.nan) + 1e-6)
        pcr_log = np.log(pcr.clip(lower=0.01))

        # Contrarian: invert the signal
        pcr_z = _rolling_zscore(pcr_log, self.lookback)
        signal = -pcr_z  # high PCR = contrarian buy

        # Further smooth with short MA
        signal = signal.rolling(self.pcr_ma_short, min_periods=1).mean()
        return signal.rename("pcr_signal")

    def oi_put_call_signal(
        self,
        put_oi: pd.Series,
        call_oi: pd.Series,
    ) -> pd.Series:
        """
        Open interest PCR — less noisy than volume PCR.

        Higher OI PCR → structural hedging demand → contrarian bullish.

        Returns
        -------
        pd.Series
        """
        oi_pcr = put_oi / (call_oi.replace(0, np.nan) + 1e-6)
        oi_pcr_log = np.log(oi_pcr.clip(lower=0.01))
        oi_z = _rolling_zscore(oi_pcr_log, self.lookback)
        return (-oi_z).rolling(self.pcr_ma_long, min_periods=1).mean().rename("oi_pcr_signal")

    def unusual_options_activity(
        self,
        total_volume: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        Detect unusually high options activity (volume >> OI).

        High ratio → informed trading signal.

        Returns
        -------
        pd.Series
            Boolean series: True = unusual activity detected.
        """
        vol_oi_ratio = total_volume / (open_interest.replace(0, np.nan) + 1e-6)
        ratio_z = _rolling_zscore(vol_oi_ratio, self.lookback)
        return (ratio_z > self.unusual_activity_threshold).rename("unusual_options_activity")

    def smart_money_flow(
        self,
        large_block_volume: pd.Series,
        total_volume: pd.Series,
        option_type: str = "call",
    ) -> pd.Series:
        """
        Smart money proxy: fraction of volume from large block trades.

        Rising fraction of block call volume → institutional accumulation (bullish).
        Rising fraction of block put volume → institutional hedging (bearish).

        Parameters
        ----------
        large_block_volume : pd.Series
            Daily volume from block trades.
        total_volume : pd.Series
            Total options volume.
        option_type : str
            'call' or 'put'.

        Returns
        -------
        pd.Series
        """
        pct_block = large_block_volume / (total_volume.replace(0, np.nan) + 1e-6)
        pct_block_z = _rolling_zscore(pct_block, self.lookback)

        if option_type == "call":
            return pct_block_z.rename("smart_money_call_flow")
        return (-pct_block_z).rename("smart_money_put_flow")  # put flow = bearish

    def composite_signal(
        self,
        put_volume: pd.Series,
        call_volume: pd.Series,
        put_oi: Optional[pd.Series] = None,
        call_oi: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Composite options flow signal."""
        pcr_sig = self.put_call_ratio_signal(put_volume, call_volume)

        if put_oi is not None and call_oi is not None:
            oi_sig = self.oi_put_call_signal(put_oi, call_oi)
            composite = (pcr_sig + oi_sig) / 2
        else:
            composite = pcr_sig

        return composite.rename("options_flow_composite")

    def options_flow_statistics(
        self,
        put_volume: pd.Series,
        call_volume: pd.Series,
    ) -> Dict:
        """Descriptive statistics of options flow data."""
        pcr = put_volume / (call_volume.replace(0, np.nan) + 1e-6)
        return {
            "mean_pcr": round(float(pcr.mean()), 4),
            "median_pcr": round(float(pcr.median()), 4),
            "std_pcr": round(float(pcr.std()), 4),
            "pcr_90th_pct": round(float(pcr.quantile(0.90)), 4),
            "pcr_10th_pct": round(float(pcr.quantile(0.10)), 4),
            "current_pcr": round(float(pcr.iloc[-1]) if len(pcr) > 0 else np.nan, 4),
        }


# ---------------------------------------------------------------------------
# COT Signal (Commitments of Traders)
# ---------------------------------------------------------------------------

class COTSignal:
    """
    COT-based signal from CFTC Commitments of Traders report.

    Uses the positioning of commercial hedgers, large speculators,
    and small speculators to generate signals.

    Strategy: Follow commercial hedgers (often contrarian to price)
    or follow large speculators (momentum-following).

    Parameters
    ----------
    lookback : int
        Window for net position z-scoring.
    follow_commercials : bool
        If True, follow commercial hedgers (contrarian to their view).
        If False, follow large speculators.
    smoothing : int
        MA smoothing applied to COT signal.
    """

    def __init__(
        self,
        lookback: int = 52,
        follow_commercials: bool = False,
        smoothing: int = 4,
    ) -> None:
        self.lookback = lookback
        self.follow_commercials = follow_commercials
        self.smoothing = smoothing

    def compute_net_position(
        self,
        long_contracts: pd.Series,
        short_contracts: pd.Series,
    ) -> pd.Series:
        """
        Compute net position = long - short.

        Returns
        -------
        pd.Series
        """
        return (long_contracts - short_contracts).rename("net_position")

    def compute_signal(
        self,
        commercial_long: pd.Series,
        commercial_short: pd.Series,
        speculator_long: Optional[pd.Series] = None,
        speculator_short: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute COT trading signal.

        Parameters
        ----------
        commercial_long, commercial_short : pd.Series
            Commercial hedger long/short contract counts (weekly).
        speculator_long, speculator_short : pd.Series or None
            Large speculator positions.

        Returns
        -------
        pd.Series
            Weekly signal, z-scored and smoothed.
        """
        commercial_net = self.compute_net_position(commercial_long, commercial_short)
        commercial_net_z = _rolling_zscore(commercial_net, self.lookback)

        if speculator_long is not None and speculator_short is not None:
            spec_net = self.compute_net_position(speculator_long, speculator_short)
            spec_net_z = _rolling_zscore(spec_net, self.lookback)
        else:
            spec_net_z = None

        if self.follow_commercials and spec_net_z is not None:
            # Commercials are usually hedging: when they go net long, expect price rise
            # (they've covered their shorts = exhausted selling pressure)
            signal = commercial_net_z
        elif spec_net_z is not None:
            # Follow speculators (momentum)
            signal = spec_net_z
        else:
            signal = commercial_net_z

        smoothed = signal.rolling(self.smoothing, min_periods=1).mean()
        return smoothed.rename("cot_signal")

    def index_of_bullishness(
        self,
        commercial_long: pd.Series,
        commercial_short: pd.Series,
        window: int = 52,
    ) -> pd.Series:
        """
        COT Index = (net - min) / (max - min) over a rolling window.

        Returns value in [0, 1]; high = historically bullish positioning.

        Returns
        -------
        pd.Series
        """
        net = self.compute_net_position(commercial_long, commercial_short)
        rolling_min = net.rolling(window, min_periods=window // 2).min()
        rolling_max = net.rolling(window, min_periods=window // 2).max()
        cot_index = (net - rolling_min) / (rolling_max - rolling_min + 1e-10)
        return cot_index.rename("cot_index")

    def backtest(
        self,
        commercial_long: pd.Series,
        commercial_short: pd.Series,
        price: pd.Series,
        speculator_long: Optional[pd.Series] = None,
        speculator_short: Optional[pd.Series] = None,
    ) -> Dict:
        """Backtest COT signal against weekly prices."""
        signal = self.compute_signal(
            commercial_long, commercial_short, speculator_long, speculator_short
        )
        weekly_ret = price.resample("W").last().pct_change()
        combined = pd.concat([signal.shift(1), weekly_ret], axis=1).dropna()
        combined.columns = ["signal", "ret"]
        port_rets = combined["signal"] * combined["ret"]
        return {k: round(v, 4) for k, v in _stats(port_rets, freq=52).items()}


# ---------------------------------------------------------------------------
# Dark Pool Proxy
# ---------------------------------------------------------------------------

class DarkPoolSignal:
    """
    Dark pool / block trade signal based on divergence between
    reported volume and exchange-visible volume.

    Dark pool activity is estimated from the residual of a regression
    of stock volume on lagged variables (total volume, volatility, price change).

    A positive residual suggests hidden accumulation (bullish);
    a negative residual suggests hidden distribution (bearish).

    Parameters
    ----------
    lookback : int
        Window for regression-based volume model.
    signal_window : int
        Smoothing window for signal.
    min_excess : float
        Minimum absolute z-score of dark volume to trigger signal.
    """

    def __init__(
        self,
        lookback: int = 21,
        signal_window: int = 5,
        min_excess: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.signal_window = signal_window
        self.min_excess = min_excess

    def estimate_dark_volume(
        self,
        total_volume: pd.Series,
        lit_volume: pd.Series,
    ) -> pd.Series:
        """
        Estimate dark pool volume = total - lit.

        If only total volume is available, use residuals from a
        volume model as proxy.

        Parameters
        ----------
        total_volume : pd.Series
            Total daily trading volume.
        lit_volume : pd.Series
            Exchange-visible (lit) volume.  If unavailable, pass a Series of NaN
            and residual-based proxy will be used.

        Returns
        -------
        pd.Series
            Estimated dark volume.
        """
        if lit_volume.isna().all():
            # Residual proxy: model expected lit volume and take residual
            return self._residual_dark_proxy(total_volume)
        dark = total_volume - lit_volume
        return dark.clip(lower=0).rename("dark_volume")

    def _residual_dark_proxy(self, volume: pd.Series) -> pd.Series:
        """
        Use rolling OLS residuals as dark volume proxy.

        Model: log(volume_t) = a + b*log(volume_{t-1}) + c*vol_ma + eps
        Positive eps = excess unexplained volume ≈ dark pool activity.
        """
        log_vol = np.log(volume.replace(0, np.nan))
        vol_ma = log_vol.rolling(10, min_periods=3).mean()

        residuals = pd.Series(0.0, index=volume.index)
        for i in range(self.lookback, len(volume)):
            window_log = log_vol.iloc[i - self.lookback:i].dropna()
            if len(window_log) < 10:
                continue
            y = window_log.values
            x_lag = log_vol.iloc[i - self.lookback - 1:i - 1].reindex(window_log.index).values
            x_ma = vol_ma.iloc[i - self.lookback:i].reindex(window_log.index).values
            valid = ~(np.isnan(x_lag) | np.isnan(x_ma))
            if valid.sum() < 5:
                continue
            X = np.column_stack([np.ones(valid.sum()), x_lag[valid], x_ma[valid]])
            y_v = y[valid]
            try:
                beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
                fitted = X @ beta
                resid = y_v[-1] - fitted[-1]
                residuals.iloc[i] = resid
            except Exception:
                pass

        return residuals.rename("dark_volume_proxy")

    def generate_signals(
        self,
        total_volume: pd.Series,
        price: pd.Series,
        lit_volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate trading signals from dark volume patterns.

        Dark volume + price rising → confirms trend (bullish).
        Dark volume + price flat/falling → hidden accumulation (bullish signal).
        Low dark volume + price rising → weak hands (bearish signal).

        Returns
        -------
        pd.Series
        """
        if lit_volume is None:
            lit_volume = pd.Series(np.nan, index=total_volume.index)

        dark_vol = self.estimate_dark_volume(total_volume, lit_volume)
        dark_z = _rolling_zscore(dark_vol, self.lookback)

        price_change = price.pct_change(5)
        price_z = _rolling_zscore(price_change, self.lookback)

        # Signal logic
        signal = pd.Series(0.0, index=total_volume.index)

        # High dark volume without price move → accumulation → bullish
        accumulation = (dark_z > self.min_excess) & (price_z.abs() < 0.5)
        signal[accumulation] = 1.0

        # High dark volume + price rising → confirm momentum
        momentum_confirm = (dark_z > self.min_excess) & (price_z > 0.5)
        signal[momentum_confirm] = 0.5

        # Low dark volume + price rising → distribution → bearish
        distribution = (dark_z < -self.min_excess) & (price_z > 0.5)
        signal[distribution] = -1.0

        smoothed = signal.rolling(self.signal_window, min_periods=1).mean()
        return smoothed.rename("dark_pool_signal")

    def backtest(
        self,
        total_volume: pd.Series,
        price: pd.Series,
        lit_volume: Optional[pd.Series] = None,
    ) -> Dict:
        """Backtest dark pool signal."""
        signal = self.generate_signals(total_volume, price, lit_volume)
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}

    def dark_volume_statistics(
        self,
        total_volume: pd.Series,
        lit_volume: Optional[pd.Series] = None,
    ) -> Dict:
        """Descriptive statistics of dark volume series."""
        if lit_volume is None:
            lit_volume = pd.Series(np.nan, index=total_volume.index)
        dark = self.estimate_dark_volume(total_volume, lit_volume)
        if (dark.abs() < 1e-6).all():
            pct_dark = np.nan
        else:
            pct_dark = float((dark / (total_volume.replace(0, np.nan) + 1e-6)).mean())
        return {
            "mean_dark_volume": round(float(dark.mean()), 4),
            "std_dark_volume": round(float(dark.std()), 4),
            "pct_dark": round(pct_dark, 4) if not np.isnan(pct_dark) else np.nan,
            "autocorr_1d": round(float(dark.autocorr(1)), 4),
        }
