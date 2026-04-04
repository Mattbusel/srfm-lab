"""
crypto/onchain.py — On-chain signal-based trading strategies.

On-chain data provides unique signals about blockchain usage,
network value, and investor behavior not available in traditional markets.

References:
  - Willy Woo (2019): NVT Ratio
  - Trace Mayer (2015): Mayer Multiple
  - PlanB (2019): Stock-to-Flow model
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    indicator_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _stats(ec: np.ndarray) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    cagr = (ec[-1] / ec[0]) ** (1 / max(1, n / 252)) - 1
    r = np.diff(ec) / (ec[:-1] + 1e-9)
    r = np.concatenate([[0], r])
    std = r.std()
    sh = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar)


def _backtest(close, signal, initial=1_000_000, cost=0.001):
    n = len(close)
    equity = initial
    ec = np.full(n, initial, dtype=float)
    trades = []
    pos = 0.0
    ep = None
    for i in range(1, n):
        s = float(signal[i - 1]) if not np.isnan(signal[i - 1]) else 0.0
        if s != pos:
            if ep is not None and pos != 0:
                ret = pos * ((close[i] - ep) / ep - cost * 2)
                trades.append(ret)
            pos = s
            ep = close[i] if s != 0 else None
        if pos != 0:
            equity *= (1 + pos * (close[i] - close[i-1]) / (close[i-1] + 1e-9))
        ec[i] = equity
    return ec, trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. NVTRatio
# ─────────────────────────────────────────────────────────────────────────────

class NVTRatio:
    """
    Network Value to Transactions (NVT) Ratio.

    NVT = Market Cap / Daily Transaction Volume (USD)

    Similar to P/E ratio for crypto. High NVT = network overvalued relative
    to usage. Low NVT = undervalued. Invented by Willy Woo (2019).

    The NVT Signal (Dmitry Kalichkin variant) smooths transaction volume
    with a 90-day moving average to reduce noise.

    Parameters
    ----------
    lookback     : window for NVT mean/std normalization (default 365)
    entry_z      : z-score threshold for overvalued signal (default 1.5)
    exit_z       : z-score to exit (default 0.5)
    smooth_vol   : smoothing window for transaction volume (default 28)
    """

    def __init__(
        self,
        lookback: int = 365,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        smooth_vol: int = 28,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.smooth_vol = smooth_vol

    def compute_nvt(
        self,
        network_value: pd.Series,
        tx_volume: pd.Series,
    ) -> pd.Series:
        """
        Compute NVT ratio.

        Parameters
        ----------
        network_value : market capitalization (price × circulating supply)
        tx_volume     : daily on-chain transaction volume in USD
        """
        # Smooth transaction volume (NVT Signal variant)
        smooth_tx = tx_volume.rolling(self.smooth_vol, min_periods=5).mean()
        nvt = network_value / (smooth_tx + 1e-9)
        return nvt

    def nvt_zscore(self, nvt: pd.Series) -> pd.Series:
        """Z-score of log-NVT against rolling window."""
        log_nvt = np.log(nvt.clip(lower=1e-3))
        rolling_mean = log_nvt.rolling(self.lookback, min_periods=self.lookback // 3).mean()
        rolling_std = log_nvt.rolling(self.lookback, min_periods=self.lookback // 3).std()
        return (log_nvt - rolling_mean) / (rolling_std + 1e-9)

    def generate_signals(
        self,
        network_value: pd.Series,
        tx_volume: pd.Series,
    ) -> pd.Series:
        """
        Signal: +1 when NVT low (undervalued), -1 when NVT high (overvalued).
        """
        nvt = self.compute_nvt(network_value, tx_volume)
        z = self.nvt_zscore(nvt)
        signal = pd.Series(0.0, index=nvt.index)
        position = 0

        for i in range(self.lookback, len(z)):
            zi = float(z.iloc[i])
            if np.isnan(zi):
                continue
            if position == 0:
                if zi > self.entry_z:   # NVT high → overvalued → short/exit
                    position = -1
                elif zi < -self.entry_z:  # NVT low → undervalued → long
                    position = 1
            elif position == -1:
                if zi < self.exit_z:
                    position = 0
            elif position == 1:
                if zi > -self.exit_z:
                    position = 0
            signal.iloc[i] = float(position)

        signal.iloc[:self.lookback] = np.nan
        return signal

    def backtest(
        self,
        network_value: pd.Series,
        tx_volume: pd.Series,
        price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(network_value, tx_volume)
        nvt = self.compute_nvt(network_value, tx_volume)
        z = self.nvt_zscore(nvt)

        ec, trades = _backtest(price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price.index[1:]),
            signals=signal, indicator_series=z,
            params={"lookback": self.lookback, "entry_z": self.entry_z},
        )

    def current_signal(
        self,
        network_value: pd.Series,
        tx_volume: pd.Series,
    ) -> dict:
        """Get current NVT assessment."""
        nvt = self.compute_nvt(network_value, tx_volume)
        z = self.nvt_zscore(nvt)
        current_nvt = float(nvt.iloc[-1])
        current_z = float(z.iloc[-1])
        return {
            "nvt_ratio": current_nvt,
            "nvt_zscore": current_z,
            "signal": "OVERVALUED" if current_z > self.entry_z else
                      "UNDERVALUED" if current_z < -self.entry_z else "NEUTRAL",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MayerMultiple
# ─────────────────────────────────────────────────────────────────────────────

class MayerMultiple:
    """
    Mayer Multiple: Price / 200-day Moving Average.

    Originally proposed by Trace Mayer for Bitcoin valuation.
    The 200-day MA represents the long-term trend.

    Interpretation:
    - MM > 2.4: historically extreme (sell signal)
    - MM 1.0-2.4: normal bull market
    - MM < 1.0: bear market (accumulate)

    Parameters
    ----------
    ma_period    : period for long-term MA (default 200)
    overbought   : Mayer multiple level to sell (default 2.4)
    oversold     : level to buy (default 0.8)
    """

    def __init__(
        self,
        ma_period: int = 200,
        overbought: float = 2.4,
        oversold: float = 0.8,
    ):
        self.ma_period = ma_period
        self.overbought = overbought
        self.oversold = oversold

    def compute_mayer_multiple(self, price: pd.Series) -> pd.Series:
        """Compute the Mayer Multiple: price / 200d MA."""
        ma = price.rolling(self.ma_period, min_periods=self.ma_period // 2).mean()
        return price / (ma + 1e-9)

    def generate_signals(self, price: pd.Series) -> pd.Series:
        """Signal: +1 buy (MM low), -1 sell/reduce (MM high), 0 hold."""
        mm = self.compute_mayer_multiple(price)
        signal = pd.Series(1.0, index=price.index)  # default: long

        signal[mm > self.overbought] = -1.0   # extreme overbought → sell
        signal[mm < self.oversold] = 1.0      # extreme oversold → strong buy
        signal[(mm >= self.oversold) & (mm <= self.overbought)] = 1.0  # hold

        signal.iloc[:self.ma_period] = np.nan
        return signal

    def backtest(
        self,
        price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(price)
        mm = self.compute_mayer_multiple(price)

        ec, trades = _backtest(price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price.index[1:]),
            signals=signal, indicator_series=mm,
            params={"ma_period": self.ma_period, "overbought": self.overbought, "oversold": self.oversold},
        )

    def historical_distribution(self, price: pd.Series) -> dict:
        """Distribution statistics of the Mayer Multiple."""
        mm = self.compute_mayer_multiple(price).dropna()
        return {
            "mean": float(mm.mean()),
            "median": float(mm.median()),
            "std": float(mm.std()),
            "p10": float(np.percentile(mm, 10)),
            "p25": float(np.percentile(mm, 25)),
            "p75": float(np.percentile(mm, 75)),
            "p90": float(np.percentile(mm, 90)),
            "pct_above_2.4": float((mm > 2.4).mean()),
            "pct_below_0.8": float((mm < 0.8).mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. StockToFlowDeviation
# ─────────────────────────────────────────────────────────────────────────────

class StockToFlowDeviation:
    """
    Stock-to-Flow (S2F) deviation trading strategy.

    S2F = stock (circulating supply) / flow (new supply per year).
    S2F model predicts market cap as a power law of S2F.

    Trading signal: buy when price is below S2F model, sell above.

    The classic S2F model:
    ln(market_cap) = a + b * ln(S2F)

    Parameters
    ----------
    model_a    : S2F model intercept (default 14.6 — calibrated to BTC)
    model_b    : S2F model slope (default 3.36)
    threshold  : deviation threshold to trigger signal (default 0.3 = 30%)
    """

    def __init__(
        self,
        model_a: float = 14.6,
        model_b: float = 3.36,
        threshold: float = 0.3,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.threshold = threshold

    def compute_s2f(
        self,
        circulating_supply: pd.Series,
        daily_issuance: pd.Series,
    ) -> pd.Series:
        """
        Compute Stock-to-Flow ratio.

        Parameters
        ----------
        circulating_supply : total supply in circulation
        daily_issuance     : new coins mined per day
        """
        annual_flow = daily_issuance * 365
        s2f = circulating_supply / (annual_flow + 1e-9)
        return s2f

    def s2f_model_price(
        self,
        s2f: pd.Series,
        circulating_supply: pd.Series,
    ) -> pd.Series:
        """
        S2F model price.

        model_market_cap = exp(a + b * ln(S2F))
        model_price = model_market_cap / circulating_supply
        """
        log_s2f = np.log(s2f.clip(lower=0.1))
        log_model_mc = self.model_a + self.model_b * log_s2f
        model_mc = np.exp(log_model_mc)
        model_price = model_mc / (circulating_supply + 1e-9)
        return model_price

    def compute_deviation(
        self,
        actual_price: pd.Series,
        model_price: pd.Series,
    ) -> pd.Series:
        """
        Compute percentage deviation from S2F model.
        Positive = above model (overvalued). Negative = below (undervalued).
        """
        return (actual_price - model_price) / (model_price + 1e-9)

    def generate_signals(
        self,
        actual_price: pd.Series,
        s2f_model_price: pd.Series,
    ) -> pd.Series:
        """
        Signal: +1 when actual_price << model_price (undervalued by S2F)
                -1 when actual_price >> model_price (overvalued)
        """
        deviation = self.compute_deviation(actual_price, s2f_model_price)
        signal = pd.Series(0.0, index=actual_price.index)

        signal[deviation < -self.threshold] = 1.0   # below model → buy
        signal[deviation > self.threshold] = -1.0   # above model → sell/reduce

        return signal

    def backtest(
        self,
        actual_price: pd.Series,
        s2f_model_price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(actual_price, s2f_model_price)
        deviation = self.compute_deviation(actual_price, s2f_model_price)

        ec, trades = _backtest(actual_price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=actual_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=actual_price.index[1:]),
            signals=signal, indicator_series=deviation,
            params={"model_a": self.model_a, "model_b": self.model_b, "threshold": self.threshold},
        )

    def fit_s2f_model(
        self,
        price: pd.Series,
        s2f: pd.Series,
    ) -> Tuple[float, float, float]:
        """
        Fit S2F model using OLS regression of ln(price) on ln(S2F).
        Returns (intercept, slope, r_squared).
        """
        log_s2f = np.log(s2f.clip(lower=0.1))
        log_price = np.log(price.clip(lower=0.01))

        # Align
        common_idx = log_s2f.index.intersection(log_price.index)
        x = log_s2f.reindex(common_idx).dropna()
        y = log_price.reindex(common_idx).dropna()
        common = x.index.intersection(y.index)
        x = x.loc[common].values
        y = y.loc[common].values

        if len(x) < 10:
            return self.model_a, self.model_b, 0.0

        n = len(x)
        xm = x.mean(); ym = y.mean()
        sxx = np.sum((x - xm) ** 2)
        sxy = np.sum((x - xm) * (y - ym))
        slope = sxy / (sxx + 1e-12)
        intercept = ym - slope * xm

        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - ym) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)

        return float(intercept), float(slope), float(r2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FearGreedSignal
# ─────────────────────────────────────────────────────────────────────────────

class FearGreedSignal:
    """
    Crypto Fear & Greed Index contrarian trading strategy.

    The Fear & Greed Index (0-100) measures market sentiment.
    - 0-25: Extreme Fear (contrarian buy)
    - 26-49: Fear (potential buy)
    - 50: Neutral
    - 51-74: Greed (potential sell)
    - 75-100: Extreme Greed (contrarian sell)

    Contrarian strategy: buy at extreme fear, sell at extreme greed.

    Parameters
    ----------
    entry_threshold  : FGI level for contrarian buy signal (default 25)
    exit_threshold   : FGI level for contrarian sell signal (default 75)
    smoothing        : days to smooth the FGI (default 3)
    """

    def __init__(
        self,
        entry_threshold: float = 25.0,
        exit_threshold: float = 75.0,
        smoothing: int = 3,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.smoothing = smoothing

    def smooth_fgi(self, fear_greed_index: pd.Series) -> pd.Series:
        """Apply smoothing to reduce noise in FGI."""
        if self.smoothing > 1:
            return fear_greed_index.ewm(span=self.smoothing, adjust=False).mean()
        return fear_greed_index

    def generate_signals(self, fear_greed_index: pd.Series) -> pd.Series:
        """
        Contrarian signals:
        +1 = buy at extreme fear
        -1 = sell at extreme greed
         0 = neutral between thresholds
        """
        fgi = self.smooth_fgi(fear_greed_index)
        signal = pd.Series(0.0, index=fgi.index)
        position = 0

        for i in range(1, len(fgi)):
            f = float(fgi.iloc[i])
            if np.isnan(f):
                continue

            if position == 0:
                if f <= self.entry_threshold:   # extreme fear → buy
                    position = 1
                elif f >= self.exit_threshold:  # extreme greed → sell
                    position = -1

            elif position == 1:
                if f >= 50:   # return to neutral/greed → exit long
                    position = 0

            elif position == -1:
                if f <= 50:   # return to neutral/fear → exit short
                    position = 0

            signal.iloc[i] = float(position)

        return signal

    def backtest(
        self,
        fear_greed_index: pd.Series,
        price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(fear_greed_index)
        fgi_smooth = self.smooth_fgi(fear_greed_index)

        ec, trades = _backtest(price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price.index[1:]),
            signals=signal, indicator_series=fgi_smooth,
            params={"entry_threshold": self.entry_threshold, "exit_threshold": self.exit_threshold},
        )

    def regime_statistics(
        self,
        fear_greed_index: pd.Series,
        price: pd.Series,
    ) -> pd.DataFrame:
        """Statistics of returns in each FGI regime."""
        fgi = self.smooth_fgi(fear_greed_index)
        bins = [0, 25, 49, 51, 75, 100]
        labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        regimes = pd.cut(fgi, bins=bins, labels=labels, include_lowest=True)
        returns = price.pct_change().fillna(0)
        forward_ret = returns.shift(-5).rolling(5).sum()

        rows = []
        for reg in labels:
            mask = regimes == reg
            fwd = forward_ret[mask].dropna()
            if len(fwd) < 5:
                continue
            rows.append({
                "regime": reg,
                "n_obs": len(fwd),
                "mean_5d_return": float(fwd.mean()),
                "hit_rate": float((fwd > 0).mean()),
                "avg_fgi": float(fgi[mask].mean()),
            })
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2021-01-01", periods=n, freq="D")

    # Bitcoin-like price
    price = pd.Series(40000.0 * np.cumprod(1 + rng.normal(0.001, 0.04, n)), index=idx)

    # Simulate on-chain metrics
    supply = pd.Series(np.linspace(18.5e6, 19.2e6, n), index=idx)  # growing supply
    daily_issuance = pd.Series(np.linspace(900, 450, n), index=idx)  # halving trend
    market_cap = price * supply
    tx_volume = market_cap * pd.Series(np.abs(rng.normal(0.03, 0.01, n)), index=idx)

    # NVT
    nvt = NVTRatio(lookback=365, entry_z=1.5)
    res1 = nvt.backtest(market_cap, tx_volume, price)
    print("NVT:", res1.summary())

    # Mayer Multiple
    mm = MayerMultiple(ma_period=200, overbought=2.4, oversold=0.8)
    res2 = mm.backtest(price)
    print("Mayer Multiple:", res2.summary())
    print("Historical distribution:", mm.historical_distribution(price))

    # S2F
    s2f_vals = StockToFlowDeviation()
    s2f = s2f_vals.compute_s2f(supply, daily_issuance)
    model_p = s2f_vals.s2f_model_price(s2f, supply)
    res3 = s2f_vals.backtest(price, model_p)
    print("S2F Deviation:", res3.summary())
    a, b, r2 = s2f_vals.fit_s2f_model(price, s2f)
    print(f"S2F fit: a={a:.2f} b={b:.2f} R²={r2:.3f}")

    # Fear & Greed
    fgi = pd.Series(rng.integers(0, 100, n).astype(float), index=idx)
    fgi_smoother = fgi.ewm(span=7).mean()
    fg = FearGreedSignal(entry_threshold=25, exit_threshold=75)
    res4 = fg.backtest(fgi_smoother, price)
    print("Fear & Greed:", res4.summary())
