"""
momentum_reversal.py -- Dual momentum with reversal detection.

References:
  - Moskowitz, Ooi, Pedersen (2012): Time Series Momentum
  - Jegadeesh, Titman (1993): Cross-sectional momentum
  - Antonacci (2014): Dual Momentum Investing
  - Hurst (1951): Long-term storage capacity of reservoirs

BH constants used for reversal detection:
  BH_MASS_THRESH = 1.92
  BH_DECAY       = 0.924
  BH_COLLAPSE    = 0.992

Strategy combines:
  1. CrossSectionalMomentum  -- rank-based, long top quartile, short bottom
  2. TimeSeriesMomentum      -- sign(12m return) * vol-scaled position
  3. ReversalDetector        -- flip signal when Hurst < 0.38 or BH collapse
  4. DualMomentumFilter      -- absolute momentum gate using benchmark
  5. PositionSizer           -- Kelly fraction with Hurst confidence scaling

LARSA v18 compatible.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# BH physics constants
# ---------------------------------------------------------------------------
BH_MASS_THRESH = 1.92
BH_DECAY       = 0.924
BH_COLLAPSE    = 0.992

HURST_TRENDING    = 0.58  # H > 0.58 -> trending regime
HURST_REVERTING   = 0.38  # H < 0.38 -> strong mean reversion


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class MomentumSignal:
    """Unified signal from MomentumReversalStrategy."""
    symbol: str          = ""
    ts_signal: float     = 0.0  # time-series momentum raw
    cs_rank: float       = 0.0  # cross-sectional rank (0..1)
    reversal_flag: bool  = False  # True if reversal mode active
    hurst: float         = 0.5
    bh_mass: float       = 0.0
    final_signal: float  = 0.0  # signed position (-1..+1)
    position_size: float = 0.0  # Kelly-scaled position
    reason: str          = ""


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    attribution: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
            f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
            f"Trades={self.n_trades}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n = len(equity_curve)
    initial = equity_curve[0]
    final   = equity_curve[-1]
    total_return = final / initial - 1.0
    n_years = max(1, n / 252)
    cagr    = (final / initial) ** (1.0 / n_years) - 1.0
    rets    = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    rets    = np.concatenate([[0.0], rets])
    std     = rets.std()
    sharpe  = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down    = rets[rets < 0]
    sortino_d = np.std(down) if len(down) > 0 else 1e-9
    sortino   = rets.mean() / sortino_d * math.sqrt(252)
    pk  = np.maximum.accumulate(equity_curve)
    dd  = (equity_curve - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0.0
    wins    = [r for r in trade_returns if r > 0]
    losses  = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown=mdd, calmar=calmar, win_rate=win_rate, profit_factor=pf,
        n_trades=len(trade_returns),
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        returns=pd.Series(rets),
    )


def _ewm_vol(returns: pd.Series, span: int = 63) -> pd.Series:
    """EWM annualized volatility."""
    return returns.ewm(span=span, adjust=False).std() * math.sqrt(252)


def _compute_hurst(prices: np.ndarray, min_len: int = 100) -> float:
    """
    Hurst exponent via rescaled range (R/S) analysis.
    Returns H in [0, 1]:  0.5 = random walk, > 0.5 trending, < 0.5 mean-reverting.
    """
    n = len(prices)
    if n < min_len:
        return 0.5
    log_prices = np.log(prices + 1e-9)
    lags       = [max(2, n // k) for k in range(2, min(20, n // 10 + 1))]
    rs_vals    = []
    lag_vals   = []
    for lag in sorted(set(lags)):
        if lag >= n:
            continue
        chunk = log_prices[:lag]
        diff  = np.diff(chunk)
        if len(diff) < 2:
            continue
        mean_diff = diff.mean()
        cumdev    = np.cumsum(diff - mean_diff)
        R         = cumdev.max() - cumdev.min()
        S         = diff.std()
        if S < 1e-12:
            continue
        rs_vals.append(math.log(R / S + 1e-12))
        lag_vals.append(math.log(lag))
    if len(rs_vals) < 4:
        return 0.5
    h = float(np.polyfit(lag_vals, rs_vals, 1)[0])
    return max(0.01, min(0.99, h))


# ---------------------------------------------------------------------------
# 1. CrossSectionalMomentum
# ---------------------------------------------------------------------------

class CrossSectionalMomentum:
    """
    Rank instruments by 12-1 month return (standard Jegadeesh-Titman).
    Long top quartile, short bottom quartile.

    Parameters
    ----------
    lookback     : formation period in bars (default 252 -- 12 months)
    skip_recent  : bars to skip at end (default 21 -- 1 month)
    top_frac     : fraction of universe to go long (default 0.25)
    bottom_frac  : fraction of universe to short (default 0.25)
    rebal_freq   : rebalance every N bars (default 21)
    """

    def __init__(
        self,
        lookback: int    = 252,
        skip_recent: int = 21,
        top_frac: float  = 0.25,
        bottom_frac: float = 0.25,
        rebal_freq: int  = 21,
    ):
        self.lookback    = lookback
        self.skip_recent = skip_recent
        self.top_frac    = top_frac
        self.bottom_frac = bottom_frac
        self.rebal_freq  = rebal_freq

    def rank_returns(self, prices: pd.DataFrame, at_i: int) -> pd.Series:
        """Return cross-sectional momentum ranks (0=worst, 1=best) at bar index at_i."""
        if at_i < self.lookback + self.skip_recent:
            return pd.Series(0.5, index=prices.columns)
        end_i   = at_i - self.skip_recent
        start_i = at_i - self.lookback
        if start_i < 0:
            return pd.Series(0.5, index=prices.columns)
        row_end   = prices.iloc[end_i]
        row_start = prices.iloc[start_i]
        mom = (row_end - row_start) / (row_start.abs() + 1e-9)
        ranked = mom.rank(pct=True)  # percentile rank 0..1
        return ranked

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute portfolio weights over time.
        Returns DataFrame with same index/columns as prices.
        """
        n_assets = prices.shape[1]
        n_top    = max(1, int(n_assets * self.top_frac))
        n_bottom = max(1, int(n_assets * self.bottom_frac))
        weights  = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        warmup = self.lookback + self.skip_recent
        for i in range(warmup, len(prices), self.rebal_freq):
            ranks   = self.rank_returns(prices, i)
            w       = pd.Series(0.0, index=prices.columns)
            long_a  = ranks.nlargest(n_top).index
            short_a = ranks.nsmallest(n_bottom).index
            w[long_a]  =  1.0 / n_top
            w[short_a] = -1.0 / n_bottom
            end_i = min(i + self.rebal_freq, len(prices))
            weights.iloc[i:end_i] = w.values

        return weights

    def momentum_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Return full time-series of cross-sectional momentum scores.
        Scores are percentile ranks at each bar.
        """
        scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        warmup = self.lookback + self.skip_recent
        for i in range(warmup, len(prices)):
            scores.iloc[i] = self.rank_returns(prices, i).values
        return scores.fillna(0.5)


# ---------------------------------------------------------------------------
# 2. TimeSeriesMomentum
# ---------------------------------------------------------------------------

class TimeSeriesMomentum:
    """
    sign(12-month return) * vol-scaled position.

    Parameters
    ----------
    lookback    : lookback in bars (default 252)
    skip_recent : bars to skip (default 21)
    target_vol  : annualized vol target (default 0.40)
    vol_span    : EWM span for vol estimate (default 63)
    """

    def __init__(
        self,
        lookback: int    = 252,
        skip_recent: int = 21,
        target_vol: float = 0.40,
        vol_span: int    = 63,
    ):
        self.lookback    = lookback
        self.skip_recent = skip_recent
        self.target_vol  = target_vol
        self.vol_span    = vol_span

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Returns vol-scaled signed position series."""
        close   = df["close"]
        returns = close.pct_change()

        # Past return: from lookback to skip_recent bars ago
        past_return = (close.shift(self.skip_recent) - close.shift(self.lookback)) / (
            close.shift(self.lookback).abs() + 1e-9
        )
        direction    = np.sign(past_return)
        realized_vol = _ewm_vol(returns, span=self.vol_span).replace(0, np.nan).ffill()
        safe_vol     = realized_vol.where(realized_vol > 0.01, 0.01)
        position     = direction * (self.target_vol / safe_vol)
        position     = position.clip(-2.0, 2.0)
        position.iloc[: self.lookback] = np.nan
        return position

    def raw_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Return raw (unscaled) momentum signal."""
        close = df["close"]
        return np.sign(
            (close.shift(self.skip_recent) - close.shift(self.lookback))
            / (close.shift(self.lookback).abs() + 1e-9)
        )


# ---------------------------------------------------------------------------
# 3. ReversalDetector
# ---------------------------------------------------------------------------

class ReversalDetector:
    """
    Detects reversal regimes based on Hurst exponent and BH mass dynamics.

    Reversal triggered when:
      1. Hurst < hurst_reversal_threshold (default 0.38) -- strong mean reversion
      2. BH mass was recently > BH_COLLAPSE and is now decaying -- collapse aftermath

    When reversal is active, the momentum signal is flipped:
      buy recent losers, sell recent winners.

    Parameters
    ----------
    hurst_reversal_threshold : Hurst threshold for reversal (default 0.38)
    hurst_window             : bars used to compute rolling Hurst (default 120)
    bh_decay_lookback        : bars to look back for BH mass peak (default 10)
    """

    def __init__(
        self,
        hurst_reversal_threshold: float = 0.38,
        hurst_window: int               = 120,
        bh_decay_lookback: int          = 10,
    ):
        self.hurst_threshold  = hurst_reversal_threshold
        self.hurst_window     = hurst_window
        self.bh_decay_lookback = bh_decay_lookback

    def compute_hurst_series(self, prices: pd.Series) -> pd.Series:
        """
        Rolling Hurst exponent over hurst_window bars.
        Returns NaN where insufficient data.
        """
        hurst_vals = np.full(len(prices), np.nan)
        arr = prices.values
        for i in range(self.hurst_window, len(arr)):
            window = arr[i - self.hurst_window: i]
            hurst_vals[i] = _compute_hurst(window)
        return pd.Series(hurst_vals, index=prices.index)

    def compute_bh_mass(self, returns: pd.Series, window: int = 21) -> pd.Series:
        """
        Simplified BH mass proxy: rolling cumulative absolute return scaled by decay.

        BH mass accumulates when large moves occur and decays at BH_DECAY rate.
        High mass > BH_COLLAPSE indicates a potential market collapse regime.
        """
        abs_ret   = returns.abs()
        cum_ret   = abs_ret.rolling(window, min_periods=5).mean()
        # Normalize to [0, 2] range scaled against BH_MASS_THRESH
        bh_mass   = (cum_ret / (cum_ret.rolling(252, min_periods=50).quantile(0.95) + 1e-9)).clip(0, 2.0)
        return bh_mass.fillna(0.0)

    def detect(self, prices: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Return boolean Series: True where reversal mode should be active.

        Reversal active when:
          - Hurst < threshold (strong mean reversion)
          - OR: BH mass peaked above BH_COLLAPSE in past bh_decay_lookback bars
            and is now declining (collapse aftermath)
        """
        hurst  = self.compute_hurst_series(prices)
        bh_mass = self.compute_bh_mass(returns)

        hurst_reversal = hurst < self.hurst_threshold

        # BH collapse aftermath: was mass > BH_COLLAPSE recently, now decaying
        bh_peak_recent = bh_mass.rolling(self.bh_decay_lookback).max().shift(1)
        bh_now         = bh_mass
        bh_reversal    = (bh_peak_recent > BH_COLLAPSE) & (bh_now < bh_peak_recent * BH_DECAY)

        reversal_active = hurst_reversal | bh_reversal
        return reversal_active.fillna(False)


# ---------------------------------------------------------------------------
# 4. DualMomentumFilter
# ---------------------------------------------------------------------------

class DualMomentumFilter:
    """
    Absolute momentum gate (Antonacci 2014).

    If the benchmark 12-month return < 0, signal goes to cash (0).
    This prevents holding risk assets in bear markets.

    Parameters
    ----------
    lookback        : lookback in bars (default 252)
    min_return      : minimum benchmark return to allow long exposure (default 0.0)
    benchmark_col   : column name for benchmark in the prices DataFrame
    """

    def __init__(
        self,
        lookback: int       = 252,
        min_return: float   = 0.0,
        benchmark_col: str  = "SPY",
    ):
        self.lookback       = lookback
        self.min_return     = min_return
        self.benchmark_col  = benchmark_col

    def get_filter_mask(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns boolean Series: True = allow signals, False = go to cash.
        If benchmark column not present, always returns True.
        """
        if self.benchmark_col not in prices.columns:
            return pd.Series(True, index=prices.index)
        bench = prices[self.benchmark_col]
        past_return = (bench - bench.shift(self.lookback)) / (bench.shift(self.lookback).abs() + 1e-9)
        allow = past_return > self.min_return
        allow.iloc[: self.lookback] = False
        return allow.fillna(False)

    def apply(self, signals: pd.Series, prices: pd.DataFrame) -> pd.Series:
        """
        Apply the absolute momentum filter to a signal series.
        Signals outside the allowed zone are set to 0.
        """
        mask   = self.get_filter_mask(prices)
        filtered = signals.copy()
        filtered[~mask] = 0.0
        return filtered


# ---------------------------------------------------------------------------
# 5. PositionSizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Kelly fraction position sizer with Hurst confidence scaling.

    Full Kelly can be aggressive -- uses fractional Kelly (default 0.5).
    Position scaled by Hurst confidence: trending (H > 0.58) gets full size,
    mean-reverting (H < 0.42) gets reduced size for momentum, flipped for reversal.

    Parameters
    ----------
    kelly_fraction  : fraction of Kelly criterion to use (default 0.5)
    max_leverage    : maximum total leverage (default 2.0)
    target_vol      : annual vol target for base sizing (default 0.20)
    vol_span        : EWM span for vol estimate (default 63)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_leverage: float   = 2.0,
        target_vol: float     = 0.20,
        vol_span: int         = 63,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_leverage   = max_leverage
        self.target_vol     = target_vol
        self.vol_span       = vol_span

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly: f* = p/|loss| - (1-p)/win
        """
        if avg_win < 1e-9 or avg_loss < 1e-9:
            return 0.0
        p  = win_rate
        q  = 1.0 - p
        f  = p / abs(avg_loss) - q / avg_win
        return max(0.0, f * self.kelly_fraction)

    def size_signal(
        self,
        signal: pd.Series,
        returns: pd.Series,
        hurst: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Scale signal by Hurst confidence and vol targeting.

        When Hurst is high (trending): scale up to max_leverage.
        When Hurst is low (mean reverting): reduce position size.

        Parameters
        ----------
        signal  : signed position signal (-1..+1)
        returns : return series for vol estimation
        hurst   : optional Hurst series (aligned with signal/returns)
        """
        # Volatility-scaled size
        vol      = _ewm_vol(returns, span=self.vol_span).replace(0, np.nan).ffill().fillna(0.20)
        vol_size = (self.target_vol / vol).clip(upper=self.max_leverage)

        # Hurst confidence multiplier
        if hurst is not None:
            hurst_aligned = hurst.reindex(signal.index).ffill().fillna(0.5)
            # Trending regime: H > 0.58 -> multiplier increases from 1.0
            # Mean-reverting: H < 0.42 -> multiplier decreases toward 0.5
            h_multiplier = np.where(
                hurst_aligned > HURST_TRENDING,
                1.0 + (hurst_aligned - HURST_TRENDING) * 2.0,
                np.where(
                    hurst_aligned < HURST_REVERTING,
                    0.5,
                    1.0,
                ),
            )
            h_multiplier = pd.Series(h_multiplier, index=signal.index).clip(0.5, 2.0)
        else:
            h_multiplier = pd.Series(1.0, index=signal.index)

        sized = signal * vol_size * h_multiplier
        return sized.clip(-self.max_leverage, self.max_leverage)


# ---------------------------------------------------------------------------
# 6. Combined MomentumReversalStrategy
# ---------------------------------------------------------------------------

class MomentumReversalStrategy:
    """
    Combines cross-sectional momentum (rank-based) with time-series momentum
    and detects reversals using BH mass / Hurst exponent signals.

    Architecture:
      1. TimeSeriesMomentum generates the primary signal
      2. ReversalDetector checks if regime calls for signal flip
      3. DualMomentumFilter applies absolute momentum cash gate
      4. PositionSizer scales by Hurst confidence and Kelly

    Parameters
    ----------
    config : dict with parameter overrides
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.ts_mom    = TimeSeriesMomentum(
            lookback=cfg.get("lookback", 252),
            skip_recent=cfg.get("skip_recent", 21),
            target_vol=cfg.get("ts_target_vol", 0.40),
            vol_span=cfg.get("vol_span", 63),
        )
        self.reversal  = ReversalDetector(
            hurst_reversal_threshold=cfg.get("hurst_reversal_threshold", 0.38),
            hurst_window=cfg.get("hurst_window", 120),
            bh_decay_lookback=cfg.get("bh_decay_lookback", 10),
        )
        self.abs_filter = DualMomentumFilter(
            lookback=cfg.get("abs_lookback", 252),
            min_return=cfg.get("abs_min_return", 0.0),
            benchmark_col=cfg.get("benchmark_col", "SPY"),
        )
        self.sizer     = PositionSizer(
            kelly_fraction=cfg.get("kelly_fraction", 0.5),
            max_leverage=cfg.get("max_leverage", 2.0),
            target_vol=cfg.get("target_vol", 0.20),
            vol_span=cfg.get("vol_span", 63),
        )
        self.config    = cfg

    def generate_signals(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate combined signals for a single instrument.

        Parameters
        ----------
        df           : OHLCV DataFrame with 'close' column
        benchmark_df : optional SPY/benchmark DataFrame for DualMomentumFilter

        Returns
        -------
        DataFrame with columns: ts_signal, reversal_flag, hurst, bh_mass,
                                 filtered_signal, final_signal
        """
        close   = df["close"]
        returns = close.pct_change()

        # Time-series momentum signal
        ts_sig  = self.ts_mom.generate_signals(df)

        # Reversal detection
        hurst_series = self.reversal.compute_hurst_series(close)
        bh_mass      = self.reversal.compute_bh_mass(returns)
        rev_mask     = self.reversal.detect(close, returns)

        # Flip momentum signal during reversal regime
        adj_signal = ts_sig.copy()
        adj_signal[rev_mask] = -adj_signal[rev_mask]

        # Absolute momentum filter
        if benchmark_df is not None:
            bench_close = benchmark_df["close"].rename("SPY")
            prices_for_filter = pd.DataFrame({"SPY": bench_close}).reindex(df.index).ffill()
            adj_signal = self.abs_filter.apply(adj_signal, prices_for_filter)

        # Position sizing with Hurst confidence
        final = self.sizer.size_signal(adj_signal, returns, hurst=hurst_series)

        result = pd.DataFrame({
            "ts_signal":      ts_sig,
            "reversal_flag":  rev_mask.astype(float),
            "hurst":          hurst_series,
            "bh_mass":        bh_mass,
            "filtered_signal": adj_signal,
            "final_signal":   final,
        }, index=df.index)
        return result

    def backtest(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0002,
    ) -> BacktestResult:
        """
        Single-instrument backtest with full performance attribution.
        """
        sig_df = self.generate_signals(df, benchmark_df=benchmark_df)
        close  = df["close"].values
        signals = sig_df["final_signal"].values

        # Run equity simulation
        n          = len(close)
        equity     = initial_equity
        eq_curve   = np.full(n, initial_equity, dtype=float)
        trade_ret  = []
        prev_sig   = 0.0
        entry_px   = None

        for i in range(1, n):
            if np.isnan(signals[i - 1]):
                eq_curve[i] = equity
                continue
            new_sig = float(signals[i - 1])
            if abs(new_sig - prev_sig) > 0.01 and entry_px is not None:
                # Close previous position
                ret = prev_sig * ((close[i] - entry_px) / (entry_px + 1e-9) - commission_pct * 2)
                trade_ret.append(float(ret))
                entry_px = close[i] if new_sig != 0 else None
            elif entry_px is None and new_sig != 0:
                entry_px = close[i]
                equity  *= (1.0 - commission_pct)

            if new_sig != 0:
                bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
                equity *= (1 + new_sig * bar_ret)

            prev_sig    = new_sig
            eq_curve[i] = equity

        stats = _compute_stats(eq_curve, trade_ret)

        # Attribution: split performance by regime
        reversal_flags = sig_df["reversal_flag"].values
        attr = self._attribution(eq_curve, reversal_flags)

        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=sig_df["final_signal"],
            attribution=attr,
            params=self.config,
        )

    def _attribution(self, equity_curve: np.ndarray, reversal_flags: np.ndarray) -> dict:
        """Compute return attribution: momentum regime vs reversal regime."""
        rets          = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        rev_flags     = reversal_flags[1:].astype(bool)
        mom_rets      = rets[~rev_flags]
        rev_rets      = rets[rev_flags]
        return {
            "momentum_mean_daily":  float(mom_rets.mean()) if len(mom_rets) > 0 else 0.0,
            "reversal_mean_daily":  float(rev_rets.mean()) if len(rev_rets) > 0 else 0.0,
            "pct_time_reversal":    float(rev_flags.mean()),
            "momentum_sharpe":      float(mom_rets.mean() / (mom_rets.std() + 1e-9) * math.sqrt(252)),
            "reversal_sharpe":      float(rev_rets.mean() / (rev_rets.std() + 1e-9) * math.sqrt(252)),
        }


# ---------------------------------------------------------------------------
# 7. MomentumReversalBacktest  # multi-instrument with attribution
# ---------------------------------------------------------------------------

class MomentumReversalBacktest:
    """
    15-year multi-instrument backtest with full performance attribution.

    Runs MomentumReversalStrategy on each instrument independently,
    then combines into an equal-weighted portfolio.

    Parameters
    ----------
    config         : passed to MomentumReversalStrategy
    initial_equity : per-instrument starting equity (default 1_000_000)
    commission_pct : commission per side (default 0.0002)
    """

    def __init__(
        self,
        config: Optional[dict]  = None,
        initial_equity: float   = 1_000_000.0,
        commission_pct: float   = 0.0002,
    ):
        self.config         = config or {}
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct

    def run(
        self,
        instrument_data: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for each instrument in instrument_data.

        Returns
        -------
        dict mapping symbol -> BacktestResult, plus "portfolio" key
        """
        results = {}
        strat   = MomentumReversalStrategy(config=self.config)

        for sym, df in instrument_data.items():
            try:
                res = strat.backtest(df, benchmark_df=benchmark_df,
                                     initial_equity=self.initial_equity,
                                     commission_pct=self.commission_pct)
                results[sym] = res
            except Exception as e:
                warnings.warn(f"Backtest failed for {sym}: {e}")

        # Build equal-weight portfolio from individual returns
        if len(results) > 1:
            all_rets = pd.DataFrame({
                sym: r.returns for sym, r in results.items() if len(r.returns) > 0
            }).dropna()
            if len(all_rets) > 0:
                port_rets  = all_rets.mean(axis=1)
                port_equity = self.initial_equity * (1 + port_rets).cumprod()
                port_equity = port_equity.values
                # Build a synthetic BacktestResult for the portfolio
                stats = _compute_stats(port_equity, list(port_rets[port_rets != 0]))
                results["portfolio"] = BacktestResult(
                    **{k: v for k, v in stats.items() if k != "returns"},
                    equity_curve=pd.Series(port_equity, index=all_rets.index),
                    returns=port_rets,
                    params={"type": "equal_weight_portfolio"},
                )

        return results

    def attribution_table(
        self, results: Dict[str, BacktestResult]
    ) -> pd.DataFrame:
        """Summary table of all results including attribution."""
        rows = []
        for sym, r in results.items():
            row = {
                "symbol":       sym,
                "total_return": r.total_return,
                "cagr":         r.cagr,
                "sharpe":       r.sharpe,
                "max_drawdown": r.max_drawdown,
                "win_rate":     r.win_rate,
                "n_trades":     r.n_trades,
            }
            if r.attribution:
                row.update({
                    "mom_sharpe": r.attribution.get("momentum_sharpe", 0.0),
                    "rev_sharpe": r.attribution.get("reversal_sharpe", 0.0),
                    "pct_reversal": r.attribution.get("pct_time_reversal", 0.0),
                })
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n   = 3780  # ~15 years of trading days
    idx = pd.date_range("2009-01-01", periods=n, freq="B")

    # Simulate price with trending and mean-reverting periods
    regimes    = np.repeat([1, -1, 0.5, -0.5, 1], n // 5)[:n]
    daily_ret  = rng.normal(regimes * 0.0002, 0.012, n)
    close      = 100.0 * np.cumprod(1 + daily_ret)
    spy_close  = 200.0 * np.cumprod(1 + rng.normal(0.0003, 0.010, n))

    df       = pd.DataFrame({"close": close}, index=idx)
    spy_df   = pd.DataFrame({"close": spy_close}, index=idx)

    strat    = MomentumReversalStrategy()
    result   = strat.backtest(df, benchmark_df=spy_df)
    print("Single instrument:", result.summary())
    print("Attribution:", result.attribution)

    # Multi-instrument
    instruments = {}
    for sym in ["A", "B", "C", "D"]:
        c = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.011, n))
        instruments[sym] = pd.DataFrame({"close": c}, index=idx)

    bt      = MomentumReversalBacktest()
    results = bt.run(instruments, benchmark_df=spy_df)
    tbl     = bt.attribution_table(results)
    print(tbl.to_string())
