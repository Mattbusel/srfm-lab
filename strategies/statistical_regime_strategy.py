"""
statistical_regime_strategy.py -- Meta-strategy that switches between sub-strategies
based on detected statistical regime.

Regime types:
  TREND       -- BH mass strategy + momentum
  MEAN_REVERT -- pairs + Hurst-guided reversal
  HIGH_VOL    -- volatility breakout + reduced size
  CRISIS      -- move to cash + short VIX proxy

Regime classification uses 5 features:
  1. Hurst exponent
  2. vol_regime (realized / historical vol ratio)
  3. bh_mass (black hole mass proxy)
  4. adx (average directional index)
  5. vix_level (implied or realized vol level)

BH constants:
  BH_MASS_THRESH = 1.92
  BH_DECAY       = 0.924
  BH_COLLAPSE    = 0.992

Hurst thresholds:
  H > 0.58 -> trending
  H < 0.38 -> strong mean reversion

LARSA v18 compatible.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
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

HURST_TRENDING  = 0.58
HURST_REVERTING = 0.38


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    TREND       = "TREND"
    MEAN_REVERT = "MEAN_REVERT"
    HIGH_VOL    = "HIGH_VOL"
    CRISIS      = "CRISIS"
    UNKNOWN     = "UNKNOWN"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    regime: Regime       = Regime.UNKNOWN
    hurst: float         = 0.5
    vol_regime: float    = 1.0  # realized_vol / long_run_vol
    bh_mass: float       = 0.0
    adx: float           = 0.0
    vix_level: float     = 0.20  # implied or realized vol level
    confidence: float    = 0.0  # 0..1 classifier confidence
    features: dict       = field(default_factory=dict)


@dataclass
class RegimeBacktestResult:
    total_return: float   = 0.0
    cagr: float           = 0.0
    sharpe: float         = 0.0
    sortino: float        = 0.0
    max_drawdown: float   = 0.0
    calmar: float         = 0.0
    win_rate: float       = 0.0
    profit_factor: float  = 0.0
    n_trades: int         = 0
    avg_trade_return: float = 0.0
    n_regime_changes: int = 0
    equity_curve: pd.Series        = field(default_factory=pd.Series)
    returns: pd.Series             = field(default_factory=pd.Series)
    regime_series: pd.Series       = field(default_factory=pd.Series)
    sub_weights: pd.DataFrame      = field(default_factory=pd.DataFrame)
    regime_stats: Dict[str, dict]  = field(default_factory=dict)
    params: dict                   = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
            f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
            f"Trades={self.n_trades} RegimeChanges={self.n_regime_changes}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n  = len(equity_curve)
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


def _compute_hurst(prices: np.ndarray) -> float:
    """Hurst exponent via R/S analysis."""
    n = len(prices)
    if n < 50:
        return 0.5
    log_p = np.log(prices + 1e-9)
    lags  = sorted(set([max(2, n // k) for k in range(2, min(20, n // 10 + 1))]))
    rs_vals, lag_vals = [], []
    for lag in lags:
        if lag >= n:
            continue
        diff = np.diff(log_p[:lag])
        if len(diff) < 2:
            continue
        cumdev = np.cumsum(diff - diff.mean())
        R = cumdev.max() - cumdev.min()
        S = diff.std()
        if S < 1e-12:
            continue
        rs_vals.append(math.log(R / S + 1e-12))
        lag_vals.append(math.log(lag))
    if len(rs_vals) < 4:
        return 0.5
    h = float(np.polyfit(lag_vals, rs_vals, 1)[0])
    return max(0.01, min(0.99, h))


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average Directional Index (Wilder).
    Returns array of ADX values (0..100).
    """
    n   = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2:
        return adx

    tr   = np.zeros(n)
    dm_p = np.zeros(n)
    dm_m = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        tr[i]   = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        dm_p[i] = max(h_diff, 0) if h_diff > l_diff else 0
        dm_m[i] = max(l_diff, 0) if l_diff > h_diff else 0

    # Smooth with Wilder EMA
    alpha = 1.0 / period
    tr_s  = np.zeros(n)
    dmp_s = np.zeros(n)
    dmm_s = np.zeros(n)
    tr_s[period]  = tr[1:period + 1].sum()
    dmp_s[period] = dm_p[1:period + 1].sum()
    dmm_s[period] = dm_m[1:period + 1].sum()
    for i in range(period + 1, n):
        tr_s[i]  = tr_s[i - 1]  * (1 - alpha) + tr[i]
        dmp_s[i] = dmp_s[i - 1] * (1 - alpha) + dm_p[i]
        dmm_s[i] = dmm_s[i - 1] * (1 - alpha) + dm_m[i]

    di_p  = 100 * dmp_s / (tr_s + 1e-9)
    di_m  = 100 * dmm_s / (tr_s + 1e-9)
    dx    = 100 * np.abs(di_p - di_m) / (np.abs(di_p + di_m) + 1e-9)

    # ADX = smoothed DX
    adx[period * 2] = dx[period + 1: period * 2 + 1].mean()
    for i in range(period * 2 + 1, n):
        adx[i] = adx[i - 1] * (1 - alpha) + dx[i] * alpha

    return adx


# ---------------------------------------------------------------------------
# 1. RegimeClassifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    5-feature rule-cascade classifier.

    Feature vector: (hurst, vol_regime, bh_mass, adx, vix_level)

    Rule priority:
      1. CRISIS:      bh_mass > BH_COLLAPSE  OR  vix_level > 0.45
      2. HIGH_VOL:    vol_regime > 1.5        OR  vix_level > 0.30
      3. TREND:       Hurst > 0.58            AND adx > 25
      4. MEAN_REVERT: Hurst < 0.42
      5. TREND:       adx > 20 (fallback if moderate trending)
      6. Default:     MEAN_REVERT

    Parameters
    ----------
    hurst_window  : bars for rolling Hurst (default 120)
    vol_window    : bars for short-run vol (default 21)
    long_vol_window : bars for long-run vol (default 252)
    adx_period    : ADX period (default 14)
    """

    def __init__(
        self,
        hurst_window: int    = 120,
        vol_window: int      = 21,
        long_vol_window: int = 252,
        adx_period: int      = 14,
    ):
        self.hurst_window    = hurst_window
        self.vol_window      = vol_window
        self.long_vol_window = long_vol_window
        self.adx_period      = adx_period

    def _classify_features(self, f: RegimeState) -> RegimeState:
        """Apply rule cascade to feature set."""
        hurst     = f.hurst
        vol_r     = f.vol_regime
        bh_mass   = f.bh_mass
        adx       = f.adx
        vix       = f.vix_level

        # Rule 1: CRISIS
        if bh_mass > BH_COLLAPSE or vix > 0.45:
            f.regime     = Regime.CRISIS
            f.confidence = min(1.0, (bh_mass / BH_COLLAPSE + vix / 0.45) / 2.0)
            return f

        # Rule 2: HIGH_VOL
        if vol_r > 1.5 or vix > 0.30:
            f.regime     = Regime.HIGH_VOL
            f.confidence = min(1.0, max(vol_r / 1.5, vix / 0.30))
            return f

        # Rule 3: TREND (strong)
        if hurst > HURST_TRENDING and adx > 25:
            f.regime     = Regime.TREND
            f.confidence = min(1.0, (hurst - HURST_TRENDING) * 5 + (adx - 25) / 50)
            return f

        # Rule 4: MEAN_REVERT
        if hurst < 0.42:
            f.regime     = Regime.MEAN_REVERT
            f.confidence = min(1.0, (0.42 - hurst) * 5)
            return f

        # Rule 5: TREND (moderate)
        if adx > 20:
            f.regime     = Regime.TREND
            f.confidence = min(1.0, (adx - 20) / 30)
            return f

        # Default
        f.regime     = Regime.MEAN_REVERT
        f.confidence = 0.3
        return f

    def compute_bh_mass(self, returns: np.ndarray, window: int = 21) -> float:
        """Simplified BH mass from recent return magnitude."""
        if len(returns) < window:
            return 0.0
        recent = np.abs(returns[-window:]).mean()
        long_q = np.percentile(np.abs(returns), 95) if len(returns) > 50 else recent
        return float(recent / (long_q + 1e-9)) * 1.5  # scale to [0, ~2]

    def classify_at(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        vix_level: Optional[float] = None,
    ) -> RegimeState:
        """
        Classify regime at current bar.

        prices, returns: full historical arrays up to current bar
        high, low: optional OHLC arrays for ADX computation
        vix_level: optional VIX or realized vol proxy (annualized)
        """
        hurst = _compute_hurst(prices[-self.hurst_window:])

        # Volatility regime: short vol / long vol
        if len(returns) >= self.long_vol_window:
            short_vol = float(np.std(returns[-self.vol_window:])) * math.sqrt(252)
            long_vol  = float(np.std(returns[-self.long_vol_window:])) * math.sqrt(252)
            vol_regime = short_vol / (long_vol + 1e-9)
        else:
            short_vol  = float(np.std(returns[-max(5, len(returns)):])) * math.sqrt(252)
            vol_regime = 1.0

        bh_mass   = self.compute_bh_mass(returns)

        # ADX
        adx_val = 0.0
        if high is not None and low is not None and len(high) > self.adx_period * 2:
            adx_arr = _compute_adx(high, low, prices, self.adx_period)
            if not np.isnan(adx_arr[-1]):
                adx_val = float(adx_arr[-1])

        # VIX level  # fallback to realized short vol
        vix = vix_level if vix_level is not None else short_vol

        state = RegimeState(
            hurst=hurst,
            vol_regime=vol_regime,
            bh_mass=bh_mass,
            adx=adx_val,
            vix_level=vix,
            features={
                "hurst": hurst, "vol_regime": vol_regime,
                "bh_mass": bh_mass, "adx": adx_val, "vix_level": vix
            },
        )
        return self._classify_features(state)

    def classify_series(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute regime for each bar in df.

        df must have 'close', optionally 'high', 'low'.
        Returns Series of Regime strings.
        """
        n      = len(df)
        close  = df["close"].values
        rets   = np.concatenate([[0.0], np.diff(np.log(close + 1e-9))])
        high   = df["high"].values  if "high"  in df.columns else None
        low    = df["low"].values   if "low"   in df.columns else None
        regimes = [Regime.UNKNOWN.value] * n
        warmup  = max(self.hurst_window, self.long_vol_window, self.adx_period * 2)

        for i in range(warmup, n):
            vix = None
            if vix_series is not None and df.index[i] in vix_series.index:
                vix = float(vix_series.loc[df.index[i]])
            h_arr  = high[:i + 1] if high is not None else None
            l_arr  = low[:i + 1]  if low  is not None else None
            state  = self.classify_at(close[:i + 1], rets[:i + 1], h_arr, l_arr, vix)
            regimes[i] = state.regime.value

        return pd.Series(regimes, index=df.index)


# ---------------------------------------------------------------------------
# 2. SubStrategyAllocator
# ---------------------------------------------------------------------------

class SubStrategyAllocator:
    """
    Given a regime, return allocation weights across sub-strategies.

    Sub-strategy names:
      "trend_momentum"  : BH mass signal + momentum
      "pairs_reversal"  : pairs trading + Hurst reversal
      "vol_breakout"    : volatility breakout
      "cash"            : risk-free / cash
      "short_vix"       : short volatility proxy (SVXY-like)

    Parameters
    ----------
    custom_allocations : dict mapping Regime -> {strategy: weight}
                         Overrides defaults if provided.
    """

    DEFAULT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
        Regime.TREND.value: {
            "trend_momentum":  0.70,
            "pairs_reversal":  0.10,
            "vol_breakout":    0.20,
            "cash":            0.00,
            "short_vix":       0.00,
        },
        Regime.MEAN_REVERT.value: {
            "trend_momentum":  0.10,
            "pairs_reversal":  0.65,
            "vol_breakout":    0.05,
            "cash":            0.20,
            "short_vix":       0.00,
        },
        Regime.HIGH_VOL.value: {
            "trend_momentum":  0.10,
            "pairs_reversal":  0.10,
            "vol_breakout":    0.50,
            "cash":            0.30,
            "short_vix":       0.00,
        },
        Regime.CRISIS.value: {
            "trend_momentum":  0.00,
            "pairs_reversal":  0.00,
            "vol_breakout":    0.00,
            "cash":            0.80,
            "short_vix":       0.20,  # short vol in crisis
        },
        Regime.UNKNOWN.value: {
            "trend_momentum":  0.25,
            "pairs_reversal":  0.25,
            "vol_breakout":    0.25,
            "cash":            0.25,
            "short_vix":       0.00,
        },
    }

    def __init__(self, custom_allocations: Optional[Dict[str, Dict[str, float]]] = None):
        self._alloc = dict(self.DEFAULT_ALLOCATIONS)
        if custom_allocations:
            for k, v in custom_allocations.items():
                self._alloc[k] = v

    def get_weights(self, regime: Regime) -> Dict[str, float]:
        """Return weight dict for the given regime."""
        return dict(self._alloc.get(regime.value, self._alloc[Regime.UNKNOWN.value]))

    def weights_series(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        Convert regime series to a DataFrame of strategy weights over time.
        """
        strategy_names = list(self.DEFAULT_ALLOCATIONS[Regime.TREND.value].keys())
        weights = pd.DataFrame(0.0, index=regime_series.index, columns=strategy_names)
        for i, regime_str in enumerate(regime_series):
            r = Regime(regime_str) if regime_str in [e.value for e in Regime] else Regime.UNKNOWN
            w = self.get_weights(r)
            for col, val in w.items():
                if col in weights.columns:
                    weights.iloc[i][col] = val
        return weights


# ---------------------------------------------------------------------------
# 3. TransitionCostModel
# ---------------------------------------------------------------------------

class TransitionCostModel:
    """
    Penalize frequent regime changes with a transaction friction cost.

    When regime changes:
      cost = friction_per_flip * sum(|w_new - w_old|) / 2

    Parameters
    ----------
    friction_per_flip : friction fraction per regime change (default 0.001 = 0.1%)
    min_regime_hold   : minimum bars before allowing another regime change (default 5)
    """

    def __init__(self, friction_per_flip: float = 0.001, min_regime_hold: int = 5):
        self.friction_per_flip = friction_per_flip
        self.min_regime_hold   = min_regime_hold
        self._last_change_i    = -999
        self._prev_regime      = Regime.UNKNOWN

    def cost(
        self,
        bar_i: int,
        new_regime: Regime,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """
        Return friction cost for transitioning to new_regime.
        Returns 0.0 if regime has not changed or min_regime_hold not met.
        """
        if new_regime == self._prev_regime:
            return 0.0
        if (bar_i - self._last_change_i) < self.min_regime_hold:
            return 0.0  # too soon to switch

        # Turnover cost
        all_keys = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(k, 0.0) - old_weights.get(k, 0.0))
            for k in all_keys
        ) / 2.0

        self._prev_regime   = new_regime
        self._last_change_i = bar_i
        return turnover * self.friction_per_flip

    def reset(self):
        self._last_change_i = -999
        self._prev_regime   = Regime.UNKNOWN


# ---------------------------------------------------------------------------
# 4. Simplified sub-strategy return generators
# ---------------------------------------------------------------------------

def _trend_momentum_returns(df: pd.DataFrame) -> pd.Series:
    """
    Simplified trend + momentum signal returns.
    EMA(50) slope direction * vol-scaled position.
    """
    close   = df["close"]
    ema     = close.ewm(span=50, adjust=False).mean()
    slope   = ema.diff()
    signal  = np.sign(slope)
    returns = close.pct_change()
    vol     = returns.ewm(span=21).std() * math.sqrt(252)
    vol     = vol.replace(0, np.nan).ffill().fillna(0.20)
    size    = (0.20 / vol).clip(upper=2.0)
    return (signal * size * returns).fillna(0.0)


def _pairs_reversal_returns(df: pd.DataFrame) -> pd.Series:
    """
    Simplified pairs / mean-reversion returns.
    Use Bollinger-band mean reversion on close.
    """
    close   = df["close"]
    returns = close.pct_change()
    ma      = close.rolling(20).mean()
    std     = close.rolling(20).std()
    z       = (close - ma) / (std + 1e-9)
    signal  = np.where(z > 1.5, -1.0, np.where(z < -1.5, 1.0, 0.0))
    signal  = pd.Series(signal, index=close.index).shift(1).fillna(0)
    return (signal * returns).fillna(0.0)


def _vol_breakout_returns(df: pd.DataFrame) -> pd.Series:
    """
    Simplified vol breakout returns.
    Enter long/short on N-bar channel breakout.
    """
    close   = df["close"]
    returns = close.pct_change()
    ch_high = close.rolling(20).max().shift(1)
    ch_low  = close.rolling(20).min().shift(1)
    signal  = np.where(close > ch_high, 1.0, np.where(close < ch_low, -1.0, 0.0))
    signal  = pd.Series(signal, index=close.index).shift(1).fillna(0)
    return (signal * returns).fillna(0.0)


def _cash_returns(df: pd.DataFrame, risk_free: float = 0.04) -> pd.Series:
    """Risk-free daily return for cash allocation."""
    daily_rf = risk_free / 252
    return pd.Series(daily_rf, index=df.index)


def _short_vix_returns(df: pd.DataFrame) -> pd.Series:
    """
    Proxy short-vol returns: collect volatility premium.
    Simple model: earn +vol_premium - jumps.
    """
    returns  = df["close"].pct_change().fillna(0)
    vol      = returns.rolling(21).std() * math.sqrt(252)
    # Short vol: earn daily theta, lose on up-moves in realized vol
    daily_vp = vol.shift(1) * 0.05 / 252  # 5% of vol as premium per day
    jump     = returns.abs().where(returns.abs() > 0.03, 0.0) * 2.0
    return (daily_vp - jump).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. StatisticalRegimeStrategy
# ---------------------------------------------------------------------------

class StatisticalRegimeStrategy:
    """
    Meta-strategy that selects from:
      - TREND:       BH mass strategy + momentum
      - MEAN_REVERT: pairs + Hurst-guided reversal
      - HIGH_VOL:    volatility breakout + reduced size
      - CRISIS:      move to cash + short VIX proxy

    Parameters
    ----------
    config : dict of parameter overrides
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.classifier  = RegimeClassifier(
            hurst_window=cfg.get("hurst_window", 120),
            vol_window=cfg.get("vol_window", 21),
            long_vol_window=cfg.get("long_vol_window", 252),
            adx_period=cfg.get("adx_period", 14),
        )
        self.allocator   = SubStrategyAllocator(
            custom_allocations=cfg.get("custom_allocations", None)
        )
        self.cost_model  = TransitionCostModel(
            friction_per_flip=cfg.get("friction_per_flip", 0.001),
            min_regime_hold=cfg.get("min_regime_hold", 5),
        )
        self.config = cfg

    def generate_regime_series(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute rolling regime for each bar in df."""
        return self.classifier.classify_series(df, vix_series=vix_series)

    def generate_sub_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute returns for each sub-strategy on df.
        Returns DataFrame: columns = sub-strategy names, index = df.index.
        """
        return pd.DataFrame({
            "trend_momentum":  _trend_momentum_returns(df),
            "pairs_reversal":  _pairs_reversal_returns(df),
            "vol_breakout":    _vol_breakout_returns(df),
            "cash":            _cash_returns(df, self.config.get("risk_free", 0.04)),
            "short_vix":       _short_vix_returns(df),
        }, index=df.index)

    def backtest(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        initial_equity: float            = 1_000_000,
    ) -> RegimeBacktestResult:
        """
        Full backtest with regime-labeled equity curve.

        Parameters
        ----------
        df           : OHLCV DataFrame with 'close' (and optionally 'high', 'low')
        vix_series   : optional VIX/vol-level series for regime classification
        initial_equity : starting equity
        """
        regime_series = self.generate_regime_series(df, vix_series=vix_series)
        sub_returns   = self.generate_sub_returns(df)
        weights_df    = self.allocator.weights_series(regime_series)

        n          = len(df)
        equity     = initial_equity
        eq_curve   = np.full(n, initial_equity, dtype=float)
        trade_ret  = []
        n_regime_changes = 0
        prev_weights     = {s: 0.0 for s in weights_df.columns}
        self.cost_model.reset()

        for i in range(1, n):
            regime_str = regime_series.iloc[i]
            r_enum     = Regime(regime_str) if regime_str in [e.value for e in Regime] else Regime.UNKNOWN
            new_weights = self.allocator.get_weights(r_enum)

            # Transition cost
            tc = self.cost_model.cost(i, r_enum, prev_weights, new_weights)
            if tc > 0:
                n_regime_changes += 1
                equity *= (1.0 - tc)

            # Portfolio return = weighted sum of sub-strategy returns
            port_ret = 0.0
            for strat_name, w in new_weights.items():
                if strat_name in sub_returns.columns:
                    port_ret += w * float(sub_returns[strat_name].iloc[i])

            equity      *= (1.0 + port_ret)
            eq_curve[i]  = equity
            prev_weights = new_weights

            if port_ret != 0:
                trade_ret.append(float(port_ret))

        stats = _compute_stats(eq_curve, trade_ret)
        return RegimeBacktestResult(
            n_regime_changes=n_regime_changes,
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            regime_series=regime_series,
            sub_weights=weights_df,
            params=self.config,
        )


# ---------------------------------------------------------------------------
# 6. StatisticalRegimeBacktest
# ---------------------------------------------------------------------------

class StatisticalRegimeBacktest:
    """
    Full backtest with regime-labeled equity curve and per-regime statistics.

    Parameters
    ----------
    config         : passed to StatisticalRegimeStrategy
    initial_equity : starting equity (default 1_000_000)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        initial_equity: float  = 1_000_000.0,
    ):
        self.config         = config or {}
        self.initial_equity = initial_equity

    def run(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> RegimeBacktestResult:
        strat  = StatisticalRegimeStrategy(config=self.config)
        result = strat.backtest(df, vix_series=vix_series, initial_equity=self.initial_equity)
        # Add per-regime stats
        result.regime_stats = self._compute_regime_stats(result)
        return result

    def _compute_regime_stats(self, result: RegimeBacktestResult) -> Dict[str, dict]:
        """Compute per-regime equity statistics."""
        rets        = result.returns
        regime_s    = result.regime_series.reindex(rets.index)
        regime_stats = {}

        for regime in Regime:
            mask = regime_s == regime.value
            r    = rets[mask]
            if len(r) < 5:
                continue
            std  = r.std()
            regime_stats[regime.value] = {
                "n_bars":       int(mask.sum()),
                "mean_daily":   float(r.mean()),
                "sharpe":       float(r.mean() / (std + 1e-9) * math.sqrt(252)),
                "total_return": float((1 + r).prod() - 1),
                "pct_time":     float(mask.mean()),
            }
        return regime_stats

    def summary_table(self, result: RegimeBacktestResult) -> pd.DataFrame:
        """Per-regime statistics as a DataFrame."""
        rows = []
        for regime, stats in result.regime_stats.items():
            rows.append({"regime": regime, **stats})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. RegimeDrawdownAnalysis
# ---------------------------------------------------------------------------

class RegimeDrawdownAnalysis:
    """
    Max drawdown per regime and regime duration statistics.

    Given a RegimeBacktestResult, compute:
      - Max drawdown within each regime
      - Median, mean, max regime duration in bars
    """

    def compute(self, result: RegimeBacktestResult) -> Dict[str, dict]:
        """
        Returns dict mapping regime -> {
          max_drawdown, avg_duration_bars, median_duration_bars, n_episodes
        }
        """
        eq     = result.equity_curve.values
        regime = result.regime_series.values
        n      = len(eq)

        analysis = {}

        for r_enum in Regime:
            r_val = r_enum.value
            # Find contiguous episodes of this regime
            in_regime    = (regime == r_val)
            episode_rets  = []
            episode_durs  = []
            episode_dd    = []

            start = None
            for i in range(n):
                if in_regime[i] and start is None:
                    start = i
                elif not in_regime[i] and start is not None:
                    ep_eq = eq[start: i]
                    episode_durs.append(i - start)
                    pk = np.maximum.accumulate(ep_eq)
                    dd = (ep_eq - pk) / (pk + 1e-9)
                    episode_dd.append(float(dd.min()))
                    ret = ep_eq[-1] / ep_eq[0] - 1 if len(ep_eq) > 0 else 0.0
                    episode_rets.append(ret)
                    start = None

            # Close any open episode at end
            if start is not None:
                ep_eq = eq[start:]
                if len(ep_eq) > 0:
                    episode_durs.append(len(ep_eq))
                    pk = np.maximum.accumulate(ep_eq)
                    dd = (ep_eq - pk) / (pk + 1e-9)
                    episode_dd.append(float(dd.min()))

            if not episode_durs:
                continue

            analysis[r_val] = {
                "n_episodes":          len(episode_durs),
                "avg_duration_bars":   float(np.mean(episode_durs)),
                "median_duration_bars": float(np.median(episode_durs)),
                "max_duration_bars":   int(max(episode_durs)),
                "max_drawdown":        float(min(episode_dd)),
                "avg_episode_return":  float(np.mean(episode_rets)) if episode_rets else 0.0,
            }

        return analysis

    def drawdown_table(self, result: RegimeBacktestResult) -> pd.DataFrame:
        analysis = self.compute(result)
        rows = [{"regime": k, **v} for k, v in analysis.items()]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 2520  # ~10 years
    idx = pd.date_range("2014-01-01", periods=n, freq="B")

    # Simulate price regimes
    seg = n // 5
    vols   = [0.008, 0.015, 0.020, 0.010, 0.012]
    drifts = [0.0004, 0.0001, -0.0003, 0.0003, 0.0002]
    rets_raw = np.concatenate([
        rng.normal(d, v, seg) for d, v in zip(drifts, vols)
    ])
    close  = 100.0 * np.cumprod(1 + rets_raw[:n])
    high   = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low    = close * (1 - np.abs(rng.normal(0, 0.003, n)))

    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

    # Simulated VIX proxy
    vix = pd.Series(0.15 + np.abs(rng.normal(0, 0.05, n)), index=idx)

    bt     = StatisticalRegimeBacktest(initial_equity=1_000_000)
    result = bt.run(df, vix_series=vix)
    print("Regime meta-strategy:", result.summary())

    # Per-regime table
    tbl = bt.summary_table(result)
    print(tbl.to_string())

    # Drawdown analysis
    dd_analysis = RegimeDrawdownAnalysis()
    dd_tbl      = dd_analysis.drawdown_table(result)
    print(dd_tbl.to_string())
