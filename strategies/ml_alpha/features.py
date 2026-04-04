"""
ml_alpha/features.py — Full feature engineering for ML alpha models.

500+ features across:
- Price features (returns, vol, skew, kurt, drawdown, Sharpe per lookback)
- Volume features (OBV, VWAP, volume momentum)
- Microstructure features (Amihud, Roll spread, Corwin-Schultz)
- Options features (VIX, skew, term structure)
- Macro features (yield curve, credit spreads, DXY)
- BH physics features (mass, momentum, ctl, regime duration)
- Calendar features (day/month/quarter, holiday proximity)
- Cross-sectional features (momentum rank, vol rank, market correlation)
"""

from __future__ import annotations
import math
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


LOOKBACKS = [5, 10, 20, 60, 120, 252]
ANNUALIZE = math.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b, fill=0.0):
    """Safe division with fill value for zeros."""
    result = np.where(np.abs(b) > 1e-12, a / b, fill)
    return result


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness."""
    def skew_fn(x):
        if len(x) < 3:
            return 0.0
        mu = x.mean(); sig = x.std()
        if sig == 0:
            return 0.0
        return float(((x - mu) ** 3).mean() / sig ** 3)
    return series.rolling(window, min_periods=max(3, window // 3)).apply(skew_fn, raw=True)


def _rolling_kurt(series: pd.Series, window: int) -> pd.Series:
    """Rolling excess kurtosis."""
    def kurt_fn(x):
        if len(x) < 4:
            return 0.0
        mu = x.mean(); sig = x.std()
        if sig == 0:
            return 0.0
        return float(((x - mu) ** 4).mean() / sig ** 4) - 3.0
    return series.rolling(window, min_periods=max(4, window // 3)).apply(kurt_fn, raw=True)


def _rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Rolling maximum drawdown over a window."""
    def mdd_fn(rets):
        equity = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-9)
        return float(dd.min())
    return returns.rolling(window, min_periods=max(5, window // 3)).apply(mdd_fn, raw=True)


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    mu = returns.rolling(window, min_periods=max(5, window // 3)).mean()
    sigma = returns.rolling(window, min_periods=max(5, window // 3)).std()
    return (mu / (sigma + 1e-9)) * ANNUALIZE


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngine
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Master feature engineering class for ML alpha models.

    All methods return pd.DataFrame aligned with the input DataFrame's index.
    NaN values appear in the warmup period.

    Usage:
        fe = FeatureEngine()
        price_feats = fe.price_features(df, lookbacks=[5, 20, 60])
        vol_feats = fe.volume_features(df)
        all_feats = pd.concat([price_feats, vol_feats, ...], axis=1)
    """

    def __init__(self, prefix_separator: str = "_"):
        self.prefix_separator = prefix_separator
        self._feature_names: List[str] = []

    # ── Price features ──────────────────────────────────────────────────────

    def price_features(
        self,
        df: pd.DataFrame,
        lookbacks: List[int] = None,
    ) -> pd.DataFrame:
        """
        Generate price-based features for each lookback period.

        For each lookback L:
        - return_L          : L-bar return
        - vol_L             : realized vol (annualized)
        - skew_L            : return skewness
        - kurt_L            : return excess kurtosis
        - max_drawdown_L    : maximum drawdown over window
        - sharpe_L          : Sharpe ratio (annualized)
        - high_low_ratio_L  : rolling high/low ratio (range compression)
        - rsi_L             : relative strength index
        - momentum_z_L      : z-scored return (return / vol)

        Returns pd.DataFrame.
        """
        if lookbacks is None:
            lookbacks = LOOKBACKS

        close = df["close"]
        returns = close.pct_change().fillna(0)
        log_returns = np.log(close / close.shift(1)).fillna(0)

        feats = {}

        for L in lookbacks:
            pfx = f"price{self.prefix_separator}{L}"

            # Return
            feats[f"{pfx}_return"] = close.pct_change(L).fillna(0)

            # Vol
            vol = log_returns.rolling(L, min_periods=max(3, L // 3)).std() * ANNUALIZE
            feats[f"{pfx}_vol"] = vol

            # Skew
            feats[f"{pfx}_skew"] = _rolling_skew(log_returns, L)

            # Kurtosis
            feats[f"{pfx}_kurt"] = _rolling_kurt(log_returns, L)

            # Max drawdown
            feats[f"{pfx}_maxdd"] = _rolling_max_drawdown(returns, L)

            # Sharpe
            feats[f"{pfx}_sharpe"] = _rolling_sharpe(log_returns, L)

            # High/low ratio (trend)
            if "high" in df.columns and "low" in df.columns:
                rolling_high = df["high"].rolling(L, min_periods=max(3, L // 3)).max()
                rolling_low = df["low"].rolling(L, min_periods=max(3, L // 3)).min()
                feats[f"{pfx}_hl_ratio"] = (rolling_high - rolling_low) / (rolling_low + 1e-9)

            # RSI
            feats[f"{pfx}_rsi"] = self._rsi(close, L)

            # Momentum z-score
            mom_z = feats[f"{pfx}_return"] / (vol + 1e-9)
            feats[f"{pfx}_mom_z"] = mom_z.clip(-5, 5)

        return pd.DataFrame(feats, index=df.index)

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Compute RSI."""
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    # ── Volume features ─────────────────────────────────────────────────────

    def volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features.

        Features:
        - obv              : On-Balance Volume (normalized)
        - obv_mom_L        : OBV momentum for each lookback
        - vol_momentum_L   : volume change over L periods
        - vol_ratio_L      : current vol / rolling average vol
        - vwap_dev         : deviation from VWAP
        - vwap_dev_5       : short-term VWAP deviation
        - vol_price_corr_L : correlation of volume with price change
        - volume_zscore_L  : z-score of volume
        """
        if "volume" not in df.columns:
            return pd.DataFrame(index=df.index)

        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change().fillna(0)

        feats = {}

        # OBV
        direction = np.sign(returns)
        obv_raw = (direction * volume).cumsum()
        # Normalize OBV by rolling std
        obv_std = obv_raw.rolling(252, min_periods=20).std()
        feats["obv_norm"] = obv_raw / (obv_std + 1e-9)

        # VWAP and deviation
        if "high" in df.columns and "low" in df.columns:
            typical_price = (df["high"] + df["low"] + close) / 3
        else:
            typical_price = close

        # Daily VWAP approximation (rolling)
        for L in [5, 20]:
            vwap_n = (typical_price * volume).rolling(L, min_periods=2).sum()
            vwap_d = volume.rolling(L, min_periods=2).sum()
            vwap = vwap_n / (vwap_d + 1e-9)
            feats[f"vwap_dev_{L}"] = (close - vwap) / (vwap + 1e-9)

        # Volume features per lookback
        for L in [5, 10, 20, 60]:
            pfx = f"vol{self.prefix_separator}{L}"

            # Volume momentum
            vol_mom = volume.pct_change(L).fillna(0)
            feats[f"{pfx}_momentum"] = vol_mom.clip(-5, 5)

            # Volume ratio (current vs average)
            vol_avg = volume.rolling(L, min_periods=max(3, L // 3)).mean()
            feats[f"{pfx}_ratio"] = (volume / (vol_avg + 1e-9)).clip(0, 10)

            # Volume z-score
            vol_std = volume.rolling(L, min_periods=max(3, L // 3)).std()
            feats[f"{pfx}_zscore"] = ((volume - vol_avg) / (vol_std + 1e-9)).clip(-5, 5)

            # Volume-price correlation
            def vp_corr(w):
                if len(w) < 5:
                    return 0.0
                n = len(w) // 2
                v_w = w[:n]; r_w = w[n:]
                if np.std(v_w) < 1e-9 or np.std(r_w) < 1e-9:
                    return 0.0
                return float(np.corrcoef(v_w, r_w)[0, 1])

            combined = pd.concat([volume, returns], axis=1)
            feats[f"{pfx}_vp_corr"] = combined.rolling(L, min_periods=max(5, L // 3)).apply(
                lambda x: vp_corr(x.values), raw=True
            ).iloc[:, 0]

            # OBV momentum
            feats[f"obv_mom_{L}"] = feats["obv_norm"] - feats["obv_norm"].shift(L)

        return pd.DataFrame(feats, index=df.index)

    # ── Microstructure features ──────────────────────────────────────────────

    def microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market microstructure features.

        Features:
        - amihud_L         : Amihud illiquidity ratio per lookback
        - roll_spread_L    : Roll (1984) bid-ask spread estimate
        - cs_spread_L      : Corwin-Schultz spread estimate
        - turnover_L       : volume turnover (volume / price)
        - kyle_lambda_L    : Kyle's lambda (price impact)
        """
        if "volume" not in df.columns:
            return pd.DataFrame(index=df.index)

        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change().fillna(0)

        feats = {}

        for L in [5, 20, 60]:
            pfx = f"mstruct{self.prefix_separator}{L}"

            # Amihud (2002) illiquidity: |return| / (price * volume)
            dollar_vol = close * volume
            amihud = (returns.abs() / (dollar_vol + 1e-9)).rolling(L, min_periods=max(3, L // 3)).mean()
            # Log-scale for stability
            feats[f"{pfx}_amihud"] = np.log1p(amihud * 1e6)

            # Roll (1984) spread estimate: 2 * sqrt(-cov(r_t, r_{t-1}))
            # If cov is negative → spread estimate. If positive → set to 0.
            def roll_spread(rets):
                if len(rets) < 5:
                    return 0.0
                cov = np.cov(rets[1:], rets[:-1])[0, 1]
                if cov < 0:
                    return float(2 * math.sqrt(-cov))
                return 0.0
            feats[f"{pfx}_roll_spread"] = returns.rolling(L, min_periods=max(5, L // 3)).apply(
                roll_spread, raw=True
            )

            # Corwin-Schultz (2012) spread: uses high/low over two consecutive periods
            # Approximation when H/L data available
            if "high" in df.columns and "low" in df.columns:
                h = df["high"]; lo = df["low"]
                # Single-day log H/L range
                beta = np.log(h / lo) ** 2
                gamma = np.log(h.rolling(2).max() / lo.rolling(2).min()) ** 2
                alpha = ((math.sqrt(2) - 1) / (3 - 2 * math.sqrt(2)) *
                         (np.sqrt(beta) - np.sqrt(gamma / 2)))
                cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
                feats[f"{pfx}_cs_spread"] = cs_spread.rolling(L, min_periods=max(3, L // 3)).mean()
            else:
                feats[f"{pfx}_cs_spread"] = pd.Series(0.0, index=df.index)

            # Turnover
            feats[f"{pfx}_turnover"] = (volume / (close + 1e-9)).rolling(L, min_periods=max(3, L // 3)).mean()

            # Kyle's lambda (price impact)
            def kyle_lambda(combined):
                n = len(combined) // 2
                rets = combined[:n]; vol = combined[n:]
                if np.std(vol) < 1e-9 or len(rets) < 5:
                    return 0.0
                # Lambda = cov(r, signed_vol) / var(signed_vol)
                signed_vol = np.sign(rets) * vol
                sv_var = np.var(signed_vol)
                if sv_var < 1e-12:
                    return 0.0
                return float(np.cov(rets, signed_vol)[0, 1] / sv_var)

            combined = pd.concat([returns, volume], axis=1)
            feats[f"{pfx}_kyle_lambda"] = combined.rolling(L, min_periods=max(5, L // 3)).apply(
                lambda x: kyle_lambda(x.values), raw=True
            ).iloc[:, 0].clip(-0.01, 0.01)

        return pd.DataFrame(feats, index=df.index)

    # ── Options features ─────────────────────────────────────────────────────

    def options_features(
        self,
        vix: pd.Series,
        skew: Optional[pd.Series] = None,
        term_structure: Optional[pd.Series] = None,
        put_call_ratio: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Options market features.

        Parameters
        ----------
        vix            : VIX index level
        skew           : put-call skew (or CBOE SKEW index)
        term_structure : VIX term structure slope (3M - 1M)
        put_call_ratio : put/call volume ratio

        Features generated:
        - vix_level        : raw VIX level
        - vix_zscore_L     : VIX z-score vs rolling mean
        - vix_percentile   : rolling percentile rank
        - vix_mom_L        : VIX momentum
        - vrp              : vol risk premium (VIX - realized vol)
        - term_structure   : VIX term structure slope
        - skew             : put/call skew
        - pcr              : put/call volume ratio
        """
        feats = {}
        idx = vix.index

        log_vix = np.log(vix.clip(lower=1.0))
        feats["vix_level"] = vix
        feats["vix_log"] = log_vix

        for L in [20, 63, 252]:
            mean = log_vix.rolling(L, min_periods=max(5, L // 3)).mean()
            std = log_vix.rolling(L, min_periods=max(5, L // 3)).std()
            feats[f"vix_zscore_{L}"] = ((log_vix - mean) / (std + 1e-9)).clip(-5, 5)
            feats[f"vix_pctile_{L}"] = log_vix.rolling(L, min_periods=max(5, L // 3)).rank(pct=True)
            feats[f"vix_mom_{L}"] = log_vix - log_vix.shift(L)

        if skew is not None:
            feats["skew_level"] = skew
            skew_mean = skew.rolling(63, min_periods=10).mean()
            skew_std = skew.rolling(63, min_periods=10).std()
            feats["skew_zscore"] = ((skew - skew_mean) / (skew_std + 1e-9)).clip(-5, 5)

        if term_structure is not None:
            feats["vix_term_structure"] = term_structure
            feats["vix_ts_sign"] = np.sign(term_structure)

        if put_call_ratio is not None:
            feats["put_call_ratio"] = put_call_ratio.clip(0, 5)
            pcr_mean = put_call_ratio.rolling(20, min_periods=5).mean()
            feats["pcr_zscore"] = ((put_call_ratio - pcr_mean) /
                                   (put_call_ratio.rolling(20).std() + 1e-9)).clip(-5, 5)

        return pd.DataFrame(feats, index=idx)

    # ── Macro features ───────────────────────────────────────────────────────

    def macro_features(
        self,
        yield_curve: Optional[pd.Series] = None,
        credit_spreads: Optional[pd.Series] = None,
        dxy: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Macro economic features.

        Parameters
        ----------
        yield_curve    : 2s10s or 5s30s yield curve slope
        credit_spreads : IG or HY credit spread level
        dxy            : DXY dollar index

        Features:
        - yield_curve_level     : curve slope level
        - yield_curve_trend_L   : trend in curve slope
        - credit_spread_level   : credit spread
        - credit_spread_z_L     : z-score of credit spread
        - dxy_level             : DXY level
        - dxy_mom_L             : DXY momentum
        - dxy_regime            : above/below 200d MA
        """
        feats = {}
        idx = None

        if yield_curve is not None:
            idx = yield_curve.index
            feats["yield_curve_level"] = yield_curve
            feats["yield_curve_sign"] = np.sign(yield_curve)
            for L in [20, 60]:
                feats[f"yield_curve_trend_{L}"] = yield_curve - yield_curve.shift(L)
            # Inversion signal
            feats["yield_curve_inverted"] = (yield_curve < 0).astype(float)

        if credit_spreads is not None:
            if idx is None:
                idx = credit_spreads.index
            log_cs = np.log(credit_spreads.clip(lower=0.01))
            feats["credit_spread_level"] = credit_spreads
            for L in [20, 63, 252]:
                cs_mean = log_cs.rolling(L, min_periods=max(5, L // 3)).mean()
                cs_std = log_cs.rolling(L, min_periods=max(5, L // 3)).std()
                feats[f"credit_spread_z_{L}"] = ((log_cs - cs_mean) / (cs_std + 1e-9)).clip(-5, 5)
            feats["credit_spread_trend_20"] = credit_spreads.pct_change(20).clip(-2, 2)

        if dxy is not None:
            if idx is None:
                idx = dxy.index
            feats["dxy_level"] = dxy
            ma_200 = dxy.rolling(200, min_periods=50).mean()
            feats["dxy_regime"] = (dxy > ma_200).astype(float)
            for L in [20, 60]:
                feats[f"dxy_mom_{L}"] = dxy.pct_change(L).clip(-0.2, 0.2)

        if idx is not None:
            return pd.DataFrame(feats, index=idx)
        return pd.DataFrame()

    # ── BH Physics features ──────────────────────────────────────────────────

    def bh_features(
        self,
        prices: pd.DataFrame,
        cf_tiers: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Black Hole physics features from srfm_core.

        For 3 timeframes (short, medium, long), computes:
        - bh_mass_TF      : BH mass
        - bh_active_TF    : is BH active (0/1)
        - bh_dir_TF       : BH direction (+1/-1/0)
        - ctl_TF          : consecutive timelike bars
        - beta_TF         : Minkowski beta (speed)
        - bh_mass_mom     : momentum of BH mass
        - regime_duration : bars in current regime

        Parameters
        ----------
        prices    : OHLCV DataFrame
        cf_tiers  : dict of timeframe → cf parameter
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../lib"))
        try:
            from srfm_core import MinkowskiClassifier, BlackHoleDetector
        except ImportError:
            # Fallback: build simplified versions
            return self._bh_features_fallback(prices)

        if cf_tiers is None:
            cf_tiers = {"short": 0.001, "mid": 0.0005, "long": 0.0002}

        close = prices["close"]
        n = len(close)
        feats = {}

        for tf_name, cf in cf_tiers.items():
            mc = MinkowskiClassifier(cf=cf)
            bh = BlackHoleDetector(bh_form=1.5, bh_collapse=1.0, bh_decay=0.95)

            masses = np.zeros(n)
            actives = np.zeros(n)
            dirs = np.zeros(n)
            ctls = np.zeros(n)
            betas = np.zeros(n)

            prev_close = None
            for i in range(n):
                c = float(close.iloc[i])
                bit = mc.update(c)
                if prev_close is not None and prev_close > 0:
                    bh.update(bit, c, prev_close)
                masses[i] = bh.bh_mass
                actives[i] = float(bh.bh_active)
                dirs[i] = float(bh.bh_dir)
                ctls[i] = float(bh.ctl)
                betas[i] = mc.beta
                prev_close = c

            feats[f"bh_mass_{tf_name}"] = masses
            feats[f"bh_active_{tf_name}"] = actives
            feats[f"bh_dir_{tf_name}"] = dirs
            feats[f"bh_ctl_{tf_name}"] = ctls
            feats[f"bh_beta_{tf_name}"] = betas

            # BH mass momentum
            mass_s = pd.Series(masses, index=close.index)
            feats[f"bh_mass_mom_{tf_name}_5"] = (mass_s - mass_s.shift(5)).values
            feats[f"bh_mass_mom_{tf_name}_20"] = (mass_s - mass_s.shift(20)).values

        # Regime duration (how long in current state)
        if "bh_active_short" in feats:
            active_s = pd.Series(feats["bh_active_short"], index=close.index)
            regime_dur = np.zeros(n)
            count = 0
            prev_state = -1
            for i in range(n):
                state = int(active_s.iloc[i])
                if state == prev_state:
                    count += 1
                else:
                    count = 1
                    prev_state = state
                regime_dur[i] = count
            feats["bh_regime_duration"] = regime_dur

        return pd.DataFrame(feats, index=close.index)

    def _bh_features_fallback(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Fallback BH features when srfm_core is not available."""
        close = prices["close"]
        returns = close.pct_change().fillna(0)
        feats = {}
        for tf_name, cf in [("short", 0.001), ("mid", 0.0005), ("long", 0.0002)]:
            beta = returns.abs() / cf
            feats[f"bh_mass_{tf_name}"] = beta.ewm(span=20).mean().values
            feats[f"bh_active_{tf_name}"] = (feats[f"bh_mass_{tf_name}"] > 1.5).astype(float)
            feats[f"bh_dir_{tf_name}"] = np.sign(returns.ewm(span=5).mean()).values
            feats[f"bh_ctl_{tf_name}"] = (returns > 0).rolling(5, min_periods=1).sum().values
            feats[f"bh_beta_{tf_name}"] = beta.clip(0, 5).values
        return pd.DataFrame(feats, index=close.index)

    # ── Calendar features ────────────────────────────────────────────────────

    def calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calendar-based features.

        Features:
        - day_of_week       : 0=Monday, 4=Friday
        - day_of_month      : 1-31
        - week_of_year      : 1-52
        - month             : 1-12
        - quarter           : 1-4
        - is_month_end      : bool
        - is_month_start    : bool
        - is_quarter_end    : bool
        - is_quarter_start  : bool
        - days_to_year_end  : days remaining in year
        - days_from_year_start: days since Jan 1
        - is_monday         : bool
        - is_friday         : bool
        - is_turn_of_month  : last/first 5 days of month
        """
        feats = {}
        idx = pd.DatetimeIndex(index)

        feats["day_of_week"] = idx.dayofweek.astype(float)
        feats["day_of_month"] = idx.day.astype(float)
        feats["week_of_year"] = idx.isocalendar().week.astype(float).values
        feats["month"] = idx.month.astype(float)
        feats["quarter"] = idx.quarter.astype(float)
        feats["is_month_end"] = idx.is_month_end.astype(float)
        feats["is_month_start"] = idx.is_month_start.astype(float)
        feats["is_quarter_end"] = idx.is_quarter_end.astype(float)
        feats["is_quarter_start"] = idx.is_quarter_start.astype(float)

        year_start = pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in idx.year])
        year_end = pd.DatetimeIndex([pd.Timestamp(y, 12, 31) for y in idx.year])
        feats["days_from_year_start"] = (idx - year_start).days.astype(float)
        feats["days_to_year_end"] = (year_end - idx).days.astype(float)

        feats["is_monday"] = (idx.dayofweek == 0).astype(float)
        feats["is_friday"] = (idx.dayofweek == 4).astype(float)

        # Turn of month: last 3 or first 3 days
        feats["is_turn_of_month"] = ((idx.day <= 3) | (idx.day >= 28)).astype(float)

        # Cyclical encoding (avoid ordinal discontinuities)
        feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
        feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
        feats["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
        feats["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

        return pd.DataFrame(feats, index=index)

    # ── Cross-sectional features ─────────────────────────────────────────────

    def cross_sectional_features(
        self,
        universe_df: pd.DataFrame,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """
        Cross-sectional features: rank each asset relative to its universe.

        universe_df: DataFrame where each column is an asset's close price.

        Features per asset:
        - mom_rank_L     : momentum rank (percentile) across universe
        - vol_rank_L     : volatility rank across universe
        - market_corr_L  : correlation with equal-weight market index
        - relative_strength : asset return - market return
        - beta_to_market  : rolling beta to market portfolio
        """
        returns = universe_df.pct_change().fillna(0)
        n_assets = len(universe_df.columns)
        market = returns.mean(axis=1)  # equal-weight market index

        feats = {}

        for L in [20, 60, 252]:
            # Momentum
            mom = universe_df.pct_change(L)
            mom_rank = mom.rank(axis=1, pct=True)

            # Vol
            vol = returns.rolling(L, min_periods=max(5, L // 3)).std() * ANNUALIZE
            vol_rank = vol.rank(axis=1, pct=True)

            for col in universe_df.columns:
                pfx = f"{col}_cs{self.prefix_separator}{L}"
                feats[f"{pfx}_mom_rank"] = mom_rank[col]
                feats[f"{pfx}_vol_rank"] = vol_rank[col]

                # Market correlation
                def corr_fn(x):
                    n = len(x) // 2
                    if np.std(x[:n]) < 1e-9 or np.std(x[n:]) < 1e-9:
                        return 0.0
                    return float(np.corrcoef(x[:n], x[n:])[0, 1])

                combo = pd.concat([returns[col], market], axis=1)
                feats[f"{pfx}_market_corr"] = combo.rolling(L, min_periods=max(5, L // 3)).apply(
                    lambda x: corr_fn(x.values), raw=True
                ).iloc[:, 0]

                # Relative strength
                feats[f"{pfx}_rel_strength"] = (returns[col] - market).rolling(L, min_periods=5).mean()

                # Beta to market
                def beta_fn(x):
                    n = len(x) // 2
                    if np.var(x[n:]) < 1e-12:
                        return 1.0
                    return float(np.cov(x[:n], x[n:])[0, 1] / np.var(x[n:]))

                combo2 = pd.concat([returns[col], market], axis=1)
                feats[f"{pfx}_beta"] = combo2.rolling(L, min_periods=max(5, L // 3)).apply(
                    lambda x: beta_fn(x.values), raw=True
                ).iloc[:, 0].clip(-5, 5)

        return pd.DataFrame(feats, index=universe_df.index)

    # ── Master feature builder ───────────────────────────────────────────────

    def build_all_features(
        self,
        df: pd.DataFrame,
        vix: Optional[pd.Series] = None,
        yield_curve: Optional[pd.Series] = None,
        credit_spreads: Optional[pd.Series] = None,
        dxy: Optional[pd.Series] = None,
        include_bh: bool = True,
        include_calendar: bool = True,
    ) -> pd.DataFrame:
        """
        Build all feature groups and concatenate.

        Parameters
        ----------
        df             : OHLCV DataFrame
        vix            : VIX level (optional)
        yield_curve    : 2s10s slope (optional)
        credit_spreads : IG credit spreads (optional)
        dxy            : DXY index (optional)
        include_bh     : whether to include BH features
        include_calendar: whether to include calendar features

        Returns
        -------
        pd.DataFrame of all features, shape (n_bars, n_features).
        """
        groups = []

        # Price features
        groups.append(self.price_features(df, LOOKBACKS))

        # Volume features
        if "volume" in df.columns:
            groups.append(self.volume_features(df))

        # Microstructure
        if "volume" in df.columns:
            groups.append(self.microstructure_features(df))

        # Options
        if vix is not None:
            groups.append(self.options_features(vix))

        # Macro
        if any(x is not None for x in [yield_curve, credit_spreads, dxy]):
            groups.append(self.macro_features(yield_curve, credit_spreads, dxy))

        # BH
        if include_bh:
            groups.append(self.bh_features(df))

        # Calendar
        if include_calendar and hasattr(df.index, "dayofweek"):
            groups.append(self.calendar_features(df.index))

        # Concatenate
        result = pd.concat([g.reindex(df.index) for g in groups if len(g) > 0], axis=1)

        # Fill NaN with 0 after a reasonable warmup
        self._feature_names = list(result.columns)
        return result

    def feature_names(self) -> List[str]:
        """Return list of feature names from last build_all_features() call."""
        return self._feature_names

    def n_features(self) -> int:
        """Return number of features."""
        return len(self._feature_names)

    def winsorize(self, df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
        """Winsorize features to [lower, upper] quantile range."""
        q_low = df.quantile(lower)
        q_high = df.quantile(upper)
        return df.clip(lower=q_low, upper=q_high, axis=1)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-mean, unit-variance normalization (cross-sectional per row or time-series)."""
        return (df - df.mean()) / (df.std() + 1e-9)

    def drop_low_variance(self, df: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
        """Drop features with very low variance."""
        variances = df.var()
        return df.loc[:, variances > threshold]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    df = pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.003, 0.003, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.integers(1000, 100000, n).astype(float),
    }, index=idx)

    fe = FeatureEngine()

    print("Building price features...")
    pf = fe.price_features(df)
    print(f"  Shape: {pf.shape}, Cols: {list(pf.columns[:5])}...")

    print("Building volume features...")
    vf = fe.volume_features(df)
    print(f"  Shape: {vf.shape}")

    print("Building microstructure features...")
    mf = fe.microstructure_features(df)
    print(f"  Shape: {mf.shape}")

    vix = pd.Series(rng.uniform(12, 30, n), index=idx)
    print("Building options features...")
    of = fe.options_features(vix)
    print(f"  Shape: {of.shape}")

    print("Building calendar features...")
    cf = fe.calendar_features(df.index)
    print(f"  Shape: {cf.shape}")

    print("Building BH features...")
    bh = fe.bh_features(df)
    print(f"  Shape: {bh.shape}")

    print("Building all features...")
    all_f = fe.build_all_features(df, vix=vix)
    print(f"  Total shape: {all_f.shape}")
    print(f"  N features: {fe.n_features()}")
    print(f"  Sample (last row): {all_f.iloc[-1].describe()}")
