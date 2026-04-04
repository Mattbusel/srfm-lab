"""
Macro alternative data signals.

Implements:
- Yield curve features (level, slope, curvature, inversion indicator)
- Credit spread signals (HY spreads, IG spreads, OAS)
- DXY (Dollar Index) regime and momentum signals
- Cross-asset momentum (equity/bond/commodity/FX composite)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
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
# Yield Curve Features
# ---------------------------------------------------------------------------

class YieldCurveFeatures:
    """
    Construct trading signals from the yield curve.

    Key features:
    - Level: long-term yield (10Y)
    - Slope: 10Y - 2Y (steepness)
    - Curvature: 2*(5Y) - (2Y + 10Y) (butterfly)
    - Inversion: 10Y < 2Y → recession signal
    - Change: rate of change in slope / level

    Parameters
    ----------
    level_yield : str
        Column name for level (e.g., '10Y').
    short_yield : str
        Column name for short end (e.g., '2Y').
    mid_yield : str
        Column name for mid point (e.g., '5Y') — used for curvature.
    lookback : int
        Window for z-score normalization.
    """

    def __init__(
        self,
        level_yield: str = "10Y",
        short_yield: str = "2Y",
        mid_yield: str = "5Y",
        lookback: int = 63,
    ) -> None:
        self.level_yield = level_yield
        self.short_yield = short_yield
        self.mid_yield = mid_yield
        self.lookback = lookback

    def compute_features(self, yields_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute yield curve features.

        Parameters
        ----------
        yields_df : pd.DataFrame
            Daily yields for multiple maturities (columns = maturity labels).

        Returns
        -------
        pd.DataFrame
            Feature columns: level, slope, curvature, inverted,
            slope_z, level_z, slope_chg, level_chg.
        """
        feats = pd.DataFrame(index=yields_df.index)

        # Level (long-end yield)
        if self.level_yield in yields_df.columns:
            feats["yc_level"] = yields_df[self.level_yield]
        elif len(yields_df.columns) > 0:
            feats["yc_level"] = yields_df.iloc[:, -1]

        # Slope
        if self.short_yield in yields_df.columns and self.level_yield in yields_df.columns:
            feats["yc_slope"] = yields_df[self.level_yield] - yields_df[self.short_yield]
        elif yields_df.shape[1] >= 2:
            feats["yc_slope"] = yields_df.iloc[:, -1] - yields_df.iloc[:, 0]

        # Curvature
        if all(c in yields_df.columns for c in [self.short_yield, self.mid_yield, self.level_yield]):
            feats["yc_curvature"] = (
                2 * yields_df[self.mid_yield]
                - yields_df[self.short_yield]
                - yields_df[self.level_yield]
            )
        elif yields_df.shape[1] >= 3:
            feats["yc_curvature"] = (
                2 * yields_df.iloc[:, yields_df.shape[1] // 2]
                - yields_df.iloc[:, 0]
                - yields_df.iloc[:, -1]
            )

        # Inversion indicator
        if "yc_slope" in feats:
            feats["yc_inverted"] = (feats["yc_slope"] < 0).astype(float)

        # Z-scores
        if "yc_slope" in feats:
            feats["yc_slope_z"] = _rolling_zscore(feats["yc_slope"], self.lookback)
        if "yc_level" in feats:
            feats["yc_level_z"] = _rolling_zscore(feats["yc_level"], self.lookback)

        # Rate of change
        if "yc_slope" in feats:
            feats["yc_slope_chg_1m"] = feats["yc_slope"].diff(21)
            feats["yc_slope_chg_3m"] = feats["yc_slope"].diff(63)
        if "yc_level" in feats:
            feats["yc_level_chg_1m"] = feats["yc_level"].diff(21)

        # Term premium proxy: slope percentile
        if "yc_slope" in feats:
            feats["yc_slope_pct"] = feats["yc_slope"].rank(pct=True)

        return feats

    def recession_signal(
        self,
        yields_df: pd.DataFrame,
        inversion_window: int = 63,
        min_inversion_days: int = 21,
    ) -> pd.Series:
        """
        Recession signal based on yield curve inversion.

        Returns +1 when curve has been inverted for min_inversion_days
        within the past inversion_window trading days.

        Returns
        -------
        pd.Series
            0 or 1 recession signal.
        """
        feats = self.compute_features(yields_df)
        if "yc_inverted" not in feats:
            return pd.Series(0.0, index=yields_df.index)

        rolling_inv = feats["yc_inverted"].rolling(inversion_window, min_periods=1).sum()
        return (rolling_inv >= min_inversion_days).astype(float).rename("recession_signal")

    def yield_curve_regime(
        self,
        yields_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Classify yield curve shape into regime.

        Returns
        -------
        pd.Series
            'STEEP', 'FLAT', 'INVERTED', 'NORMAL' labels.
        """
        feats = self.compute_features(yields_df)
        if "yc_slope" not in feats:
            return pd.Series("NORMAL", index=yields_df.index)

        slope = feats["yc_slope"]
        labels = pd.Series("NORMAL", index=yields_df.index)
        q25 = slope.quantile(0.25)
        q75 = slope.quantile(0.75)
        labels[slope > q75] = "STEEP"
        labels[slope < 0] = "INVERTED"
        labels[(slope >= 0) & (slope < q25)] = "FLAT"
        return labels

    def equity_yield_spread(
        self,
        yields_df: pd.DataFrame,
        earnings_yield: pd.Series,
    ) -> pd.Series:
        """
        Fed Model: earnings yield - 10Y yield.

        Positive → equities cheap vs bonds (bullish equity signal).

        Returns
        -------
        pd.Series
        """
        if self.level_yield in yields_df.columns:
            bond_yield = yields_df[self.level_yield]
        else:
            bond_yield = yields_df.iloc[:, -1]

        spread = earnings_yield.reindex(bond_yield.index) - bond_yield
        spread_z = _rolling_zscore(spread, self.lookback)
        return spread_z.rename("equity_yield_spread_z")

    def backtest(
        self,
        yields_df: pd.DataFrame,
        price: pd.Series,
        feature: str = "yc_slope",
        signal_direction: float = 1.0,
    ) -> Dict:
        """Backtest a yield curve feature as a trading signal."""
        feats = self.compute_features(yields_df)
        if feature not in feats:
            return {}
        signal = feats[feature] * signal_direction
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}


# ---------------------------------------------------------------------------
# Credit Spread Signal
# ---------------------------------------------------------------------------

class CreditSpreadSignal:
    """
    Trading signals from credit spreads.

    Signals:
    - Z-score of IG/HY spreads (wide = risk-off, narrow = risk-on)
    - Spread momentum (trend in spreads)
    - Cross-market spread (HY - IG differential)
    - Distress ratio (fraction of issuers above distress threshold)

    Parameters
    ----------
    lookback : int
        Window for z-score computation.
    spread_type : str
        'hy' (High Yield), 'ig' (Investment Grade), or 'oas'.
    momentum_window : int
        Window for spread momentum.
    distress_threshold : float
        OAS threshold above which an issuer is considered distressed (bps).
    """

    def __init__(
        self,
        lookback: int = 126,
        spread_type: str = "hy",
        momentum_window: int = 21,
        distress_threshold: float = 1000.0,
    ) -> None:
        self.lookback = lookback
        self.spread_type = spread_type
        self.momentum_window = momentum_window
        self.distress_threshold = distress_threshold

    def compute_signals(self, spreads: pd.Series) -> pd.DataFrame:
        """
        Compute spread-based signals.

        Parameters
        ----------
        spreads : pd.Series
            Daily OAS or spread in basis points.

        Returns
        -------
        pd.DataFrame
            Features: spread_level, spread_z, spread_momentum,
            spread_pct, spread_regime.
        """
        feats = pd.DataFrame(index=spreads.index)

        feats["spread_level"] = spreads
        feats["spread_z"] = _rolling_zscore(spreads, self.lookback)

        # Spread momentum (rising spreads = risk-off)
        spread_ma = spreads.rolling(self.momentum_window, min_periods=1).mean()
        spread_ma_slow = spreads.rolling(self.momentum_window * 3, min_periods=1).mean()
        feats["spread_momentum"] = spread_ma - spread_ma_slow

        # Percentile rank
        feats["spread_pct"] = spreads.rank(pct=True)

        # Regime
        q25 = spreads.quantile(0.25)
        q75 = spreads.quantile(0.75)
        feats["spread_regime"] = "NORMAL"
        feats.loc[spreads > q75, "spread_regime"] = "WIDE"
        feats.loc[spreads < q25, "spread_regime"] = "NARROW"

        # Risk-on/off signal: narrow spreads = risk-on (+1), wide = risk-off (-1)
        feats["credit_signal"] = -feats["spread_z"].clip(-3, 3)

        return feats

    def hy_ig_differential(
        self,
        hy_spreads: pd.Series,
        ig_spreads: pd.Series,
    ) -> pd.Series:
        """
        HY-IG spread differential as a credit quality signal.

        Widening differential → risk-off (bearish equities).

        Returns
        -------
        pd.Series
            Z-scored differential.
        """
        diff = hy_spreads - ig_spreads
        return _rolling_zscore(diff, self.lookback).rename("hy_ig_differential_z")

    def distress_ratio(
        self,
        issuer_spreads: pd.DataFrame,
    ) -> pd.Series:
        """
        Fraction of issuers with spreads above distress_threshold.

        High distress ratio → systemic credit stress.

        Parameters
        ----------
        issuer_spreads : pd.DataFrame
            (dates x issuers) spread levels.

        Returns
        -------
        pd.Series
        """
        distressed = (issuer_spreads > self.distress_threshold).mean(axis=1)
        return distressed.rename("distress_ratio")

    def credit_cycle_phase(self, spreads: pd.Series) -> pd.Series:
        """
        Classify credit cycle phase based on spread level and momentum.

        Returns
        -------
        pd.Series
            'EXPANSION', 'LATE_CYCLE', 'CONTRACTION', 'RECOVERY' labels.
        """
        feats = self.compute_signals(spreads)
        level = feats["spread_z"]
        momentum = feats["spread_momentum"]
        mom_z = _rolling_zscore(momentum, self.lookback)

        labels = pd.Series("EXPANSION", index=spreads.index)
        labels[(level > 0) & (mom_z > 0)] = "CONTRACTION"
        labels[(level > 0) & (mom_z < 0)] = "RECOVERY"
        labels[(level < 0) & (mom_z > 0)] = "LATE_CYCLE"

        return labels

    def backtest(self, spreads: pd.Series, price: pd.Series) -> Dict:
        """Backtest credit signal against equity returns."""
        feats = self.compute_signals(spreads)
        signal = feats["credit_signal"]
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}


# ---------------------------------------------------------------------------
# DXY Regime and Momentum
# ---------------------------------------------------------------------------

class DXYSignal:
    """
    US Dollar Index (DXY) regime detection and momentum signals.

    A strong USD typically:
    - Pressures EM equities and commodities (negative signal)
    - Benefits USD-denominated asset holders
    - Indicates risk-off globally

    Parameters
    ----------
    lookback : int
        Window for z-score normalization.
    momentum_periods : list of int
        Lookback windows for momentum signals.
    trend_window : int
        Window for trend classification.
    """

    def __init__(
        self,
        lookback: int = 63,
        momentum_periods: Optional[List[int]] = None,
        trend_window: int = 200,
    ) -> None:
        self.lookback = lookback
        self.momentum_periods = momentum_periods or [21, 63, 126]
        self.trend_window = trend_window

    def compute_features(self, dxy: pd.Series) -> pd.DataFrame:
        """
        Compute DXY-based features.

        Parameters
        ----------
        dxy : pd.Series
            Daily DXY index level.

        Returns
        -------
        pd.DataFrame
            Feature columns.
        """
        feats = pd.DataFrame(index=dxy.index)
        feats["dxy_level"] = dxy
        feats["dxy_ret"] = dxy.pct_change()
        feats["dxy_z"] = _rolling_zscore(dxy, self.lookback)

        # Trend
        dxy_ma = dxy.rolling(self.trend_window, min_periods=self.trend_window // 2).mean()
        feats["dxy_above_ma"] = (dxy > dxy_ma).astype(float)
        feats["dxy_ma_distance_z"] = _rolling_zscore((dxy - dxy_ma) / (dxy_ma + 1e-6) * 100,
                                                       self.lookback)

        # Momentum
        for period in self.momentum_periods:
            ret = dxy.pct_change(period)
            feats[f"dxy_mom_{period}d"] = ret
            feats[f"dxy_mom_z_{period}d"] = _rolling_zscore(ret, self.lookback)

        # Regime
        q25 = dxy.quantile(0.25)
        q75 = dxy.quantile(0.75)
        feats["dxy_regime"] = "NEUTRAL"
        feats.loc[dxy > q75, "dxy_regime"] = "STRONG_USD"
        feats.loc[dxy < q25, "dxy_regime"] = "WEAK_USD"

        return feats

    def dxy_signal(
        self,
        dxy: pd.Series,
        target_asset: str = "equities",
        momentum_window: int = 63,
    ) -> pd.Series:
        """
        Generate asset-specific signal from DXY.

        Parameters
        ----------
        dxy : pd.Series
        target_asset : str
            'equities' — strong USD bearish for EM
            'commodities' — strong USD bearish for commodities
            'bonds' — strong USD typically risk-off, bullish for UST

        Returns
        -------
        pd.Series
            Signal in normalized units.
        """
        dxy_mom = dxy.pct_change(momentum_window)
        dxy_mom_z = _rolling_zscore(dxy_mom, self.lookback)

        # Direction depends on asset class
        if target_asset in ("equities", "commodities", "em"):
            # USD up → bearish for these assets
            return (-dxy_mom_z).rename(f"dxy_signal_{target_asset}")
        elif target_asset in ("bonds", "usd_assets"):
            # USD up → risk-off → bullish bonds
            return dxy_mom_z.rename(f"dxy_signal_{target_asset}")
        else:
            return (-dxy_mom_z).rename("dxy_signal")

    def backtest(
        self,
        dxy: pd.Series,
        price: pd.Series,
        target_asset: str = "equities",
    ) -> Dict:
        """Backtest DXY signal against an asset price."""
        signal = self.dxy_signal(dxy, target_asset)
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}


# ---------------------------------------------------------------------------
# Cross-Asset Momentum
# ---------------------------------------------------------------------------

class CrossAssetMomentum:
    """
    Cross-asset time-series momentum signal.

    Combines momentum signals across equity, bond, commodity, and FX
    markets into a composite risk-on/risk-off indicator.

    Parameters
    ----------
    lookbacks : list of int
        Lookback periods for momentum computation.
    vol_scale : bool
        If True, scale each asset's signal by its rolling volatility.
    vol_window : int
        Window for volatility estimation.
    weights : dict or None
        Asset-class weights. Defaults to equal weight.
    """

    def __init__(
        self,
        lookbacks: Optional[List[int]] = None,
        vol_scale: bool = True,
        vol_window: int = 63,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lookbacks = lookbacks or [21, 63, 126, 252]
        self.vol_scale = vol_scale
        self.vol_window = vol_window
        self.weights = weights

    def compute_asset_momentum(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute multi-lookback momentum for each asset.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily price levels (dates x assets).

        Returns
        -------
        pd.DataFrame
            Average momentum signal per asset (dates x assets).
        """
        all_signals = []
        for lb in self.lookbacks:
            mom = prices.pct_change(lb)
            if self.vol_scale:
                vol = prices.pct_change().rolling(self.vol_window, min_periods=10).std()
                vol = vol.replace(0, np.nan)
                mom = mom / (vol + 1e-10)
            # Z-score cross-sectionally
            mu = mom.mean(axis=1)
            sd = mom.std(axis=1).replace(0, 1.0)
            mom_z = mom.sub(mu, axis=0).div(sd, axis=0)
            all_signals.append(mom_z)

        return pd.concat(all_signals).groupby(level=0).mean()

    def composite_signal(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        commodity_prices: Optional[pd.Series] = None,
        fx_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Build a single cross-asset risk-on/risk-off composite.

        Risk-on: equities +, bonds -, commodities +
        Risk-off: equities -, bonds +, commodities -

        Parameters
        ----------
        equity_prices, bond_prices : pd.Series
        commodity_prices, fx_prices : pd.Series, optional

        Returns
        -------
        pd.Series
            Composite cross-asset momentum signal.
        """
        components = {}

        # Equity momentum → risk-on
        eq_mom = equity_prices.pct_change(63)
        eq_z = _rolling_zscore(eq_mom, 252)
        components["equity"] = eq_z

        # Bond momentum → risk-off (invert for composite)
        bond_mom = bond_prices.pct_change(63)
        bond_z = _rolling_zscore(bond_mom, 252)
        components["bond"] = -bond_z  # bond up = risk-off = subtract

        if commodity_prices is not None:
            com_mom = commodity_prices.pct_change(63)
            com_z = _rolling_zscore(com_mom, 252)
            components["commodity"] = com_z

        if fx_prices is not None:
            fx_mom = fx_prices.pct_change(63)
            fx_z = _rolling_zscore(fx_mom, 252)
            components["fx"] = fx_z

        # Weight and combine
        if self.weights:
            comp_df = pd.concat(
                [s.rename(k) for k, s in components.items()], axis=1
            )
            total_w = sum(self.weights.get(k, 1.0) for k in components)
            composite = sum(
                comp_df[k] * self.weights.get(k, 1.0) / total_w
                for k in components
                if k in comp_df.columns
            )
        else:
            comp_df = pd.concat(
                [s.rename(k) for k, s in components.items()], axis=1
            )
            composite = comp_df.mean(axis=1, skipna=True)

        smoothed = composite.rolling(21, min_periods=5).mean()
        return smoothed.rename("cross_asset_momentum")

    def risk_regime(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        commodity_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Classify market into RISK_ON / RISK_OFF / NEUTRAL.

        Returns
        -------
        pd.Series
        """
        composite = self.composite_signal(equity_prices, bond_prices, commodity_prices)
        q33 = composite.quantile(0.33)
        q67 = composite.quantile(0.67)

        regime = pd.Series("NEUTRAL", index=composite.index)
        regime[composite > q67] = "RISK_ON"
        regime[composite < q33] = "RISK_OFF"
        return regime

    def conditional_returns(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        target_price: pd.Series,
        commodity_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute target asset returns conditional on cross-asset regime.

        Returns
        -------
        pd.DataFrame
            Returns statistics per regime.
        """
        regime = self.risk_regime(equity_prices, bond_prices, commodity_prices)
        daily_ret = target_price.pct_change()
        combined = pd.concat([daily_ret, regime], axis=1).dropna()
        combined.columns = ["ret", "regime"]

        rows = []
        for r in sorted(combined["regime"].unique()):
            subset = combined[combined["regime"] == r]["ret"]
            rows.append({
                "regime": r,
                "n_obs": len(subset),
                "mean_annual": round(subset.mean() * 252, 4),
                "vol_annual": round(subset.std() * np.sqrt(252), 4),
                "sharpe": round(subset.mean() / (subset.std() + 1e-12) * np.sqrt(252), 4),
                "pct_positive": round((subset > 0).mean(), 4),
            })

        return pd.DataFrame(rows).set_index("regime")

    def backtest(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        target_price: pd.Series,
        commodity_prices: Optional[pd.Series] = None,
    ) -> Dict:
        """Backtest cross-asset momentum composite as a trading signal."""
        signal = self.composite_signal(equity_prices, bond_prices, commodity_prices)
        daily_ret = target_price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()
        return {k: round(v, 4) for k, v in _stats(port_rets).items()}
