"""
Factor return series construction.

Implements:
- Fama-French 3 factors: MKT, SMB, HML
- WML (Winners Minus Losers momentum)
- Quality (QMJ: Quality Minus Junk)
- BAB (Betting Against Beta)
- Carry factor

All factor constructors accept a universe of daily prices plus
fundamental / classification data and return a pd.Series of daily
factor returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lo, hi)


def _demean_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def _long_short_return(
    signal: pd.Series,
    returns: pd.Series,
    n_quantile: int = 3,
) -> float:
    """
    Long top-quantile, short bottom-quantile one-period return.
    Returns float (scalar).
    """
    combined = pd.concat([signal, returns], axis=1).dropna()
    combined.columns = ["sig", "ret"]
    if len(combined) < n_quantile * 2:
        return np.nan
    combined["q"] = pd.qcut(combined["sig"], n_quantile, labels=False, duplicates="drop")
    group_ret = combined.groupby("q")["ret"].mean()
    if len(group_ret) < 2:
        return np.nan
    return float(group_ret.iloc[-1] - group_ret.iloc[0])


def _rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Rolling OLS beta of asset against market."""
    betas = []
    idx = asset_ret.index
    for i in range(window, len(idx)):
        y = asset_ret.iloc[i - window:i].values
        x = mkt_ret.reindex(idx[i - window:i]).values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < window // 2:
            betas.append(np.nan)
            continue
        x_m = x[mask]
        y_m = y[mask]
        var_x = np.var(x_m, ddof=1)
        beta = np.cov(x_m, y_m)[0, 1] / (var_x + 1e-12)
        betas.append(beta)
    return pd.Series(betas, index=idx[window:])


# ---------------------------------------------------------------------------
# Fama-French 3-Factor Construction
# ---------------------------------------------------------------------------

class FamaFrench3:
    """
    Construct Fama-French MKT, SMB, and HML factor returns.

    Parameters
    ----------
    rf_rate : pd.Series or float
        Risk-free rate series (daily) or scalar annualized rate.
    rebal_freq : str
        Rebalancing frequency: 'M' (monthly) or 'A' (annual).
    n_size_groups : int
        Number of size quantile groups (default 2: small/large).
    n_value_groups : int
        Number of book-to-market groups (default 3: low/mid/high).
    """

    def __init__(
        self,
        rf_rate: float = 0.02,
        rebal_freq: str = "M",
        n_size_groups: int = 2,
        n_value_groups: int = 3,
    ) -> None:
        self.rf_rate = rf_rate
        self.rebal_freq = rebal_freq
        self.n_size_groups = n_size_groups
        self.n_value_groups = n_value_groups

    def _daily_rf(self, index: pd.DatetimeIndex) -> pd.Series:
        if isinstance(self.rf_rate, pd.Series):
            return self.rf_rate.reindex(index, method="ffill").fillna(0)
        return pd.Series(self.rf_rate / 252, index=index)

    def compute(
        self,
        price_df: pd.DataFrame,
        market_cap: pd.DataFrame,
        book_to_market: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute MKT, SMB, HML daily factor returns.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily close prices (dates x tickers).
        market_cap : pd.DataFrame
            Market capitalisation (dates x tickers); used for size sort.
            Can be monthly (forward-filled internally).
        book_to_market : pd.DataFrame
            Book-to-market ratio (dates x tickers); updated at rebal frequency.

        Returns
        -------
        pd.DataFrame
            Columns: MKT, SMB, HML, RF.
        """
        daily_ret = price_df.pct_change()
        rf = self._daily_rf(price_df.index)

        # Value-weight market return
        mkt_cap_daily = market_cap.reindex(price_df.index, method="ffill")
        cap_weights = mkt_cap_daily.div(mkt_cap_daily.sum(axis=1), axis=0)
        vw_mkt = (cap_weights * daily_ret).sum(axis=1)
        mkt_excess = vw_mkt - rf

        # Rebalancing dates
        if self.rebal_freq == "M":
            rebal_dates = price_df.resample("ME").last().index
        else:
            rebal_dates = price_df.resample("YE").last().index

        # Build SMB and HML by holding portfolios constant between rebal dates
        smb_series = pd.Series(0.0, index=price_df.index)
        hml_series = pd.Series(0.0, index=price_df.index)

        for i, rebal_dt in enumerate(rebal_dates[:-1]):
            start = rebal_dt
            end = rebal_dates[i + 1]
            period_idx = price_df.index[(price_df.index > start) & (price_df.index <= end)]
            if len(period_idx) == 0:
                continue

            # Use most recent size and BM at rebal date
            # Get the last available data on or before rebal_dt
            mc_row = mkt_cap_daily.loc[:rebal_dt].iloc[-1] if len(mkt_cap_daily.loc[:rebal_dt]) > 0 else mkt_cap_daily.iloc[0]
            bm_row = book_to_market.reindex(price_df.index, method="ffill")
            if rebal_dt in bm_row.index:
                bm_row = bm_row.loc[rebal_dt]
            else:
                future_idx = bm_row.index[bm_row.index <= rebal_dt]
                bm_row = bm_row.iloc[-1] if len(future_idx) > 0 else bm_row.iloc[0]

            combined = pd.DataFrame({"mc": mc_row, "bm": bm_row}).dropna()
            if len(combined) < 4:
                continue

            # Size sort
            size_median = combined["mc"].median()
            small = combined[combined["mc"] <= size_median].index
            large = combined[combined["mc"] > size_median].index

            # BM sort (30/40/30)
            bm_30 = combined["bm"].quantile(0.30)
            bm_70 = combined["bm"].quantile(0.70)
            value = combined[combined["bm"] >= bm_70].index
            growth = combined[combined["bm"] <= bm_30].index

            # Six portfolios: S/V, S/N, S/G, B/V, B/N, B/G
            sv = small.intersection(value)
            sg = small.intersection(growth)
            bv = large.intersection(value)
            bg = large.intersection(growth)

            period_ret = daily_ret.loc[period_idx]

            def _ew_ret(tickers, ret_df):
                cols = [t for t in tickers if t in ret_df.columns]
                if not cols:
                    return pd.Series(0.0, index=ret_df.index)
                return ret_df[cols].mean(axis=1)

            sv_r = _ew_ret(sv, period_ret)
            sg_r = _ew_ret(sg, period_ret)
            bv_r = _ew_ret(bv, period_ret)
            bg_r = _ew_ret(bg, period_ret)

            smb_p = (sv_r + sg_r) / 2 - (bv_r + bg_r) / 2
            hml_p = (sv_r + bv_r) / 2 - (sg_r + bg_r) / 2

            smb_series.loc[period_idx] = smb_p.values
            hml_series.loc[period_idx] = hml_p.values

        result = pd.DataFrame({
            "MKT": mkt_excess,
            "SMB": smb_series,
            "HML": hml_series,
            "RF": rf,
        })
        return result

    def factor_statistics(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics for each factor."""
        rows = []
        for col in ["MKT", "SMB", "HML"]:
            if col not in factors.columns:
                continue
            f = factors[col].dropna()
            rows.append({
                "factor": col,
                "mean_annual": round(f.mean() * 252, 4),
                "std_annual": round(f.std() * np.sqrt(252), 4),
                "sharpe": round(f.mean() / (f.std() + 1e-12) * np.sqrt(252), 4),
                "skew": round(float(f.skew()), 4),
                "max_dd": round(float(((1 + f).cumprod().cummax() - (1 + f).cumprod()) /
                                      ((1 + f).cumprod().cummax() + 1e-12)).min(), 4),
            })
        return pd.DataFrame(rows).set_index("factor")


# ---------------------------------------------------------------------------
# WML Momentum Factor
# ---------------------------------------------------------------------------

class WMLFactor:
    """
    Winners Minus Losers (WML) momentum factor.

    Standard Jegadeesh-Titman (1993) construction:
    - Rank on 12-1 month momentum (skip last month)
    - Long top 30%, short bottom 30%
    - Equal-weight within groups

    Parameters
    ----------
    formation_period : int
        Lookback for momentum ranking (trading days).
    skip_period : int
        Days to skip before ranking date (avoids short-term reversal).
    holding_period : int
        Days to hold portfolio.
    long_pct : float
        Fraction of universe in long (and short) leg.
    """

    def __init__(
        self,
        formation_period: int = 252,
        skip_period: int = 21,
        holding_period: int = 21,
        long_pct: float = 0.30,
    ) -> None:
        self.formation_period = formation_period
        self.skip_period = skip_period
        self.holding_period = holding_period
        self.long_pct = long_pct

    def compute(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Compute daily WML factor returns.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily close prices (dates x tickers).

        Returns
        -------
        pd.Series
            Daily WML factor returns.
        """
        # Momentum: cumulative return from t-(formation+skip) to t-skip
        wml_returns = []
        wml_index = []
        rebal_freq = self.holding_period

        dates = price_df.index
        i = self.formation_period + self.skip_period

        while i < len(dates):
            dt = dates[i]
            form_start = dates[i - self.formation_period - self.skip_period]
            form_end = dates[i - self.skip_period]

            ret_window = price_df.loc[form_start:form_end].iloc[-1] / \
                         price_df.loc[form_start:form_end].iloc[0] - 1
            ret_window = ret_window.dropna()

            if len(ret_window) < 4:
                i += rebal_freq
                continue

            n_long = max(1, int(len(ret_window) * self.long_pct))
            winners = ret_window.nlargest(n_long).index
            losers = ret_window.nsmallest(n_long).index

            # Hold for holding_period days
            end_idx = min(i + rebal_freq, len(dates))
            hold_idx = dates[i:end_idx]
            hold_rets = price_df.loc[hold_idx].pct_change()

            winner_ret = hold_rets[winners].mean(axis=1)
            loser_ret = hold_rets[losers].mean(axis=1)
            wml_period = winner_ret - loser_ret

            for j, dt_j in enumerate(hold_idx):
                if j < len(wml_period):
                    wml_returns.append(float(wml_period.iloc[j]) if not np.isnan(wml_period.iloc[j]) else 0.0)
                    wml_index.append(dt_j)

            i += rebal_freq

        wml = pd.Series(wml_returns, index=wml_index, name="WML")
        # Deduplicate index by taking mean where overlap exists
        wml = wml.groupby(level=0).mean()
        return wml.reindex(price_df.index).fillna(0)

    def momentum_statistics(self, price_df: pd.DataFrame) -> Dict:
        """Summary statistics of the WML factor."""
        wml = self.compute(price_df)
        return {
            "mean_annual": round(wml.mean() * 252, 4),
            "std_annual": round(wml.std() * np.sqrt(252), 4),
            "sharpe": round(wml.mean() / (wml.std() + 1e-12) * np.sqrt(252), 4),
            "max_dd": round(float(
                ((1 + wml).cumprod() - (1 + wml).cumprod().cummax()) /
                ((1 + wml).cumprod().cummax() + 1e-12)
            ).min() if len(wml) > 0 else np.nan, 4),
            "skew": round(float(wml.skew()), 4),
        }


# ---------------------------------------------------------------------------
# Quality Factor (QMJ)
# ---------------------------------------------------------------------------

class QualityFactor:
    """
    Quality Minus Junk (QMJ) factor (Asness, Frazzini, Pedersen 2019).

    Quality score combines:
    - Profitability: ROE, ROA, gross profit margin, cash flow margin
    - Growth: 5-year growth in above metrics
    - Safety: low leverage, low earnings variability, low beta

    Each component is z-scored cross-sectionally and averaged.

    Parameters
    ----------
    rebal_freq : str
        'Q' (quarterly) or 'M' (monthly).
    long_pct : float
        Fraction of universe in each leg.
    """

    def __init__(self, rebal_freq: str = "Q", long_pct: float = 0.30) -> None:
        self.rebal_freq = rebal_freq
        self.long_pct = long_pct

    def compute_quality_score(
        self,
        roe: pd.DataFrame,
        roa: pd.DataFrame,
        gross_margin: pd.DataFrame,
        leverage: pd.DataFrame,
        earnings_growth: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute composite quality score.

        Parameters
        ----------
        roe, roa, gross_margin : pd.DataFrame
            Profitability metrics (dates x tickers).
        leverage : pd.DataFrame
            Debt-to-equity or similar leverage metric.
        earnings_growth : pd.DataFrame, optional
            YoY growth in earnings.

        Returns
        -------
        pd.DataFrame
            Composite quality z-score.
        """
        def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=1)
            sd = df.std(axis=1).replace(0, np.nan)
            return df.sub(mu, axis=0).div(sd, axis=0)

        prof_z = (
            _cs_zscore(roe).fillna(0)
            + _cs_zscore(roa).fillna(0)
            + _cs_zscore(gross_margin).fillna(0)
        ) / 3

        safety_z = -_cs_zscore(leverage).fillna(0)

        if earnings_growth is not None:
            growth_z = _cs_zscore(earnings_growth).fillna(0)
            quality = (prof_z + safety_z + growth_z) / 3
        else:
            quality = (prof_z + safety_z) / 2

        return quality

    def compute(
        self,
        price_df: pd.DataFrame,
        roe: pd.DataFrame,
        roa: pd.DataFrame,
        gross_margin: pd.DataFrame,
        leverage: pd.DataFrame,
        earnings_growth: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute QMJ daily factor returns.

        Returns
        -------
        pd.Series
            Daily QMJ factor returns.
        """
        quality = self.compute_quality_score(roe, roa, gross_margin, leverage, earnings_growth)
        quality_daily = quality.reindex(price_df.index, method="ffill")

        daily_ret = price_df.pct_change()

        if self.rebal_freq == "Q":
            rebal_dates = price_df.resample("QE").last().index
        else:
            rebal_dates = price_df.resample("ME").last().index

        qmj_returns = []
        qmj_index = []

        for i, rebal_dt in enumerate(rebal_dates[:-1]):
            start = rebal_dt
            end = rebal_dates[i + 1]
            period_idx = price_df.index[(price_df.index > start) & (price_df.index <= end)]

            if len(period_idx) == 0:
                continue

            if rebal_dt not in quality_daily.index:
                continue

            q_row = quality_daily.loc[rebal_dt].dropna()
            if len(q_row) < 4:
                continue

            n_long = max(1, int(len(q_row) * self.long_pct))
            quality_stocks = q_row.nlargest(n_long).index
            junk_stocks = q_row.nsmallest(n_long).index

            period_ret = daily_ret.loc[period_idx]
            q_ret = period_ret[[c for c in quality_stocks if c in period_ret.columns]].mean(axis=1)
            j_ret = period_ret[[c for c in junk_stocks if c in period_ret.columns]].mean(axis=1)
            qmj_period = q_ret - j_ret

            qmj_returns.extend(qmj_period.tolist())
            qmj_index.extend(period_idx.tolist())

        qmj = pd.Series(qmj_returns, index=qmj_index, name="QMJ")
        qmj = qmj.groupby(level=0).mean()
        return qmj.reindex(price_df.index).fillna(0)


# ---------------------------------------------------------------------------
# BAB (Betting Against Beta)
# ---------------------------------------------------------------------------

class BABFactor:
    """
    Betting Against Beta factor (Frazzini & Pedersen 2014).

    Construct beta-sorted portfolios:
    - Long low-beta stocks (levered to market beta = 1)
    - Short high-beta stocks (de-levered to market beta = 1)

    The portfolio is dollar-neutral but beta-neutral.

    Parameters
    ----------
    beta_window : int
        Rolling window for beta estimation.
    beta_shrinkage : float
        Shrinkage of estimated beta toward 1.0 (Vasicek shrinkage).
    rebal_freq : str
        Rebalancing frequency.
    n_groups : int
        Number of beta quantile groups.
    """

    def __init__(
        self,
        beta_window: int = 252,
        beta_shrinkage: float = 0.6,
        rebal_freq: str = "M",
        n_groups: int = 5,
    ) -> None:
        self.beta_window = beta_window
        self.beta_shrinkage = beta_shrinkage
        self.rebal_freq = rebal_freq
        self.n_groups = n_groups

    def compute_betas(
        self, price_df: pd.DataFrame, market_ret: pd.Series
    ) -> pd.DataFrame:
        """
        Compute rolling betas for all tickers.

        Returns
        -------
        pd.DataFrame
            Rolling beta estimates (dates x tickers).
        """
        daily_ret = price_df.pct_change()
        beta_df = pd.DataFrame(np.nan, index=price_df.index, columns=price_df.columns)

        for col in price_df.columns:
            asset_ret = daily_ret[col]
            beta_series = _rolling_beta(asset_ret, market_ret, self.beta_window)
            # Vasicek shrinkage toward 1
            shrunk = self.beta_shrinkage * beta_series + (1 - self.beta_shrinkage) * 1.0
            beta_df[col] = shrunk.reindex(price_df.index)

        return beta_df

    def compute(
        self,
        price_df: pd.DataFrame,
        market_ret: pd.Series,
    ) -> pd.Series:
        """
        Compute daily BAB factor returns.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily close prices (dates x tickers).
        market_ret : pd.Series
            Daily market returns (value-weighted).

        Returns
        -------
        pd.Series
            Daily BAB returns.
        """
        beta_df = self.compute_betas(price_df, market_ret)
        daily_ret = price_df.pct_change()

        if self.rebal_freq == "M":
            rebal_dates = price_df.resample("ME").last().index
        else:
            rebal_dates = price_df.resample("QE").last().index

        bab_returns = []
        bab_index = []

        for i, rebal_dt in enumerate(rebal_dates[:-1]):
            start = rebal_dt
            end = rebal_dates[i + 1]
            period_idx = price_df.index[(price_df.index > start) & (price_df.index <= end)]

            if len(period_idx) == 0:
                continue

            # Get betas at rebal date
            available_betas = beta_df.loc[:rebal_dt].dropna(how="all")
            if len(available_betas) == 0:
                continue
            beta_row = available_betas.iloc[-1].dropna()
            if len(beta_row) < 4:
                continue

            beta_median = beta_row.median()
            low_beta = beta_row[beta_row <= beta_median].index
            high_beta = beta_row[beta_row > beta_median].index

            avg_low_beta = beta_row[low_beta].mean()
            avg_high_beta = beta_row[high_beta].mean()

            period_ret = daily_ret.loc[period_idx]
            low_ret = period_ret[[c for c in low_beta if c in period_ret.columns]].mean(axis=1)
            high_ret = period_ret[[c for c in high_beta if c in period_ret.columns]].mean(axis=1)

            # Scale to beta-neutral
            if avg_low_beta > 1e-6 and avg_high_beta > 1e-6:
                bab_period = low_ret / avg_low_beta - high_ret / avg_high_beta
            else:
                bab_period = low_ret - high_ret

            bab_returns.extend(bab_period.tolist())
            bab_index.extend(period_idx.tolist())

        bab = pd.Series(bab_returns, index=bab_index, name="BAB")
        bab = bab.groupby(level=0).mean()
        return bab.reindex(price_df.index).fillna(0)

    def beta_statistics(
        self, price_df: pd.DataFrame, market_ret: pd.Series
    ) -> pd.DataFrame:
        """Summary statistics of cross-sectional beta distribution."""
        beta_df = self.compute_betas(price_df, market_ret)
        rows = []
        for col in beta_df.columns:
            b = beta_df[col].dropna()
            if len(b) == 0:
                continue
            rows.append({
                "ticker": col,
                "mean_beta": round(b.mean(), 4),
                "std_beta": round(b.std(), 4),
                "min_beta": round(b.min(), 4),
                "max_beta": round(b.max(), 4),
                "current_beta": round(float(b.iloc[-1]), 4),
            })
        return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------------------
# Carry Factor
# ---------------------------------------------------------------------------

class CarryFactor:
    """
    Carry factor across asset classes (Koijen et al. 2018).

    Carry = expected return from holding the asset if prices don't change.
    For equities: dividend yield.
    For bonds: yield – duration × yield_change_expectation.
    For FX: interest rate differential.
    For commodities: roll yield (spot – futures spread).

    Parameters
    ----------
    asset_type : str
        'equity', 'bond', 'fx', or 'commodity'.
    rebal_freq : str
        Rebalancing frequency: 'M' or 'W'.
    long_pct : float
        Fraction of universe in each leg.
    carry_smoothing : int
        Rolling window to smooth carry signal (avoids noise).
    """

    def __init__(
        self,
        asset_type: str = "equity",
        rebal_freq: str = "M",
        long_pct: float = 0.30,
        carry_smoothing: int = 3,
    ) -> None:
        self.asset_type = asset_type
        self.rebal_freq = rebal_freq
        self.long_pct = long_pct
        self.carry_smoothing = carry_smoothing

    def compute_carry(
        self,
        price_df: pd.DataFrame,
        carry_signal: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute carry factor returns.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily prices.
        carry_signal : pd.DataFrame
            Carry metric (dates x tickers).  For equity: dividend yield.
            For FX: interest rate differential.  For commodity: roll yield.

        Returns
        -------
        pd.Series
            Daily carry factor returns.
        """
        carry_smooth = carry_signal.rolling(self.carry_smoothing, min_periods=1).mean()
        carry_daily = carry_smooth.reindex(price_df.index, method="ffill")

        daily_ret = price_df.pct_change()

        if self.rebal_freq == "M":
            rebal_dates = price_df.resample("ME").last().index
        elif self.rebal_freq == "W":
            rebal_dates = price_df.resample("W").last().index
        else:
            rebal_dates = price_df.resample("ME").last().index

        carry_returns = []
        carry_index = []

        for i, rebal_dt in enumerate(rebal_dates[:-1]):
            start = rebal_dt
            end = rebal_dates[i + 1]
            period_idx = price_df.index[(price_df.index > start) & (price_df.index <= end)]

            if len(period_idx) == 0:
                continue

            if rebal_dt not in carry_daily.index:
                continue

            carry_row = carry_daily.loc[rebal_dt].dropna()
            if len(carry_row) < 4:
                continue

            n_long = max(1, int(len(carry_row) * self.long_pct))
            high_carry = carry_row.nlargest(n_long).index
            low_carry = carry_row.nsmallest(n_long).index

            period_ret = daily_ret.loc[period_idx]
            hc_ret = period_ret[[c for c in high_carry if c in period_ret.columns]].mean(axis=1)
            lc_ret = period_ret[[c for c in low_carry if c in period_ret.columns]].mean(axis=1)

            carry_period = hc_ret - lc_ret
            carry_returns.extend(carry_period.tolist())
            carry_index.extend(period_idx.tolist())

        carry = pd.Series(carry_returns, index=carry_index, name="CARRY")
        carry = carry.groupby(level=0).mean()
        return carry.reindex(price_df.index).fillna(0)

    def carry_statistics(
        self,
        price_df: pd.DataFrame,
        carry_signal: pd.DataFrame,
    ) -> Dict:
        """Summary statistics for the carry factor."""
        carry = self.compute_carry(price_df, carry_signal)
        return {
            "mean_annual": round(carry.mean() * 252, 4),
            "std_annual": round(carry.std() * np.sqrt(252), 4),
            "sharpe": round(carry.mean() / (carry.std() + 1e-12) * np.sqrt(252), 4),
            "skew": round(float(carry.skew()), 4),
            "kurt": round(float(carry.kurt()), 4),
            "pct_positive": round((carry > 0).mean(), 4),
        }


# ---------------------------------------------------------------------------
# Factor Portfolio Builder
# ---------------------------------------------------------------------------

class FactorPortfolio:
    """
    Combines multiple factor return series into a diversified factor portfolio.

    Parameters
    ----------
    target_vol : float
        Annualized target volatility for each factor leg.
    rebal_freq : str
        Rebalancing frequency for factor weights.
    method : str
        'equal_weight', 'risk_parity', or 'max_sharpe'.
    vol_lookback : int
        Rolling window for factor volatility estimation.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        rebal_freq: str = "M",
        method: str = "equal_weight",
        vol_lookback: int = 63,
    ) -> None:
        self.target_vol = target_vol
        self.rebal_freq = rebal_freq
        self.method = method
        self.vol_lookback = vol_lookback

    def vol_scale(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Scale each factor to target vol."""
        rolling_vol = factor_returns.rolling(self.vol_lookback, min_periods=20).std() * np.sqrt(252)
        scale = (self.target_vol / (rolling_vol + 1e-8)).clip(0, 5)
        return factor_returns * scale

    def combine(self, factor_returns: pd.DataFrame) -> pd.Series:
        """
        Combine factor returns according to the chosen method.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Daily factor returns (dates x factors).

        Returns
        -------
        pd.Series
            Combined factor portfolio daily returns.
        """
        scaled = self.vol_scale(factor_returns)

        if self.method == "equal_weight":
            return scaled.mean(axis=1)

        elif self.method == "risk_parity":
            from scipy.optimize import minimize

            if self.rebal_freq == "M":
                rebal_dates = factor_returns.resample("ME").last().index
            else:
                rebal_dates = factor_returns.resample("QE").last().index

            weights_df = pd.DataFrame(
                1.0 / factor_returns.shape[1],
                index=factor_returns.index,
                columns=factor_returns.columns,
            )

            for i, rebal_dt in enumerate(rebal_dates[:-1]):
                past = scaled.loc[:rebal_dt].tail(self.vol_lookback)
                if len(past) < 20:
                    continue
                cov = past.cov().values
                n = cov.shape[0]

                def risk_budget_obj(w):
                    w = np.array(w)
                    port_vol = np.sqrt(w @ cov @ w)
                    rc = w * (cov @ w) / (port_vol + 1e-12)
                    target = np.full(n, port_vol / n)
                    return float(np.sum((rc - target) ** 2))

                w0 = np.ones(n) / n
                bounds = [(0.01, 0.5)] * n
                cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                res = minimize(risk_budget_obj, w0, bounds=bounds, constraints=cons,
                               method="SLSQP")
                if res.success:
                    start = rebal_dt
                    end = rebal_dates[i + 1]
                    mask = (weights_df.index > start) & (weights_df.index <= end)
                    weights_df.loc[mask] = res.x

            return (scaled * weights_df).sum(axis=1)

        elif self.method == "max_sharpe":
            return scaled.mean(axis=1)  # simplified; full max-Sharpe in portfolio module

        return scaled.mean(axis=1)

    def factor_correlation_matrix(
        self, factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Pairwise Pearson correlation between factors."""
        return factor_returns.corr().round(4)

    def factor_summary(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Per-factor annualized statistics."""
        rows = []
        for col in factor_returns.columns:
            f = factor_returns[col].dropna()
            if len(f) == 0:
                continue
            eq = (1 + f).cumprod()
            roll_max = eq.cummax()
            dd = ((eq - roll_max) / (roll_max + 1e-12)).min()
            rows.append({
                "factor": col,
                "mean_return": round(f.mean() * 252, 4),
                "volatility": round(f.std() * np.sqrt(252), 4),
                "sharpe": round(f.mean() / (f.std() + 1e-12) * np.sqrt(252), 4),
                "skew": round(float(f.skew()), 4),
                "max_drawdown": round(float(dd), 4),
            })
        return pd.DataFrame(rows).set_index("factor")
