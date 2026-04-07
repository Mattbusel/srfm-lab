# research/pipeline/cross_sectional_study.py
# SRFM -- Cross-sectional returns analysis: quintiles, IC, Fama-MacBeth regression

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuintileResult:
    """Results from a quintile sort analysis."""

    # Mean returns per quintile: index 0 = lowest factor score (short),
    # index 4 = highest factor score (long)
    quintile_returns: List[float]

    # Long-short spread: Q5 - Q1
    spread: float

    # Annualized information ratio of the spread
    information_ratio: float

    # t-statistic on the spread being different from zero
    t_stat: float

    # Fraction of consecutive quintile pairs (Qi, Qi+1) where Qi < Qi+1
    # Perfect monotonicity = 1.0; range [0, 1]
    monotonicity: float

    # Additional diagnostics
    quintile_sharpes: List[float] = None
    n_observations: int = 0
    spread_pvalue: float = 1.0

    def __post_init__(self):
        if self.quintile_sharpes is None:
            self.quintile_sharpes = []


@dataclass
class DecileResult:
    """Results from a decile sort analysis."""

    decile_returns: List[float]
    spread: float            # D10 - D1
    information_ratio: float
    t_stat: float
    monotonicity: float
    n_observations: int = 0
    spread_pvalue: float = 1.0


@dataclass
class FamaMacBethResult:
    """Results from a Fama-MacBeth panel regression."""

    # Mean cross-sectional slope coefficient on the factor
    coefficient: float

    # Standard error of the mean slope (time-series std / sqrt(T))
    std_error: float

    # t-statistic: coefficient / std_error
    t_stat: float

    # p-value from two-sided t-test
    p_value: float

    # Newey-West HAC standard error with 4 lags
    newey_west_se: float

    # Average cross-sectional R-squared
    r_squared: float

    # Number of time periods used
    n_periods: int

    # Per-period slope coefficients
    period_coefficients: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class CrossSectionalStudy:
    """
    Cross-sectional returns analysis toolkit.

    Provides quintile/decile portfolio sorts, information coefficient (IC)
    time series, Fama-MacBeth panel regressions, and long-short portfolio
    construction.
    """

    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    # ------------------------------------------------------------------
    # Quintile returns
    # ------------------------------------------------------------------

    def quintile_returns(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        forward_horizon: int = 1,
    ) -> QuintileResult:
        """
        Sort stocks into quintiles by factor score and compute mean returns per quintile.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers); higher = more positive exposure.
        returns : pd.DataFrame
            Daily stock returns (dates x tickers).
        forward_horizon : int
            Number of days to look forward for returns.

        Returns
        -------
        QuintileResult
        """
        return self._bucket_returns(factor, returns, forward_horizon, n_buckets=5)

    def decile_returns(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        forward_horizon: int = 1,
    ) -> DecileResult:
        """
        Sort stocks into deciles by factor score and compute mean returns per decile.
        """
        result = self._bucket_returns(factor, returns, forward_horizon, n_buckets=10)
        return DecileResult(
            decile_returns=result.decile_returns,
            spread=result.spread,
            information_ratio=result.information_ratio,
            t_stat=result.t_stat,
            monotonicity=result.monotonicity,
            n_observations=result.n_observations,
            spread_pvalue=result.spread_pvalue,
        )

    def _bucket_returns(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        forward_horizon: int,
        n_buckets: int,
    ):
        """
        Internal helper for quintile/decile sorts.
        """
        fwd_returns = returns.shift(-forward_horizon)

        common_dates = factor.index.intersection(fwd_returns.index)
        bucket_ret_lists = [[] for _ in range(n_buckets)]
        spread_series = []

        for date in common_dates:
            f = factor.loc[date].dropna()
            r = fwd_returns.loc[date].reindex(f.index).dropna()
            f = f.reindex(r.index)

            n = len(f)
            if n < n_buckets * 2:
                continue

            labels = pd.qcut(f, q=n_buckets, labels=False, duplicates="drop")
            if labels.isna().all():
                continue

            for b in range(n_buckets):
                mask = labels == b
                if mask.sum() > 0:
                    bucket_ret_lists[b].append(r[mask].mean())

            # Spread: top - bottom bucket
            top_mask = labels == (n_buckets - 1)
            bot_mask = labels == 0
            if top_mask.sum() > 0 and bot_mask.sum() > 0:
                spread_series.append(r[top_mask].mean() - r[bot_mask].mean())

        if not spread_series:
            empty = [0.0] * n_buckets
            if n_buckets == 5:
                return QuintileResult(
                    quintile_returns=empty, spread=0.0,
                    information_ratio=0.0, t_stat=0.0,
                    monotonicity=0.0, n_observations=0,
                )
            return DecileResult(
                decile_returns=empty, spread=0.0,
                information_ratio=0.0, t_stat=0.0,
                monotonicity=0.0, n_observations=0,
            )

        mean_rets = [
            float(np.mean(bucket_ret_lists[b])) if bucket_ret_lists[b] else 0.0
            for b in range(n_buckets)
        ]

        spread_arr = np.array(spread_series)
        spread_mean = float(spread_arr.mean())
        spread_std = float(spread_arr.std())
        ir = spread_mean / (spread_std + 1e-9) * np.sqrt(self.annualization_factor)
        t_stat, p_val = stats.ttest_1samp(spread_arr, 0.0)

        # Monotonicity: fraction of pairs where Q_i < Q_{i+1}
        n_pairs = n_buckets - 1
        mono_count = sum(
            1 for i in range(n_pairs) if mean_rets[i] < mean_rets[i + 1]
        )
        monotonicity = mono_count / n_pairs

        if n_buckets == 5:
            sharpes = []
            for b in range(n_buckets):
                arr = np.array(bucket_ret_lists[b])
                if len(arr) > 1 and arr.std() > 0:
                    sharpes.append(float(arr.mean() / arr.std() * np.sqrt(self.annualization_factor)))
                else:
                    sharpes.append(0.0)
            return QuintileResult(
                quintile_returns=mean_rets,
                spread=spread_mean,
                information_ratio=ir,
                t_stat=float(t_stat),
                monotonicity=monotonicity,
                quintile_sharpes=sharpes,
                n_observations=len(spread_series),
                spread_pvalue=float(p_val),
            )

        return DecileResult(
            decile_returns=mean_rets,
            spread=spread_mean,
            information_ratio=ir,
            t_stat=float(t_stat),
            monotonicity=monotonicity,
            n_observations=len(spread_series),
            spread_pvalue=float(p_val),
        )

    # ------------------------------------------------------------------
    # Information Coefficient
    # ------------------------------------------------------------------

    def information_coefficient(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = "spearman",
        forward_horizon: int = 1,
    ) -> pd.Series:
        """
        Compute a time series of cross-sectional IC between factor and forward returns.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers).
        forward_returns : pd.DataFrame
            Stock returns (dates x tickers); will be shifted by forward_horizon.
        method : str
            "spearman" (rank correlation) or "pearson" (linear correlation).
        forward_horizon : int
            Days ahead to measure returns.

        Returns
        -------
        pd.Series
            Daily IC values indexed by date.
        """
        if method not in ("spearman", "pearson"):
            raise ValueError("method must be 'spearman' or 'pearson'")

        fwd = forward_returns.shift(-forward_horizon)
        common = factor.index.intersection(fwd.index)
        ic_dict = {}

        for date in common:
            f = factor.loc[date].dropna()
            r = fwd.loc[date].reindex(f.index).dropna()
            f = f.reindex(r.index)
            if len(f) < 5:
                continue
            if method == "spearman":
                rho, _ = stats.spearmanr(f, r)
            else:
                rho, _ = stats.pearsonr(f, r)
            ic_dict[date] = float(rho)

        return pd.Series(ic_dict, name="ic").sort_index()

    # ------------------------------------------------------------------
    # Fama-MacBeth regression
    # ------------------------------------------------------------------

    def fama_macbeth(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        controls: Optional[pd.DataFrame] = None,
        forward_horizon: int = 1,
        nw_lags: int = 4,
    ) -> FamaMacBethResult:
        """
        Fama-MacBeth (1973) two-pass cross-sectional regression.

        Step 1: For each period t, regress cross-sectional returns on factor
                (and optional controls) to obtain period-specific slope gamma_t.
        Step 2: Estimate mean gamma = mean(gamma_t) and compute standard error
                as std(gamma_t) / sqrt(T), plus Newey-West HAC SE.

        Parameters
        ----------
        factor : pd.DataFrame
            Test factor scores (dates x tickers).
        returns : pd.DataFrame
            Stock returns (dates x tickers).
        controls : pd.DataFrame, optional
            Additional control factors stacked along columns as a
            MultiIndex (dates x (tickers, factor_names)). NOT YET SUPPORTED --
            use a flat DataFrame matching dates x tickers for single control.
        forward_horizon : int
            Days ahead for return measurement.
        nw_lags : int
            Lags for Newey-West HAC standard error adjustment.

        Returns
        -------
        FamaMacBethResult
        """
        fwd = returns.shift(-forward_horizon)
        common = factor.index.intersection(fwd.index)

        gammas = []
        r_squareds = []
        dates_used = []

        for date in common:
            f = factor.loc[date].dropna()
            r = fwd.loc[date].reindex(f.index).dropna()
            f = f.reindex(r.index)

            if len(f) < 10:
                continue

            # Standardize factor cross-sectionally
            f_std = (f - f.mean()) / (f.std() + 1e-9)
            y = r.values
            X = np.column_stack([np.ones(len(f_std)), f_std.values])

            if controls is not None and date in controls.index:
                ctrl = controls.loc[date].reindex(f.index).fillna(0.0)
                ctrl_std = (ctrl - ctrl.mean()) / (ctrl.std() + 1e-9)
                X = np.column_stack([X, ctrl_std.values])

            try:
                beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = 1.0 - ss_res / (ss_tot + 1e-12)

            gammas.append(beta[1])  # slope on the test factor
            r_squareds.append(r_sq)
            dates_used.append(date)

        if len(gammas) < 4:
            logger.warning("Fama-MacBeth: insufficient observations (%d periods)", len(gammas))
            return FamaMacBethResult(
                coefficient=0.0, std_error=0.0, t_stat=0.0, p_value=1.0,
                newey_west_se=0.0, r_squared=0.0, n_periods=0,
            )

        gamma_arr = np.array(gammas)
        T = len(gamma_arr)
        coef = float(gamma_arr.mean())
        ols_se = float(gamma_arr.std() / np.sqrt(T))
        t_stat = coef / (ols_se + 1e-12)
        _, p_val = stats.ttest_1samp(gamma_arr, 0.0)
        nw_se = self._newey_west_se(gamma_arr, lags=nw_lags)
        mean_r2 = float(np.mean(r_squareds))

        return FamaMacBethResult(
            coefficient=coef,
            std_error=ols_se,
            t_stat=float(t_stat),
            p_value=float(p_val),
            newey_west_se=nw_se,
            r_squared=mean_r2,
            n_periods=T,
            period_coefficients=pd.Series(gamma_arr, index=pd.DatetimeIndex(dates_used)),
        )

    def _newey_west_se(self, series: np.ndarray, lags: int = 4) -> float:
        """
        Newey-West HAC standard error for a scalar time series.

        Computes the long-run variance estimator:
        Var_NW = gamma_0 + 2 * sum_{j=1}^{lags} (1 - j/(lags+1)) * gamma_j
        where gamma_j is the lag-j autocovariance of (series - mean(series)).

        Returns SE = sqrt(Var_NW / T).
        """
        T = len(series)
        if T < 2:
            return 0.0
        demeaned = series - series.mean()
        gamma_0 = np.dot(demeaned, demeaned) / T
        nw_var = gamma_0
        for j in range(1, lags + 1):
            w = 1.0 - j / (lags + 1.0)
            gamma_j = np.dot(demeaned[j:], demeaned[:-j]) / T
            nw_var += 2.0 * w * gamma_j
        nw_var = max(nw_var, 0.0)
        return float(np.sqrt(nw_var / T))

    # ------------------------------------------------------------------
    # Long-short portfolio
    # ------------------------------------------------------------------

    def long_short_portfolio(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        n_long: int = 20,
        n_short: int = 20,
        forward_horizon: int = 1,
        weighting: str = "equal",
    ) -> pd.Series:
        """
        Construct a daily long-short portfolio return series.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers); top scores go long.
        returns : pd.DataFrame
            Daily stock returns.
        n_long : int
            Number of stocks in the long leg.
        n_short : int
            Number of stocks in the short leg.
        forward_horizon : int
            Days ahead for return measurement.
        weighting : str
            "equal" for equal-weight legs; "factor" for factor-score weighting.

        Returns
        -------
        pd.Series
            Daily long-short portfolio returns indexed by date.
        """
        if weighting not in ("equal", "factor"):
            raise ValueError("weighting must be 'equal' or 'factor'")

        fwd = returns.shift(-forward_horizon)
        common = factor.index.intersection(fwd.index)
        portfolio_rets = {}

        for date in common:
            f = factor.loc[date].dropna()
            r = fwd.loc[date].reindex(f.index).dropna()
            f = f.reindex(r.index)
            n = len(f)

            if n < n_long + n_short:
                continue

            sorted_f = f.sort_values()
            short_tickers = sorted_f.iloc[:n_short].index
            long_tickers = sorted_f.iloc[-n_long:].index

            if weighting == "equal":
                long_ret = r.loc[long_tickers].mean()
                short_ret = r.loc[short_tickers].mean()
            else:
                long_scores = f.loc[long_tickers].clip(lower=0)
                short_scores = (-f.loc[short_tickers]).clip(lower=0)
                long_ret = (r.loc[long_tickers] * long_scores).sum() / (long_scores.sum() + 1e-9)
                short_ret = (r.loc[short_tickers] * short_scores).sum() / (short_scores.sum() + 1e-9)

            portfolio_rets[date] = float(long_ret) - float(short_ret)

        return pd.Series(portfolio_rets, name="ls_portfolio").sort_index()

    # ------------------------------------------------------------------
    # Summary statistics helper
    # ------------------------------------------------------------------

    def summarize_ic(
        self, ic_series: pd.Series, annualize: bool = True
    ) -> dict:
        """
        Compute summary statistics for an IC time series.

        Returns a dict with: mean, std, icir, t_stat, p_value,
        hit_rate (fraction positive), max_drawdown_ic.
        """
        ic = ic_series.dropna()
        if len(ic) < 4:
            return {}

        mean = float(ic.mean())
        std = float(ic.std())
        icir_raw = mean / (std + 1e-9)
        icir = icir_raw * np.sqrt(self.annualization_factor) if annualize else icir_raw
        t_stat, p_value = stats.ttest_1samp(ic, 0.0)

        # Hit rate
        hit_rate = float((ic > 0).mean())

        # Max drawdown in cumulative IC (measures consistency)
        cum_ic = ic.cumsum()
        roll_max = cum_ic.cummax()
        drawdown = (cum_ic - roll_max)
        max_dd = float(drawdown.min())

        return {
            "mean": mean,
            "std": std,
            "icir": icir,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "hit_rate": hit_rate,
            "max_drawdown_ic": max_dd,
            "n": len(ic),
        }
