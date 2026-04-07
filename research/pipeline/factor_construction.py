# research/pipeline/factor_construction.py
# SRFM -- Systematic factor construction: building, combining, and neutralizing factors

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FactorBuilder -- constructs individual factors from raw data
# ---------------------------------------------------------------------------

class FactorBuilder:
    """
    Builds individual factor exposures from raw market and fundamental data.

    All returned DataFrames are indexed by date, columns by ticker.
    Raw scores are NOT yet neutralized or standardized -- apply FactorNeutralizer
    downstream for production use.
    """

    # ------------------------------------------------------------------
    # Momentum factor
    # ------------------------------------------------------------------

    def build_momentum_factor(
        self,
        returns: pd.DataFrame,
        lookback: int,
        skip: int = 1,
    ) -> pd.DataFrame:
        """
        Standard price momentum factor.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily total returns, (dates x tickers).
        lookback : int
            Rolling window length in trading days.
        skip : int
            Days to skip before the lookback window (avoids 1d reversal bias).
            Default is 1 (skip most recent day).

        Returns
        -------
        pd.DataFrame
            Cumulative return over [t - lookback - skip, t - skip].
        """
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if skip < 0:
            raise ValueError("skip must be non-negative")

        cum_return = (1.0 + returns).rolling(lookback).apply(
            lambda x: x.prod() - 1.0, raw=True
        )
        # Shift by skip days to avoid short-term reversal contamination
        factor = cum_return.shift(skip)
        factor.name = f"momentum_{lookback}d_skip{skip}"
        return factor

    # ------------------------------------------------------------------
    # Quality factor
    # ------------------------------------------------------------------

    def build_quality_factor(
        self, fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Composite quality factor from ROE, gross margin, and accruals.

        Parameters
        ----------
        fundamentals : pd.DataFrame
            MultiIndex (date, ticker) DataFrame with columns:
            ["roe", "gross_margin", "total_assets", "net_income",
             "operating_cf", "book_equity"].
            Alternatively a panel dict is accepted.

        Returns
        -------
        pd.DataFrame
            Cross-sectionally z-scored quality scores, (dates x tickers).
        """
        required_cols = {"roe", "gross_margin", "total_assets"}
        if not required_cols.issubset(fundamentals.columns):
            raise ValueError(f"fundamentals must contain columns: {required_cols}")

        neutralizer = FactorNeutralizer()

        # -- Return on equity (higher = better)
        roe = fundamentals["roe"].unstack(level=-1) if isinstance(
            fundamentals.index, pd.MultiIndex
        ) else pd.DataFrame(
            np.tile(fundamentals["roe"].values.reshape(-1, 1), (1, 1)),
            index=fundamentals.index,
            columns=["roe"],
        )

        # -- Gross margin (higher = better)
        gm = fundamentals["gross_margin"].unstack(level=-1) if isinstance(
            fundamentals.index, pd.MultiIndex
        ) else pd.DataFrame(
            fundamentals["gross_margin"].values.reshape(-1, 1),
            index=fundamentals.index,
            columns=["gm"],
        )

        # -- Accruals ratio (lower = better quality, so negate)
        if "operating_cf" in fundamentals.columns and "net_income" in fundamentals.columns:
            accruals_raw = (
                fundamentals["net_income"] - fundamentals["operating_cf"]
            ) / (fundamentals["total_assets"].clip(lower=1.0))
            accruals = accruals_raw.unstack(level=-1) if isinstance(
                fundamentals.index, pd.MultiIndex
            ) else pd.DataFrame(
                accruals_raw.values.reshape(-1, 1),
                index=fundamentals.index,
                columns=["accruals"],
            )
            accruals = -accruals  # negate: low accruals = high quality
        else:
            accruals = pd.DataFrame(0.0, index=roe.index, columns=roe.columns)

        # Align all to common index and columns
        all_idx = roe.index.union(gm.index).union(accruals.index)
        all_cols = roe.columns.union(gm.columns).union(accruals.columns)
        roe = roe.reindex(index=all_idx, columns=all_cols)
        gm = gm.reindex(index=all_idx, columns=all_cols)
        accruals = accruals.reindex(index=all_idx, columns=all_cols)

        # Z-score each component cross-sectionally and average
        roe_z = neutralizer.standardize(roe)
        gm_z = neutralizer.standardize(gm)
        ac_z = neutralizer.standardize(accruals)

        quality = (roe_z + gm_z + ac_z) / 3.0
        return quality

    # ------------------------------------------------------------------
    # Value factor
    # ------------------------------------------------------------------

    def build_value_factor(
        self,
        prices: pd.DataFrame,
        book_values: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Book-to-market value factor.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily closing prices (dates x tickers).
        book_values : pd.DataFrame
            Book value per share, typically quarterly but can be daily (dates x tickers).
            Will be forward-filled to match price dates.

        Returns
        -------
        pd.DataFrame
            Book-to-price ratio (high = cheap = positive value factor).
        """
        book = book_values.reindex(index=prices.index, columns=prices.columns).ffill()
        price_clipped = prices.clip(lower=0.01)
        bp_ratio = book / price_clipped
        # Winsorize extreme values before returning
        neutralizer = FactorNeutralizer()
        return neutralizer.winsorize(bp_ratio, clip_pct=0.01)

    # ------------------------------------------------------------------
    # Low-volatility factor
    # ------------------------------------------------------------------

    def build_low_vol_factor(
        self,
        returns: pd.DataFrame,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Low-realized-volatility factor.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x tickers).
        window : int
            Rolling window for realized vol estimation, default 63 (3 months).

        Returns
        -------
        pd.DataFrame
            Inverted percentile rank of realized vol: high score = low vol stock.
        """
        if window < 2:
            raise ValueError("window must be at least 2")

        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        # Rank cross-sectionally; invert so low vol -> high rank
        low_vol = realized_vol.rank(axis=1, ascending=False, pct=True)
        return low_vol

    # ------------------------------------------------------------------
    # Liquidity factor
    # ------------------------------------------------------------------

    def build_liquidity_factor(
        self,
        volumes: pd.DataFrame,
        market_caps: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Amihud illiquidity factor (inverted for use as a liquidity premium signal).

        Parameters
        ----------
        volumes : pd.DataFrame
            Share volume (dates x tickers).
        market_caps : pd.DataFrame
            Market capitalisation in USD (dates x tickers).
        window : int
            Rolling window for illiquidity estimation.

        Returns
        -------
        pd.DataFrame
            Percentile rank of illiquidity (high = more illiquid = premium potential).
        """
        # Amihud ratio = |return| / dollar_volume
        # We approximate returns from market cap changes
        mc = market_caps.reindex(columns=volumes.columns)
        price_proxy = mc / (volumes + 1.0)  # rough per-share price proxy
        ret_abs = price_proxy.pct_change().abs()
        dollar_vol = volumes * price_proxy.shift(1)
        amihud = ret_abs / (dollar_vol + 1.0)
        amihud_smooth = amihud.rolling(window).mean()
        # Rank: high amihud = illiquid = potential premium
        illiq_rank = amihud_smooth.rank(axis=1, pct=True)
        return illiq_rank

    # ------------------------------------------------------------------
    # Sentiment factor
    # ------------------------------------------------------------------

    def build_sentiment_factor(
        self,
        news_scores: pd.DataFrame,
        lookback: int = 5,
    ) -> pd.DataFrame:
        """
        News-based sentiment factor with EMA smoothing.

        Parameters
        ----------
        news_scores : pd.DataFrame
            Raw NLP sentiment scores in [-1, 1] range (dates x tickers).
        lookback : int
            EMA span for smoothing raw daily scores.

        Returns
        -------
        pd.DataFrame
            EMA-smoothed sentiment percentile rank.
        """
        if lookback < 1:
            raise ValueError("lookback must be at least 1")

        smoothed = news_scores.ewm(span=lookback, min_periods=1).mean()
        ranked = smoothed.rank(axis=1, pct=True)
        return ranked


# ---------------------------------------------------------------------------
# FactorCombiner -- combines multiple factors into a composite
# ---------------------------------------------------------------------------

class FactorCombiner:
    """
    Combines multiple factor DataFrames into a single composite factor.

    All input factors should have the same (date x ticker) shape.
    """

    def _align_factors(
        self, factors: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align all factor DataFrames to a common index and columns."""
        all_idx = None
        all_cols = None
        for df in factors.values():
            all_idx = df.index if all_idx is None else all_idx.union(df.index)
            all_cols = df.columns if all_cols is None else all_cols.union(df.columns)
        return {
            name: df.reindex(index=all_idx, columns=all_cols)
            for name, df in factors.items()
        }

    def equal_weight(
        self, factors: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Equal-weight composite: simple average of all factor scores after
        cross-sectional z-scoring each factor.
        """
        if not factors:
            raise ValueError("factors dict is empty")

        neutralizer = FactorNeutralizer()
        aligned = self._align_factors(factors)
        standardized = {name: neutralizer.standardize(df) for name, df in aligned.items()}
        stack = pd.concat(list(standardized.values()), axis=0, keys=list(standardized.keys()))
        composite = stack.groupby(level=1).mean()
        return composite

    def ic_weight(
        self,
        factors: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """
        IC-weighted composite: weight each factor by its trailing ICIR.

        Parameters
        ----------
        factors : dict
            Named factor DataFrames.
        returns : pd.DataFrame
            Forward returns for IC calculation.
        lookback : int
            Rolling window for ICIR estimation.

        Returns
        -------
        pd.DataFrame
            ICIR-weighted composite factor.
        """
        if not factors:
            raise ValueError("factors dict is empty")

        aligned = self._align_factors(factors)
        neutralizer = FactorNeutralizer()

        factor_arrays = {}
        for name, df in aligned.items():
            factor_arrays[name] = neutralizer.standardize(df)

        common_dates = sorted(
            set.intersection(*[set(df.index) for df in factor_arrays.values()])
        )

        composite_rows = {}

        for i, date in enumerate(common_dates):
            if i < lookback:
                # Not enough history -- fall back to equal weight
                n = len(factor_arrays)
                weights = {name: 1.0 / n for name in factor_arrays}
            else:
                window_dates = common_dates[i - lookback: i]
                ic_window = {}
                for name, df in factor_arrays.items():
                    ic_vals = []
                    for wd in window_dates:
                        if wd not in df.index or wd not in returns.index:
                            continue
                        s = df.loc[wd].dropna()
                        r = returns.loc[wd].reindex(s.index).dropna()
                        s = s.reindex(r.index)
                        if len(s) < 5:
                            continue
                        rho, _ = stats.spearmanr(s, r)
                        ic_vals.append(rho)
                    if ic_vals and np.std(ic_vals) > 1e-9:
                        icir = np.mean(ic_vals) / np.std(ic_vals)
                    else:
                        icir = 0.0
                    ic_window[name] = max(icir, 0.0)  # only positive ICIR signals

                total_icir = sum(ic_window.values())
                if total_icir > 0:
                    weights = {n: v / total_icir for n, v in ic_window.items()}
                else:
                    n = len(factor_arrays)
                    weights = {n: 1.0 / n for n in factor_arrays}

            row = pd.Series(0.0, index=returns.columns)
            for name, df in factor_arrays.items():
                if date in df.index:
                    row = row.add(df.loc[date].fillna(0.0) * weights[name], fill_value=0.0)
            composite_rows[date] = row

        composite = pd.DataFrame(composite_rows).T
        return composite

    def rank_combine(
        self, factors: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Rank-combination: cross-sectionally rank each factor separately,
        then average the ranks. Robust to outliers.
        """
        if not factors:
            raise ValueError("factors dict is empty")

        aligned = self._align_factors(factors)
        ranked = {
            name: df.rank(axis=1, pct=True)
            for name, df in aligned.items()
        }
        stack = pd.concat(list(ranked.values()), axis=0, keys=list(ranked.keys()))
        composite = stack.groupby(level=1).mean()
        return composite

    def optimize_weights(
        self,
        factors: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        method: str = "max_ic",
    ) -> Dict[str, float]:
        """
        Optimize factor combination weights via quadratic programming.

        Parameters
        ----------
        factors : dict
            Named factor DataFrames.
        returns : pd.DataFrame
            Forward returns used as the optimization target.
        method : str
            "max_ic"  -- maximize expected IC subject to w >= 0, sum(w) = 1.
            "max_icir" -- maximize IC / std(IC) using the full IC time series.

        Returns
        -------
        dict
            Optimized weights for each factor name.
        """
        if not factors:
            raise ValueError("factors dict is empty")

        aligned = self._align_factors(factors)
        names = list(aligned.keys())
        n = len(names)

        if n == 1:
            return {names[0]: 1.0}

        # Build IC matrix: one row per date, one column per factor
        neutralizer = FactorNeutralizer()
        common_dates = sorted(
            set.intersection(*[set(df.index) for df in aligned.values()])
            .intersection(set(returns.index))
        )

        ic_matrix = np.zeros((len(common_dates), n))
        for j, name in enumerate(names):
            df = neutralizer.standardize(aligned[name])
            for i, date in enumerate(common_dates):
                if date not in df.index:
                    continue
                s = df.loc[date].dropna()
                r = returns.loc[date].reindex(s.index).dropna()
                s = s.reindex(r.index)
                if len(s) < 5:
                    continue
                rho, _ = stats.spearmanr(s, r)
                ic_matrix[i, j] = rho

        mean_ic = ic_matrix.mean(axis=0)
        cov_ic = np.cov(ic_matrix.T) + np.eye(n) * 1e-6

        if method == "max_ic":
            # Maximize w^T * mean_ic subject to sum(w)=1, w>=0
            # Equivalent to minimizing -w^T * mean_ic
            def neg_ic(w):
                return -w @ mean_ic

            def neg_ic_grad(w):
                return -mean_ic

            result = minimize(
                neg_ic,
                x0=np.ones(n) / n,
                jac=neg_ic_grad,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                options={"ftol": 1e-9, "maxiter": 500},
            )
        else:
            # Maximize ICIR = w^T * mean_ic / sqrt(w^T * cov * w)
            # Minimize negative ICIR
            def neg_icir(w):
                portfolio_ic = w @ mean_ic
                portfolio_var = w @ cov_ic @ w
                return -portfolio_ic / (np.sqrt(portfolio_var) + 1e-9)

            result = minimize(
                neg_icir,
                x0=np.ones(n) / n,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                options={"ftol": 1e-9, "maxiter": 500},
            )

        if result.success:
            weights = np.clip(result.x, 0.0, 1.0)
            weights /= weights.sum()
        else:
            logger.warning("Weight optimization did not converge; falling back to equal weight")
            weights = np.ones(n) / n

        return dict(zip(names, weights.tolist()))


# ---------------------------------------------------------------------------
# FactorNeutralizer -- removes unwanted exposures from factor scores
# ---------------------------------------------------------------------------

class FactorNeutralizer:
    """
    Removes market, sector, and style biases from factor exposures.
    Also provides winsorization and cross-sectional standardization.
    """

    def neutralize_market(
        self,
        factor: pd.DataFrame,
        market_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Remove market beta from each factor row via cross-sectional regression.

        For each date, regress factor[t] on market_return[t] (scalar) and
        take the residuals. Since market return is a scalar on a given day,
        this removes the average market-wide component.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers).
        market_returns : pd.Series
            Daily market returns indexed by date.

        Returns
        -------
        pd.DataFrame
            Market-neutralized factor scores.
        """
        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < 3:
                continue
            mkt = market_returns.loc[date] if date in market_returns.index else 0.0
            # OLS: factor_i = alpha + beta * mkt + eps_i
            # With scalar mkt, residual = factor_i - mean(factor_i)
            # But proper treatment uses cross-sectional OLS with market as regressor
            X = np.column_stack([np.ones(len(row)), np.full(len(row), mkt)])
            y = row.values
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
                result.loc[date, row.index] = residuals
            except np.linalg.LinAlgError:
                pass
        return result

    def neutralize_sector(
        self,
        factor: pd.DataFrame,
        sector_dummies: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove sector effects from factor via cross-sectional OLS.

        For each date, regress factor[t, :] on sector dummies and return residuals.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers).
        sector_dummies : pd.DataFrame
            Binary sector membership matrix (tickers x sectors).

        Returns
        -------
        pd.DataFrame
            Sector-neutralized factor scores.
        """
        result = factor.copy()
        sector_matrix = sector_dummies.values.astype(float)

        for date in factor.index:
            row = factor.loc[date]
            valid_mask = row.notna()
            valid_tickers = row.index[valid_mask]
            if len(valid_tickers) < sector_dummies.shape[1] + 2:
                continue

            # Align tickers
            common = [t for t in valid_tickers if t in sector_dummies.index]
            if len(common) < sector_dummies.shape[1] + 2:
                continue

            y = row.loc[common].values
            X = sector_dummies.loc[common].values.astype(float)
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
                result.loc[date, common] = residuals
            except np.linalg.LinAlgError:
                pass

        return result

    def winsorize(
        self,
        factor: pd.DataFrame,
        clip_pct: float = 0.01,
    ) -> pd.DataFrame:
        """
        Winsorize factor cross-sectionally at clip_pct and 1-clip_pct quantiles.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers).
        clip_pct : float
            Fraction to clip at each tail. Default 0.01 (1%).

        Returns
        -------
        pd.DataFrame
            Winsorized factor scores.
        """
        if not (0.0 < clip_pct < 0.5):
            raise ValueError("clip_pct must be in (0, 0.5)")

        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < 4:
                continue
            # Use 'lower'/'higher' so thresholds are actual data points that
            # exclude the top/bottom tail values from the clip boundaries.
            lo = float(np.percentile(row.values, clip_pct * 100.0, method="higher"))
            hi = float(np.percentile(row.values, (1.0 - clip_pct) * 100.0, method="lower"))
            result.loc[date] = factor.loc[date].clip(lower=lo, upper=hi)
        return result

    def standardize(
        self,
        factor: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cross-sectional z-score: for each date subtract mean and divide by std.

        Parameters
        ----------
        factor : pd.DataFrame
            Factor scores (dates x tickers).

        Returns
        -------
        pd.DataFrame
            Z-scored factor scores (mean=0, std=1 cross-sectionally per date).
        """
        mean = factor.mean(axis=1)
        std = factor.std(axis=1).clip(lower=1e-9)
        result = factor.sub(mean, axis=0).div(std, axis=0)
        return result

    def full_neutralize(
        self,
        factor: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        sector_dummies: Optional[pd.DataFrame] = None,
        clip_pct: float = 0.01,
    ) -> pd.DataFrame:
        """
        Convenience method: winsorize -> sector neutralize -> market neutralize -> standardize.
        """
        f = self.winsorize(factor, clip_pct=clip_pct)
        if sector_dummies is not None:
            f = self.neutralize_sector(f, sector_dummies)
        if market_returns is not None:
            f = self.neutralize_market(f, market_returns)
        f = self.standardize(f)
        return f
