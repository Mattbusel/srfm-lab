"""
ml_alpha/pipeline.py — End-to-end ML alpha pipeline.

Walk-forward cross-validation with purged splits.
IC computation, ICIR, factor return analysis, decile analysis.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class MLResult:
    """Container for ML pipeline results."""
    ic_series: pd.Series = field(default_factory=pd.Series)
    icir: float = 0.0
    factor_returns: pd.Series = field(default_factory=pd.Series)
    decile_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.Series = field(default_factory=pd.Series)
    feature_importance: pd.Series = field(default_factory=pd.Series)
    train_ic_per_fold: List[float] = field(default_factory=list)
    val_ic_per_fold: List[float] = field(default_factory=list)
    n_splits: int = 0
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"ICIR={self.icir:.3f} IC_mean={self.ic_series.mean():.3f} "
                f"IC_std={self.ic_series.std():.3f} "
                f"IC_positive_rate={float((self.ic_series > 0).mean()):.2%} "
                f"n_splits={self.n_splits}")


def _spearman_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation IC."""
    from scipy.stats import spearmanr
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if valid.sum() < 5:
        return 0.0
    ic, _ = spearmanr(y_pred[valid], y_true[valid])
    return float(ic)


def _pearson_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pearson correlation IC."""
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if valid.sum() < 5:
        return 0.0
    corr = np.corrcoef(y_pred[valid], y_true[valid])[0, 1]
    return float(corr)


# ─────────────────────────────────────────────────────────────────────────────
# Purged K-Fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Purged K-Fold cross-validator for time-series.

    Prevents information leakage:
    1. Ensures no overlap between train and test periods
    2. Adds an embargo period between train end and test start

    Parameters
    ----------
    n_splits       : number of folds (default 5)
    purge_pct      : fraction of training data to purge at the boundary (default 0.05)
    embargo_pct    : fraction of test data to embargo after training (default 0.01)
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray, y: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Returns list of (train_idx, test_idx) tuples.
        Training sets are all data before the test set, with purging.
        """
        n = len(X)
        # Compute fold boundaries
        fold_size = n // self.n_splits
        splits = []

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n

            # Embargo: bars at the end of training close to test
            embargo_size = max(1, int(fold_size * self.embargo_pct))
            purge_size = max(1, int(fold_size * self.purge_pct))

            # Training: everything before test_start, minus purge
            train_end = max(0, test_start - embargo_size)
            train_start = 0

            train_idx = np.arange(train_start, max(0, train_end - purge_size))
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 10 and len(test_idx) > 5:
                splits.append((train_idx, test_idx))

        return splits

    def combinatorial_purged_splits(
        self,
        n: int,
        n_test_splits: int = 2,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorial purged cross-validation.
        Tests multiple combinations of folds for more robust evaluation.
        """
        fold_size = n // (self.n_splits + n_test_splits)
        splits = []
        embargo = max(1, int(fold_size * self.embargo_pct))

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + n_test_splits) * fold_size

            train_idx = np.concatenate([
                np.arange(0, max(0, test_start - embargo)),
                np.arange(min(n, test_end + embargo), n),
            ])
            test_idx = np.arange(test_start, min(n, test_end))

            if len(train_idx) > 20 and len(test_idx) > 5:
                splits.append((train_idx, test_idx))

        return splits


# ─────────────────────────────────────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MLPipeline:
    """
    End-to-end ML pipeline for alpha generation.

    Steps:
    1. Feature engineering (via FeatureEngine)
    2. Target computation (forward returns)
    3. Purged K-Fold walk-forward cross-validation
    4. Model training and prediction
    5. Performance evaluation (IC, ICIR, factor returns, decile analysis)

    Parameters
    ----------
    model_class     : model class to use (default: LinearAlpha)
    model_kwargs    : kwargs for model constructor
    n_splits        : CV splits (default 5)
    target_horizon  : forward return horizon (default 5 bars)
    purge_pct       : purging fraction (default 0.05)
    winsorize_pct   : feature winsorization percentile (default 0.01)
    """

    def __init__(
        self,
        model_class=None,
        model_kwargs: Optional[Dict] = None,
        n_splits: int = 5,
        target_horizon: int = 5,
        purge_pct: float = 0.05,
        winsorize_pct: float = 0.01,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.n_splits = n_splits
        self.target_horizon = target_horizon
        self.purge_pct = purge_pct
        self.winsorize_pct = winsorize_pct
        self._models: List = []
        self._feature_names: List[str] = []

    def compute_target(
        self,
        prices: pd.Series,
        horizon: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute forward return target.

        Parameters
        ----------
        prices  : close price series
        horizon : forward horizon (default: self.target_horizon)

        Returns
        -------
        pd.Series of forward returns.
        """
        h = horizon or self.target_horizon
        fwd_return = prices.shift(-h) / prices - 1
        return fwd_return

    def run(
        self,
        prices_df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None,
        target_horizon: Optional[int] = None,
        n_splits: Optional[int] = None,
    ) -> MLResult:
        """
        Full walk-forward ML pipeline.

        Parameters
        ----------
        prices_df     : OHLCV DataFrame with at minimum 'close' column
        feature_df    : pre-computed feature DataFrame (if None, builds from prices_df)
        target_horizon: forward return horizon
        n_splits      : number of cross-validation splits

        Returns
        -------
        MLResult with IC series, predictions, factor returns, etc.
        """
        from strategies.ml_alpha.features import FeatureEngine
        from strategies.ml_alpha.models import LinearAlpha

        horizon = target_horizon or self.target_horizon
        k = n_splits or self.n_splits

        # Default model
        if self.model_class is None:
            model_cls = LinearAlpha
            model_kw = {"method": "ridge", "alpha": 0.1}
        else:
            model_cls = self.model_class
            model_kw = self.model_kwargs

        # Compute features
        if feature_df is None:
            fe = FeatureEngine()
            feature_df = fe.build_all_features(prices_df, include_bh=False)

        # Compute target
        target = self.compute_target(prices_df["close"], horizon)

        # Align
        common_idx = feature_df.index.intersection(target.index)
        X_df = feature_df.loc[common_idx]
        y = target.loc[common_idx]

        # Winsorize features
        if self.winsorize_pct > 0:
            q_low = X_df.quantile(self.winsorize_pct)
            q_high = X_df.quantile(1 - self.winsorize_pct)
            X_df = X_df.clip(lower=q_low, upper=q_high, axis=1)

        X = X_df.fillna(0).values.astype(np.float32)
        y_arr = y.fillna(np.nan).values.astype(np.float32)

        self._feature_names = list(X_df.columns)

        # Walk-forward cross-validation
        cv = PurgedKFold(n_splits=k, purge_pct=self.purge_pct)
        splits = cv.split(X, y_arr)

        all_predictions = np.full(len(X), np.nan)
        ic_by_fold = []
        train_ic_by_fold = []
        feature_importances = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X[train_idx]; y_train = y_arr[train_idx]
            X_test = X[test_idx]; y_test = y_arr[test_idx]

            # Filter valid training samples
            valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            if valid_train.sum() < 20:
                continue

            model = model_cls(**model_kw)
            try:
                model.train(X_train[valid_train], y_train[valid_train],
                            feature_names=self._feature_names)
            except TypeError:
                model.train(X_train[valid_train], y_train[valid_train])

            preds = model.predict(X_test)
            all_predictions[test_idx] = preds

            # IC for this fold
            valid_test = ~(np.isnan(preds) | np.isnan(y_test))
            if valid_test.sum() > 5:
                ic = _spearman_ic(preds[valid_test], y_test[valid_test])
                ic_by_fold.append(ic)

            # Train IC
            train_ic = getattr(model, "_train_ic", 0.0)
            train_ic_by_fold.append(train_ic)

            # Feature importance
            try:
                fi = model.feature_importance()
                feature_importances.append(fi)
            except (AttributeError, RuntimeError):
                pass

            self._models.append(model)

        # Build prediction series
        pred_series = pd.Series(all_predictions, index=common_idx)

        # Rolling IC (monthly)
        ic_series = self.compute_ic(pred_series, y)
        icir = self.compute_icir(ic_series)

        # Factor returns
        factor_returns = self.compute_factor_returns(pred_series, prices_df["close"].pct_change())

        # Decile analysis
        decile_df = self.decile_analysis(pred_series, y)

        # Average feature importance
        if feature_importances:
            avg_fi = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        else:
            avg_fi = pd.Series(dtype=float)

        return MLResult(
            ic_series=ic_series,
            icir=icir,
            factor_returns=factor_returns,
            decile_returns=decile_df,
            predictions=pred_series,
            feature_importance=avg_fi,
            train_ic_per_fold=train_ic_by_fold,
            val_ic_per_fold=ic_by_fold,
            n_splits=len(splits),
            params={
                "horizon": horizon,
                "n_splits": k,
                "model": model_cls.__name__,
            },
        )

    def compute_ic(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """
        Compute rolling Information Coefficient (IC).

        IC = Spearman rank correlation between predictions and realized returns.
        Computed monthly (window=21 bars).

        Parameters
        ----------
        predictions : predicted return/signal series
        returns     : realized return series
        window      : rolling window for IC computation

        Returns
        -------
        pd.Series of IC values.
        """
        common = predictions.index.intersection(returns.index)
        pred = predictions.reindex(common)
        ret = returns.reindex(common)

        ic_series = pd.Series(np.nan, index=common)

        for i in range(window, len(common)):
            p_win = pred.iloc[i - window:i].values
            r_win = ret.iloc[i - window:i].values
            valid = ~(np.isnan(p_win) | np.isnan(r_win))
            if valid.sum() < 5:
                continue
            ic_series.iloc[i] = _spearman_ic(p_win[valid], r_win[valid])

        return ic_series.dropna()

    def compute_icir(self, ic_series: pd.Series) -> float:
        """
        Compute IC Information Ratio (ICIR).

        ICIR = mean(IC) / std(IC) — measures consistency of predictive power.

        Parameters
        ----------
        ic_series : series of IC values per period

        Returns
        -------
        ICIR as float (annualized by * sqrt(12) for monthly IC).
        """
        clean_ic = ic_series.dropna()
        if len(clean_ic) < 3:
            return 0.0
        return float(clean_ic.mean() / (clean_ic.std() + 1e-9))

    def compute_factor_returns(
        self,
        predictions: pd.Series,
        universe_returns: pd.Series,
        n_long: int = None,
        n_short: int = None,
    ) -> pd.Series:
        """
        Compute factor portfolio returns from signal predictions.

        Goes long the top-scored assets and short the bottom.
        For single-asset: uses the signal direction as weight.

        Parameters
        ----------
        predictions      : signal scores
        universe_returns : realized returns
        n_long           : number of long positions (default: top 20%)
        n_short          : number of short positions (default: bottom 20%)

        Returns
        -------
        pd.Series of factor portfolio daily returns.
        """
        common = predictions.index.intersection(universe_returns.index)
        pred = predictions.reindex(common)
        ret = universe_returns.reindex(common)

        # For single asset: use sign of prediction as position
        # Scale by magnitude for stronger predictions
        def compute_position(p: float, threshold: float = 0.0) -> float:
            if np.isnan(p):
                return 0.0
            return float(np.sign(p)) if abs(p) > threshold else 0.0

        positions = pred.apply(compute_position).shift(1).fillna(0)
        factor_rets = positions * ret
        return factor_rets

    def decile_analysis(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Decile (quantile) analysis of alpha signal.

        Sort predictions into n_bins groups. Show mean return per group.
        A good factor should show monotonic relationship between
        rank and forward return.

        Parameters
        ----------
        predictions : signal scores
        returns     : forward returns
        n_bins      : number of quantile bins (default 10)

        Returns
        -------
        pd.DataFrame with columns: mean_return, std_return, n_obs, sharpe
        """
        common = predictions.index.intersection(returns.index)
        pred = predictions.reindex(common).dropna()
        ret = returns.reindex(common).dropna()

        common2 = pred.index.intersection(ret.index)
        pred = pred.reindex(common2)
        ret = ret.reindex(common2)

        valid = ~(pred.isna() | ret.isna())
        if valid.sum() < n_bins * 5:
            return pd.DataFrame()

        pred_clean = pred[valid]
        ret_clean = ret[valid]

        # Assign deciles based on prediction rank
        try:
            deciles = pd.qcut(pred_clean, q=n_bins, labels=False, duplicates="drop")
        except Exception:
            return pd.DataFrame()

        rows = []
        for d in range(n_bins):
            mask = deciles == d
            if mask.sum() < 3:
                continue
            r = ret_clean[mask]
            rows.append({
                "decile": d + 1,
                "n_obs": len(r),
                "mean_return": float(r.mean()),
                "std_return": float(r.std()),
                "sharpe": float(r.mean() / (r.std() + 1e-9) * math.sqrt(252)),
                "hit_rate": float((r > 0).mean()),
            })

        if rows:
            df = pd.DataFrame(rows).set_index("decile")
            df["spread"] = df["mean_return"] - df["mean_return"].mean()
            return df
        return pd.DataFrame()

    def rolling_prediction(
        self,
        prices_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        retrain_freq: int = 63,
        min_train_size: int = 252,
    ) -> pd.Series:
        """
        Generate predictions using a rolling expanding window.

        Retrains the model every retrain_freq bars.

        Parameters
        ----------
        prices_df       : OHLCV data
        feature_df      : pre-computed features
        retrain_freq    : retrain every N bars
        min_train_size  : minimum training window

        Returns
        -------
        pd.Series of out-of-sample predictions.
        """
        from strategies.ml_alpha.models import LinearAlpha

        model_cls = self.model_class or LinearAlpha
        model_kw = self.model_kwargs or {"method": "ridge", "alpha": 0.1}

        target = self.compute_target(prices_df["close"])
        X_df = feature_df.reindex(prices_df.index).fillna(0)
        y = target.reindex(prices_df.index)

        n = len(prices_df)
        all_preds = pd.Series(np.nan, index=prices_df.index)
        model = None

        for i in range(min_train_size, n):
            if (i - min_train_size) % retrain_freq == 0:
                # Retrain
                X_train = X_df.iloc[:i].values.astype(np.float32)
                y_train = y.iloc[:i].values.astype(np.float32)
                valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
                if valid.sum() < 50:
                    continue
                model = model_cls(**model_kw)
                try:
                    model.train(X_train[valid], y_train[valid], feature_names=list(X_df.columns))
                except TypeError:
                    model.train(X_train[valid], y_train[valid])

            if model is not None:
                x = X_df.iloc[i:i+1].values.astype(np.float32)
                if not np.isnan(x).any():
                    p = model.predict(x)
                    all_preds.iloc[i] = float(p[0])

        return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

    rng = np.random.default_rng(42)
    n = 1500
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1000, 100000, n).astype(float)
    }, index=idx)

    # Build features
    from strategies.ml_alpha.features import FeatureEngine
    fe = FeatureEngine()
    features = fe.build_all_features(df, include_bh=False)
    print(f"Features shape: {features.shape}")

    # Run pipeline
    pipeline = MLPipeline(n_splits=5, target_horizon=5)
    result = pipeline.run(df, feature_df=features)
    print("\nML Pipeline Result:")
    print(result.summary())
    print(f"\nIC Stats:")
    print(f"  Mean IC: {result.ic_series.mean():.4f}")
    print(f"  ICIR: {result.icir:.3f}")
    print(f"  IC > 0: {(result.ic_series > 0).mean():.2%}")
    if len(result.decile_returns) > 0:
        print(f"\nDecile Analysis:")
        print(result.decile_returns.to_string())
    if len(result.feature_importance) > 0:
        print(f"\nTop 10 Features:")
        print(result.feature_importance.head(10).to_string())
