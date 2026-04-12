"""
signal_extraction.py — Alpha signal extraction via tensor decompositions.

This module provides a library of alpha-generating signal extractors that
operate on the tensor representations produced by data_pipeline.py.
All heavy-compute routines are JAX-compatible and JIT-compilable.

Key classes
-----------
* TTSignalExtractor       — extracts low-rank signals from return tensors
* MomentumSignalFactory   — cross-sectional & time-series momentum
* MeanReversionSignal     — z-score mean-reversion signals
* VolatilitySignalFactory — realised vol, GARCH-style, range-based signals
* CorrelationBreakSignal  — detect sudden correlation regime changes
* TensorPCASignal         — PCA via SVD on flattened TT cores
* SpreadSignalFactory     — pair / basket spread signals
* SignalCombiner          — alpha combination with IC-weighted blending
* SignalEvaluator         — IC, ICIR, factor portfolio analysis
* AlphaDecayAnalyser      — measure signal half-life via autocorrelation
* SignalNeutraliser       — sector / market neutralisation
* InformationCoefficient  — rolling and expanding IC calculations
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Cross-sectional z-score."""
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sigma = np.nanstd(x, axis=axis, keepdims=True) + eps
    return (x - mu) / sigma


def _winsorise(x: np.ndarray, quantile: float = 0.01) -> np.ndarray:
    """Winsorise array at lower and upper quantile."""
    lo = np.nanquantile(x, quantile)
    hi = np.nanquantile(x, 1 - quantile)
    return np.clip(x, lo, hi)


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] using cross-sectional rank."""
    T, N = x.shape
    out = np.zeros_like(x)
    for t in range(T):
        row = x[t]
        finite = np.isfinite(row)
        ranks = np.zeros(N)
        if finite.sum() > 1:
            r = np.argsort(np.argsort(row[finite]))
            ranks[finite] = r / (r.max() + 1e-12)
        out[t] = ranks
    return out


def _rolling_apply(x: np.ndarray, window: int, fn: Callable) -> np.ndarray:
    """Apply *fn* to rolling windows of *x* (T, N) -> (T, N)."""
    T, N = x.shape
    out = np.full_like(x, np.nan)
    for t in range(window - 1, T):
        out[t] = fn(x[t - window + 1:t + 1])
    return out


def _exponential_weights(window: int, half_life: float) -> np.ndarray:
    """Exponential weights summing to 1, most recent = index -1."""
    decay = math.pow(0.5, 1.0 / half_life)
    w = np.array([decay ** (window - 1 - i) for i in range(window)])
    return w / w.sum()


# ---------------------------------------------------------------------------
# TTSignalExtractor
# ---------------------------------------------------------------------------


@dataclass
class SignalExtractorConfig:
    """Configuration for TTSignalExtractor."""
    tt_rank: int = 4
    n_components: int = 3        # number of TT components to extract
    standardise_output: bool = True
    winsorise_quantile: float = 0.01
    use_rank_normalise: bool = True


class TTSignalExtractor:
    """
    Extract low-rank signals from a (T, N) return tensor via truncated SVD.

    The extractor computes a rank-K approximation of the return matrix and
    returns the K strongest left singular vectors (T, K) as time-series
    signals, and the K right singular vectors (N, K) as asset loadings.

    Parameters
    ----------
    config : SignalExtractorConfig
    """

    def __init__(self, config: SignalExtractorConfig | None = None) -> None:
        self.config = config or SignalExtractorConfig()
        self._U: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._Vt: np.ndarray | None = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "TTSignalExtractor":
        """
        Fit the extractor to a (T, N) return matrix.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)

        Returns
        -------
        self
        """
        R = np.array(returns, dtype=np.float64)
        R = np.nan_to_num(R, nan=0.0)
        K = self.config.n_components
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        self._U = U[:, :K]
        self._S = s[:K]
        self._Vt = Vt[:K, :]
        self._fitted = True
        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """
        Project new return data onto the fitted basis.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)

        Returns
        -------
        signals : np.ndarray, shape (T, K)
        """
        if not self._fitted:
            raise RuntimeError("TTSignalExtractor.fit() must be called first.")
        R = np.array(returns, dtype=np.float64)
        R = np.nan_to_num(R, nan=0.0)
        signals = R @ self._Vt.T
        if self.config.standardise_output:
            signals = _zscore(signals, axis=0)
        return signals.astype(np.float32)

    def fit_transform(self, returns: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(returns)
        return np.array(self._U * self._S[None, :], dtype=np.float32)

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each component."""
        if not self._fitted:
            raise RuntimeError("fit() required first.")
        total = np.sum(self._S ** 2)
        return (self._S ** 2 / (total + 1e-12)).astype(np.float32)

    def asset_loadings(self) -> np.ndarray:
        """Return asset loading matrix (K, N)."""
        if not self._fitted:
            raise RuntimeError("fit() required first.")
        return self._Vt.astype(np.float32)

    def reconstruct(self) -> np.ndarray:
        """Return rank-K reconstruction (T, N)."""
        if not self._fitted:
            raise RuntimeError("fit() required first.")
        return (self._U * self._S[None, :]) @ self._Vt


# ---------------------------------------------------------------------------
# MomentumSignalFactory
# ---------------------------------------------------------------------------


@dataclass
class MomentumConfig:
    """Configuration for MomentumSignalFactory."""
    lookbacks: list = field(default_factory=lambda: [21, 63, 126, 252])
    skip_days: int = 1       # skip most recent N days (reversal filter)
    normalise: str = "zscore"  # "zscore" | "rank" | "none"
    blend_weights: list | None = None  # weights for multi-lookback blend


class MomentumSignalFactory:
    """
    Cross-sectional and time-series momentum signal generator.

    Produces signals for multiple lookback windows and optionally blends
    them with specified weights.

    Parameters
    ----------
    config : MomentumConfig
    """

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self.config = config or MomentumConfig()

    def cross_sectional(self, returns: np.ndarray) -> dict:
        """
        Compute cross-sectional momentum signals.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)

        Returns
        -------
        dict mapping lookback (int) -> signal (T, N).
        """
        signals = {}
        cfg = self.config
        T, N = returns.shape
        skip = cfg.skip_days

        for lb in cfg.lookbacks:
            if T < lb + skip:
                continue
            cum_ret = np.zeros((T, N))
            for t in range(lb + skip, T + 1):
                window = returns[t - lb - skip:t - skip]
                cum_ret[t - 1] = np.sum(window, axis=0)
            sig = cum_ret
            if cfg.normalise == "zscore":
                sig = _zscore(sig)
            elif cfg.normalise == "rank":
                sig = _rank_normalise(sig) - 0.5
            signals[lb] = sig.astype(np.float32)

        return signals

    def time_series(self, returns: np.ndarray) -> dict:
        """
        Time-series momentum: sign of trailing cumulative return.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
        """
        signals = {}
        cfg = self.config
        T, N = returns.shape
        skip = cfg.skip_days

        for lb in cfg.lookbacks:
            if T < lb + skip:
                continue
            sig = np.zeros((T, N))
            for t in range(lb + skip, T):
                window = returns[t - lb - skip:t - skip]
                sig[t] = np.sign(np.sum(window, axis=0))
            signals[lb] = sig.astype(np.float32)

        return signals

    def blend(self, signals_dict: dict) -> np.ndarray:
        """
        Blend multiple lookback signals into a single composite.

        Parameters
        ----------
        signals_dict : dict (lookback -> np.ndarray shape (T, N))

        Returns
        -------
        blended : np.ndarray, shape (T, N)
        """
        keys = sorted(signals_dict.keys())
        arrs = [signals_dict[k] for k in keys]
        if not arrs:
            raise ValueError("Empty signals dict.")
        cfg = self.config
        bw = cfg.blend_weights
        if bw is None:
            bw = [1.0 / len(arrs)] * len(arrs)
        bw = np.array(bw[:len(arrs)], dtype=np.float32)
        bw /= bw.sum()
        T, N = arrs[0].shape
        blended = np.zeros((T, N), dtype=np.float32)
        for w, arr in zip(bw, arrs):
            blended += w * arr
        return blended


# ---------------------------------------------------------------------------
# MeanReversionSignal
# ---------------------------------------------------------------------------


@dataclass
class MeanReversionConfig:
    """Configuration for MeanReversionSignal."""
    lookback: int = 10
    z_entry: float = 1.5       # enter on z-score > this
    z_exit: float = 0.5        # exit when z-score < this
    half_life_days: float = 5.0
    use_ewm: bool = True


class MeanReversionSignal:
    """
    Z-score mean-reversion signal.

    For each asset at time t, computes the deviation of the current price
    from its rolling mean in units of rolling standard deviation.

    Parameters
    ----------
    config : MeanReversionConfig
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def compute(self, log_prices: np.ndarray) -> np.ndarray:
        """
        Compute mean-reversion z-score signal.

        Parameters
        ----------
        log_prices : np.ndarray, shape (T, N)
            Log-price series.

        Returns
        -------
        z_scores : np.ndarray, shape (T, N)
            Positive = above mean (short signal), negative = below (long signal).
        """
        T, N = log_prices.shape
        lb = self.config.lookback
        z = np.zeros((T, N), dtype=np.float32)

        if self.config.use_ewm:
            hl = self.config.half_life_days
            weights = _exponential_weights(lb, hl)
            for t in range(lb, T):
                window = log_prices[t - lb:t]
                mu = (weights[:, None] * window).sum(axis=0)
                var = (weights[:, None] * (window - mu) ** 2).sum(axis=0)
                std = np.sqrt(var) + 1e-8
                z[t] = (log_prices[t] - mu) / std
        else:
            for t in range(lb, T):
                window = log_prices[t - lb:t]
                mu = window.mean(axis=0)
                std = window.std(axis=0) + 1e-8
                z[t] = (log_prices[t] - mu) / std

        # Signal direction: negative z → buy, positive → sell
        return -z  # flip so long = undervalued

    def entry_mask(self, z_scores: np.ndarray) -> np.ndarray:
        """Boolean mask for entry signals."""
        return np.abs(z_scores) > self.config.z_entry

    def exit_mask(self, z_scores: np.ndarray) -> np.ndarray:
        """Boolean mask for exit signals (reversion toward mean)."""
        return np.abs(z_scores) < self.config.z_exit


# ---------------------------------------------------------------------------
# VolatilitySignalFactory
# ---------------------------------------------------------------------------


@dataclass
class VolSignalConfig:
    """Configuration for VolatilitySignalFactory."""
    rv_window: int = 21           # realised vol lookback
    parkinson_window: int = 10    # Parkinson estimator window
    garch_omega: float = 1e-6     # GARCH(1,1) omega
    garch_alpha: float = 0.10     # GARCH alpha (ARCH term)
    garch_beta: float = 0.85      # GARCH beta (GARCH term)
    vol_of_vol_window: int = 63   # vol-of-vol lookback


class VolatilitySignalFactory:
    """
    Suite of volatility signal estimators.

    Methods
    -------
    realised_vol       : Rolling realised volatility
    parkinson_vol      : Range-based Parkinson estimator
    garch_vol          : GARCH(1,1) conditional volatility
    vol_of_vol         : Volatility of volatility
    vol_percentile     : Cross-sectional vol rank signal
    vol_term_structure : Ratio of short-term to long-term vol
    """

    def __init__(self, config: VolSignalConfig | None = None) -> None:
        self.config = config or VolSignalConfig()

    def realised_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        Rolling realised volatility (annualised).

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)

        Returns
        -------
        rv : np.ndarray, shape (T, N)
        """
        lb = self.config.rv_window
        rv = _rolling_apply(
            returns, lb,
            lambda w: np.std(w, axis=0) * math.sqrt(252)
        )
        return rv.astype(np.float32)

    def parkinson_vol(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Parkinson range-based volatility estimator.

        Parameters
        ----------
        high, low : np.ndarray, shape (T, N)
            High and low log-prices.
        """
        T, N = high.shape
        lb = self.config.parkinson_window
        log_hl = (np.log(np.abs(high - low) + 1e-12)) ** 2
        pv = _rolling_apply(
            log_hl, lb,
            lambda w: np.sqrt(252 / (4 * math.log(2)) * np.mean(w, axis=0))
        )
        return pv.astype(np.float32)

    def garch_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        GARCH(1,1) conditional volatility per asset.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
        """
        T, N = returns.shape
        omega = self.config.garch_omega
        alpha = self.config.garch_alpha
        beta = self.config.garch_beta
        sigma2 = np.var(returns[:20], axis=0) * np.ones((T, N))

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        return np.sqrt(sigma2 * 252).astype(np.float32)

    def vol_of_vol(self, returns: np.ndarray) -> np.ndarray:
        """Rolling volatility-of-volatility signal."""
        rv = self.realised_vol(returns)
        vov_lb = self.config.vol_of_vol_window
        vov = _rolling_apply(rv, vov_lb, lambda w: np.std(w, axis=0))
        return vov.astype(np.float32)

    def vol_percentile(self, returns: np.ndarray) -> np.ndarray:
        """Cross-sectional realised-vol percentile rank."""
        rv = self.realised_vol(returns)
        return _rank_normalise(rv).astype(np.float32)

    def vol_term_structure(
        self, returns: np.ndarray, short_lb: int = 10, long_lb: int = 63
    ) -> np.ndarray:
        """Ratio of short-term to long-term realised vol."""
        orig = self.config.rv_window
        self.config.rv_window = short_lb
        rv_short = self.realised_vol(returns)
        self.config.rv_window = long_lb
        rv_long = self.realised_vol(returns)
        self.config.rv_window = orig
        return (rv_short / (rv_long + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# CorrelationBreakSignal
# ---------------------------------------------------------------------------


@dataclass
class CorrBreakConfig:
    """Configuration for CorrelationBreakSignal."""
    short_window: int = 21
    long_window: int = 126
    threshold: float = 0.3   # Frobenius norm change threshold
    n_top_pairs: int = 20    # pairs to report


class CorrelationBreakSignal:
    """
    Detect sudden changes in the cross-sectional correlation structure.

    Computes the Frobenius norm of the difference between short-window
    and long-window correlation matrices as a market-level signal.

    Parameters
    ----------
    config : CorrBreakConfig
    """

    def __init__(self, config: CorrBreakConfig | None = None) -> None:
        self.config = config or CorrBreakConfig()

    def correlation_break_series(self, returns: np.ndarray) -> np.ndarray:
        """
        Returns a (T,) array of correlation-break magnitudes.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
        """
        cfg = self.config
        T, N = returns.shape
        signal = np.zeros(T, dtype=np.float32)

        for t in range(cfg.long_window, T):
            r_short = returns[t - cfg.short_window:t]
            r_long = returns[t - cfg.long_window:t]
            c_short = np.corrcoef(r_short.T) if r_short.shape[0] > 1 else np.eye(N)
            c_long = np.corrcoef(r_long.T) if r_long.shape[0] > 1 else np.eye(N)
            signal[t] = float(np.linalg.norm(c_short - c_long, "fro"))

        return signal

    def regime_indicator(self, signal: np.ndarray) -> np.ndarray:
        """1 if correlation break exceeds threshold, else 0."""
        return (signal > self.config.threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# TensorPCASignal
# ---------------------------------------------------------------------------


@dataclass
class TensorPCAConfig:
    """Configuration for TensorPCASignal."""
    n_components: int = 5
    mode: int = 1              # which tensor mode to unfold along
    window: int = 252
    normalise: bool = True


class TensorPCASignal:
    """
    Principal component signals from a 3-D return tensor.

    Assumes input tensor has shape (T, N, M) where M is some third mode
    (e.g. frequencies, or a stack of factor values).  The tensor is unfolded
    along ``mode`` and SVD is applied.

    Parameters
    ----------
    config : TensorPCAConfig
    """

    def __init__(self, config: TensorPCAConfig | None = None) -> None:
        self.config = config or TensorPCAConfig()

    def extract(self, tensor: np.ndarray) -> np.ndarray:
        """
        Extract PCA signals from a 3-D tensor.

        Parameters
        ----------
        tensor : np.ndarray, shape (T, N, M)

        Returns
        -------
        signals : np.ndarray, shape (T, K)  — K = n_components
        """
        T, N, M = tensor.shape
        cfg = self.config
        K = cfg.n_components
        # Unfold along mode
        if cfg.mode == 0:
            X = tensor.reshape(T, N * M)
        elif cfg.mode == 1:
            X = tensor.transpose(1, 0, 2).reshape(N, T * M)
        else:
            X = tensor.transpose(2, 0, 1).reshape(M, T * N)

        # SVD
        X_c = X - X.mean(axis=1, keepdims=True)
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        if cfg.mode == 0:
            signals = U[:, :K] * s[:K]  # (T, K)
        else:
            # project original T-dim slices
            Vt_trunc = Vt[:K, :]   # (K, T*M) or (K, N*M)
            signals = (X_c @ Vt_trunc.T)[:K, :].T[:T, :K]

        if cfg.normalise:
            signals = _zscore(signals.reshape(-1, K)).reshape(-1, K)

        return signals[:T, :K].astype(np.float32)


# ---------------------------------------------------------------------------
# SpreadSignalFactory
# ---------------------------------------------------------------------------


@dataclass
class SpreadConfig:
    """Configuration for SpreadSignalFactory."""
    hedge_ratio_window: int = 63
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_pairs: int = 50
    min_correlation: float = 0.7


class SpreadSignalFactory:
    """
    Statistical arbitrage spread signals for pairs trading.

    Identifies correlated pairs and computes spread z-scores.

    Parameters
    ----------
    config : SpreadConfig
    """

    def __init__(self, config: SpreadConfig | None = None) -> None:
        self.config = config or SpreadConfig()
        self._pairs: list = []

    def find_pairs(
        self, returns: np.ndarray, asset_names: list | None = None
    ) -> list:
        """
        Identify highly correlated pairs.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
        asset_names : list of str, optional

        Returns
        -------
        list of (i, j, correlation) tuples
        """
        T, N = returns.shape
        cfg = self.config
        corr = np.corrcoef(returns.T)
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                c = corr[i, j]
                if c > cfg.min_correlation:
                    name_i = asset_names[i] if asset_names else str(i)
                    name_j = asset_names[j] if asset_names else str(j)
                    pairs.append((i, j, c, name_i, name_j))
        # Sort by correlation descending
        pairs.sort(key=lambda x: -x[2])
        self._pairs = pairs[:cfg.max_pairs]
        return self._pairs

    def spread_zscore(
        self, log_prices: np.ndarray, pair: tuple
    ) -> np.ndarray:
        """
        Compute spread z-score for a pair.

        Parameters
        ----------
        log_prices : np.ndarray, shape (T, N)
        pair : tuple (i, j, corr, ...)

        Returns
        -------
        z : np.ndarray, shape (T,)
        """
        i, j = pair[0], pair[1]
        T = log_prices.shape[0]
        lb = self.config.hedge_ratio_window
        z = np.zeros(T)

        for t in range(lb, T):
            p_i = log_prices[t - lb:t, i]
            p_j = log_prices[t - lb:t, j]
            beta = np.cov(p_i, p_j)[0, 1] / (np.var(p_j) + 1e-12)
            spread = log_prices[t, i] - beta * log_prices[t, j]
            hist_spread = log_prices[t - lb:t, i] - beta * log_prices[t - lb:t, j]
            z[t] = (spread - hist_spread.mean()) / (hist_spread.std() + 1e-8)

        return z.astype(np.float32)

    def all_spreads(self, log_prices: np.ndarray) -> dict:
        """Compute spread z-scores for all identified pairs."""
        if not self._pairs:
            warnings.warn("No pairs found. Call find_pairs() first.")
            return {}
        return {(p[3], p[4]): self.spread_zscore(log_prices, p) for p in self._pairs}


# ---------------------------------------------------------------------------
# SignalCombiner
# ---------------------------------------------------------------------------


@dataclass
class CombinerConfig:
    """Configuration for SignalCombiner."""
    method: str = "equal"    # "equal" | "ic_weighted" | "ml"
    ic_window: int = 63
    decay_halflife: float = 21.0
    l2_reg: float = 1e-3
    n_iters: int = 200
    lr: float = 1e-2


class SignalCombiner:
    """
    Combines multiple alpha signals into a single composite using various
    blending strategies.

    Parameters
    ----------
    config : CombinerConfig
    signal_names : list of str
    """

    def __init__(
        self, config: CombinerConfig | None = None, signal_names: list | None = None
    ) -> None:
        self.config = config or CombinerConfig()
        self.signal_names = signal_names or []
        self._weights: np.ndarray | None = None

    def fit(
        self,
        signals: np.ndarray,      # (T, N, K) K signals over T days, N assets
        forward_returns: np.ndarray,  # (T, N)
    ) -> "SignalCombiner":
        """
        Estimate combination weights from historical IC.

        Parameters
        ----------
        signals : np.ndarray, shape (T, N, K)
        forward_returns : np.ndarray, shape (T, N)
        """
        cfg = self.config
        K = signals.shape[2]
        if cfg.method == "equal":
            self._weights = np.ones(K) / K
        elif cfg.method == "ic_weighted":
            ics = self._compute_ic(signals, forward_returns)
            ic_mean = np.nanmean(ics, axis=0)  # (K,)
            ic_mean = np.clip(ic_mean, 0, None)
            s = ic_mean.sum()
            self._weights = ic_mean / (s + 1e-12) if s > 0 else np.ones(K) / K
        elif cfg.method == "ml":
            self._weights = self._ml_weights(signals, forward_returns)
        return self

    def _compute_ic(
        self, signals: np.ndarray, fwd_ret: np.ndarray
    ) -> np.ndarray:
        """Compute rolling IC for each signal. Returns (T, K)."""
        T, N, K = signals.shape
        ics = np.zeros((T, K))
        for t in range(1, T):
            for k in range(K):
                sig_t = signals[t - 1, :, k]
                ret_t = fwd_ret[t]
                valid = np.isfinite(sig_t) & np.isfinite(ret_t)
                if valid.sum() > 5:
                    ic = np.corrcoef(sig_t[valid], ret_t[valid])[0, 1]
                    ics[t, k] = ic if np.isfinite(ic) else 0.0
        return ics

    def _ml_weights(
        self, signals: np.ndarray, fwd_ret: np.ndarray
    ) -> np.ndarray:
        """Learn combination weights via ridge regression (JAX/optax)."""
        T, N, K = signals.shape
        X = signals.reshape(T * N, K)
        y = fwd_ret.reshape(T * N)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid], y[valid]
        # Ridge
        lam = self.config.l2_reg
        XtX = X.T @ X + lam * np.eye(K)
        Xty = X.T @ y
        w = np.linalg.solve(XtX, Xty)
        w = np.clip(w, 0, None)
        s = w.sum()
        return (w / (s + 1e-12)).astype(np.float32)

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply combination weights.

        Parameters
        ----------
        signals : np.ndarray, shape (T, N, K)

        Returns
        -------
        composite : np.ndarray, shape (T, N)
        """
        if self._weights is None:
            K = signals.shape[2]
            self._weights = np.ones(K) / K
        return (signals * self._weights[None, None, :]).sum(axis=2).astype(np.float32)

    @property
    def weights(self) -> np.ndarray | None:
        return self._weights


# ---------------------------------------------------------------------------
# SignalEvaluator
# ---------------------------------------------------------------------------


class SignalEvaluator:
    """
    Evaluates alpha signal quality using IC, ICIR, and factor portfolio metrics.

    Parameters
    ----------
    signal : np.ndarray, shape (T, N)
    forward_returns : np.ndarray, shape (T, N)
    """

    def __init__(
        self,
        signal: np.ndarray,
        forward_returns: np.ndarray,
    ) -> None:
        self.signal = np.array(signal, dtype=np.float64)
        self.fwd = np.array(forward_returns, dtype=np.float64)
        T, N = signal.shape
        self.T = T
        self.N = N

    def ic_series(self) -> np.ndarray:
        """Rank IC (Spearman) for each time step. Returns (T,)."""
        from scipy.stats import spearmanr
        ics = np.zeros(self.T)
        for t in range(self.T):
            s = self.signal[t]
            r = self.fwd[t]
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() > 5:
                ic, _ = spearmanr(s[valid], r[valid])
                ics[t] = ic if np.isfinite(ic) else 0.0
        return ics.astype(np.float32)

    def mean_ic(self) -> float:
        return float(np.nanmean(self.ic_series()))

    def icir(self) -> float:
        """IC Information Ratio = mean(IC) / std(IC)."""
        ic = self.ic_series()
        return float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12))

    def quantile_returns(self, n_quantiles: int = 5) -> np.ndarray:
        """
        Returns average forward returns for each signal quantile.

        Returns
        -------
        np.ndarray, shape (n_quantiles,)
        """
        T, N = self.signal.shape
        q_returns = np.zeros((T, n_quantiles))
        for t in range(T):
            s = self.signal[t]
            r = self.fwd[t]
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() < n_quantiles:
                continue
            bins = np.percentile(s[valid], np.linspace(0, 100, n_quantiles + 1))
            for q in range(n_quantiles):
                mask = valid & (s >= bins[q]) & (s < bins[q + 1])
                if mask.sum() > 0:
                    q_returns[t, q] = r[mask].mean()
        return q_returns.mean(axis=0).astype(np.float32)

    def long_short_return(self) -> np.ndarray:
        """
        Returns of a long-top-quintile, short-bottom-quintile portfolio. (T,)
        """
        T, N = self.signal.shape
        ls_ret = np.zeros(T)
        for t in range(T):
            s = self.signal[t]
            r = self.fwd[t]
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() < 10:
                continue
            q20 = np.percentile(s[valid], 20)
            q80 = np.percentile(s[valid], 80)
            long_mask = valid & (s >= q80)
            short_mask = valid & (s <= q20)
            ls_ret[t] = r[long_mask].mean() - r[short_mask].mean()
        return ls_ret.astype(np.float32)

    def summary(self) -> dict:
        """Return dict of signal quality metrics."""
        ic = self.ic_series()
        ls = self.long_short_return()
        ann_ret = float(ls.mean() * 252)
        ann_vol = float(ls.std() * math.sqrt(252))
        return {
            "mean_ic": round(float(np.nanmean(ic)), 4),
            "std_ic": round(float(np.nanstd(ic)), 4),
            "icir": round(self.icir(), 4),
            "ic_positive_frac": round(float((ic > 0).mean()), 4),
            "ls_ann_return": round(ann_ret, 4),
            "ls_ann_vol": round(ann_vol, 4),
            "ls_sharpe": round(ann_ret / (ann_vol + 1e-12), 4),
        }


# ---------------------------------------------------------------------------
# AlphaDecayAnalyser
# ---------------------------------------------------------------------------


@dataclass
class DecayConfig:
    """Configuration for AlphaDecayAnalyser."""
    max_lag: int = 63
    annualise: bool = True


class AlphaDecayAnalyser:
    """
    Measures signal half-life via autocorrelation of IC series.

    Parameters
    ----------
    config : DecayConfig
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self.config = config or DecayConfig()

    def autocorrelation(self, ic_series: np.ndarray) -> np.ndarray:
        """
        Compute IC autocorrelation up to max_lag.

        Returns
        -------
        acf : np.ndarray, shape (max_lag + 1,)
        """
        ic = np.array(ic_series, dtype=np.float64)
        ic = ic - np.nanmean(ic)
        var = np.nanvar(ic) + 1e-12
        max_lag = self.config.max_lag
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[0] = 1.0
            else:
                cov = np.nanmean(ic[lag:] * ic[:-lag])
                acf[lag] = cov / var
        return acf.astype(np.float32)

    def half_life(self, ic_series: np.ndarray) -> float:
        """
        Estimate signal half-life as the lag at which autocorrelation
        first drops below 0.5.

        Returns
        -------
        half_life : float  (in days, or np.inf if never drops below 0.5)
        """
        acf = self.autocorrelation(ic_series)
        for lag, val in enumerate(acf):
            if val < 0.5 and lag > 0:
                return float(lag)
        return float("inf")


# ---------------------------------------------------------------------------
# SignalNeutraliser
# ---------------------------------------------------------------------------


@dataclass
class NeutralisationConfig:
    """Configuration for SignalNeutraliser."""
    neutralise_market: bool = True
    neutralise_sector: bool = False
    market_cap_weight: bool = False


class SignalNeutraliser:
    """
    Neutralises alpha signals by removing market-wide and sector effects.

    Parameters
    ----------
    config : NeutralisationConfig
    sector_ids : np.ndarray, shape (N,), optional
        Integer sector labels per asset.
    """

    def __init__(
        self,
        config: NeutralisationConfig | None = None,
        sector_ids: np.ndarray | None = None,
    ) -> None:
        self.config = config or NeutralisationConfig()
        self.sector_ids = sector_ids

    def neutralise(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove market (and optionally sector) effects from signal.

        Parameters
        ----------
        signal : np.ndarray, shape (T, N)

        Returns
        -------
        neutralised : np.ndarray, shape (T, N)
        """
        sig = np.array(signal, dtype=np.float64)
        T, N = sig.shape

        if self.config.neutralise_market:
            market_mean = np.nanmean(sig, axis=1, keepdims=True)
            sig = sig - market_mean

        if self.config.neutralise_sector and self.sector_ids is not None:
            for t in range(T):
                for sector in np.unique(self.sector_ids):
                    mask = self.sector_ids == sector
                    sector_mean = np.nanmean(sig[t, mask])
                    sig[t, mask] -= sector_mean

        return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# InformationCoefficient
# ---------------------------------------------------------------------------


class InformationCoefficient:
    """
    Rolling and expanding IC calculations.

    Parameters
    ----------
    window : int
        Rolling window length (days).
    """

    def __init__(self, window: int = 63) -> None:
        self.window = window

    def rolling_ic(
        self, signal: np.ndarray, fwd_returns: np.ndarray
    ) -> np.ndarray:
        """
        Rolling Pearson IC.

        Parameters
        ----------
        signal : np.ndarray, shape (T, N)
        fwd_returns : np.ndarray, shape (T, N)

        Returns
        -------
        rolling_ic : np.ndarray, shape (T,)
        """
        T, N = signal.shape
        ics = np.zeros(T, dtype=np.float32)
        for t in range(self.window, T):
            s = signal[t - self.window:t].ravel()
            r = fwd_returns[t - self.window:t].ravel()
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() > 10:
                ic = float(np.corrcoef(s[valid], r[valid])[0, 1])
                ics[t] = ic if np.isfinite(ic) else 0.0
        return ics

    def expanding_ic(
        self, signal: np.ndarray, fwd_returns: np.ndarray
    ) -> np.ndarray:
        """Expanding-window IC."""
        T, N = signal.shape
        ics = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            s = signal[:t].ravel()
            r = fwd_returns[:t].ravel()
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() > 10:
                ic = float(np.corrcoef(s[valid], r[valid])[0, 1])
                ics[t] = ic if np.isfinite(ic) else 0.0
        return ics

    def icir(self, signal: np.ndarray, fwd_returns: np.ndarray) -> float:
        """IC Information Ratio over full history."""
        ics = self.rolling_ic(signal, fwd_returns)
        return float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12))


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Utilities
    "_zscore",
    "_winsorise",
    "_rank_normalise",
    "_rolling_apply",
    "_exponential_weights",
    # Extractors
    "SignalExtractorConfig",
    "TTSignalExtractor",
    # Momentum
    "MomentumConfig",
    "MomentumSignalFactory",
    # Mean Reversion
    "MeanReversionConfig",
    "MeanReversionSignal",
    # Volatility
    "VolSignalConfig",
    "VolatilitySignalFactory",
    # Correlation break
    "CorrBreakConfig",
    "CorrelationBreakSignal",
    # Tensor PCA
    "TensorPCAConfig",
    "TensorPCASignal",
    # Spread / pairs
    "SpreadConfig",
    "SpreadSignalFactory",
    # Combination
    "CombinerConfig",
    "SignalCombiner",
    # Evaluation
    "SignalEvaluator",
    # Decay
    "DecayConfig",
    "AlphaDecayAnalyser",
    # Neutralisation
    "NeutralisationConfig",
    "SignalNeutraliser",
    # IC
    "InformationCoefficient",
]
