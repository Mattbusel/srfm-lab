"""
data_pipeline.py — Financial data loaders and tensor construction for TensorNet (Project AETERNUS).

Provides:
  - Multi-asset return tensor construction from CSV/Parquet files
  - Rolling-window tensor batches
  - Tensor augmentation (noise injection, time warping, scaling)
  - DataLoader-style iteration over tensor batches
  - Normalization statistics (Z-score, min-max, robust)
  - Integration with Chronos LOB output format
  - High-dimensional covariance tensor construction
  - Factor model tensor construction
  - Streaming data pipeline with memory bounds
  - Data validation and integrity checks
"""

from __future__ import annotations

import math
import os
import warnings
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List, Optional,
    Sequence, Tuple, Union,
)

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ReturnTensor:
    """Multi-asset return tensor with metadata."""
    data: np.ndarray          # shape: (n_windows, window_size, n_assets)
    asset_names: List[str]
    timestamps: np.ndarray    # shape: (n_total_time,)
    window_start_indices: np.ndarray  # which time steps start each window
    normalization_stats: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return self.data.shape[0]

    @property
    def window_size(self) -> int:
        return self.data.shape[1]

    @property
    def n_assets(self) -> int:
        return self.data.shape[2]

    def __repr__(self) -> str:
        return (
            f"ReturnTensor(windows={self.n_windows}, "
            f"window_size={self.window_size}, "
            f"n_assets={self.n_assets})"
        )


@dataclass
class NormalizationStats:
    """Statistics for data normalization."""
    method: str
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    iqr: Optional[np.ndarray] = None

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self.method == "zscore":
            return (data - self.mean) / (self.std + 1e-8)
        elif self.method == "minmax":
            rng = self.max_val - self.min_val
            return (data - self.min_val) / (rng + 1e-8)
        elif self.method == "robust":
            return (data - self.median) / (self.iqr + 1e-8)
        else:
            return data

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.method == "zscore":
            return data * (self.std + 1e-8) + self.mean
        elif self.method == "minmax":
            rng = self.max_val - self.min_val
            return data * (rng + 1e-8) + self.min_val
        elif self.method == "robust":
            return data * (self.iqr + 1e-8) + self.median
        else:
            return data


@dataclass
class DataPipelineConfig:
    """Configuration for the financial data pipeline."""
    window_size: int = 60
    stride: int = 1
    normalize: bool = True
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust"
    augment: bool = False
    noise_std: float = 0.01
    time_warp_sigma: float = 0.2
    scale_range: Tuple[float, float] = (0.9, 1.1)
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    dtype: str = "float32"
    fillna_method: str = "forward"  # "forward", "zero", "mean"
    return_type: str = "log"  # "log", "simple", "percentage"
    clip_returns: Optional[float] = 0.2  # clip at ±20% by default
    seed: int = 42


# ============================================================================
# CSV / Parquet loading
# ============================================================================

def load_price_csv(
    path: str,
    date_col: str = "date",
    asset_cols: Optional[List[str]] = None,
    parse_dates: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load asset price data from a CSV file.

    Args:
        path: Path to CSV file.
        date_col: Name of the date/timestamp column.
        asset_cols: Columns to use as assets. If None, use all non-date columns.
        parse_dates: Parse the date column.

    Returns:
        (prices, asset_names, timestamps) arrays.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for CSV loading.")

    df = pd.read_csv(path, parse_dates=[date_col] if parse_dates else False)
    df = df.sort_values(date_col).reset_index(drop=True)

    timestamps = df[date_col].values

    if asset_cols is None:
        asset_cols = [c for c in df.columns if c != date_col]

    prices = df[asset_cols].values.astype(np.float64)
    return prices, list(asset_cols), timestamps


def load_price_parquet(
    path: str,
    date_col: str = "date",
    asset_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load asset price data from a Parquet file.

    Args:
        path: Path to Parquet file.
        date_col: Date column name.
        asset_cols: Asset columns to use.

    Returns:
        (prices, asset_names, timestamps).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Parquet loading.")

    df = pd.read_parquet(path)
    df = df.sort_values(date_col).reset_index(drop=True)
    timestamps = df[date_col].values

    if asset_cols is None:
        asset_cols = [c for c in df.columns if c != date_col]

    prices = df[asset_cols].values.astype(np.float64)
    return prices, list(asset_cols), timestamps


def prices_to_returns(
    prices: np.ndarray,
    return_type: str = "log",
    clip: Optional[float] = None,
) -> np.ndarray:
    """Compute asset returns from price series.

    Args:
        prices: Price array of shape (n_time, n_assets).
        return_type: "log" (log returns), "simple" (arithmetic), or "percentage".
        clip: Optional clipping threshold (symmetric).

    Returns:
        Returns array of shape (n_time - 1, n_assets).
    """
    prices = np.asarray(prices, dtype=float)
    p1 = prices[1:]
    p0 = prices[:-1]

    if return_type == "log":
        rets = np.log(p1 / (p0 + 1e-15))
    elif return_type == "simple":
        rets = (p1 - p0) / (p0 + 1e-15)
    elif return_type == "percentage":
        rets = (p1 - p0) / (p0 + 1e-15) * 100.0
    else:
        raise ValueError(f"Unknown return_type: {return_type}")

    if clip is not None:
        rets = np.clip(rets, -clip, clip)

    return rets


def fill_missing_returns(
    returns: np.ndarray,
    method: str = "forward",
) -> np.ndarray:
    """Fill missing (NaN) values in return series.

    Args:
        returns: Returns array (n_time, n_assets).
        method: "forward", "zero", or "mean".

    Returns:
        Filled returns array.
    """
    rets = np.array(returns, dtype=float)

    if method == "zero":
        rets = np.nan_to_num(rets, nan=0.0)
    elif method == "forward":
        for j in range(rets.shape[1]):
            col = rets[:, j]
            mask = np.isnan(col)
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
            rets[:, j] = col
        rets = np.nan_to_num(rets, nan=0.0)
    elif method == "mean":
        col_means = np.nanmean(rets, axis=0)
        for j in range(rets.shape[1]):
            nan_mask = np.isnan(rets[:, j])
            rets[nan_mask, j] = col_means[j]
    else:
        raise ValueError(f"Unknown fillna_method: {method}")

    return rets


# ============================================================================
# Normalization
# ============================================================================

def compute_normalization_stats(
    data: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> NormalizationStats:
    """Compute normalization statistics from training data.

    Args:
        data: Data array of shape (n_samples, ...).
        method: Normalization method.
        axis: Axis to compute statistics over.

    Returns:
        NormalizationStats object.
    """
    if method == "zscore":
        return NormalizationStats(
            method=method,
            mean=np.nanmean(data, axis=axis, keepdims=True),
            std=np.nanstd(data, axis=axis, keepdims=True),
        )
    elif method == "minmax":
        return NormalizationStats(
            method=method,
            min_val=np.nanmin(data, axis=axis, keepdims=True),
            max_val=np.nanmax(data, axis=axis, keepdims=True),
        )
    elif method == "robust":
        return NormalizationStats(
            method=method,
            median=np.nanmedian(data, axis=axis, keepdims=True),
            iqr=np.nanpercentile(data, 75, axis=axis, keepdims=True)
            - np.nanpercentile(data, 25, axis=axis, keepdims=True),
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# Rolling window tensor construction
# ============================================================================

def build_rolling_window_tensor(
    returns: np.ndarray,
    window_size: int,
    stride: int = 1,
    config: Optional[DataPipelineConfig] = None,
) -> np.ndarray:
    """Build rolling window tensor from return series.

    Args:
        returns: Return array of shape (n_time, n_assets).
        window_size: Length of each window.
        stride: Step between consecutive windows.
        config: Optional pipeline config.

    Returns:
        Tensor of shape (n_windows, window_size, n_assets).
    """
    n_time, n_assets = returns.shape
    n_windows = max(0, (n_time - window_size) // stride + 1)

    windows = np.empty((n_windows, window_size, n_assets), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i] = returns[start : start + window_size]

    return windows


def build_covariance_tensor(
    returns: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Build rolling covariance matrix tensor.

    Args:
        returns: Return array (n_time, n_assets).
        window_size: Window for covariance estimation.
        stride: Stride between windows.

    Returns:
        Tensor of shape (n_windows, n_assets, n_assets).
    """
    n_time, n_assets = returns.shape
    n_windows = max(0, (n_time - window_size) // stride + 1)
    cov_tensor = np.empty((n_windows, n_assets, n_assets), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        window = returns[start : start + window_size]
        cov = np.cov(window.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov_tensor[i] = cov

    return cov_tensor


def build_factor_tensor(
    returns: np.ndarray,
    n_factors: int,
    window_size: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build rolling factor model tensors via PCA.

    Args:
        returns: Return array (n_time, n_assets).
        n_factors: Number of PCA factors.
        window_size: Rolling window size.
        stride: Stride.

    Returns:
        (factor_loadings_tensor, factor_returns_tensor):
          factor_loadings: (n_windows, n_assets, n_factors)
          factor_returns:  (n_windows, window_size, n_factors)
    """
    n_time, n_assets = returns.shape
    n_windows = max(0, (n_time - window_size) // stride + 1)
    n_f = min(n_factors, n_assets)

    loadings = np.empty((n_windows, n_assets, n_f), dtype=np.float32)
    factor_rets = np.empty((n_windows, window_size, n_f), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        window = returns[start : start + window_size]  # (window_size, n_assets)
        window_centered = window - window.mean(axis=0)
        U, s, Vt = np.linalg.svd(window_centered, full_matrices=False)
        loadings[i] = Vt[:n_f].T  # (n_assets, n_f)
        factor_rets[i] = window_centered @ Vt[:n_f].T  # (window_size, n_f)

    return loadings, factor_rets


# ============================================================================
# Tensor augmentation
# ============================================================================

def augment_noise_injection(
    tensor: np.ndarray,
    noise_std: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to a tensor batch.

    Args:
        tensor: Input tensor of any shape.
        noise_std: Standard deviation of noise relative to data std.
        rng: Random number generator.

    Returns:
        Noisy tensor.
    """
    if rng is None:
        rng = np.random.default_rng()
    data_std = float(np.std(tensor)) + 1e-8
    noise = rng.normal(0.0, noise_std * data_std, size=tensor.shape)
    return (tensor + noise).astype(tensor.dtype)


def augment_time_warp(
    tensor: np.ndarray,
    sigma: float = 0.2,
    n_knots: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply time warping augmentation to a (window_size, n_assets) tensor.

    Generates a smooth warp path via cubic interpolation and resamples
    the time series along it.

    Args:
        tensor: Array of shape (window_size, n_assets) or (n_windows, window_size, n_assets).
        sigma: Warp magnitude.
        n_knots: Number of warp knots.
        rng: Random number generator.

    Returns:
        Time-warped tensor.
    """
    if rng is None:
        rng = np.random.default_rng()

    single = tensor.ndim == 2
    if single:
        tensor = tensor[np.newaxis]

    n_windows, window_size, n_assets = tensor.shape
    warped = np.empty_like(tensor)

    original_steps = np.linspace(0, 1, window_size)
    knot_x = np.linspace(0, 1, n_knots + 2)

    for w in range(n_windows):
        knot_y = knot_x + rng.normal(0, sigma, size=knot_x.shape)
        knot_y[0] = 0.0
        knot_y[-1] = 1.0
        knot_y = np.sort(knot_y)  # keep monotone

        warp_path = np.interp(original_steps, knot_x, knot_y)
        warp_idx = np.clip(warp_path * (window_size - 1), 0, window_size - 1)

        for a in range(n_assets):
            warped[w, :, a] = np.interp(warp_idx, np.arange(window_size), tensor[w, :, a])

    if single:
        return warped[0]
    return warped


def augment_scaling(
    tensor: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    per_asset: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random scaling to each asset independently.

    Args:
        tensor: Array of shape (window_size, n_assets) or (n_windows, window_size, n_assets).
        scale_range: (min_scale, max_scale).
        per_asset: If True, scale each asset independently.
        rng: Random number generator.

    Returns:
        Scaled tensor.
    """
    if rng is None:
        rng = np.random.default_rng()

    single = tensor.ndim == 2
    if single:
        tensor = tensor[np.newaxis]

    n_windows, window_size, n_assets = tensor.shape
    scaled = tensor.copy()

    lo, hi = scale_range
    if per_asset:
        scales = rng.uniform(lo, hi, size=(n_windows, 1, n_assets))
    else:
        scales = rng.uniform(lo, hi, size=(n_windows, 1, 1))

    scaled = (scaled * scales).astype(tensor.dtype)

    if single:
        return scaled[0]
    return scaled


def augment_window_slice(
    tensor: np.ndarray,
    min_fraction: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random window slicing augmentation.

    Randomly slices a contiguous sub-window and zero-pads back.

    Args:
        tensor: Array of shape (window_size, n_assets).
        min_fraction: Minimum fraction of window to keep.
        rng: Random generator.

    Returns:
        Sliced and padded tensor, same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    window_size = tensor.shape[0]
    min_len = max(1, int(window_size * min_fraction))
    slice_len = rng.integers(min_len, window_size + 1)
    start = rng.integers(0, window_size - slice_len + 1)

    result = np.zeros_like(tensor)
    result[:slice_len] = tensor[start : start + slice_len]
    return result


def apply_augmentations(
    tensor: np.ndarray,
    config: DataPipelineConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply all configured augmentations.

    Args:
        tensor: Input tensor (window_size, n_assets) or (n_windows, window_size, n_assets).
        config: DataPipelineConfig with augmentation settings.
        rng: Random generator.

    Returns:
        Augmented tensor.
    """
    if not config.augment:
        return tensor

    if rng is None:
        rng = np.random.default_rng(config.seed)

    result = tensor.copy()

    if config.noise_std > 0:
        result = augment_noise_injection(result, config.noise_std, rng)

    if config.time_warp_sigma > 0:
        result = augment_time_warp(result, config.time_warp_sigma, rng=rng)

    lo, hi = config.scale_range
    if lo != 1.0 or hi != 1.0:
        result = augment_scaling(result, config.scale_range, rng=rng)

    return result


# ============================================================================
# DataLoader-style iteration
# ============================================================================

class TensorDataLoader:
    """DataLoader-style iterator over rolling window tensor batches.

    Supports shuffling, batching, augmentation, and JAX array conversion.

    Args:
        data: Tensor of shape (n_windows, window_size, n_assets).
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        drop_last: Drop the last incomplete batch.
        augment: Apply augmentations.
        config: Optional DataPipelineConfig.
        seed: Random seed.
    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        augment: bool = False,
        config: Optional[DataPipelineConfig] = None,
        seed: int = 42,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.augment = augment
        self.config = config or DataPipelineConfig(batch_size=batch_size, seed=seed)
        self.seed = seed
        self._epoch = 0
        self._n = len(data)

    @property
    def n_batches(self) -> int:
        if self.drop_last:
            return self._n // self.batch_size
        return math.ceil(self._n / self.batch_size)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[jnp.ndarray]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        idx = np.arange(self._n)
        if self.shuffle:
            rng.shuffle(idx)

        for start in range(0, self._n, self.batch_size):
            end = start + self.batch_size
            if end > self._n and self.drop_last:
                break
            end = min(end, self._n)

            batch = self.data[idx[start:end]]

            if self.augment:
                batch = apply_augmentations(batch, self.config, rng)

            yield jnp.array(batch)

    def get_all(self) -> jnp.ndarray:
        """Return the full dataset as a JAX array."""
        return jnp.array(self.data)


# ============================================================================
# Chronos LOB format integration
# ============================================================================

@dataclass
class ChronosLOBRecord:
    """A single record from Chronos LOB output."""
    timestamp: Any
    asset_id: str
    bid_prices: np.ndarray    # (n_levels,)
    ask_prices: np.ndarray    # (n_levels,)
    bid_sizes: np.ndarray     # (n_levels,)
    ask_sizes: np.ndarray     # (n_levels,)
    mid_price: float
    spread: float


def parse_chronos_lob_csv(
    path: str,
    n_levels: int = 10,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
) -> Dict[str, List[ChronosLOBRecord]]:
    """Parse Chronos LOB output CSV into records.

    Expected columns:
        timestamp, asset_id, bid_p1..bid_pN, ask_p1..ask_pN,
        bid_s1..bid_sN, ask_s1..ask_sN

    Args:
        path: Path to CSV file.
        n_levels: Number of order book levels.
        asset_col: Column containing asset identifier.
        timestamp_col: Column containing timestamps.

    Returns:
        Dict mapping asset_id -> list of ChronosLOBRecord.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Chronos LOB parsing.")

    df = pd.read_csv(path)
    records: Dict[str, List[ChronosLOBRecord]] = {}

    for _, row in df.iterrows():
        asset = str(row.get(asset_col, "unknown"))
        ts = row.get(timestamp_col, None)

        bid_p = np.array([row.get(f"bid_p{i+1}", 0.0) for i in range(n_levels)])
        ask_p = np.array([row.get(f"ask_p{i+1}", 0.0) for i in range(n_levels)])
        bid_s = np.array([row.get(f"bid_s{i+1}", 0.0) for i in range(n_levels)])
        ask_s = np.array([row.get(f"ask_s{i+1}", 0.0) for i in range(n_levels)])
        mid = float((bid_p[0] + ask_p[0]) / 2.0) if bid_p[0] > 0 and ask_p[0] > 0 else 0.0
        spread = float(ask_p[0] - bid_p[0]) if bid_p[0] > 0 and ask_p[0] > 0 else 0.0

        rec = ChronosLOBRecord(
            timestamp=ts,
            asset_id=asset,
            bid_prices=bid_p,
            ask_prices=ask_p,
            bid_sizes=bid_s,
            ask_sizes=ask_s,
            mid_price=mid,
            spread=spread,
        )
        records.setdefault(asset, []).append(rec)

    return records


def lob_records_to_tensor(
    records: Dict[str, List[ChronosLOBRecord]],
    n_levels: int = 10,
    feature_type: str = "full",
) -> Tuple[np.ndarray, List[str]]:
    """Convert LOB records to a feature tensor.

    Args:
        records: Dict asset_id -> list of records.
        n_levels: Number of LOB levels.
        feature_type: "full" (all bid/ask), "mid" (mid-price only), "spread" (spread only).

    Returns:
        (tensor, asset_names) where tensor has shape (n_time, n_assets, n_features).
    """
    assets = sorted(records.keys())
    n_assets = len(assets)

    # Determine n_time from first asset
    n_time = min(len(records[a]) for a in assets)

    if feature_type == "full":
        n_features = 4 * n_levels  # bid_p, ask_p, bid_s, ask_s
    elif feature_type == "mid":
        n_features = 1
    elif feature_type == "spread":
        n_features = 1
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    tensor = np.zeros((n_time, n_assets, n_features), dtype=np.float32)

    for j, asset in enumerate(assets):
        recs = records[asset][:n_time]
        for t, rec in enumerate(recs):
            if feature_type == "full":
                feats = np.concatenate([rec.bid_prices, rec.ask_prices, rec.bid_sizes, rec.ask_sizes])
                tensor[t, j, :] = feats[:n_features]
            elif feature_type == "mid":
                tensor[t, j, 0] = rec.mid_price
            elif feature_type == "spread":
                tensor[t, j, 0] = rec.spread

    return tensor, assets


def lob_to_return_tensor(
    records: Dict[str, List[ChronosLOBRecord]],
    window_size: int = 60,
    stride: int = 1,
    config: Optional[DataPipelineConfig] = None,
) -> ReturnTensor:
    """End-to-end: LOB records -> normalized rolling window return tensor.

    Computes mid-price returns from LOB data, builds rolling windows,
    and optionally normalizes.

    Args:
        records: Chronos LOB records per asset.
        window_size: Rolling window size.
        stride: Stride between windows.
        config: Pipeline config.

    Returns:
        ReturnTensor ready for model input.
    """
    if config is None:
        config = DataPipelineConfig(window_size=window_size, stride=stride)

    # Extract mid prices
    assets = sorted(records.keys())
    n_assets = len(assets)
    n_time_min = min(len(records[a]) for a in assets)

    mid_prices = np.zeros((n_time_min, n_assets), dtype=np.float64)
    for j, asset in enumerate(assets):
        mid_prices[:, j] = [r.mid_price for r in records[asset][:n_time_min]]

    # Compute returns
    returns = prices_to_returns(mid_prices, config.return_type, config.clip_returns)
    returns = fill_missing_returns(returns, config.fillna_method)

    # Normalize
    norm_stats = None
    if config.normalize:
        stats = compute_normalization_stats(returns, config.normalization_method)
        returns = stats.normalize(returns)
        norm_stats = {
            "method": config.normalization_method,
            "mean": stats.mean,
            "std": stats.std,
        }

    # Build rolling windows
    windows = build_rolling_window_tensor(returns, config.window_size, config.stride)
    n_windows = windows.shape[0]
    window_starts = np.arange(n_windows) * config.stride

    timestamps = np.arange(n_time_min)

    return ReturnTensor(
        data=windows,
        asset_names=assets,
        timestamps=timestamps,
        window_start_indices=window_starts,
        normalization_stats=norm_stats,
    )


# ============================================================================
# Full data pipeline from CSV/Parquet
# ============================================================================

class FinancialDataPipeline:
    """End-to-end pipeline: file -> normalized rolling window tensor batches.

    Usage::

        pipeline = FinancialDataPipeline(config)
        pipeline.load("prices.csv")
        loader = pipeline.get_dataloader(split="train")
        for batch in loader:
            # batch: jnp array (batch_size, window_size, n_assets)
            ...
    """

    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self._raw_prices: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None
        self._asset_names: Optional[List[str]] = None
        self._timestamps: Optional[np.ndarray] = None
        self._norm_stats: Optional[NormalizationStats] = None
        self._windows: Optional[np.ndarray] = None

    def load(
        self,
        path: str,
        date_col: str = "date",
        asset_cols: Optional[List[str]] = None,
    ) -> "FinancialDataPipeline":
        """Load price data from file.

        Args:
            path: Path to CSV or Parquet file.
            date_col: Date column name.
            asset_cols: Asset columns to use.

        Returns:
            self (for chaining).
        """
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            prices, names, ts = load_price_csv(path, date_col, asset_cols)
        elif ext in (".parquet", ".pq"):
            prices, names, ts = load_price_parquet(path, date_col, asset_cols)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        self._raw_prices = prices
        self._asset_names = names
        self._timestamps = ts
        return self

    def load_array(
        self,
        prices: np.ndarray,
        asset_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> "FinancialDataPipeline":
        """Load directly from a numpy array.

        Args:
            prices: Price array (n_time, n_assets).
            asset_names: Optional asset names.
            timestamps: Optional timestamp array.

        Returns:
            self.
        """
        self._raw_prices = np.asarray(prices, dtype=np.float64)
        n_assets = prices.shape[1] if prices.ndim > 1 else 1
        self._asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]
        self._timestamps = timestamps if timestamps is not None else np.arange(len(prices))
        return self

    def process(self) -> "FinancialDataPipeline":
        """Run the full processing pipeline.

        Steps:
            1. Compute returns
            2. Fill missing values
            3. Compute normalization stats on training period
            4. Normalize
            5. Build rolling windows

        Returns:
            self.
        """
        if self._raw_prices is None:
            raise RuntimeError("Call load() before process().")

        rets = prices_to_returns(
            self._raw_prices,
            return_type=self.config.return_type,
            clip=self.config.clip_returns,
        )
        rets = fill_missing_returns(rets, self.config.fillna_method)

        if self.config.normalize:
            self._norm_stats = compute_normalization_stats(rets, self.config.normalization_method)
            rets = self._norm_stats.normalize(rets)

        self._returns = rets.astype(np.float32)
        self._windows = build_rolling_window_tensor(
            self._returns,
            self.config.window_size,
            self.config.stride,
        )

        return self

    def split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split windows into train/val/test.

        Args:
            train_frac: Fraction for training.
            val_frac: Fraction for validation.

        Returns:
            (train_windows, val_windows, test_windows).
        """
        if self._windows is None:
            self.process()

        n = len(self._windows)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = self._windows[:n_train]
        val = self._windows[n_train : n_train + n_val]
        test = self._windows[n_train + n_val :]

        return train, val, test

    def get_dataloader(
        self,
        split: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> TensorDataLoader:
        """Get a DataLoader for a specific split.

        Args:
            split: "train", "val", or "test".
            train_frac: Training fraction.
            val_frac: Validation fraction.

        Returns:
            TensorDataLoader.
        """
        tr, va, te = self.split(train_frac, val_frac)
        split_data = {"train": tr, "val": va, "test": te}[split]
        shuffle = split == "train"
        augment = split == "train" and self.config.augment

        return TensorDataLoader(
            data=split_data,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=self.config.drop_last,
            augment=augment,
            config=self.config,
            seed=self.config.seed,
        )

    @property
    def asset_names(self) -> List[str]:
        return self._asset_names or []

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)

    @property
    def n_windows(self) -> int:
        return len(self._windows) if self._windows is not None else 0

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization on data.

        Args:
            data: Normalized data array.

        Returns:
            Denormalized data.
        """
        if self._norm_stats is None:
            return data
        return self._norm_stats.denormalize(data)


# ============================================================================
# Streaming data pipeline (memory bounded)
# ============================================================================

class StreamingDataPipeline:
    """Memory-bounded streaming data pipeline for large datasets.

    Processes data in chunks to avoid loading the full dataset into memory.
    Yields batches of rolling-window tensors one chunk at a time.

    Args:
        source_path: Path to data file (CSV or Parquet).
        config: DataPipelineConfig.
        chunk_size: Number of time steps per chunk.
    """

    def __init__(
        self,
        source_path: str,
        config: DataPipelineConfig,
        chunk_size: int = 10000,
    ):
        self.source_path = source_path
        self.config = config
        self.chunk_size = chunk_size
        self._norm_stats: Optional[NormalizationStats] = None

    def fit_normalization(self, n_rows: int = 10000) -> NormalizationStats:
        """Fit normalization statistics on the first n_rows.

        Args:
            n_rows: Number of rows to use for fitting.

        Returns:
            NormalizationStats.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for streaming pipeline.")

        ext = Path(self.source_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(self.source_path, nrows=n_rows)
        else:
            df = pd.read_parquet(self.source_path)
            df = df.iloc[:n_rows]

        price_cols = [c for c in df.columns if c not in ("date", "timestamp", "time")]
        prices = df[price_cols].values.astype(float)
        returns = prices_to_returns(prices, self.config.return_type)
        returns = fill_missing_returns(returns, self.config.fillna_method)

        self._norm_stats = compute_normalization_stats(returns, self.config.normalization_method)
        return self._norm_stats

    def stream_batches(
        self,
        overlap: Optional[int] = None,
    ) -> Generator[jnp.ndarray, None, None]:
        """Yield batches of rolling window tensors, streaming from file.

        Args:
            overlap: Overlap between consecutive chunks (defaults to window_size - 1).

        Yields:
            Batches of shape (batch_size, window_size, n_assets) as JAX arrays.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for streaming pipeline.")

        if overlap is None:
            overlap = self.config.window_size - 1

        ext = Path(self.source_path).suffix.lower()

        # Determine total rows (or use chunked reading)
        chunk_iter = pd.read_csv(
            self.source_path, chunksize=self.chunk_size
        ) if ext == ".csv" else [pd.read_parquet(self.source_path)]

        price_cols = None
        prev_tail: Optional[np.ndarray] = None
        loader_buffer: List[np.ndarray] = []

        for chunk_df in chunk_iter:
            if price_cols is None:
                price_cols = [c for c in chunk_df.columns if c not in ("date", "timestamp", "time")]

            prices_chunk = chunk_df[price_cols].values.astype(float)

            # Prepend overlap from previous chunk
            if prev_tail is not None:
                prices_chunk = np.concatenate([prev_tail, prices_chunk], axis=0)

            # Compute returns
            if len(prices_chunk) < 2:
                continue

            rets = prices_to_returns(prices_chunk, self.config.return_type)
            rets = fill_missing_returns(rets, self.config.fillna_method)

            if self._norm_stats is not None and self.config.normalize:
                rets = self._norm_stats.normalize(rets)

            windows = build_rolling_window_tensor(
                rets, self.config.window_size, self.config.stride
            )

            # Yield batches
            for start in range(0, len(windows), self.config.batch_size):
                end = min(start + self.config.batch_size, len(windows))
                yield jnp.array(windows[start:end].astype(np.float32))

            # Save tail for overlap
            prev_tail = prices_chunk[-overlap:] if len(prices_chunk) >= overlap else prices_chunk


# ============================================================================
# Data validation
# ============================================================================

def validate_return_tensor(
    returns: np.ndarray,
    check_nan: bool = True,
    check_inf: bool = True,
    check_extreme: bool = True,
    extreme_threshold: float = 5.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Validate a return tensor for data quality issues.

    Args:
        returns: Return array (n_time, n_assets).
        check_nan: Check for NaN values.
        check_inf: Check for Inf values.
        check_extreme: Check for statistically extreme values.
        extreme_threshold: Z-score threshold for extreme values.
        verbose: Print report.

    Returns:
        Dict with validation results.
    """
    result: Dict[str, Any] = {"valid": True, "issues": []}

    if check_nan:
        n_nan = int(np.sum(np.isnan(returns)))
        if n_nan > 0:
            result["valid"] = False
            result["issues"].append(f"NaN values: {n_nan}")
        result["n_nan"] = n_nan

    if check_inf:
        n_inf = int(np.sum(np.isinf(returns)))
        if n_inf > 0:
            result["valid"] = False
            result["issues"].append(f"Inf values: {n_inf}")
        result["n_inf"] = n_inf

    if check_extreme:
        finite_mask = np.isfinite(returns)
        if finite_mask.any():
            z_scores = (returns - np.nanmean(returns)) / (np.nanstd(returns) + 1e-8)
            n_extreme = int(np.sum(np.abs(z_scores) > extreme_threshold))
            result["n_extreme"] = n_extreme
            if n_extreme > returns.size * 0.01:
                result["issues"].append(f"Many extreme values: {n_extreme} ({100*n_extreme/returns.size:.2f}%)")

    result["shape"] = returns.shape
    result["dtype"] = str(returns.dtype)
    result["mean"] = float(np.nanmean(returns))
    result["std"] = float(np.nanstd(returns))
    result["min"] = float(np.nanmin(returns))
    result["max"] = float(np.nanmax(returns))

    if verbose:
        print("Return tensor validation:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    return result


# ============================================================================
# Advanced financial tensor construction
# ============================================================================

def build_cross_sectional_momentum_tensor(
    returns: np.ndarray,
    lookback_windows: List[int],
    skip_window: int = 1,
) -> np.ndarray:
    """Build cross-sectional momentum tensor across multiple lookback windows.

    For each lookback window L, computes the normalized momentum score
    for each asset at each time step.

    Args:
        returns: Return array (n_time, n_assets).
        lookback_windows: List of lookback period lengths.
        skip_window: Skip most recent period (standard momentum skip).

    Returns:
        Momentum tensor (n_time, n_assets, n_windows).
    """
    n_time, n_assets = returns.shape
    n_windows = len(lookback_windows)
    max_lookback = max(lookback_windows) + skip_window

    momentum = np.zeros((n_time, n_assets, n_windows), dtype=np.float32)

    for w_idx, lookback in enumerate(lookback_windows):
        for t in range(max_lookback, n_time):
            start = t - lookback - skip_window
            end = t - skip_window
            if start < 0:
                continue
            window_rets = returns[start:end]
            cumulative_ret = np.prod(1.0 + window_rets, axis=0) - 1.0
            # Cross-sectional normalize
            cs_mean = cumulative_ret.mean()
            cs_std = cumulative_ret.std() + 1e-8
            momentum[t, :, w_idx] = (cumulative_ret - cs_mean) / cs_std

    return momentum


def build_volatility_surface_tensor(
    returns: np.ndarray,
    vol_windows: List[int],
    quantiles: Optional[List[float]] = None,
) -> np.ndarray:
    """Build volatility surface tensor across time windows and quantiles.

    Args:
        returns: Return array (n_time, n_assets).
        vol_windows: List of window sizes for volatility estimation.
        quantiles: Quantiles to compute. Defaults to [0.25, 0.5, 0.75].

    Returns:
        Tensor of shape (n_time, n_assets, n_windows * n_quantiles).
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]

    n_time, n_assets = returns.shape
    n_windows = len(vol_windows)
    n_q = len(quantiles)
    n_features = n_windows * n_q
    max_window = max(vol_windows)

    surface = np.zeros((n_time, n_assets, n_features), dtype=np.float32)

    for w_idx, win in enumerate(vol_windows):
        for t in range(max_window, n_time):
            window = returns[t - win : t]  # (win, n_assets)
            for q_idx, q in enumerate(quantiles):
                feat_idx = w_idx * n_q + q_idx
                surface[t, :, feat_idx] = np.quantile(np.abs(window), q, axis=0)

    return surface


def build_correlation_change_tensor(
    returns: np.ndarray,
    short_window: int = 20,
    long_window: int = 60,
    stride: int = 5,
) -> np.ndarray:
    """Build correlation change tensor (short vs long correlation regime).

    Measures the difference between short-term and long-term correlation,
    capturing correlation breakdowns and regime changes.

    Args:
        returns: Return array (n_time, n_assets).
        short_window: Short correlation window.
        long_window: Long correlation window.
        stride: Stride between windows.

    Returns:
        Tensor (n_windows, n_assets, n_assets) of correlation differences.
    """
    n_time, n_assets = returns.shape
    n_windows = max(0, (n_time - long_window) // stride + 1)
    diff_tensor = np.zeros((n_windows, n_assets, n_assets), dtype=np.float32)

    for i in range(n_windows):
        t = i * stride + long_window
        short_rets = returns[t - short_window : t]
        long_rets = returns[t - long_window : t]

        corr_short = np.corrcoef(short_rets.T) if short_rets.shape[0] >= 2 else np.eye(n_assets)
        corr_long = np.corrcoef(long_rets.T) if long_rets.shape[0] >= 2 else np.eye(n_assets)

        if np.any(np.isnan(corr_short)):
            corr_short = np.eye(n_assets)
        if np.any(np.isnan(corr_long)):
            corr_long = np.eye(n_assets)

        diff_tensor[i] = (corr_short - corr_long).astype(np.float32)

    return diff_tensor


def build_lead_lag_tensor(
    returns: np.ndarray,
    max_lag: int = 5,
    window_size: int = 60,
    stride: int = 10,
) -> np.ndarray:
    """Build lead-lag relationship tensor across assets.

    Computes cross-correlation between all asset pairs at multiple lags,
    capturing lead-lag relationships in the return series.

    Args:
        returns: Return array (n_time, n_assets).
        max_lag: Maximum lag (in both directions).
        window_size: Rolling window for cross-correlation estimation.
        stride: Stride between windows.

    Returns:
        Tensor (n_windows, n_assets, n_assets, 2*max_lag+1) of cross-correlations.
    """
    n_time, n_assets = returns.shape
    n_lags = 2 * max_lag + 1
    n_windows = max(0, (n_time - window_size) // stride + 1)
    lead_lag = np.zeros((n_windows, n_assets, n_assets, n_lags), dtype=np.float32)

    for w_idx in range(n_windows):
        start = w_idx * stride
        window = returns[start : start + window_size]

        for i in range(n_assets):
            for j in range(n_assets):
                xi = window[:, i]
                xj = window[:, j]
                # Normalize
                xi = (xi - xi.mean()) / (xi.std() + 1e-8)
                xj = (xj - xj.mean()) / (xj.std() + 1e-8)

                for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
                    if lag == 0:
                        corr = float(np.corrcoef(xi, xj)[0, 1])
                    elif lag > 0:
                        if len(xi) > lag:
                            corr = float(np.corrcoef(xi[lag:], xj[:-lag])[0, 1])
                        else:
                            corr = 0.0
                    else:
                        l = -lag
                        if len(xi) > l:
                            corr = float(np.corrcoef(xi[:-l], xj[l:])[0, 1])
                        else:
                            corr = 0.0

                    if np.isnan(corr):
                        corr = 0.0
                    lead_lag[w_idx, i, j, lag_idx] = corr

    return lead_lag


# ============================================================================
# Multi-frequency tensor construction
# ============================================================================

def resample_returns(
    returns: np.ndarray,
    original_freq: str,
    target_freq: str,
    timestamps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resample return series to a different frequency.

    Converts high-frequency returns to lower frequency by compounding.

    Args:
        returns: Return array (n_time, n_assets).
        original_freq: Original frequency string ("1min", "5min", "1h", "1d").
        target_freq: Target frequency.
        timestamps: Optional timestamp array for alignment.

    Returns:
        Resampled returns.
    """
    freq_minutes = {
        "1min": 1, "5min": 5, "15min": 15, "30min": 30,
        "1h": 60, "4h": 240, "1d": 390,  # ~6.5h trading day
    }

    orig_m = freq_minutes.get(original_freq, 1)
    target_m = freq_minutes.get(target_freq, 1)

    if target_m <= orig_m:
        return returns

    ratio = target_m // orig_m
    n_time, n_assets = returns.shape
    n_periods = n_time // ratio

    resampled = np.zeros((n_periods, n_assets), dtype=returns.dtype)
    for i in range(n_periods):
        chunk = returns[i * ratio : (i + 1) * ratio]
        # Compound returns
        resampled[i] = np.prod(1.0 + chunk, axis=0) - 1.0

    return resampled


def build_multi_frequency_tensor(
    returns: np.ndarray,
    frequencies: List[str],
    base_freq: str = "1min",
    window_size: int = 50,
    stride: int = 5,
) -> np.ndarray:
    """Build a multi-frequency tensor combining returns at different time scales.

    Args:
        returns: High-frequency return array (n_time, n_assets).
        frequencies: Target frequencies to include.
        base_freq: Original data frequency.
        window_size: Window size at each frequency.
        stride: Stride at the base frequency.

    Returns:
        Tensor (n_windows, window_size, n_assets, n_frequencies).
    """
    resampled_list = []
    for freq in frequencies:
        resampled = resample_returns(returns, base_freq, freq)
        resampled_list.append(resampled)

    # Find minimum length across resampled series
    min_len = min(len(r) for r in resampled_list)
    resampled_list = [r[:min_len] for r in resampled_list]

    n_assets = returns.shape[1]
    n_freq = len(frequencies)
    n_windows = max(0, (min_len - window_size) // stride + 1)

    tensor = np.zeros((n_windows, window_size, n_assets, n_freq), dtype=np.float32)

    for f_idx, rets in enumerate(resampled_list):
        for w_idx in range(n_windows):
            start = w_idx * stride
            tensor[w_idx, :, :, f_idx] = rets[start : start + window_size]

    return tensor


# ============================================================================
# Factor model data tensors
# ============================================================================

def compute_residual_tensor(
    returns: np.ndarray,
    factor_loadings: np.ndarray,
    factor_returns: np.ndarray,
) -> np.ndarray:
    """Compute idiosyncratic residual tensor after factor removal.

    Args:
        returns: Return array (n_time, n_assets).
        factor_loadings: Factor loadings (n_assets, n_factors).
        factor_returns: Factor returns (n_time, n_factors).

    Returns:
        Residual returns (n_time, n_assets).
    """
    systematic = factor_returns @ factor_loadings.T
    residuals = returns - systematic
    return residuals.astype(np.float32)


def build_factor_residual_rolling_tensor(
    returns: np.ndarray,
    n_factors: int = 5,
    window_size: int = 60,
    stride: int = 5,
    include_factors: bool = True,
) -> Dict[str, np.ndarray]:
    """Build rolling tensors for factors and residuals.

    Performs rolling PCA to extract factor returns and residuals
    at each window.

    Args:
        returns: Return array (n_time, n_assets).
        n_factors: Number of PCA factors.
        window_size: Window size.
        stride: Stride.
        include_factors: Whether to include factor returns in output.

    Returns:
        Dict with 'residuals', 'factor_returns', 'factor_loadings' arrays.
    """
    n_time, n_assets = returns.shape
    n_windows = max(0, (n_time - window_size) // stride + 1)
    n_f = min(n_factors, n_assets)

    residual_windows = np.zeros((n_windows, window_size, n_assets), dtype=np.float32)
    factor_ret_windows = np.zeros((n_windows, window_size, n_f), dtype=np.float32)
    loading_windows = np.zeros((n_windows, n_assets, n_f), dtype=np.float32)

    for w_idx in range(n_windows):
        start = w_idx * stride
        window = returns[start : start + window_size]
        w_mean = window.mean(axis=0)
        w_centered = window - w_mean

        U, s, Vt = np.linalg.svd(w_centered, full_matrices=False)
        r = min(n_f, len(s))

        f_rets = U[:, :r] * s[:r]  # (window_size, n_f)
        loadings = Vt[:r].T          # (n_assets, n_f)

        # Residuals
        systematic = f_rets @ loadings.T  # (window_size, n_assets)
        residuals = w_centered - systematic

        residual_windows[w_idx] = residuals.astype(np.float32)
        factor_ret_windows[w_idx, :, :r] = f_rets.astype(np.float32)
        loading_windows[w_idx, :, :r] = loadings.astype(np.float32)

    return {
        "residuals": residual_windows,
        "factor_returns": factor_ret_windows,
        "factor_loadings": loading_windows,
        "n_factors": n_f,
        "n_windows": n_windows,
        "window_size": window_size,
        "n_assets": n_assets,
    }


# ============================================================================
# Tensor augmentation extensions
# ============================================================================

def augment_mixup(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Mixup augmentation: convex combination of two tensors.

    Args:
        tensor_a: First tensor.
        tensor_b: Second tensor (same shape).
        alpha: Beta distribution concentration parameter.
        rng: Random generator.

    Returns:
        Mixed tensor.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = rng.beta(alpha, alpha)
    return (lam * tensor_a + (1 - lam) * tensor_b).astype(tensor_a.dtype)


def augment_cutmix(
    tensor: np.ndarray,
    tensor_b: np.ndarray,
    cut_frac: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """CutMix augmentation: replace a random time segment with another sample.

    Args:
        tensor: First tensor (window_size, n_assets).
        tensor_b: Second tensor (same shape).
        cut_frac: Fraction of window to cut.
        rng: Random generator.

    Returns:
        Mixed tensor.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = tensor.copy()
    window_size = tensor.shape[0]
    cut_len = max(1, int(window_size * cut_frac))
    start = rng.integers(0, max(1, window_size - cut_len + 1))
    result[start : start + cut_len] = tensor_b[start : start + cut_len]
    return result


def augment_asset_dropout(
    tensor: np.ndarray,
    dropout_rate: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly zero out some asset channels.

    Simulates missing data or illiquid assets.

    Args:
        tensor: Input tensor (window_size, n_assets).
        dropout_rate: Fraction of assets to zero out.
        rng: Random generator.

    Returns:
        Tensor with some assets zeroed.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = tensor.copy()
    n_assets = tensor.shape[-1]
    n_drop = max(0, int(n_assets * dropout_rate))
    if n_drop > 0:
        drop_idx = rng.choice(n_assets, size=n_drop, replace=False)
        result[..., drop_idx] = 0.0
    return result


# ---------------------------------------------------------------------------
# Section: Advanced financial feature engineering
# ---------------------------------------------------------------------------

import numpy as np
import warnings


def compute_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index (RSI) for each asset.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
        Price series.
    window : int
        RSI lookback window.

    Returns
    -------
    rsi : np.ndarray, shape (T, N)
        RSI values in [0, 100].
    """
    T, N = prices.shape
    delta = np.diff(prices, axis=0)   # (T-1, N)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        avg_gain = gain[t - window:t].mean(axis=0)
        avg_loss = loss[t - window:t].mean(axis=0)
        rs = avg_gain / (avg_loss + 1e-12)
        rsi[t] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_macd(
    prices: np.ndarray,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> tuple:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    fast_window : int
    slow_window : int
    signal_window : int

    Returns
    -------
    macd_line : np.ndarray, shape (T, N)
    signal_line : np.ndarray, shape (T, N)
    histogram : np.ndarray, shape (T, N)
    """
    T, N = prices.shape

    def ema(arr, span):
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for t in range(1, len(arr)):
            result[t] = alpha * arr[t] + (1 - alpha) * result[t - 1]
        return result

    macd_line = np.zeros((T, N), dtype=np.float32)
    signal_line = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        fast = ema(prices[:, n], fast_window)
        slow = ema(prices[:, n], slow_window)
        macd_line[:, n] = fast - slow
        signal_line[:, n] = ema(macd_line[:, n], signal_window)

    histogram = (macd_line - signal_line).astype(np.float32)
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: np.ndarray,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    window : int
        Rolling window for mean and std.
    n_std : float
        Number of standard deviations for the bands.

    Returns
    -------
    upper : np.ndarray, shape (T, N)
    middle : np.ndarray, shape (T, N)
    lower : np.ndarray, shape (T, N)
    bandwidth : np.ndarray, shape (T, N)
        (upper - lower) / middle, normalised width.
    """
    T, N = prices.shape
    middle = np.zeros((T, N), dtype=np.float32)
    upper = np.zeros((T, N), dtype=np.float32)
    lower = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        window_prices = prices[t - window:t]
        mu = window_prices.mean(axis=0)
        sigma = window_prices.std(axis=0)
        middle[t] = mu
        upper[t] = mu + n_std * sigma
        lower[t] = mu - n_std * sigma

    bandwidth = (upper - lower) / (middle + 1e-12)
    return upper.astype(np.float32), middle.astype(np.float32), lower.astype(np.float32), bandwidth.astype(np.float32)


def compute_average_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Compute Average True Range (ATR).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    atr : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    prev_close = np.roll(close, 1, axis=0)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    atr = np.zeros((T, N), dtype=np.float32)
    for t in range(window, T):
        atr[t] = tr[t - window:t].mean(axis=0)

    return atr


def compute_stochastic_oscillator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple:
    """
    Compute Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    k_window : int
    d_window : int

    Returns
    -------
    pct_k : np.ndarray, shape (T, N)
    pct_d : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    pct_k = np.zeros((T, N), dtype=np.float32)

    for t in range(k_window, T):
        h_max = high[t - k_window:t].max(axis=0)
        l_min = low[t - k_window:t].min(axis=0)
        pct_k[t] = 100.0 * (close[t] - l_min) / (h_max - l_min + 1e-12)

    # %D = simple moving average of %K
    pct_d = np.zeros((T, N), dtype=np.float32)
    for t in range(k_window + d_window, T):
        pct_d[t] = pct_k[t - d_window:t].mean(axis=0)

    return pct_k, pct_d


def compute_on_balance_volume(
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute On Balance Volume (OBV).

    Parameters
    ----------
    close : np.ndarray, shape (T, N)
    volume : np.ndarray, shape (T, N)

    Returns
    -------
    obv : np.ndarray, shape (T, N)
    """
    T, N = close.shape
    obv = np.zeros((T, N), dtype=np.float64)
    obv[0] = volume[0]
    for t in range(1, T):
        direction = np.sign(close[t] - close[t - 1])
        obv[t] = obv[t - 1] + direction * volume[t]
    return obv.astype(np.float32)


def compute_accumulation_distribution(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute Accumulation/Distribution Line.

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    volume : np.ndarray, shape (T, N)

    Returns
    -------
    ad : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    clv = ((close - low) - (high - close)) / (high - low + 1e-12)
    mfv = clv * volume
    ad = np.cumsum(mfv, axis=0)
    return ad.astype(np.float32)


def compute_williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Compute Williams %R oscillator.

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    wr : np.ndarray, shape (T, N)  in [-100, 0]
    """
    T, N = high.shape
    wr = np.zeros((T, N), dtype=np.float32)
    for t in range(window, T):
        h_max = high[t - window:t].max(axis=0)
        l_min = low[t - window:t].min(axis=0)
        wr[t] = -100.0 * (h_max - close[t]) / (h_max - l_min + 1e-12)
    return wr


def compute_commodity_channel_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    constant: float = 0.015,
) -> np.ndarray:
    """
    Compute Commodity Channel Index (CCI).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int
    constant : float

    Returns
    -------
    cci : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    typical = (high + low + close) / 3.0
    cci = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        tp_window = typical[t - window:t]
        mean_tp = tp_window.mean(axis=0)
        mean_dev = np.abs(tp_window - mean_tp).mean(axis=0)
        cci[t] = (typical[t] - mean_tp) / (constant * mean_dev + 1e-12)

    return cci


def build_technical_indicator_tensor(
    prices: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    include_rsi: bool = True,
    include_macd: bool = True,
    include_bb: bool = True,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    bb_window: int = 20,
) -> np.ndarray:
    """
    Construct a tensor of technical indicators from OHLCV data.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
        Close prices.
    high, low : np.ndarray, shape (T, N), optional
        High and low prices.
    volume : np.ndarray, shape (T, N), optional
        Volume data.
    include_rsi, include_macd, include_bb : bool
        Which indicators to include.

    Returns
    -------
    indicator_tensor : np.ndarray, shape (T, N, K)
        Stacked indicator features.
    """
    T, N = prices.shape
    feature_list = []

    if include_rsi:
        rsi = compute_rsi(prices, window=rsi_window)
        feature_list.append(rsi[:, :, None])

    if include_macd:
        macd_line, signal_line, hist = compute_macd(
            prices, fast_window=macd_fast, slow_window=macd_slow
        )
        feature_list.append(macd_line[:, :, None])
        feature_list.append(signal_line[:, :, None])
        feature_list.append(hist[:, :, None])

    if include_bb and high is not None and low is not None:
        upper, middle, lower, bw = compute_bollinger_bands(prices, window=bb_window)
        feature_list.append(bw[:, :, None])
    elif include_bb:
        upper, middle, lower, bw = compute_bollinger_bands(prices, window=bb_window)
        feature_list.append(bw[:, :, None])

    if not feature_list:
        return prices[:, :, None]

    return np.concatenate(feature_list, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Section: Multi-asset return attribution
# ---------------------------------------------------------------------------


def compute_factor_betas(
    returns: np.ndarray,
    factors: np.ndarray,
    window: int = 63,
) -> np.ndarray:
    """
    Rolling OLS factor betas.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    factors : np.ndarray, shape (T, K)
    window : int

    Returns
    -------
    betas : np.ndarray, shape (T, N, K)
        Rolling factor loadings.
    """
    T, N = returns.shape
    K = factors.shape[1]
    betas = np.zeros((T, N, K), dtype=np.float32)

    for t in range(window, T):
        F = factors[t - window:t]       # (window, K)
        R = returns[t - window:t]       # (window, N)
        # OLS: beta = (F^T F)^{-1} F^T R
        FtF = F.T @ F + 1e-6 * np.eye(K)
        FtR = F.T @ R
        beta = np.linalg.solve(FtF, FtR)  # (K, N)
        betas[t] = beta.T  # (N, K)

    return betas


def compute_idiosyncratic_returns(
    returns: np.ndarray,
    factors: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """
    Compute idiosyncratic (residual) returns after removing factor exposures.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    factors : np.ndarray, shape (T, K)
    betas : np.ndarray, shape (T, N, K)

    Returns
    -------
    idio : np.ndarray, shape (T, N)
    """
    T, N, K = betas.shape
    # systematic = sum_k beta_{t,n,k} * f_{t,k}
    systematic = np.einsum("tnk,tk->tn", betas, factors)
    return (returns - systematic).astype(np.float32)


def build_factor_contribution_tensor(
    returns: np.ndarray,
    factors: np.ndarray,
    factor_names: list | None = None,
    window: int = 63,
) -> dict:
    """
    Decompose asset returns into factor contributions + idiosyncratic.

    Returns a dict with:
    * ``"betas"`` : (T, N, K)
    * ``"factor_returns"`` : (T, N, K) = betas * factor_realizations
    * ``"idiosyncratic"`` : (T, N)
    * ``"r_squared"`` : (T, N)
    * ``"factor_names"`` : list of str
    """
    T, N = returns.shape
    K = factors.shape[1]
    betas = compute_factor_betas(returns, factors, window)
    factor_rets = np.einsum("tnk,tk->tnk", betas, factors)
    idio = compute_idiosyncratic_returns(returns, factors, betas)

    # R-squared
    systematic = factor_rets.sum(axis=2)
    ss_tot = np.var(returns, axis=0, keepdims=True).repeat(T, axis=0) + 1e-12
    ss_res = idio ** 2
    r_sq = 1 - ss_res / ss_tot

    return {
        "betas": betas,
        "factor_returns": factor_rets,
        "idiosyncratic": idio,
        "r_squared": r_sq.astype(np.float32),
        "factor_names": factor_names or [f"f{k}" for k in range(K)],
    }


# ---------------------------------------------------------------------------
# Section: Regime-conditioned data splits
# ---------------------------------------------------------------------------


def split_by_regime(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int | None = None,
) -> dict:
    """
    Split return tensor by regime label.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    regime_labels : np.ndarray, shape (T,)
        Integer regime assignments.
    n_regimes : int, optional
        Number of regimes. If None, inferred from labels.

    Returns
    -------
    dict mapping regime_id (int) -> np.ndarray of shape (T_r, N)
    """
    if n_regimes is None:
        n_regimes = int(regime_labels.max()) + 1
    result = {}
    for k in range(n_regimes):
        mask = regime_labels == k
        if mask.any():
            result[k] = returns[mask]
    return result


def regime_conditional_statistics(
    returns: np.ndarray,
    regime_labels: np.ndarray,
) -> dict:
    """
    Compute per-regime return statistics.

    Returns dict mapping regime_id -> dict with ``mean``, ``std``,
    ``sharpe``, ``n_obs``.
    """
    splits = split_by_regime(returns, regime_labels)
    stats = {}
    for k, r in splits.items():
        mu = r.mean(axis=0)
        sigma = r.std(axis=0) + 1e-12
        stats[k] = {
            "mean": mu.astype(np.float32),
            "std": sigma.astype(np.float32),
            "sharpe": (mu / sigma * np.sqrt(252)).astype(np.float32),
            "n_obs": r.shape[0],
        }
    return stats


# ---------------------------------------------------------------------------
# Section: Tensor data augmentation — extended
# ---------------------------------------------------------------------------


def augment_time_warp(
    tensor: np.ndarray,
    warp_factor: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Time-warp augmentation: randomly stretch/compress time axis.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, ...) or (T, N)
    warp_factor : float
        Max fractional change in time axis length.
    rng : np.random.Generator, optional

    Returns
    -------
    warped : np.ndarray, same shape as input
    """
    if rng is None:
        rng = np.random.default_rng()
    T = tensor.shape[0]
    new_T = int(T * (1.0 + rng.uniform(-warp_factor, warp_factor)))
    new_T = max(2, new_T)
    from scipy.interpolate import interp1d
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_T)
    shape = tensor.shape
    flat = tensor.reshape(T, -1)
    fn = interp1d(old_idx, flat, axis=0, kind="linear", fill_value="extrapolate")
    warped_flat = fn(new_idx)
    # Resize back to original T
    fn2 = interp1d(new_idx, warped_flat, axis=0, kind="linear", fill_value="extrapolate")
    result = fn2(old_idx)
    return result.reshape(shape).astype(np.float32)


def augment_magnitude_warp(
    tensor: np.ndarray,
    warp_std: float = 0.05,
    n_knots: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Magnitude-warp augmentation: smooth random multiplicative distortion.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N)
    warp_std : float
        Standard deviation of warp noise at knots.
    n_knots : int
        Number of cubic spline knots.
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    T, N = tensor.shape
    knot_locs = np.linspace(0, T - 1, n_knots)
    warp = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        knot_vals = 1.0 + rng.normal(0, warp_std, n_knots)
        warp[:, n] = np.interp(np.arange(T), knot_locs, knot_vals)
    return (tensor * warp).astype(np.float32)


def augment_window_slice(
    tensor: np.ndarray,
    reduce_ratio: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Window-slice augmentation: extract a random sub-window and resize back.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N)
    reduce_ratio : float
        Fraction of T to sample.
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    T, N = tensor.shape
    window_size = max(2, int(T * reduce_ratio))
    start = rng.integers(0, T - window_size + 1)
    sliced = tensor[start:start + window_size]
    # Resize back to T via interpolation
    old_idx = np.linspace(0, 1, window_size)
    new_idx = np.linspace(0, 1, T)
    result = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        result[:, n] = np.interp(new_idx, old_idx, sliced[:, n])
    return result


def augment_jitter_and_scale(
    tensor: np.ndarray,
    jitter_std: float = 0.001,
    scale_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Combined jitter (additive noise) and scaling augmentation.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N) or (T, N, M)
    jitter_std : float
        Standard deviation of additive noise.
    scale_std : float
        Standard deviation of multiplicative scale (applied per-asset).
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    jitter = rng.normal(0, jitter_std, tensor.shape).astype(np.float32)
    scale = rng.normal(1.0, scale_std, tensor.shape[1:]).astype(np.float32)
    return ((tensor + jitter) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Section: Cross-asset tensor construction
# ---------------------------------------------------------------------------


def build_cross_asset_return_tensor(
    returns_dict: dict,
    asset_universe: list,
    T: int,
) -> np.ndarray:
    """
    Stack multiple asset return series into a unified (T, N) tensor.

    Parameters
    ----------
    returns_dict : dict mapping asset_id (str) -> np.ndarray, shape (T_i,)
    asset_universe : list of str
        Ordered list of asset IDs.
    T : int
        Target number of time steps (most recent).

    Returns
    -------
    tensor : np.ndarray, shape (T, N)
        NaN-filled where data is missing.
    """
    N = len(asset_universe)
    tensor = np.full((T, N), np.nan, dtype=np.float32)
    for n, asset_id in enumerate(asset_universe):
        if asset_id in returns_dict:
            r = np.array(returns_dict[asset_id], dtype=np.float32)
            length = min(T, len(r))
            tensor[-length:, n] = r[-length:]
    return tensor


def align_multi_frequency_returns(
    daily_returns: np.ndarray,
    weekly_returns: np.ndarray,
    monthly_returns: np.ndarray,
) -> np.ndarray:
    """
    Align daily, weekly, and monthly return series into a 3-D tensor.

    Parameters
    ----------
    daily_returns : np.ndarray, shape (T, N)
    weekly_returns : np.ndarray, shape (T//5, N)
    monthly_returns : np.ndarray, shape (T//21, N)

    Returns
    -------
    tensor : np.ndarray, shape (T, N, 3)
    """
    T, N = daily_returns.shape
    tensor = np.zeros((T, N, 3), dtype=np.float32)
    tensor[:, :, 0] = daily_returns

    # Upsample weekly to daily
    T_w = weekly_returns.shape[0]
    idx_w = np.linspace(0, T - 1, T_w)
    for n in range(N):
        tensor[:, n, 1] = np.interp(np.arange(T), idx_w, weekly_returns[:, n])

    # Upsample monthly to daily
    T_m = monthly_returns.shape[0]
    idx_m = np.linspace(0, T - 1, T_m)
    for n in range(N):
        tensor[:, n, 2] = np.interp(np.arange(T), idx_m, monthly_returns[:, n])

    return tensor


def build_sector_return_tensor(
    returns: np.ndarray,
    sector_ids: np.ndarray,
    n_sectors: int | None = None,
) -> np.ndarray:
    """
    Compute sector-average return tensor.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    sector_ids : np.ndarray, shape (N,)
        Integer sector labels.
    n_sectors : int, optional

    Returns
    -------
    sector_returns : np.ndarray, shape (T, n_sectors)
    """
    T, N = returns.shape
    if n_sectors is None:
        n_sectors = int(sector_ids.max()) + 1
    sector_ret = np.zeros((T, n_sectors), dtype=np.float32)
    for k in range(n_sectors):
        mask = sector_ids == k
        if mask.any():
            sector_ret[:, k] = returns[:, mask].mean(axis=1)
    return sector_ret


# ---------------------------------------------------------------------------
# Section: Data quality utilities
# ---------------------------------------------------------------------------


def detect_price_jumps(
    prices: np.ndarray,
    threshold: float = 5.0,
    window: int = 20,
) -> np.ndarray:
    """
    Detect abnormal price jumps (returns > threshold * rolling std).

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    threshold : float
        Z-score threshold for jump detection.
    window : int

    Returns
    -------
    jump_mask : np.ndarray, shape (T, N)  bool
    """
    T, N = prices.shape
    log_ret = np.diff(np.log(np.abs(prices) + 1e-12), axis=0)
    jump_mask = np.zeros((T, N), dtype=bool)

    for t in range(window, T - 1):
        hist_ret = log_ret[t - window:t]
        sigma = hist_ret.std(axis=0) + 1e-12
        z = np.abs(log_ret[t]) / sigma
        jump_mask[t + 1] = z > threshold

    return jump_mask


def fill_price_jumps(
    prices: np.ndarray,
    jump_mask: np.ndarray,
    method: str = "interpolate",
) -> np.ndarray:
    """
    Fill detected price jumps.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    jump_mask : np.ndarray, shape (T, N)  bool
    method : str
        "interpolate" | "prev" | "median"

    Returns
    -------
    filled : np.ndarray, shape (T, N)
    """
    T, N = prices.shape
    filled = prices.copy()

    for n in range(N):
        jump_idx = np.where(jump_mask[:, n])[0]
        if len(jump_idx) == 0:
            continue
        if method == "prev":
            for t in jump_idx:
                if t > 0:
                    filled[t, n] = filled[t - 1, n]
        elif method == "interpolate":
            for t in jump_idx:
                t_prev = t - 1 if t > 0 else 0
                t_next = t + 1 if t < T - 1 else T - 1
                filled[t, n] = (filled[t_prev, n] + filled[t_next, n]) / 2.0
        elif method == "median":
            rolling_median = np.median(prices[:, n])
            for t in jump_idx:
                filled[t, n] = rolling_median

    return filled.astype(np.float32)


def compute_data_quality_report(returns: np.ndarray) -> dict:
    """
    Comprehensive data quality report for a return tensor.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)

    Returns
    -------
    dict with quality metrics per asset and overall summary.
    """
    T, N = returns.shape
    nan_frac = np.isnan(returns).mean(axis=0)
    inf_frac = np.isinf(returns).mean(axis=0)
    mean_ret = np.nanmean(returns, axis=0)
    std_ret = np.nanstd(returns, axis=0)
    min_ret = np.nanmin(returns, axis=0)
    max_ret = np.nanmax(returns, axis=0)
    skew = (np.nanmean((returns - mean_ret) ** 3, axis=0) /
            (std_ret ** 3 + 1e-12))
    kurt = (np.nanmean((returns - mean_ret) ** 4, axis=0) /
            (std_ret ** 4 + 1e-12)) - 3

    return {
        "T": T,
        "N": N,
        "overall_nan_frac": float(nan_frac.mean()),
        "overall_inf_frac": float(inf_frac.mean()),
        "assets_with_nans": int((nan_frac > 0).sum()),
        "per_asset": {
            "nan_frac": nan_frac.tolist(),
            "inf_frac": inf_frac.tolist(),
            "mean": mean_ret.tolist(),
            "std": std_ret.tolist(),
            "min": min_ret.tolist(),
            "max": max_ret.tolist(),
            "skewness": skew.tolist(),
            "excess_kurtosis": kurt.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Section: Pipeline orchestration helpers
# ---------------------------------------------------------------------------


class DataPipelineOrchestrator:
    """
    High-level orchestrator that chains multiple pipeline steps.

    Steps are registered as callables and executed in order.

    Usage::

        orch = DataPipelineOrchestrator()
        orch.register("load", lambda: load_prices())
        orch.register("returns", lambda x: prices_to_returns(x))
        orch.register("normalise", lambda x: (x - x.mean()) / x.std())
        result = orch.run()
    """

    def __init__(self) -> None:
        self._steps: list = []   # list of (name, fn)
        self._outputs: dict = {}
        self._timings: dict = {}

    def register(self, name: str, fn, depends_on: str | None = None) -> None:
        """Register a pipeline step."""
        self._steps.append((name, fn, depends_on))

    def run(self, initial_input=None) -> dict:
        """
        Execute all registered steps in order.

        Returns dict of step_name -> output.
        """
        import time
        current = initial_input
        for name, fn, depends_on in self._steps:
            if depends_on is not None:
                inp = self._outputs.get(depends_on, current)
            else:
                inp = current
            t0 = time.monotonic()
            if inp is None:
                output = fn()
            else:
                output = fn(inp)
            self._timings[name] = time.monotonic() - t0
            self._outputs[name] = output
            current = output
        return self._outputs

    def get_output(self, step_name: str):
        """Return output of a specific step."""
        return self._outputs.get(step_name)

    def timing_report(self) -> dict:
        """Return timing info for each step."""
        return dict(self._timings)

    def reset(self) -> None:
        self._outputs = {}
        self._timings = {}


class DataPipelineCache:
    """
    Simple on-disk cache for pipeline outputs using numpy .npz format.

    Parameters
    ----------
    cache_dir : str
        Directory for cached files.
    """

    def __init__(self, cache_dir: str = "/tmp/tensor_net_cache") -> None:
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        import os
        return os.path.join(self.cache_dir, f"{key}.npz")

    def exists(self, key: str) -> bool:
        import os
        return os.path.exists(self._path(key))

    def save(self, key: str, arrays: dict) -> None:
        np.savez_compressed(self._path(key), **arrays)

    def load(self, key: str) -> dict:
        data = np.load(self._path(key))
        return dict(data)

    def delete(self, key: str) -> None:
        import os
        p = self._path(key)
        if os.path.exists(p):
            os.remove(p)

    def list_keys(self) -> list:
        import os
        return [
            f.replace(".npz", "")
            for f in os.listdir(self.cache_dir)
            if f.endswith(".npz")
        ]

