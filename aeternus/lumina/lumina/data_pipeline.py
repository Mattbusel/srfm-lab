"""
lumina/data_pipeline.py

Data pipeline for Lumina Financial Foundation Model:

  - FinancialDataset          : base OHLCV dataset
  - MultiAssetDataset         : multi-asset aligned dataset
  - TimeSeriesAugmenter       : time warping, magnitude scaling, jitter, cutout
  - NormalizationStatistics   : per-feature normalization stats
  - WindowedDataset           : sliding window dataset
  - DataLoaderFactory         : builds train/val/test loaders
  - OnlineDataStreamer         : streaming data interface (for live trading)
  - FinancialDataModule        : PyTorch Lightning-compatible data module
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Data
    data_dir:           str   = "./data"
    assets:             List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    freq:               str   = "1h"       # "1m" | "5m" | "1h" | "1d"
    n_channels:         int   = 5         # OHLCV
    lookback:           int   = 256
    horizon:            int   = 1

    # Train/val/test split
    train_frac:         float = 0.7
    val_frac:           float = 0.15
    test_frac:          float = 0.15

    # Normalization
    norm_mode:          str   = "zscore"  # "zscore" | "minmax" | "returns" | "robust"
    norm_window:        int   = 0         # 0 = global, >0 = rolling window

    # Augmentation
    use_augmentation:   bool  = True
    jitter_std:         float = 0.01
    scale_range:        Tuple[float, float] = (0.9, 1.1)
    time_warp_sigma:    float = 0.2
    cutout_n_holes:     int   = 1
    cutout_max_len:     int   = 8
    aug_prob:           float = 0.5      # probability of applying each augmentation

    # DataLoader
    batch_size:         int   = 32
    num_workers:        int   = 4
    prefetch_factor:    int   = 2
    pin_memory:         bool  = True
    drop_last:          bool  = True

    # Cache
    use_cache:          bool  = True
    cache_dir:          str   = "./cache/lumina"


# ---------------------------------------------------------------------------
# Normalization Statistics
# ---------------------------------------------------------------------------

class NormalizationStatistics:
    """Stores and applies per-feature normalization statistics."""

    def __init__(self, mode: str = "zscore"):
        self.mode   = mode
        self.params: Dict[str, np.ndarray] = {}

    def fit(self, data: np.ndarray) -> None:
        """Fit normalization to data. data: (T, F)."""
        if self.mode == "zscore":
            self.params["mean"] = data.mean(axis=0)
            self.params["std"]  = data.std(axis=0).clip(1e-8)
        elif self.mode == "minmax":
            self.params["min"]  = data.min(axis=0)
            self.params["max"]  = data.max(axis=0)
            self.params["range"] = (self.params["max"] - self.params["min"]).clip(1e-8)
        elif self.mode == "robust":
            self.params["median"] = np.median(data, axis=0)
            q75, q25              = np.percentile(data, [75, 25], axis=0)
            self.params["iqr"]    = (q75 - q25).clip(1e-8)
        elif self.mode == "returns":
            # Convert to returns before normalizing
            pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization. data: (..., F)."""
        if self.mode == "zscore":
            return (data - self.params["mean"]) / self.params["std"]
        elif self.mode == "minmax":
            return (data - self.params["min"]) / self.params["range"]
        elif self.mode == "robust":
            return (data - self.params["median"]) / self.params["iqr"]
        elif self.mode == "returns":
            returns       = np.zeros_like(data)
            returns[1:]   = (data[1:] - data[:-1]) / (data[:-1] + 1e-8)
            mu            = returns.mean(axis=0)
            std           = returns.std(axis=0).clip(1e-8)
            return (returns - mu) / std
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert normalization."""
        if self.mode == "zscore":
            return data * self.params["std"] + self.params["mean"]
        elif self.mode == "minmax":
            return data * self.params["range"] + self.params["min"]
        elif self.mode == "robust":
            return data * self.params["iqr"] + self.params["median"]
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def save(self, path: Union[str, Path]) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({"mode": self.mode, "params": self.params}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormalizationStatistics":
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj        = cls(data["mode"])
        obj.params = data["params"]
        return obj


# ---------------------------------------------------------------------------
# Time Series Augmentation
# ---------------------------------------------------------------------------

class TimeSeriesAugmenter:
    """
    Data augmentation for financial time series.

    Augmentations:
      - Gaussian jitter: adds noise to each channel
      - Magnitude scaling: scales amplitude randomly
      - Time warping: smooth nonlinear time distortion
      - Window slicing: crop random subwindow
      - Magnitude warp: smooth amplitude distortion
      - Cutout: random time intervals zeroed out
      - Frequency domain augmentation: mask random frequency components
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def jitter(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to each feature."""
        noise = np.random.randn(*x.shape) * self.cfg.jitter_std
        return x + noise

    def magnitude_scale(self, x: np.ndarray) -> np.ndarray:
        """Random amplitude scaling (applied to OHLC, not volume)."""
        lo, hi = self.cfg.scale_range
        scale  = np.random.uniform(lo, hi)
        out    = x.copy()
        out[:, :4] = x[:, :4] * scale   # OHLC only
        return out

    def time_warp(self, x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Smooth time warping: distort the time axis with a smooth random warp.
        Uses cubic interpolation of random warp points.
        """
        from scipy.interpolate import CubicSpline
        sigma = sigma or self.cfg.time_warp_sigma
        T, C  = x.shape

        # Random warp anchors
        n_anchors  = max(3, T // 50)
        orig_steps = np.linspace(0, T - 1, n_anchors)
        warp_steps = orig_steps + np.random.randn(n_anchors) * sigma * T / n_anchors
        warp_steps = np.clip(np.sort(warp_steps), 0, T - 1)
        # Ensure endpoints are fixed
        warp_steps[0]  = 0
        warp_steps[-1] = T - 1

        cs          = CubicSpline(orig_steps, warp_steps)
        new_steps   = cs(np.arange(T))
        new_steps   = np.clip(new_steps, 0, T - 1)

        # Interpolate each channel
        out = np.zeros_like(x)
        for c in range(C):
            out[:, c] = np.interp(np.arange(T), new_steps, x[:, c])
        return out

    def magnitude_warp(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Smooth random amplitude distortion per channel."""
        from scipy.interpolate import CubicSpline
        T, C = x.shape

        n_knots = max(3, T // 30)
        knots   = np.linspace(0, T - 1, n_knots)
        out     = x.copy()

        for c in range(C):
            warp_vals = 1.0 + np.random.randn(n_knots) * sigma
            cs        = CubicSpline(knots, warp_vals)
            warp      = cs(np.arange(T)).clip(0.1, 3.0)
            out[:, c] = x[:, c] * warp

        return out

    def window_slice(self, x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Randomly crop a subwindow and resize back."""
        T, C     = x.shape
        target_T = max(3, int(T * reduce_ratio))
        start    = np.random.randint(0, T - target_T + 1)
        sliced   = x[start:start + target_T, :]

        # Resample back to T using interpolation
        out = np.zeros_like(x)
        for c in range(C):
            out[:, c] = np.interp(
                np.linspace(0, target_T - 1, T),
                np.arange(target_T),
                sliced[:, c]
            )
        return out

    def cutout(self, x: np.ndarray) -> np.ndarray:
        """Zero out random time intervals."""
        out = x.copy()
        T   = x.shape[0]
        for _ in range(self.cfg.cutout_n_holes):
            max_len = min(self.cfg.cutout_max_len, T // 4)
            length  = np.random.randint(1, max_len + 1)
            start   = np.random.randint(0, T - length + 1)
            out[start:start + length, :] = 0.0
        return out

    def frequency_mask(self, x: np.ndarray, mask_frac: float = 0.1) -> np.ndarray:
        """Mask random frequency components in each channel."""
        out = x.copy()
        T   = x.shape[0]
        n_mask = max(1, int(T // 2 * mask_frac))

        for c in range(x.shape[1]):
            freq  = np.fft.rfft(x[:, c])
            idx   = np.random.choice(len(freq), n_mask, replace=False)
            freq[idx] = 0
            out[:, c] = np.fft.irfft(freq, n=T)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations with probability aug_prob."""
        if np.random.rand() > self.cfg.aug_prob:
            return x

        aug_fns = [
            self.jitter,
            self.magnitude_scale,
        ]

        # Heavier augmentations with lower probability
        if np.random.rand() < 0.3:
            try:
                x = self.time_warp(x)
            except Exception:
                pass

        if np.random.rand() < 0.3:
            try:
                x = self.magnitude_warp(x)
            except Exception:
                pass

        if np.random.rand() < 0.2:
            x = self.cutout(x)

        for fn in aug_fns:
            if np.random.rand() < 0.5:
                x = fn(x)

        return x


# ---------------------------------------------------------------------------
# Base Financial Dataset
# ---------------------------------------------------------------------------

class FinancialDataset(Dataset):
    """
    Sliding window dataset over a single-asset OHLCV series.

    Returns windows of shape (lookback, n_channels) with optional labels.
    """

    def __init__(
        self,
        ohlcv:       np.ndarray,           # (T, n_channels)
        cfg:         DataConfig,
        split:       str = "train",
        augmenter:   Optional[TimeSeriesAugmenter] = None,
        norm_stats:  Optional[NormalizationStatistics] = None,
        timestamps:  Optional[np.ndarray] = None,
    ):
        self.cfg        = cfg
        self.split      = split
        self.augmenter  = augmenter
        self.timestamps = timestamps

        # Normalize
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = NormalizationStatistics(cfg.norm_mode)
            self.norm_stats.fit(ohlcv)

        self.ohlcv = self.norm_stats.transform(ohlcv).astype(np.float32)

        # Build index of valid windows
        T = len(self.ohlcv)
        L = cfg.lookback
        H = cfg.horizon

        # Split by time
        train_end = int(T * cfg.train_frac)
        val_end   = int(T * (cfg.train_frac + cfg.val_frac))

        if split == "train":
            start, end = 0, train_end
        elif split == "val":
            start, end = train_end, val_end
        else:  # test
            start, end = val_end, T

        self.indices = list(range(start, end - L - H + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i    = self.indices[idx]
        x    = self.ohlcv[i: i + self.cfg.lookback].copy()
        item = {}

        if self.split == "train" and self.augmenter is not None:
            x = self.augmenter(x)

        item["ohlcv"] = torch.from_numpy(x).float()

        if self.timestamps is not None:
            item["timestamps"] = torch.from_numpy(
                self.timestamps[i: i + self.cfg.lookback].astype(np.float32)
            )

        return item


class LabeledFinancialDataset(FinancialDataset):
    """Financial dataset with automatic label generation."""

    def __init__(self, ohlcv: np.ndarray, cfg: DataConfig, task: str = "direction", **kwargs):
        super().__init__(ohlcv, cfg, **kwargs)
        self.task   = task
        self.labels = self._compute_labels(ohlcv)

    def _compute_labels(self, ohlcv: np.ndarray) -> np.ndarray:
        close   = ohlcv[:, 3]
        T       = len(ohlcv)
        H       = self.cfg.horizon
        labels  = np.zeros(T)

        if self.task == "direction":
            ret    = np.zeros(T)
            ret[:T - H] = (close[H:] - close[:T - H]) / (close[:T - H] + 1e-8)
            labels = np.where(ret > 0.005, 0, np.where(ret < -0.005, 1, 2))
        elif self.task == "return":
            labels[:T - H] = (close[H:] - close[:T - H]) / (close[:T - H] + 1e-8)
        elif self.task == "volatility":
            ret = np.diff(np.log(close + 1e-8))
            for t in range(H, T):
                labels[t] = ret[max(0, t - H):t].std() * np.sqrt(252)

        return labels.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        i    = self.indices[idx]
        y    = self.labels[i + self.cfg.lookback]

        if self.task == "direction":
            item["y"] = torch.tensor(int(y), dtype=torch.long)
        else:
            item["y"] = torch.tensor(y, dtype=torch.float)

        return item


# ---------------------------------------------------------------------------
# Multi-Asset Dataset
# ---------------------------------------------------------------------------

class MultiAssetDataset(Dataset):
    """
    Dataset for multiple synchronized asset series.
    Handles different-length series by padding to shortest or longest.
    """

    def __init__(
        self,
        ohlcv_dict:  Dict[str, np.ndarray],   # {asset_name: (T, 5) array}
        cfg:         DataConfig,
        split:       str = "train",
        augmenter:   Optional[TimeSeriesAugmenter] = None,
    ):
        self.cfg        = cfg
        self.split      = split
        self.augmenter  = augmenter
        self.asset_names = list(ohlcv_dict.keys())

        # Normalize each asset independently
        self.norm_stats: Dict[str, NormalizationStatistics] = {}
        self.data: Dict[str, np.ndarray] = {}

        T_min = min(v.shape[0] for v in ohlcv_dict.values())

        for name, ohlcv in ohlcv_dict.items():
            ns = NormalizationStatistics(cfg.norm_mode)
            ns.fit(ohlcv[:T_min])
            self.norm_stats[name] = ns
            self.data[name]       = ns.transform(ohlcv[:T_min]).astype(np.float32)

        # Build common index
        T = T_min
        L = cfg.lookback
        H = cfg.horizon

        train_end = int(T * cfg.train_frac)
        val_end   = int(T * (cfg.train_frac + cfg.val_frac))

        if split == "train":
            start, end = 0, train_end
        elif split == "val":
            start, end = train_end, val_end
        else:
            start, end = val_end, T

        self.indices = list(range(start, end - L - H + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i   = self.indices[idx]
        L   = self.cfg.lookback
        out = {}

        for j, name in enumerate(self.asset_names):
            x = self.data[name][i: i + L].copy()
            if self.split == "train" and self.augmenter is not None:
                x = self.augmenter(x)
            out[f"ohlcv_{name}"] = torch.from_numpy(x).float()

        out["asset_names"] = self.asset_names
        out["idx"]         = i
        return out

    def get_collate_fn(self) -> Callable:
        """Returns a collate function that stacks multi-asset data."""
        asset_names = self.asset_names

        def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            out = {}
            for name in asset_names:
                key = f"ohlcv_{name}"
                out[key] = torch.stack([b[key] for b in batch])
            return out

        return collate


# ---------------------------------------------------------------------------
# Streaming Dataset (for live inference)
# ---------------------------------------------------------------------------

class StreamingOHLCVDataset(IterableDataset):
    """
    Streaming dataset that reads OHLCV data on-the-fly.
    Suitable for real-time inference in a trading system.
    """

    def __init__(
        self,
        data_source:  Any,               # callable returning batches
        lookback:     int  = 256,
        batch_size:   int  = 32,
        max_batches:  Optional[int] = None,
    ):
        self.data_source = data_source
        self.lookback    = lookback
        self.batch_size  = batch_size
        self.max_batches = max_batches

    def __iter__(self) -> Iterator:
        buffer = []
        count  = 0

        for ohlcv_bar in self.data_source:
            buffer.append(ohlcv_bar)
            if len(buffer) >= self.lookback:
                window = np.array(buffer[-self.lookback:]).astype(np.float32)
                yield {"ohlcv": torch.from_numpy(window).float().unsqueeze(0)}
                count += 1
                if self.max_batches is not None and count >= self.max_batches:
                    break


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

class DataLoaderFactory:
    """Factory for building PyTorch DataLoaders with proper configuration."""

    @staticmethod
    def build(
        dataset:      Dataset,
        cfg:          DataConfig,
        split:        str = "train",
        collate_fn:   Optional[Callable] = None,
    ) -> DataLoader:
        is_train = split == "train"
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            drop_last=cfg.drop_last and is_train,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=collate_fn,
        )

    @staticmethod
    def build_all(
        ohlcv:       np.ndarray,
        cfg:         DataConfig,
        task:        str = "pretrain",
        timestamps:  Optional[np.ndarray] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build train, val, test loaders from a single OHLCV array."""
        augmenter = TimeSeriesAugmenter(cfg) if cfg.use_augmentation else None

        if task == "pretrain":
            train_ds = FinancialDataset(ohlcv, cfg, "train", augmenter, timestamps=timestamps)
            val_ds   = FinancialDataset(ohlcv, cfg, "val",   None,      timestamps=timestamps)
            test_ds  = FinancialDataset(ohlcv, cfg, "test",  None,      timestamps=timestamps)
        else:
            train_ds = LabeledFinancialDataset(ohlcv, cfg, task, split="train", augmenter=augmenter)
            val_ds   = LabeledFinancialDataset(ohlcv, cfg, task, split="val",   augmenter=None)
            test_ds  = LabeledFinancialDataset(ohlcv, cfg, task, split="test",  augmenter=None)

        train_loader = DataLoaderFactory.build(train_ds, cfg, "train")
        val_loader   = DataLoaderFactory.build(val_ds,   cfg, "val")
        test_loader  = DataLoaderFactory.build(test_ds,  cfg, "test")

        return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Synthetic data generation (for testing)
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    T:         int  = 10000,
    n_assets:  int  = 1,
    freq:      str  = "1h",
    trend:     float = 0.0001,
    vol:       float = 0.01,
    seed:      int   = 42,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate synthetic OHLCV data using a geometric Brownian motion model.
    Returns (T, 5) array or list of arrays for multiple assets.
    """
    np.random.seed(seed)

    def _gen_single(start_price: float = 100.0) -> np.ndarray:
        prices = np.zeros(T + 1)
        prices[0] = start_price

        for t in range(1, T + 1):
            shock     = np.random.randn() * vol
            prices[t] = prices[t - 1] * np.exp(trend + shock)

        ohlcv = np.zeros((T, 5))
        for t in range(T):
            o = prices[t]
            c = prices[t + 1]
            h = max(o, c) * (1 + abs(np.random.randn()) * vol * 0.5)
            l = min(o, c) * (1 - abs(np.random.randn()) * vol * 0.5)
            v = abs(np.random.randn()) * 1000 + 500

            ohlcv[t] = [o, h, l, c, v]

        return ohlcv.astype(np.float32)

    if n_assets == 1:
        return _gen_single()

    assets = []
    for i in range(n_assets):
        start = 10.0 * (i + 1) * 10 ** np.random.uniform(0, 2)
        assets.append(_gen_single(start_price=start))
    return assets


def generate_correlated_assets(
    n_assets:    int   = 5,
    T:           int   = 10000,
    base_corr:   float = 0.6,
    vol:         float = 0.01,
    seed:        int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated synthetic asset returns.
    Returns (T, n_assets) returns array and (n_assets, n_assets) correlation matrix.
    """
    np.random.seed(seed)

    # Build correlation matrix
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    # Add individual idiosyncratic variation
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                corr[i, j] += np.random.uniform(-0.2, 0.2)
    corr = np.clip(corr, -0.95, 0.95)
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition for correlated returns
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Make positive definite
        eigvals = np.linalg.eigvalsh(corr)
        corr    = corr + np.eye(n_assets) * (abs(min(eigvals)) + 0.01)
        L       = np.linalg.cholesky(corr)

    # Generate correlated returns
    uncorr  = np.random.randn(T, n_assets) * vol
    returns = uncorr @ L.T

    return returns.astype(np.float32), corr.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch collation utilities
# ---------------------------------------------------------------------------

def collate_financial_batch(
    batch: List[Dict[str, Any]],
    pad_to: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for financial data batches.
    Handles variable-length sequences and optional padding.
    """
    out     = {}
    keys    = batch[0].keys()

    for key in keys:
        if key == "asset_names":
            out[key] = batch[0][key]
            continue
        values = [b[key] for b in batch if key in b]
        if not values:
            continue

        if isinstance(values[0], torch.Tensor):
            if all(v.shape == values[0].shape for v in values):
                out[key] = torch.stack(values)
            else:
                # Pad to max length
                max_len  = max(v.shape[0] for v in values)
                padded   = []
                for v in values:
                    if v.shape[0] < max_len:
                        pad = torch.zeros(max_len - v.shape[0], *v.shape[1:])
                        v   = torch.cat([v, pad], dim=0)
                    padded.append(v)
                out[key] = torch.stack(padded)
        else:
            out[key] = values

    return out


# ---------------------------------------------------------------------------
# Data Module (PyTorch Lightning compatible)
# ---------------------------------------------------------------------------

class LuminaDataModule:
    """
    Data module that manages datasets and loaders for Lumina training.
    Compatible with PyTorch Lightning DataModule interface.
    """

    def __init__(
        self,
        cfg:        DataConfig,
        data:       Optional[Dict[str, np.ndarray]] = None,
        task:       str = "pretrain",
    ):
        self.cfg  = cfg
        self.data = data or {}
        self.task = task

        self.train_loader: Optional[DataLoader] = None
        self.val_loader:   Optional[DataLoader] = None
        self.test_loader:  Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets for given stage."""
        augmenter = TimeSeriesAugmenter(self.cfg) if self.cfg.use_augmentation else None

        if "main" in self.data:
            ohlcv = self.data["main"]
            if self.task == "pretrain":
                train_ds = FinancialDataset(ohlcv, self.cfg, "train", augmenter)
                val_ds   = FinancialDataset(ohlcv, self.cfg, "val",   None)
                test_ds  = FinancialDataset(ohlcv, self.cfg, "test",  None)
            else:
                train_ds = LabeledFinancialDataset(ohlcv, self.cfg, self.task, split="train", augmenter=augmenter)
                val_ds   = LabeledFinancialDataset(ohlcv, self.cfg, self.task, split="val",   augmenter=None)
                test_ds  = LabeledFinancialDataset(ohlcv, self.cfg, self.task, split="test",  augmenter=None)

            self.train_loader = DataLoaderFactory.build(train_ds, self.cfg, "train")
            self.val_loader   = DataLoaderFactory.build(val_ds,   self.cfg, "val")
            self.test_loader  = DataLoaderFactory.build(test_ds,  self.cfg, "test")

    def train_dataloader(self) -> DataLoader:
        assert self.train_loader is not None, "Call setup() first"
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        assert self.val_loader is not None, "Call setup() first"
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        assert self.test_loader is not None, "Call setup() first"
        return self.test_loader

    @property
    def n_train_batches(self) -> int:
        if self.train_loader is None:
            return 0
        return len(self.train_loader)


# ---------------------------------------------------------------------------
# Technical Indicator Computation
# ---------------------------------------------------------------------------

class TechnicalIndicators:
    """Collection of technical analysis indicators for financial time series.

    All indicators operate on raw OHLCV data (pandas-style numpy arrays).
    Methods return numpy arrays of the same length as input (with NaN padding).

    Available indicators:
    - Trend:       SMA, EMA, DEMA, TEMA, WMA, HMA, KAMA
    - Momentum:    RSI, MACD, Stochastic, ROC, MFI
    - Volatility:  ATR, Bollinger Bands, Keltner Channels, NATR
    - Volume:      OBV, VWAP, CMF, Force Index
    - Pattern:     Doji, Hammer, Engulfing (simplified)

    Example:
        >>> close = np.random.randn(200).cumsum() + 100
        >>> rsi = TechnicalIndicators.rsi(close, period=14)
        >>> macd_line, signal, hist = TechnicalIndicators.macd(close)
    """

    @staticmethod
    def sma(values: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average.

        SMA(t) = mean(values[t-period+1 : t+1])

        Args:
            values: (T,) price array
            period: lookback period

        Returns:
            sma: (T,) array with NaN for first period-1 values
        """
        result = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            result[i] = values[i - period + 1:i + 1].mean()
        return result

    @staticmethod
    def ema(values: np.ndarray, period: int, smoothing: float = 2.0) -> np.ndarray:
        """Exponential Moving Average.

        EMA(t) = price * k + EMA(t-1) * (1-k)
        where k = smoothing / (period + 1)

        Args:
            values:    (T,) price array
            period:    lookback period
            smoothing: smoothing factor (2.0 = standard EMA)

        Returns:
            ema: (T,) array
        """
        k = smoothing / (period + 1)
        result = np.full_like(values, np.nan)
        # First valid EMA = SMA of first period values
        result[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def dema(values: np.ndarray, period: int) -> np.ndarray:
        """Double Exponential Moving Average.

        DEMA = 2 * EMA(period) - EMA(EMA(period))

        Reduces lag compared to single EMA.

        Args:
            values: (T,) price array
            period: period

        Returns:
            dema: (T,) array
        """
        ema1 = TechnicalIndicators.ema(values, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        return 2 * ema1 - ema2

    @staticmethod
    def wma(values: np.ndarray, period: int) -> np.ndarray:
        """Weighted Moving Average (linearly weighted).

        WMA(t) = sum(values[t-i] * (period-i) for i in 0..period-1) / sum(1..period)

        Args:
            values: (T,) price array
            period: lookback period

        Returns:
            wma: (T,) array with NaN for first period-1 values
        """
        weights = np.arange(1, period + 1, dtype=np.float64)
        total_weight = weights.sum()
        result = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            result[i] = (window * weights).sum() / total_weight
        return result

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index.

        RSI = 100 - 100 / (1 + RS)
        RS = avg_gain / avg_loss over period bars

        Args:
            close:  (T,) closing prices
            period: RSI period (typically 14)

        Returns:
            rsi: (T,) array, values 0-100
        """
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        result = np.full_like(close, np.nan)
        # Wilder's smoothed MA
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        for i in range(period, len(close)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs = avg_gain / (avg_loss + 1e-10)
            result[i] = 100.0 - 100.0 / (1.0 + rs)

        return result

    @staticmethod
    def macd(
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD: Moving Average Convergence/Divergence.

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line

        Args:
            close:  (T,) closing prices
            fast:   fast EMA period (typically 12)
            slow:   slow EMA period (typically 26)
            signal: signal EMA period (typically 9)

        Returns:
            macd_line: (T,) MACD line values
            signal_line: (T,) signal line values
            histogram: (T,) histogram values
        """
        ema_fast = TechnicalIndicators.ema(close, fast)
        ema_slow = TechnicalIndicators.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(
            np.where(np.isnan(macd_line), 0.0, macd_line), signal
        )
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Average True Range.

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = EMA(True Range, period)

        Args:
            high:   (T,) high prices
            low:    (T,) low prices
            close:  (T,) closing prices
            period: ATR period (typically 14)

        Returns:
            atr: (T,) ATR values
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        return TechnicalIndicators.ema(tr, period)

    @staticmethod
    def bollinger_bands(
        close: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands.

        Middle Band = SMA(close, period)
        Upper Band = Middle + num_std * StdDev(close, period)
        Lower Band = Middle - num_std * StdDev(close, period)

        Args:
            close:   (T,) closing prices
            period:  lookback period (typically 20)
            num_std: number of standard deviations (typically 2.0)

        Returns:
            upper_band: (T,) upper band
            middle_band: (T,) middle band (SMA)
            lower_band: (T,) lower band
        """
        middle = TechnicalIndicators.sma(close, period)
        std = np.full_like(close, np.nan)
        for i in range(period - 1, len(close)):
            std[i] = close[i - period + 1:i + 1].std()

        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def keltner_channels(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels.

        Middle = EMA(close, period)
        Upper = Middle + multiplier * ATR(atr_period)
        Lower = Middle - multiplier * ATR(atr_period)

        Args:
            high, low, close: OHLC arrays
            period:           EMA period
            atr_period:       ATR period
            multiplier:       ATR multiplier

        Returns:
            upper, middle, lower band arrays
        """
        middle = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, atr_period)
        return middle + multiplier * atr, middle, middle - multiplier * atr

    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """%K and %D Stochastic Oscillator.

        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %D = SMA(%K, d_period)

        Args:
            high, low, close: price arrays
            k_period: lookback for %K (typically 14)
            d_period: smoothing for %D (typically 3)

        Returns:
            k_line: %K array
            d_line: %D array
        """
        k = np.full_like(close, np.nan)
        for i in range(k_period - 1, len(close)):
            hh = high[i - k_period + 1:i + 1].max()
            ll = low[i - k_period + 1:i + 1].min()
            k[i] = 100.0 * (close[i] - ll) / (hh - ll + 1e-10)
        d = TechnicalIndicators.sma(k, d_period)
        return k, d

    @staticmethod
    def roc(values: np.ndarray, period: int = 12) -> np.ndarray:
        """Rate of Change.

        ROC(t) = 100 * (values[t] - values[t-period]) / values[t-period]

        Args:
            values: (T,) price array
            period: lookback period

        Returns:
            roc: (T,) rate of change in percent
        """
        result = np.full_like(values, np.nan)
        for i in range(period, len(values)):
            if values[i - period] != 0:
                result[i] = 100.0 * (values[i] - values[i - period]) / abs(values[i - period])
        return result

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume.

        OBV(t) = OBV(t-1) + volume[t] * sign(close[t] - close[t-1])

        Args:
            close:  (T,) closing prices
            volume: (T,) volume

        Returns:
            obv: (T,) OBV values
        """
        result = np.zeros_like(close)
        result[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                result[i] = result[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = result[i - 1] - volume[i]
            else:
                result[i] = result[i - 1]
        return result

    @staticmethod
    def vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Volume-Weighted Average Price.

        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3

        Args:
            high, low, close, volume: OHLCV arrays

        Returns:
            vwap: (T,) VWAP values
        """
        typical = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tpv = np.cumsum(typical * volume)
        return cum_tpv / np.maximum(cum_vol, 1e-10)

    @staticmethod
    def mfi(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Money Flow Index.

        MFI = 100 - 100 / (1 + positive_money_flow / negative_money_flow)

        Args:
            high, low, close, volume: OHLCV arrays
            period: MFI period (typically 14)

        Returns:
            mfi: (T,) MFI values (0-100)
        """
        typical = (high + low + close) / 3.0
        money_flow = typical * volume

        result = np.full_like(close, np.nan)
        for i in range(period, len(close)):
            pos = 0.0
            neg = 0.0
            for j in range(i - period + 1, i + 1):
                if typical[j] > typical[j - 1]:
                    pos += money_flow[j]
                elif typical[j] < typical[j - 1]:
                    neg += money_flow[j]
            mfr = pos / (neg + 1e-10)
            result[i] = 100.0 - 100.0 / (1.0 + mfr)
        return result

    @staticmethod
    def compute_all(
        ohlcv: np.ndarray,
        include_slow: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute a standard set of technical indicators.

        Args:
            ohlcv:        (T, 5) array [open, high, low, close, volume]
            include_slow: include slow-to-compute indicators (Stochastic, MFI)

        Returns:
            indicators: dict of indicator name → (T,) array
        """
        open_, high, low, close, volume = ohlcv.T

        indicators = {}

        # Trend
        for p in [5, 10, 20, 50]:
            indicators[f"sma_{p}"] = TechnicalIndicators.sma(close, p)
            indicators[f"ema_{p}"] = TechnicalIndicators.ema(close, p)

        # Momentum
        indicators["rsi_14"] = TechnicalIndicators.rsi(close)
        macd_l, sig_l, hist = TechnicalIndicators.macd(close)
        indicators["macd"] = macd_l
        indicators["macd_signal"] = sig_l
        indicators["macd_hist"] = hist
        for p in [1, 5, 10, 20]:
            indicators[f"roc_{p}"] = TechnicalIndicators.roc(close, p)

        # Volatility
        indicators["atr_14"] = TechnicalIndicators.atr(high, low, close)
        bb_upper, bb_mid, bb_lower = TechnicalIndicators.bollinger_bands(close)
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_mid
        indicators["bb_lower"] = bb_lower
        indicators["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        indicators["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Volume
        indicators["obv"] = TechnicalIndicators.obv(close, volume)
        indicators["vwap"] = TechnicalIndicators.vwap(high, low, close, volume)

        if include_slow:
            stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
            indicators["stoch_k"] = stoch_k
            indicators["stoch_d"] = stoch_d
            indicators["mfi_14"] = TechnicalIndicators.mfi(high, low, close, volume)

        return indicators


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------

class RollingStatistics:
    """Compute rolling statistical features for financial time series.

    Features:
    - Rolling mean, std, skewness, kurtosis
    - Rolling min, max, range
    - Rolling autocorrelation (lag 1, 5, 20)
    - Rolling Hurst exponent (approximate)
    - Rolling realized volatility (sum of squared returns)

    Args:
        window: rolling window size

    Example:
        >>> rs = RollingStatistics(window=20)
        >>> returns = np.diff(np.log(close))
        >>> stats = rs.compute(returns)
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute rolling statistics.

        Args:
            values: (T,) array of values (typically log returns)

        Returns:
            stats: dict of statistic name → (T,) array
        """
        T = len(values)
        w = self.window
        stats: Dict[str, np.ndarray] = {}

        mean_arr = np.full(T, np.nan)
        std_arr = np.full(T, np.nan)
        skew_arr = np.full(T, np.nan)
        kurt_arr = np.full(T, np.nan)
        min_arr = np.full(T, np.nan)
        max_arr = np.full(T, np.nan)
        rv_arr = np.full(T, np.nan)  # realized volatility
        ac1_arr = np.full(T, np.nan)  # lag-1 autocorrelation

        for i in range(w - 1, T):
            window = values[i - w + 1:i + 1]
            mean_arr[i] = window.mean()
            std = window.std()
            std_arr[i] = std
            min_arr[i] = window.min()
            max_arr[i] = window.max()
            rv_arr[i] = (window ** 2).sum()

            if std > 1e-10:
                centered = window - mean_arr[i]
                skew_arr[i] = ((centered / std) ** 3).mean()
                kurt_arr[i] = ((centered / std) ** 4).mean() - 3

            if w > 2:
                # Lag-1 autocorrelation
                x = window[:-1]
                y = window[1:]
                xm = x.mean()
                ym = y.mean()
                num = ((x - xm) * (y - ym)).sum()
                den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
                if den > 1e-10:
                    ac1_arr[i] = num / den

        stats["rolling_mean"] = mean_arr
        stats["rolling_std"] = std_arr
        stats["rolling_skew"] = skew_arr
        stats["rolling_kurt"] = kurt_arr
        stats["rolling_min"] = min_arr
        stats["rolling_max"] = max_arr
        stats["rolling_range"] = max_arr - min_arr
        stats["realized_vol"] = rv_arr
        stats["lag1_autocorr"] = ac1_arr

        return stats


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------------------------

class FeatureEngineeringPipeline:
    """Comprehensive feature engineering pipeline for financial OHLCV data.

    Combines technical indicators, rolling statistics, and derived features
    into a feature matrix for model input.

    Pipeline steps:
    1. Compute log returns and price ratios
    2. Add technical indicators
    3. Add rolling statistics
    4. Optional: add calendar features from timestamps
    5. Handle NaN values (forward fill or zero-fill)
    6. Normalize features

    Args:
        include_technicals:  whether to include technical indicators
        include_rolling:     whether to include rolling statistics
        include_calendar:    whether to include calendar features
        rolling_window:      window for rolling statistics
        normalize:           normalize features after computation

    Example:
        >>> pipe = FeatureEngineeringPipeline()
        >>> ohlcv = np.random.randn(500, 5).cumsum(axis=0) + 100
        >>> features = pipe.transform(ohlcv)  # (500, n_features)
    """

    def __init__(
        self,
        include_technicals: bool = True,
        include_rolling: bool = True,
        include_calendar: bool = False,
        rolling_window: int = 20,
        normalize: bool = True,
    ):
        self.include_technicals = include_technicals
        self.include_rolling = include_rolling
        self.include_calendar = include_calendar
        self.rolling_stats = RollingStatistics(rolling_window)
        self.normalize = normalize
        self._feature_names: List[str] = []

    def transform(
        self,
        ohlcv: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Transform OHLCV to feature matrix.

        Args:
            ohlcv:      (T, 5) array [open, high, low, close, volume]
            timestamps: (T,) optional Unix timestamps

        Returns:
            features:      (T, F) feature matrix
            feature_names: list of F feature names
        """
        T = len(ohlcv)
        open_, high, low, close, volume = ohlcv.T
        feature_cols = []
        names = []

        # Base price features
        log_return = np.diff(np.log(np.abs(close) + 1e-10), prepend=0)
        feature_cols.append(log_return)
        names.append("log_return")

        for p in [1, 2, 5, 10]:
            ret = np.zeros(T)
            ret[p:] = np.log(close[p:] / np.maximum(close[:-p], 1e-10))
            feature_cols.append(ret)
            names.append(f"log_return_{p}")

        # Price ratios
        hl_ratio = (high - low) / np.maximum(close, 1e-10)
        oc_ratio = (close - open_) / np.maximum(close, 1e-10)
        feature_cols.extend([hl_ratio, oc_ratio])
        names.extend(["hl_ratio", "oc_ratio"])

        # Volume log
        log_vol = np.log(np.maximum(volume, 1.0))
        vol_change = np.diff(log_vol, prepend=log_vol[0])
        feature_cols.extend([log_vol, vol_change])
        names.extend(["log_volume", "volume_change"])

        # Technical indicators
        if self.include_technicals:
            indicators = TechnicalIndicators.compute_all(ohlcv)
            for name, values in indicators.items():
                feature_cols.append(values)
                names.append(name)

        # Rolling statistics on returns
        if self.include_rolling:
            rs = self.rolling_stats.compute(log_return)
            for name, values in rs.items():
                feature_cols.append(values)
                names.append(name)

        # Stack features
        features = np.stack(feature_cols, axis=1)  # (T, F)

        # Handle NaN and Inf
        features = np.where(np.isnan(features), 0.0, features)
        features = np.where(np.isinf(features), 0.0, features)
        features = np.clip(features, -100.0, 100.0)

        if self.normalize:
            # Normalize each feature to zero mean, unit std
            means = features.mean(axis=0)
            stds = features.std(axis=0)
            stds = np.maximum(stds, 1e-8)
            features = (features - means) / stds

        self._feature_names = names
        return features, names

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

class SimpleRegimeDetector:
    """Simple heuristic regime detector for financial time series.

    Classifies each time step into a market regime based on:
    - Trend: bull, bear, sideways
    - Volatility: high vol, low vol

    Regimes:
    0 = Bull + Low Vol (trending up, calm)
    1 = Bull + High Vol (trending up, volatile)
    2 = Bear + Low Vol  (trending down, calm)
    3 = Bear + High Vol (trending down, volatile)
    4 = Sideways        (no clear trend)

    Args:
        trend_window:    window for trend determination (SMA)
        vol_window:      window for volatility regime
        vol_threshold:   z-score threshold for high-vol classification

    Example:
        >>> detector = SimpleRegimeDetector(trend_window=20)
        >>> regimes = detector.fit_predict(close)
        >>> # regimes: (T,) int array with values 0-4
    """

    def __init__(
        self,
        trend_window: int = 50,
        vol_window: int = 20,
        vol_threshold: float = 1.0,
    ):
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold

    def fit_predict(self, close: np.ndarray) -> np.ndarray:
        """Classify market regimes.

        Args:
            close: (T,) closing price array

        Returns:
            regimes: (T,) integer regime labels
        """
        T = len(close)
        regimes = np.zeros(T, dtype=int)

        # Trend: price vs SMA
        sma = TechnicalIndicators.sma(close, self.trend_window)
        is_bull = close > sma

        # Volatility: rolling std of returns
        returns = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0)
        vol = np.full(T, np.nan)
        for i in range(self.vol_window - 1, T):
            vol[i] = returns[i - self.vol_window + 1:i + 1].std()

        # Normalize vol
        vol_mean = np.nanmean(vol)
        vol_std = np.nanstd(vol) + 1e-10
        vol_zscore = (vol - vol_mean) / vol_std
        is_high_vol = vol_zscore > self.vol_threshold

        # Sideways: price close to SMA (within 1 ATR)
        atr = TechnicalIndicators.atr(
            close * 1.01, close * 0.99, close, self.trend_window
        )
        is_sideways = np.abs(close - sma) < atr

        for i in range(T):
            if np.isnan(sma[i]) or np.isnan(vol[i]):
                regimes[i] = 4  # Unknown early
            elif is_sideways[i]:
                regimes[i] = 4
            elif is_bull[i] and not is_high_vol[i]:
                regimes[i] = 0
            elif is_bull[i] and is_high_vol[i]:
                regimes[i] = 1
            elif not is_bull[i] and not is_high_vol[i]:
                regimes[i] = 2
            else:
                regimes[i] = 3

        return regimes


# ---------------------------------------------------------------------------
# Walk-Forward Cross-Validation
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """Walk-forward cross-validation for financial time series.

    Respects temporal order by training on past data and validating on
    future data. Avoids look-ahead bias.

    Produces splits:
    - Train: [0, train_end]
    - Valid: [train_end, train_end + valid_size]

    Both train and valid windows move forward in time.

    Args:
        n_splits:     number of train/valid splits
        min_train:    minimum training samples (fraction or absolute)
        valid_size:   validation window size (fraction or absolute)
        gap:          gap between train end and valid start (to avoid leakage)
        step_size:    how far to advance train window each split

    Example:
        >>> cv = WalkForwardCV(n_splits=5, valid_size=0.1)
        >>> for train_idx, valid_idx in cv.split(T=1000):
        ...     train_data = data[train_idx]
        ...     valid_data = data[valid_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train: Union[int, float] = 0.5,
        valid_size: Union[int, float] = 0.1,
        gap: int = 0,
        step_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.min_train = min_train
        self.valid_size = valid_size
        self.gap = gap
        self.step_size = step_size

    def split(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits.

        Args:
            T: total dataset length

        Returns:
            splits: list of (train_indices, valid_indices) tuples
        """
        min_train = (
            int(self.min_train * T) if isinstance(self.min_train, float)
            else self.min_train
        )
        valid_size = (
            int(self.valid_size * T) if isinstance(self.valid_size, float)
            else self.valid_size
        )

        total_needed = min_train + self.gap + valid_size
        remaining = T - total_needed

        if remaining < 0:
            raise ValueError(
                f"Insufficient data: T={T}, need at least {total_needed}"
            )

        step = self.step_size or max(1, remaining // self.n_splits)

        splits = []
        for i in range(self.n_splits):
            train_end = min_train + i * step
            valid_start = train_end + self.gap
            valid_end = valid_start + valid_size

            if valid_end > T:
                break

            train_idx = np.arange(0, train_end)
            valid_idx = np.arange(valid_start, valid_end)
            splits.append((train_idx, valid_idx))

        return splits

    def get_split_sizes(self, T: int) -> List[Tuple[int, int]]:
        """Return sizes of each split without generating indices."""
        splits = self.split(T)
        return [(len(tr), len(val)) for tr, val in splits]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

from typing import Tuple, Union

__all__ = [
    "DataConfig",
    "NormalizationStatistics",
    "TimeSeriesAugmenter",
    "FinancialDataset",
    "LabeledFinancialDataset",
    "MultiAssetDataset",
    "StreamingOHLCVDataset",
    "DataLoaderFactory",
    "LuminaDataModule",
    "TechnicalIndicators",
    "RollingStatistics",
    "FeatureEngineeringPipeline",
    "SimpleRegimeDetector",
    "WalkForwardCV",
    "collate_financial_batch",
    "generate_synthetic_ohlcv",
    "generate_correlated_assets",
]


# =============================================================================
# SECTION: Advanced Feature Engineering
# =============================================================================

class MicrostructureFeatures:
    """Market microstructure feature extraction.

    Computes features capturing market quality, liquidity, and
    intraday trading dynamics from OHLCV and trade data.

    Features:
    - Bid-ask spread proxies (Corwin-Schultz, Roll measure)
    - Price impact (Amihud illiquidity ratio)
    - Intraday patterns (U-shaped volume, volatility clustering)
    - Order flow imbalance proxies

    Args:
        window: Rolling window for feature computation
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def corwin_schultz_spread(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        """Estimate bid-ask spread from high-low prices.

        Reference: Corwin & Schultz, "A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices" JF 2012.

        Args:
            high: (T,) daily high prices
            low: (T,) daily low prices
        Returns:
            (T,) spread estimates (in fraction)
        """
        T = len(high)
        spreads = np.zeros(T)
        beta = np.log(high / low) ** 2
        beta_sum = np.zeros(T)
        for t in range(1, T):
            beta_sum[t] = beta[t] + beta[t-1]
        gamma = np.log(np.maximum(high[1:], high[:-1]) / np.minimum(low[1:], low[:-1])) ** 2
        gamma = np.concatenate([[0], gamma])
        alpha = (np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        spreads = np.maximum(0, 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
        return spreads

    def roll_spread(self, close: np.ndarray) -> np.ndarray:
        """Roll (1984) effective spread estimate from price changes.

        Args:
            close: (T,) closing prices
        Returns:
            (T,) spread estimates
        """
        delta = np.diff(close)
        T = len(close)
        spreads = np.full(T, np.nan)
        half_w = self.window // 2
        for t in range(self.window, T):
            dlt = delta[t - self.window:t]
            cov = np.cov(dlt[1:], dlt[:-1])[0, 1]
            if cov < 0:
                spreads[t] = 2 * np.sqrt(-cov)
        return spreads

    def amihud_illiquidity(
        self,
        returns: np.ndarray,
        volume: np.ndarray,
        price: np.ndarray,
    ) -> np.ndarray:
        """Amihud illiquidity ratio: |return| / (price * volume).

        High values = illiquid (large price impact per unit volume).

        Reference: Amihud, "Illiquidity and stock returns" JFM 2002.

        Args:
            returns: (T,) daily returns
            volume: (T,) daily volume in shares
            price: (T,) closing prices
        Returns:
            (T,) illiquidity estimates
        """
        dollar_vol = price * volume
        illiq = np.abs(returns) / (dollar_vol + 1e-10)
        # Rolling average
        result = np.full_like(illiq, np.nan)
        for t in range(self.window, len(illiq)):
            result[t] = illiq[t - self.window:t].mean()
        return result

    def intraday_volume_pattern(
        self,
        volume_by_hour: np.ndarray,
    ) -> np.ndarray:
        """Compute deviation from typical U-shaped intraday volume.

        Args:
            volume_by_hour: (T, H) volume per hour for T days, H hours
        Returns:
            (T,) measure of deviation from typical pattern
        """
        T, H = volume_by_hour.shape
        # Typical U-shape: high at open and close
        typical = np.array([1/(1+abs(h - H/2)) for h in range(H)])
        typical = typical / typical.sum()
        result = np.zeros(T)
        for t in range(T):
            row = volume_by_hour[t]
            total = row.sum()
            if total > 0:
                norm_vol = row / total
                result[t] = np.sqrt(((norm_vol - typical) ** 2).mean())
        return result

    def order_flow_toxicity(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """VPIN (Volume-synchronized Probability of Informed Trading) proxy.

        Estimates probability of informed trading based on
        volume imbalance between buyer and seller-initiated trades.

        Reference: Easley et al., "Flow Toxicity and Liquidity in a
        High-Frequency World" RFS 2012.

        Args:
            close: (T,) prices
            volume: (T,) volumes
            window: Window for estimation
        Returns:
            (T,) VPIN-proxy estimates
        """
        w = window or self.window
        returns = np.diff(close, prepend=close[0])
        # Signed volume proxy: positive returns -> buy, negative -> sell
        buy_vol = volume * (returns > 0).astype(float)
        sell_vol = volume * (returns <= 0).astype(float)
        T = len(close)
        vpin = np.full(T, np.nan)
        for t in range(w, T):
            bv = buy_vol[t-w:t].sum()
            sv = sell_vol[t-w:t].sum()
            total = bv + sv
            if total > 0:
                vpin[t] = abs(bv - sv) / total
        return vpin

    def compute_all(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute all microstructure features.

        Args:
            close, high, low, volume: (T,) price/volume arrays
        Returns:
            Dict of feature name -> (T,) array
        """
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        features = {}
        try:
            features["cs_spread"] = self.corwin_schultz_spread(high, low)
        except Exception:
            features["cs_spread"] = np.zeros(len(close))
        try:
            features["roll_spread"] = self.roll_spread(close)
        except Exception:
            features["roll_spread"] = np.full(len(close), np.nan)
        features["amihud"] = self.amihud_illiquidity(returns, volume, close)
        features["vpin"] = self.order_flow_toxicity(close, volume)
        features["rel_spread_hl"] = (high - low) / (close + 1e-10)
        features["volume_zscore"] = (volume - volume.mean()) / (volume.std() + 1e-10)
        return features


class AlternativeDataProcessor:
    """Process alternative data sources for financial modeling.

    Handles:
    - News sentiment scores
    - Social media metrics (Twitter, Reddit)
    - Web search trends
    - Satellite imagery features (parking lot occupancy, etc.)
    - Credit card transaction data

    All data is aligned to daily trading calendar.

    Args:
        trading_calendar: Array of trading dates (YYYYMMDD format)
        fillna_method: How to fill missing values ('ffill', 'zero', 'mean')
    """

    def __init__(
        self,
        trading_calendar: np.ndarray,
        fillna_method: str = "ffill",
    ) -> None:
        self.trading_calendar = trading_calendar
        self.fillna_method = fillna_method

    def process_news_sentiment(
        self,
        dates: np.ndarray,
        sentiment_scores: np.ndarray,
        article_counts: np.ndarray,
    ) -> np.ndarray:
        """Aggregate news sentiment to daily trading frequency.

        Args:
            dates: (N,) article dates (YYYYMMDD integers)
            sentiment_scores: (N,) sentiment per article [-1, 1]
            article_counts: (N,) or ones
        Returns:
            (T,) volume-weighted daily sentiment
        """
        T = len(self.trading_calendar)
        daily_sentiment = np.zeros(T)
        daily_count = np.zeros(T)

        cal_map = {d: i for i, d in enumerate(self.trading_calendar)}
        for d, s, c in zip(dates, sentiment_scores, article_counts):
            if d in cal_map:
                t = cal_map[d]
                daily_sentiment[t] += s * c
                daily_count[t] += c

        # Normalize
        valid = daily_count > 0
        daily_sentiment[valid] /= daily_count[valid]

        # Fill missing days
        if self.fillna_method == "ffill":
            last_val = 0.0
            for t in range(T):
                if valid[t]:
                    last_val = daily_sentiment[t]
                else:
                    daily_sentiment[t] = last_val
        return daily_sentiment

    def process_search_trends(
        self,
        weekly_trends: np.ndarray,
        trend_dates: np.ndarray,
    ) -> np.ndarray:
        """Interpolate weekly Google Trends to daily frequency.

        Args:
            weekly_trends: (W,) normalized search volume (0-100)
            trend_dates: (W,) week start dates (YYYYMMDD)
        Returns:
            (T,) daily-interpolated trend scores
        """
        T = len(self.trading_calendar)
        # Map to trading day indices
        cal_arr = self.trading_calendar.astype(float)
        date_arr = trend_dates.astype(float)
        result = np.interp(cal_arr, date_arr, weekly_trends.astype(float))
        return result

    def compute_text_features(
        self,
        texts: List[str],
        dates: np.ndarray,
        vectorizer=None,
    ) -> np.ndarray:
        """Extract bag-of-words or TF-IDF features from text.

        Args:
            texts: List of text strings
            dates: (N,) dates for each text
            vectorizer: Optional sklearn TfidfVectorizer
        Returns:
            (T, vocab_size) sparse feature matrix
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return np.zeros((len(self.trading_calendar), 1))

        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        X = vectorizer.fit_transform(texts).toarray()  # (N, V)
        T = len(self.trading_calendar)
        V = X.shape[1]
        result = np.zeros((T, V))
        count = np.zeros(T)
        cal_map = {d: i for i, d in enumerate(self.trading_calendar)}
        for i, d in enumerate(dates):
            if d in cal_map:
                t = cal_map[d]
                result[t] += X[i]
                count[t] += 1
        valid = count > 0
        result[valid] /= count[valid, np.newaxis]
        return result


class CrossSectionalNormalizer:
    """Cross-sectional normalization for equity return prediction.

    For each time step, normalizes features across the cross-section
    of assets. This removes time-series level trends and focuses
    the model on relative (cross-sectional) differences.

    Methods:
    - z-score: (x - mean) / std
    - rank: percentile rank [0, 1]
    - winsorize + z-score: clip outliers then z-score
    - robust: (x - median) / MAD
    - truncated: truncate to [-n, n] sigma

    Args:
        method: Normalization method
        winsorize_pct: Percentile for winsorization (default 1%)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        method: str = "zscore",
        winsorize_pct: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        self.method = method
        self.winsorize_pct = winsorize_pct
        self.eps = eps

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Apply cross-sectional normalization.

        Args:
            X: (T, N) feature matrix where T=time, N=assets
        Returns:
            (T, N) normalized features
        """
        T, N = X.shape
        result = np.zeros_like(X)

        for t in range(T):
            row = X[t].copy()
            if np.all(np.isnan(row)):
                continue
            valid = ~np.isnan(row)

            if self.method == "zscore":
                mu = row[valid].mean()
                sigma = row[valid].std()
                result[t] = (row - mu) / (sigma + self.eps)
            elif self.method == "rank":
                ranked = np.full(N, np.nan)
                ranks = row[valid].argsort().argsort()
                valid_idx = np.where(valid)[0]
                ranked[valid_idx] = ranks / max(1, len(ranks) - 1)
                result[t] = ranked
            elif self.method == "winsorize_zscore":
                lo = np.nanpercentile(row[valid], self.winsorize_pct * 100)
                hi = np.nanpercentile(row[valid], (1 - self.winsorize_pct) * 100)
                row[valid] = np.clip(row[valid], lo, hi)
                mu = row[valid].mean()
                sigma = row[valid].std()
                result[t] = (row - mu) / (sigma + self.eps)
            elif self.method == "robust":
                median = np.nanmedian(row[valid])
                mad = np.nanmedian(np.abs(row[valid] - median))
                result[t] = (row - median) / (1.4826 * mad + self.eps)
            else:
                result[t] = row

        return result

    def transform_batch(
        self,
        X: np.ndarray,
        t_axis: int = 0,
    ) -> np.ndarray:
        """Transform with arbitrary time axis."""
        if t_axis != 0:
            X = np.moveaxis(X, t_axis, 0)
        result = self.fit_transform(X)
        if t_axis != 0:
            result = np.moveaxis(result, 0, t_axis)
        return result


class TimeSeriesDataset:
    """PyTorch-compatible financial time series dataset with lookback windows.

    Handles:
    - Multiple time series aligned to the same calendar
    - Configurable lookback window and forecast horizon
    - Train/val/test splits respecting temporal ordering
    - Missing data handling
    - Multi-asset (panel) data support

    Args:
        data: (T, N, F) array where T=time, N=assets, F=features
        labels: (T, N, H) target labels where H=forecast horizons
        lookback: Number of historical timesteps as input
        horizon: Number of future timesteps to predict
        stride: Step between samples
        normalize: Whether to normalize within each window
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        lookback: int = 60,
        horizon: int = 5,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        self.data = data
        self.labels = labels
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.normalize = normalize
        T = data.shape[0]
        # Valid sample indices
        self.indices = list(range(lookback, T - horizon, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        t = self.indices[idx]
        x = self.data[t - self.lookback:t].copy()  # (lookback, N, F)
        if self.normalize:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mean) / std
        result = {"x": x, "t": t}
        if self.labels is not None:
            y = self.labels[t:t + self.horizon]  # (horizon, N, ...)
            result["y"] = y
        return result

    def train_val_test_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> Tuple["TimeSeriesDataset", "TimeSeriesDataset", "TimeSeriesDataset"]:
        """Split dataset preserving temporal order.

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        n = len(self.indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        def subset(start, end):
            ds = TimeSeriesDataset.__new__(TimeSeriesDataset)
            ds.data = self.data
            ds.labels = self.labels
            ds.lookback = self.lookback
            ds.horizon = self.horizon
            ds.stride = self.stride
            ds.normalize = self.normalize
            ds.indices = self.indices[start:end]
            return ds

        return (
            subset(0, n_train),
            subset(n_train, n_train + n_val),
            subset(n_train + n_val, n),
        )


class DataQualityChecker:
    """Data quality checks and cleaning for financial time series.

    Detects and handles:
    - Missing values
    - Price staleness (repeated values)
    - Price jumps (extreme moves)
    - Volume anomalies
    - Survivorship bias warnings
    - Corporate action detection

    Args:
        max_missing_frac: Max fraction of missing values per series
        max_consecutive_missing: Max consecutive missing values
        price_jump_threshold: Abs return threshold for suspected error (e.g., 0.5 = 50%)
        stale_threshold: Max consecutive identical prices
    """

    def __init__(
        self,
        max_missing_frac: float = 0.05,
        max_consecutive_missing: int = 5,
        price_jump_threshold: float = 0.5,
        stale_threshold: int = 5,
    ) -> None:
        self.max_missing_frac = max_missing_frac
        self.max_consecutive_missing = max_consecutive_missing
        self.price_jump_threshold = price_jump_threshold
        self.stale_threshold = stale_threshold

    def check_series(self, prices: np.ndarray, name: str = "") -> Dict[str, Any]:
        """Run all quality checks on a price series.

        Args:
            prices: (T,) price time series
            name: Series identifier for reporting
        Returns:
            Dict with check results and issues
        """
        issues = []
        T = len(prices)

        # Missing value check
        missing = np.isnan(prices)
        missing_frac = missing.mean()
        if missing_frac > self.max_missing_frac:
            issues.append(f"High missing fraction: {missing_frac:.2%}")

        # Consecutive missing
        max_consec = 0
        consec = 0
        for m in missing:
            if m:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        if max_consec > self.max_consecutive_missing:
            issues.append(f"Long missing gap: {max_consec} consecutive")

        # Price staleness
        valid_prices = prices[~missing]
        if len(valid_prices) > 1:
            stale_count = 0
            max_stale = 0
            for t in range(1, len(valid_prices)):
                if valid_prices[t] == valid_prices[t-1]:
                    stale_count += 1
                    max_stale = max(max_stale, stale_count)
                else:
                    stale_count = 0
            if max_stale >= self.stale_threshold:
                issues.append(f"Price staleness: {max_stale} consecutive identical")

        # Price jumps
        if len(valid_prices) > 1:
            returns = np.diff(valid_prices) / (valid_prices[:-1] + 1e-10)
            extreme = (np.abs(returns) > self.price_jump_threshold)
            if extreme.any():
                issues.append(f"Extreme price moves: {extreme.sum()} instances")

        return {
            "name": name,
            "length": T,
            "missing_fraction": float(missing_frac),
            "max_consecutive_missing": max_consec,
            "issues": issues,
            "quality_score": max(0.0, 1.0 - len(issues) * 0.25),
        }

    def clean_series(
        self,
        prices: np.ndarray,
        method: str = "ffill",
    ) -> np.ndarray:
        """Clean a price series by handling detected issues.

        Args:
            prices: (T,) raw price series
            method: How to fill missing values
        Returns:
            (T,) cleaned price series
        """
        result = prices.copy().astype(float)

        # Fill NaN
        if method == "ffill":
            last_valid = np.nan
            for t in range(len(result)):
                if np.isnan(result[t]):
                    result[t] = last_valid
                else:
                    last_valid = result[t]
        elif method == "interpolate":
            nans = np.isnan(result)
            x = np.arange(len(result))
            if not nans.all():
                result[nans] = np.interp(x[nans], x[~nans], result[~nans])
        elif method == "zero":
            result = np.nan_to_num(result, nan=0.0)

        # Clip extreme moves
        returns = np.diff(result, prepend=result[0]) / (np.abs(result) + 1e-10)
        extreme = np.abs(returns) > self.price_jump_threshold
        for t in np.where(extreme)[0]:
            if t > 0:
                result[t] = result[t-1]  # Replace with previous value

        return result


class EfficientDataLoader:
    """Memory-efficient data loader for large financial datasets.

    Supports:
    - Chunked reading from disk (HDF5, Parquet, CSV)
    - In-memory caching with LRU eviction
    - Asynchronous prefetching
    - Multi-process workers

    Args:
        data_path: Path to data file or directory
        chunk_size: Number of timesteps per chunk
        cache_chunks: Number of chunks to keep in memory
        file_format: 'hdf5', 'parquet', 'csv', or 'numpy'
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 1000,
        cache_chunks: int = 10,
        file_format: str = "numpy",
    ) -> None:
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        self.file_format = file_format
        self._cache: Dict[int, Any] = {}
        self._cache_order: List[int] = []

    def _load_chunk(self, chunk_idx: int) -> np.ndarray:
        """Load a specific chunk from disk."""
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size

        if self.file_format == "numpy":
            data = np.load(self.data_path)
            return data[start:end]
        elif self.file_format == "csv":
            import pandas as pd
            df = pd.read_csv(self.data_path, skiprows=start, nrows=self.chunk_size)
            return df.values
        elif self.file_format == "parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(self.data_path)
                return df.values[start:end]
            except ImportError:
                return np.zeros((self.chunk_size, 1))
        else:
            return np.zeros((self.chunk_size, 1))

    def get_chunk(self, chunk_idx: int) -> np.ndarray:
        """Get chunk from cache or load from disk."""
        if chunk_idx in self._cache:
            return self._cache[chunk_idx]
        # Evict LRU if cache full
        if len(self._cache) >= self.cache_chunks:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        chunk = self._load_chunk(chunk_idx)
        self._cache[chunk_idx] = chunk
        self._cache_order.append(chunk_idx)
        return chunk

    def get_range(self, start: int, end: int) -> np.ndarray:
        """Get a range of data across potentially multiple chunks."""
        chunks = []
        start_chunk = start // self.chunk_size
        end_chunk = end // self.chunk_size + 1
        for ci in range(start_chunk, end_chunk):
            chunk = self.get_chunk(ci)
            chunk_start = ci * self.chunk_size
            slice_start = max(0, start - chunk_start)
            slice_end = min(self.chunk_size, end - chunk_start)
            if slice_start < slice_end and slice_start < len(chunk):
                chunks.append(chunk[slice_start:min(slice_end, len(chunk))])
        return np.concatenate(chunks, axis=0) if chunks else np.array([])


class SyntheticDataGenerator:
    """Generate synthetic financial time series for testing and augmentation.

    Generates realistic synthetic data using:
    - Geometric Brownian Motion (GBM)
    - Jump-diffusion processes (Merton)
    - Heston stochastic volatility
    - GARCH(1,1) volatility clustering
    - Regime-switching models

    Args:
        seed: Random seed for reproducibility
        dt: Time step (1/252 for daily)
    """

    def __init__(self, seed: int = 42, dt: float = 1.0 / 252) -> None:
        np.random.seed(seed)
        self.dt = dt

    def gbm(
        self,
        T: int,
        mu: float = 0.08,
        sigma: float = 0.2,
        S0: float = 100.0,
    ) -> np.ndarray:
        """Geometric Brownian Motion.

        dS = mu*S*dt + sigma*S*dW

        Args:
            T: Number of timesteps
            mu: Drift (annual)
            sigma: Volatility (annual)
            S0: Initial price
        Returns:
            (T,) price path
        """
        dt = self.dt
        Z = np.random.standard_normal(T)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        return S0 * np.exp(log_returns.cumsum())

    def heston(
        self,
        T: int,
        mu: float = 0.08,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        V0: float = 0.04,
        S0: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heston stochastic volatility model.

        dS = mu*S*dt + sqrt(V)*S*dW1
        dV = kappa*(theta-V)*dt + xi*sqrt(V)*dW2
        corr(dW1, dW2) = rho

        Args:
            T: Number of timesteps
            mu, kappa, theta, xi, rho, V0, S0: Model parameters
        Returns:
            (prices, variances) both (T,)
        """
        dt = self.dt
        prices = np.zeros(T)
        variances = np.zeros(T)
        S, V = S0, V0
        for t in range(T):
            Z1 = np.random.standard_normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal()
            V = max(1e-8, V + kappa * (theta - V) * dt + xi * np.sqrt(V * dt) * Z2)
            S = S * np.exp((mu - 0.5 * V) * dt + np.sqrt(V * dt) * Z1)
            prices[t] = S
            variances[t] = V
        return prices, variances

    def jump_diffusion(
        self,
        T: int,
        mu: float = 0.08,
        sigma: float = 0.2,
        jump_intensity: float = 5.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.05,
        S0: float = 100.0,
    ) -> np.ndarray:
        """Merton jump-diffusion model.

        dS/S = (mu - lambda*k)*dt + sigma*dW + J*dN
        where N is Poisson process, J is log-normal jump size.

        Args:
            T: Number of timesteps
            jump_intensity: Poisson intensity (jumps per year)
            jump_mean, jump_std: Log-jump size distribution
        Returns:
            (T,) price path
        """
        dt = self.dt
        S = S0
        prices = np.zeros(T)
        lam = jump_intensity * dt
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        for t in range(T):
            Z = np.random.standard_normal()
            N = np.random.poisson(lam)
            jump = sum(np.random.normal(jump_mean, jump_std) for _ in range(N))
            S = S * np.exp((mu - 0.5 * sigma**2 - lam * k) * dt +
                           sigma * np.sqrt(dt) * Z + jump)
            prices[t] = S
        return prices

    def garch_returns(
        self,
        T: int,
        omega: float = 1e-6,
        alpha: float = 0.1,
        beta: float = 0.85,
        mu: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GARCH(1,1) return process for volatility clustering.

        sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2

        Args:
            T: Number of timesteps
            omega, alpha, beta: GARCH parameters
            mu: Mean return
        Returns:
            (returns, variances) both (T,)
        """
        returns = np.zeros(T)
        variances = np.zeros(T)
        sigma2 = omega / (1 - alpha - beta)  # Unconditional variance
        eps = 0.0
        for t in range(T):
            sigma2 = omega + alpha * eps**2 + beta * sigma2
            eps = np.sqrt(sigma2) * np.random.standard_normal()
            returns[t] = mu + eps
            variances[t] = sigma2
        return returns, variances

    def regime_switching(
        self,
        T: int,
        regimes: Optional[List[Dict]] = None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-regime Markov switching model.

        Args:
            T: Number of timesteps
            regimes: List of regime parameter dicts with 'mu' and 'sigma'
            transition_matrix: (2,2) transition probability matrix
        Returns:
            (prices, regime_labels) both (T,)
        """
        if regimes is None:
            regimes = [
                {"mu": 0.10, "sigma": 0.15},   # Bull
                {"mu": -0.20, "sigma": 0.35},  # Bear
            ]
        if transition_matrix is None:
            transition_matrix = np.array([[0.97, 0.03], [0.05, 0.95]])

        prices = np.zeros(T)
        labels = np.zeros(T, dtype=int)
        state = 0
        S = 100.0
        for t in range(T):
            state = np.random.choice(2, p=transition_matrix[state])
            r = regimes[state]
            ret = np.random.normal(r["mu"] * self.dt, r["sigma"] * np.sqrt(self.dt))
            S = S * np.exp(ret)
            prices[t] = S
            labels[t] = state

        return prices, labels


_NEW_DATA_PIPELINE_EXPORTS = [
    "MicrostructureFeatures", "AlternativeDataProcessor", "CrossSectionalNormalizer",
    "TimeSeriesDataset", "DataQualityChecker", "EfficientDataLoader", "SyntheticDataGenerator",
]
