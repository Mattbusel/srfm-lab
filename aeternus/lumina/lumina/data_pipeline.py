"""
lumina/data_pipeline.py

Data pipeline for Lumina pretraining and fine-tuning:

  - SyntheticFinancialDataset
  - DataCollator
  - StreamingDataLoader
  - DataAugmentation
  - CurriculumScheduler
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# SyntheticFinancialDataset
# ---------------------------------------------------------------------------
@dataclass
class FinancialDataConfig:
    n_samples: int = 10000
    seq_len: int = 256
    n_price_features: int = 256
    n_onchain_signals: int = 3
    n_news_features: int = 256
    n_regimes: int = 8
    # Heston model
    mu: float = 0.0002
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.7
    v0: float = 0.04
    dt: float = 1 / 252
    # Hawkes process for news events
    hawkes_mu: float = 0.5
    hawkes_alpha: float = 0.3
    hawkes_beta: float = 1.5
    # Crisis injection
    crisis_prob: float = 0.05
    crisis_vol_mult: float = 5.0
    # Curriculum
    difficulty: float = 1.0   # 0.0=only calm, 1.0=full distribution


class SyntheticFinancialDataset(Dataset):
    """
    Generates paired (price_tokens, onchain, news_embedding) tuples
    using Heston SV + Hawkes news process + crisis injection.

    Designed to be a drop-in for PyTorch DataLoader.
    """

    def __init__(self, config: FinancialDataConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self._data = self._generate_all()

    # ------------------------------------------------------------------
    # Stochastic processes
    # ------------------------------------------------------------------

    def _heston_euler(self) -> Tuple[np.ndarray, np.ndarray]:
        """Euler-Maruyama discretization of Heston SDE."""
        cfg = self.config
        T = cfg.seq_len
        S = np.zeros(T + 1)
        v = np.zeros(T + 1)
        S[0] = 100.0
        v[0] = cfg.v0

        for t in range(T):
            z1 = self.rng.randn()
            z2 = cfg.rho * z1 + math.sqrt(1 - cfg.rho ** 2) * self.rng.randn()
            v_next = (v[t] + cfg.kappa * (cfg.theta - v[t]) * cfg.dt
                      + cfg.xi * math.sqrt(max(v[t], 0.0)) * math.sqrt(cfg.dt) * z2)
            v[t + 1] = max(v_next, 1e-9)
            S[t + 1] = S[t] * math.exp(
                (cfg.mu - 0.5 * v[t]) * cfg.dt
                + math.sqrt(max(v[t], 0.0) * cfg.dt) * z1
            )

        return S[:T], v[:T]

    def _hawkes_news_times(self, T_seconds: float) -> List[float]:
        """Simulate Hawkes process to get news event times (in continuous time)."""
        cfg = self.config
        times = []
        t = 0.0
        while t < T_seconds:
            # Intensity = mu + sum alpha*exp(-beta*(t-ti)) for ti < t
            lam = cfg.hawkes_mu
            for ti in times:
                lam += cfg.hawkes_alpha * math.exp(-cfg.hawkes_beta * (t - ti))
            dt = self.rng.exponential(1.0 / max(lam, 1e-6))
            t += dt
            if t < T_seconds:
                times.append(t)
        return times

    def _crisis_inject(
        self, S: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Inject a crisis episode with probability crisis_prob * difficulty."""
        effective_prob = self.config.crisis_prob * self.config.difficulty
        if self.rng.rand() < effective_prob:
            T = len(S)
            start = self.rng.randint(T // 4, 3 * T // 4)
            length = self.rng.randint(10, max(11, T // 4))
            end = min(start + length, T)
            v = v.copy()
            S = S.copy()
            v[start:end] *= self.config.crisis_vol_mult
            drawdown = np.exp(-0.003 * np.arange(end - start))
            S[start:end] *= np.cumprod(drawdown)
            return S, v, True
        return S, v, False

    def _assign_regime(self, v: np.ndarray) -> int:
        """Assign a regime index based on volatility statistics."""
        v_mean = float(v.mean())
        v_std = float(v.std())
        # Quantize mean vol into n_regimes buckets
        vol_range = (0.005, 0.50)
        bucket = int(
            (v_mean - vol_range[0]) / (vol_range[1] - vol_range[0]) * self.config.n_regimes
        )
        return int(np.clip(bucket, 0, self.config.n_regimes - 1))

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def _price_to_tokens(self, S: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Convert price/vol path to token-like feature matrix.
        In production this would run through PriceSeriesTokenizer.
        Here we create a D-dimensional representation per bar.
        """
        T = self.config.seq_len
        D = self.config.n_price_features
        tokens = np.zeros((T, D), dtype=np.float32)

        log_ret = np.diff(np.log(np.maximum(S, 1e-8)), prepend=np.log(S[0]))
        realized_vol = np.sqrt(v)

        for t in range(T):
            r = log_ret[t]
            rv = realized_vol[t]
            for d in range(D):
                if d < D // 4:
                    tokens[t, d] = math.sin(r * (d + 1) * 20)
                elif d < D // 2:
                    tokens[t, d] = math.cos(rv * (d - D // 4 + 1) * 20)
                elif d < 3 * D // 4:
                    tokens[t, d] = math.tanh(r * (d - D // 2 + 1))
                else:
                    tokens[t, d] = rv * math.sin(d * 0.1)

        return tokens

    def _onchain_signals(self, S: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Generate correlated on-chain signals."""
        T = self.config.seq_len
        S_abs_ret = np.abs(np.diff(np.log(np.maximum(S, 1e-8)), prepend=0.0))
        signals = np.zeros((T, self.config.n_onchain_signals), dtype=np.float32)

        # Whale flow: correlated with |returns| + autoregressive component
        signals[:, 0] = (S_abs_ret * 10
                         + 0.3 * np.roll(S_abs_ret * 10, 1)
                         + 0.1 * self.rng.randn(T))

        # DEX volume: lagged returns + volume surge during crisis
        signals[:, 1] = (np.roll(S_abs_ret, 3) * 5
                         + np.sqrt(v) * 2
                         + 0.05 * self.rng.randn(T))

        # LP depth: inversely related to volatility
        signals[:, 2] = 1.0 / (np.sqrt(v) + 0.01) + 0.1 * self.rng.randn(T)

        return np.clip(signals, 0, None)

    def _news_embeddings(
        self, news_times: List[float], regime: int
    ) -> np.ndarray:
        """
        Generate synthetic BERT-like news embeddings for T time steps.
        News events are placed at nearest bar based on news_times.
        """
        T = self.config.seq_len
        D = self.config.n_news_features
        embeddings = np.zeros((T, D), dtype=np.float32)

        # Base sentiment vector for this regime
        rng_base = np.random.RandomState(regime)
        regime_direction = rng_base.randn(D)
        regime_direction /= np.linalg.norm(regime_direction) + 1e-8

        # Map news event times to bar indices
        T_seconds = T * 300  # assume 5-min bars
        for t_event in news_times:
            bar_idx = int(t_event / T_seconds * T)
            bar_idx = np.clip(bar_idx, 0, T - 1)
            # Add news event embedding
            intensity = 1.0 + 0.5 * self.rng.rand()
            noise = self.rng.randn(D) * 0.3
            embeddings[bar_idx] += regime_direction * intensity + noise

        # Smooth over time (news sentiment persists)
        for d in range(D):
            for t in range(1, T):
                embeddings[t, d] = 0.8 * embeddings[t, d] + 0.2 * embeddings[t - 1, d]

        return embeddings

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> List[Dict]:
        samples = []
        T_seconds = self.config.seq_len * 300  # 5-min bars

        for i in range(self.config.n_samples):
            S, v = self._heston_euler()
            S, v, is_crisis = self._crisis_inject(S, v)
            regime = self._assign_regime(v)
            news_times = self._hawkes_news_times(T_seconds)

            price_tokens = self._price_to_tokens(S, v)
            onchain = self._onchain_signals(S, v)
            news_emb = self._news_embeddings(news_times, regime)

            samples.append({
                "price_tokens": price_tokens.astype(np.float32),
                "onchain": onchain.astype(np.float32),
                "news_emb": news_emb.astype(np.float32),
                "regime": int(regime),
                "is_crisis": int(is_crisis),
                "n_news_events": len(news_times),
            })

        return samples

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        s = self._data[idx]
        return {
            "price_tokens": torch.from_numpy(s["price_tokens"]),
            "onchain": torch.from_numpy(s["onchain"]),
            "news_emb": torch.from_numpy(s["news_emb"]),
            "regime": torch.tensor(s["regime"], dtype=torch.long),
            "is_crisis": torch.tensor(s["is_crisis"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataCollator
# ---------------------------------------------------------------------------
class DataCollator:
    """
    Pads/truncates sequences to fixed length and creates attention masks.
    Handles heterogeneous modality fields.
    """

    def __init__(
        self,
        seq_len: int = 256,
        pad_value: float = 0.0,
        modality_fields: Optional[List[str]] = None,
    ):
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.modality_fields = modality_fields or ["price_tokens", "onchain", "news_emb"]

    def __call__(self, batch: List[Dict]) -> Dict:
        B = len(batch)
        result = {}

        # Scalar labels
        for key in ["regime", "is_crisis", "labels", "direction_labels"]:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])

        # Sequence fields
        for field in self.modality_fields:
            if field not in batch[0]:
                continue

            tensors = [b[field] for b in batch]
            T_actual = tensors[0].shape[0]
            D = tensors[0].shape[-1] if tensors[0].dim() > 1 else 1

            if T_actual >= self.seq_len:
                # Truncate
                if D > 1:
                    stacked = torch.stack([t[:self.seq_len] for t in tensors])
                else:
                    stacked = torch.stack([t[:self.seq_len] for t in tensors])
                mask = torch.ones(B, self.seq_len, dtype=torch.bool)
            else:
                # Pad
                pad_len = self.seq_len - T_actual
                padded = []
                for t in tensors:
                    if t.dim() == 1:
                        pad = torch.full((pad_len,), self.pad_value)
                        padded.append(torch.cat([t, pad]))
                    else:
                        pad = torch.full((pad_len, *t.shape[1:]), self.pad_value)
                        padded.append(torch.cat([t, pad], dim=0))
                stacked = torch.stack(padded)
                mask = torch.cat([
                    torch.ones(B, T_actual, dtype=torch.bool),
                    torch.zeros(B, pad_len, dtype=torch.bool),
                ], dim=1)

            result[field] = stacked
            if field == self.modality_fields[0]:
                result["attention_mask"] = mask

        return result


# ---------------------------------------------------------------------------
# StreamingDataLoader
# ---------------------------------------------------------------------------
class StreamingFinancialDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large financial datasets.
    Generates samples on-the-fly (no pre-allocation of all samples).
    """

    def __init__(self, config: FinancialDataConfig, total_steps: int = 100000):
        self.config = config
        self.total_steps = total_steps

    def __iter__(self) -> Iterator[Dict]:
        """Yields samples indefinitely (until total_steps consumed by DataLoader)."""
        worker_info = torch.utils.data.get_worker_info()
        seed = 42 if worker_info is None else worker_info.id
        rng = np.random.RandomState(seed)

        # Create a temporary dataset with a single sample config
        sample_cfg = FinancialDataConfig(
            n_samples=1,
            seq_len=self.config.seq_len,
            n_price_features=self.config.n_price_features,
            n_onchain_signals=self.config.n_onchain_signals,
            n_news_features=self.config.n_news_features,
            n_regimes=self.config.n_regimes,
            mu=self.config.mu,
            kappa=self.config.kappa,
            theta=self.config.theta,
            xi=self.config.xi,
            rho=self.config.rho,
            v0=self.config.v0,
            crisis_prob=self.config.crisis_prob,
            crisis_vol_mult=self.config.crisis_vol_mult,
            difficulty=self.config.difficulty,
        )

        for _ in range(self.total_steps):
            ds = SyntheticFinancialDataset(sample_cfg, seed=int(rng.randint(0, 2**30)))
            yield ds[0]


class StreamingDataLoader:
    """
    Wraps StreamingFinancialDataset with configurable batch size and workers.
    """

    def __init__(
        self,
        config: FinancialDataConfig,
        batch_size: int = 32,
        seq_len: int = 256,
        num_workers: int = 0,
        total_steps: int = 100000,
    ):
        self.ds = StreamingFinancialDataset(config, total_steps * batch_size)
        self.collator = DataCollator(seq_len=seq_len)
        self.loader = DataLoader(
            self.ds,
            batch_size=batch_size,
            collate_fn=self.collator,
            num_workers=num_workers,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# ---------------------------------------------------------------------------
# DataAugmentation
# ---------------------------------------------------------------------------
class DataAugmentation:
    """
    Augmentation strategies for financial time series tensors.

    All methods accept (B, T, D) tensors and return augmented tensors.
    """

    def __init__(
        self,
        time_warp_sigma: float = 0.2,
        mag_scale_sigma: float = 0.1,
        window_reduce_ratio: float = 0.9,
        dropout_p: float = 0.05,
    ):
        self.time_warp_sigma = time_warp_sigma
        self.mag_scale_sigma = mag_scale_sigma
        self.window_reduce_ratio = window_reduce_ratio
        self.dropout_p = dropout_p

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply smooth random time warping to a sequence."""
        B, T, D = x.shape
        n_knots = max(4, T // 16)
        results = []

        for b in range(B):
            knot_positions = torch.linspace(0, T - 1, n_knots)
            # Add small random perturbations to knot values
            perturb = torch.randn(n_knots) * self.time_warp_sigma * (T / n_knots)
            warped_knots = (knot_positions + perturb).clamp(0, T - 1)
            warped_knots, _ = warped_knots.sort()

            # Build full time warp via linear interpolation
            t = torch.arange(T, dtype=torch.float32)
            new_t = torch.zeros(T)
            for i in range(n_knots - 1):
                start = int(knot_positions[i].item())
                end = int(knot_positions[i + 1].item())
                if end <= start:
                    continue
                alpha = (t[start:end] - start) / max(end - start, 1)
                new_t[start:end] = warped_knots[i] + alpha * (warped_knots[i + 1] - warped_knots[i])
            new_t[-1] = warped_knots[-1]

            # Sample x at warped positions
            idx = new_t.long().clamp(0, T - 1)
            results.append(x[b, idx])

        return torch.stack(results)

    def magnitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each feature dimension by a random scale factor."""
        scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * self.mag_scale_sigma
        return x * scale

    def window_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Slice a sub-window and interpolate back to original length."""
        B, T, D = x.shape
        T_slice = max(2, int(T * self.window_reduce_ratio))
        starts = torch.randint(0, T - T_slice + 1, (B,))

        results = []
        for b in range(B):
            sliced = x[b, starts[b]:starts[b] + T_slice]  # (T_slice, D)
            # Interpolate back to T
            sliced = sliced.unsqueeze(0).permute(0, 2, 1)  # (1, D, T_slice)
            stretched = F.interpolate(sliced, size=T, mode="linear", align_corners=False)
            results.append(stretched.squeeze(0).permute(1, 0))  # (T, D)

        return torch.stack(results)

    def dropout_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out random tokens."""
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.dropout_p).float()
        return x * mask

    def gaussian_noise(self, x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        """Add Gaussian noise to all tokens."""
        return x + torch.randn_like(x) * sigma

    def apply(self, x: torch.Tensor, ops: Optional[List[str]] = None) -> torch.Tensor:
        """Apply a sequence of augmentation operations."""
        if ops is None:
            ops = random.choices(
                ["time_warp", "magnitude_scale", "window_slice", "dropout_tokens", "gaussian_noise"],
                k=2
            )
        for op in ops:
            fn = getattr(self, op, None)
            if fn is not None:
                x = fn(x)
        return x


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------
class CurriculumScheduler:
    """
    Curriculum learning: start with easy (calm market) samples,
    gradually introduce harder (volatile, crisis) samples.

    Controls the `difficulty` parameter of FinancialDataConfig.
    """

    def __init__(
        self,
        total_steps: int = 100000,
        warmup_steps: int = 10000,
        schedule: str = "linear",
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self._step = 0

    def step(self) -> float:
        """Advance curriculum by one step, return current difficulty."""
        self._step += 1
        return self.get_difficulty()

    def get_difficulty(self) -> float:
        """Return current difficulty in [min_difficulty, max_difficulty]."""
        if self._step < self.warmup_steps:
            # Stay at min during warmup
            return self.min_difficulty

        progress = (self._step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)

        if self.schedule == "linear":
            d = self.min_difficulty + progress * (self.max_difficulty - self.min_difficulty)
        elif self.schedule == "cosine":
            d = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * (
                1 - math.cos(math.pi * progress)
            ) / 2
        elif self.schedule == "step":
            # Step function: easy → medium → hard
            if progress < 0.33:
                d = self.min_difficulty
            elif progress < 0.67:
                d = (self.min_difficulty + self.max_difficulty) / 2
            else:
                d = self.max_difficulty
        else:
            d = self.max_difficulty

        return float(np.clip(d, self.min_difficulty, self.max_difficulty))

    def update_config(self, config: FinancialDataConfig) -> FinancialDataConfig:
        """Update dataset config with current difficulty level."""
        config.difficulty = self.get_difficulty()
        return config

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict):
        self._step = state["step"]
