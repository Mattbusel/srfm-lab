"""
tests/test_distributed.py

Tests for distributed_training.py module.
Tests run in single-process mode (no distributed env vars needed).
"""

from __future__ import annotations

import math
import os
import pathlib
import tempfile
import unittest
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# Ensure imports work without the package being installed
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.distributed_training import (
    DistributedConfig,
    TrainingState,
    AMPContext,
    GradAccumManager,
    CheckpointManager,
    ModelEMA,
    MetricsAggregator,
    cosine_with_warmup_schedule,
    wsd_schedule,
    get_linear_schedule_with_warmup,
    CyclicCosineScheduler,
    build_optimizer,
    clip_grad_norm,
    compute_grad_norm,
    count_parameters,
    model_memory_estimate,
    LAMB,
    ColumnParallelLinear,
    RowParallelLinear,
    ZeROGradientSharder,
    InfiniteDistributedSampler,
    CheckpointedSequential,
)


# ---------------------------------------------------------------------------
# Simple test model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, x: Tensor, labels: Tensor = None) -> Dict[str, Tensor]:
        h = torch.relu(self.fc1(x))
        logit = self.fc2(h).squeeze(-1)
        out = {"logits": logit}
        if labels is not None:
            out["loss"] = nn.functional.binary_cross_entropy_with_logits(logit, labels.float())
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistributedConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = DistributedConfig()
        self.assertEqual(cfg.backend, "nccl")
        self.assertEqual(cfg.strategy, "ddp")
        self.assertEqual(cfg.grad_accum_steps, 1)
        self.assertTrue(cfg.use_amp)

    def test_custom_config(self):
        cfg = DistributedConfig(
            strategy="fsdp",
            grad_accum_steps=8,
            max_grad_norm=0.5,
        )
        self.assertEqual(cfg.strategy, "fsdp")
        self.assertEqual(cfg.grad_accum_steps, 8)


class TestTrainingState(unittest.TestCase):

    def test_to_dict_and_back(self):
        state = TrainingState(step=100, epoch=2, best_val_loss=0.5)
        d = state.to_dict()
        state2 = TrainingState.from_dict(d)
        self.assertEqual(state2.step, 100)
        self.assertEqual(state2.epoch, 2)
        self.assertAlmostEqual(state2.best_val_loss, 0.5)

    def test_list_fields(self):
        state = TrainingState()
        state.loss_history.extend([0.9, 0.8, 0.7])
        d = state.to_dict()
        state2 = TrainingState.from_dict(d)
        self.assertEqual(state2.loss_history, [0.9, 0.8, 0.7])


class TestAMPContext(unittest.TestCase):

    def test_cpu_amp_disabled(self):
        cfg = DistributedConfig(use_amp=True)
        amp = AMPContext(cfg)
        # On CPU, AMP should be disabled
        if not torch.cuda.is_available():
            self.assertFalse(amp.enabled)

    def test_scale_loss(self):
        cfg = DistributedConfig(use_amp=False)
        amp = AMPContext(cfg)
        loss = torch.tensor(1.0)
        scaled = amp.scale(loss)
        self.assertAlmostEqual(scaled.item(), 1.0)  # Scaler=1.0 when disabled

    def test_autocast_context(self):
        cfg = DistributedConfig(use_amp=False)
        amp = AMPContext(cfg)
        with amp.autocast():
            x = torch.randn(4, 4)
            y = x @ x.T
        self.assertEqual(y.shape, (4, 4))


class TestGradAccumManager(unittest.TestCase):

    def test_should_sync(self):
        model = TinyModel()
        accum = GradAccumManager(model, accum_steps=4)
        self.assertFalse(accum.should_sync)  # step 0: not should sync yet

        # Advance 3 steps
        for _ in range(3):
            with accum.accumulate():
                pass

        self.assertTrue(accum.should_sync)  # step 4 should sync

    def test_normalize_loss(self):
        model = TinyModel()
        accum = GradAccumManager(model, accum_steps=4)
        loss = torch.tensor(4.0)
        normed = accum.normalize_loss(loss)
        self.assertAlmostEqual(normed.item(), 1.0)

    def test_reset(self):
        model = TinyModel()
        accum = GradAccumManager(model, accum_steps=2)
        with accum.accumulate():
            pass
        accum.reset()
        self.assertEqual(accum._step, 0)


class TestCheckpointManager(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            state = TrainingState(step=100, epoch=5)

            ckpt_mgr = CheckpointManager(tmpdir, keep_last_n=3, rank=0, world_size=1)
            ckpt_path = ckpt_mgr.save(100, model, optimizer, None, state)

            self.assertTrue((ckpt_path / "checkpoint.pt").exists())
            self.assertTrue((ckpt_path / "meta.json").exists())

            # Load back
            model2 = TinyModel()
            opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            loaded_state = ckpt_mgr.load(ckpt_path, model2, opt2)

            self.assertEqual(loaded_state.step, 100)
            self.assertEqual(loaded_state.epoch, 5)

            # Check weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_pruning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters())
            state = TrainingState()

            ckpt_mgr = CheckpointManager(tmpdir, keep_last_n=2, rank=0, world_size=1)

            # Save 4 checkpoints — should keep only last 2
            for step in [100, 200, 300, 400]:
                state.step = step
                ckpt_mgr.save(step, model, optimizer, None, state)

            ckpts = ckpt_mgr.list_checkpoints()
            self.assertLessEqual(len(ckpts), 2)

    def test_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters())
            state = TrainingState()
            ckpt_mgr = CheckpointManager(tmpdir, keep_last_n=5, rank=0, world_size=1)

            self.assertIsNone(ckpt_mgr.latest_checkpoint())

            for step in [50, 100, 150]:
                state.step = step
                ckpt_mgr.save(step, model, optimizer, None, state)

            latest = ckpt_mgr.latest_checkpoint()
            self.assertIsNotNone(latest)
            self.assertIn("00000150", latest.name)


class TestSchedulers(unittest.TestCase):

    def test_cosine_warmup(self):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup_schedule(opt, warmup_steps=10, total_steps=100)

        # Before warmup
        sched.step()
        lr_at_1 = opt.param_groups[0]["lr"]
        self.assertLess(lr_at_1, 1e-3)

        # After warmup
        for _ in range(15):
            sched.step()
        lr_at_16 = opt.param_groups[0]["lr"]
        self.assertLessEqual(lr_at_16, 1e-3)

    def test_wsd_schedule(self):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = wsd_schedule(opt, warmup_steps=5, stable_steps=10, decay_steps=5)

        # During stable phase
        for _ in range(10):
            sched.step()
        lr_stable = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_stable, 1e-3, places=5)

    def test_cyclic_cosine(self):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = CyclicCosineScheduler(opt, first_cycle_steps=50, max_lr=1e-3, min_lr=1e-5)

        lrs = []
        for _ in range(60):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        self.assertGreater(max(lrs), 5e-4)   # Should reach near max
        self.assertLess(min(lrs), 1e-4)       # Should reach near min

    def test_linear_warmup(self):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
        for _ in range(10):
            sched.step()
        lr_at_peak = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_at_peak, 1e-3, places=5)


class TestOptimizer(unittest.TestCase):

    def test_build_adamw(self):
        model = TinyModel()
        opt = build_optimizer(model, lr=1e-4, optimizer_type="adamw")
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(len(opt.param_groups), 2)  # decay and no-decay groups

    def test_lamb_optimizer(self):
        model = TinyModel()
        opt = LAMB(model.parameters(), lr=1e-3)
        x = torch.randn(8, 64)
        y = torch.randint(0, 2, (8,)).float()
        out = model(x, y)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()

    def test_weight_decay_separation(self):
        model = TinyModel()
        opt = build_optimizer(model, lr=1e-4, weight_decay=0.1)
        wd_groups = [pg["weight_decay"] for pg in opt.param_groups]
        self.assertIn(0.1, wd_groups)   # At least one group with WD
        self.assertIn(0.0, wd_groups)   # At least one group without WD


class TestGradientUtils(unittest.TestCase):

    def test_compute_grad_norm(self):
        model = TinyModel()
        x = torch.randn(4, 64)
        y = torch.randint(0, 2, (4,)).float()
        out = model(x, y)
        out["loss"].backward()
        norm = compute_grad_norm(model)
        self.assertGreater(norm, 0)

    def test_clip_grad_norm(self):
        model = TinyModel()
        # Set large gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 100.0
        norm_before = compute_grad_norm(model)
        clip_grad_norm(model, max_norm=1.0)
        norm_after = compute_grad_norm(model)
        self.assertLessEqual(norm_after, 1.01)  # Should be clipped


class TestModelEMA(unittest.TestCase):

    def test_ema_update(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.99)

        # Original params
        orig_params = {n: p.clone() for n, p in model.named_parameters()}

        # Modify model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        # Update EMA
        ema.update(model)

        # EMA params should be between original and new
        for name, ema_p in ema.module.named_parameters():
            orig = orig_params[name]
            # ema = 0.99 * orig + 0.01 * (orig + 1) = orig + 0.01
            expected = orig * 0.99 + (orig + 1.0) * 0.01
            self.assertTrue(torch.allclose(ema_p, expected, atol=1e-5))

    def test_ema_state_dict(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.999)
        sd = ema.state_dict()
        self.assertIn("fc1.weight", sd)
        self.assertIn("fc2.weight", sd)


class TestMetricsAggregator(unittest.TestCase):

    def test_basic_aggregation(self):
        agg = MetricsAggregator(world_size=1)
        agg.update({"loss": 1.0, "acc": 0.9})
        agg.update({"loss": 2.0, "acc": 0.8})
        result = agg.compute(sync=False)
        self.assertAlmostEqual(result["loss"], 1.5)
        self.assertAlmostEqual(result["acc"], 0.85)

    def test_reset(self):
        agg = MetricsAggregator(world_size=1)
        agg.update({"loss": 1.0})
        agg.reset()
        result = agg.compute(sync=False)
        self.assertEqual(result, {})


class TestParallelLinear(unittest.TestCase):

    def test_column_parallel_forward(self):
        layer = ColumnParallelLinear(32, 64, world_size=2, rank=0)
        x = torch.randn(4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (4, 32))  # 64/2 = 32

    def test_row_parallel_forward(self):
        layer = RowParallelLinear(64, 32, world_size=2, rank=0)
        x = torch.randn(4, 32)   # 64/2 = 32 local input
        out = layer(x)
        self.assertEqual(out.shape, (4, 32))


class TestCountParameters(unittest.TestCase):

    def test_count_all(self):
        model = TinyModel(64)
        n = count_parameters(model, trainable_only=False)
        # fc1: 64*64 + 64 = 4160, fc2: 64*1 + 1 = 65 => total 4225
        expected = 64 * 64 + 64 + 64 * 1 + 1
        self.assertEqual(n, expected)

    def test_count_trainable(self):
        model = TinyModel(64)
        # Freeze fc1
        for p in model.fc1.parameters():
            p.requires_grad = False
        n_trainable = count_parameters(model, trainable_only=True)
        n_total = count_parameters(model, trainable_only=False)
        self.assertLess(n_trainable, n_total)

    def test_memory_estimate(self):
        model = TinyModel(64)
        mem = model_memory_estimate(model)
        self.assertIn("param_mb", mem)
        self.assertIn("total_gb", mem)
        self.assertGreater(mem["param_mb"], 0)


class TestInfiniteDistributedSampler(unittest.TestCase):

    def test_sample_count(self):
        sampler = InfiniteDistributedSampler(100, rank=0, world_size=2)
        it = iter(sampler)
        samples = [next(it) for _ in range(50)]
        self.assertEqual(len(samples), 50)
        # All should be valid indices
        self.assertTrue(all(0 <= s < 100 for s in samples))

    def test_different_ranks(self):
        sampler0 = InfiniteDistributedSampler(100, rank=0, world_size=2, seed=0)
        sampler1 = InfiniteDistributedSampler(100, rank=1, world_size=2, seed=0)
        it0, it1 = iter(sampler0), iter(sampler1)
        s0 = [next(it0) for _ in range(50)]
        s1 = [next(it1) for _ in range(50)]
        # Different ranks should get different samples
        self.assertNotEqual(s0, s1)


class TestCheckpointedSequential(unittest.TestCase):

    def test_forward(self):
        seq = CheckpointedSequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        x = torch.randn(4, 32)
        out = seq(x)
        self.assertEqual(out.shape, (4, 16))

    def test_gradient_flows(self):
        seq = CheckpointedSequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        x = torch.randn(4, 32)
        loss = seq(x).sum()
        loss.backward()
        # Gradients should exist
        for p in seq.parameters():
            self.assertIsNotNone(p.grad)


class TestZeROGradientSharder(unittest.TestCase):

    def test_assign_params_to_ranks(self):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters())
        params = list(model.parameters())
        sharder = ZeROGradientSharder(opt, params, rank=0, world_size=2, zero_stage=1)
        self.assertGreater(len(sharder._param_to_rank), 0)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
