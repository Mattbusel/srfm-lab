"""
test_distributed.py — Tests for distributed JAX operations (TensorNet AETERNUS).
"""

from __future__ import annotations

import os
import pickle
import tempfile
import pytest
import numpy as np
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor_net.distributed_jit import (
    get_available_devices,
    n_devices,
    cast_cores,
    DTYPE_MAP,
    mixed_precision_forward,
    MixedPrecisionConfig,
    GradientAccumulator,
    ShardingStrategy,
    replicate_cores,
    pipeline_partition_cores,
    fused_tt_matvec,  # imported from kernel_fusion actually, but tested here
    vmap_tt_contract,
    vmap_mps_inner,
    pmap_tt_norm,
    PipelineParallelTT,
    DistributedTrainingConfig,
    make_distributed_optimizer,
    DistributedTTTrainer,
    save_cores_checkpoint,
    load_cores_checkpoint,
    scatter_data,
    distributed_hparam_sweep,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_cores(rng):
    """Simple 3-core TT decomposition."""
    return [
        jnp.array(rng.normal(0, 0.1, (1, 4, 4)).astype(np.float32)),
        jnp.array(rng.normal(0, 0.1, (4, 4, 4)).astype(np.float32)),
        jnp.array(rng.normal(0, 0.1, (4, 4, 1)).astype(np.float32)),
    ]


@pytest.fixture
def small_dataset(rng):
    return rng.normal(0, 1, (64, 32)).astype(np.float32)


# ============================================================================
# Device utilities
# ============================================================================

class TestDeviceUtilities:

    def test_get_devices_returns_list(self):
        devices = get_available_devices()
        assert isinstance(devices, list)
        assert len(devices) >= 1

    def test_n_devices_positive(self):
        assert n_devices() >= 1

    def test_get_cpu_devices(self):
        cpu_devices = get_available_devices("cpu")
        assert isinstance(cpu_devices, list)
        assert len(cpu_devices) >= 1


# ============================================================================
# Mixed precision
# ============================================================================

class TestMixedPrecision:

    def test_cast_cores_bfloat16(self, simple_cores):
        cast = cast_cores(simple_cores, "bfloat16")
        for c in cast:
            assert c.dtype == jnp.bfloat16

    def test_cast_cores_float32(self, simple_cores):
        cast = cast_cores(simple_cores, "float32")
        for c in cast:
            assert c.dtype == jnp.float32

    def test_cast_cores_dtype_object(self, simple_cores):
        cast = cast_cores(simple_cores, jnp.float32)
        for c in cast:
            assert c.dtype == jnp.float32

    def test_cast_preserves_values(self, simple_cores):
        # Cast to float32 then back — values should be approximately the same
        cast = cast_cores(simple_cores, "float32")
        for orig, c in zip(simple_cores, cast):
            assert np.allclose(np.array(orig), np.array(c), atol=1e-3)

    def test_mixed_precision_forward_output_dtype(self, simple_cores):
        vec = jnp.ones(4)
        result = mixed_precision_forward(
            simple_cores, vec, compute_dtype="bfloat16", output_dtype="float32"
        )
        assert result.dtype == jnp.float32

    def test_mixed_precision_config_defaults(self):
        cfg = MixedPrecisionConfig()
        assert cfg.param_dtype == "float32"
        assert cfg.compute_dtype == "bfloat16"


# ============================================================================
# Gradient accumulation
# ============================================================================

class TestGradientAccumulation:

    def test_accumulator_init(self, simple_cores):
        import optax
        opt = optax.adam(1e-3)
        acc = GradientAccumulator(n_accumulation_steps=4, optimizer=opt)
        acc.init(simple_cores)
        assert acc._grad_buffer is not None
        assert acc._opt_state is not None

    def test_accumulator_returns_true_on_nth_step(self, simple_cores):
        import optax
        n_acc = 4
        opt = optax.adam(1e-3)
        acc = GradientAccumulator(n_accumulation_steps=n_acc, optimizer=opt)
        acc.init(simple_cores)

        grads = [jnp.zeros_like(c) for c in simple_cores]
        results = [acc.accumulate(grads) for _ in range(n_acc)]
        assert results[-1] is True
        assert all(not r for r in results[:-1])

    def test_accumulator_apply_update(self, simple_cores):
        import optax
        opt = optax.adam(1e-3)
        acc = GradientAccumulator(n_accumulation_steps=2, optimizer=opt)
        acc.init(simple_cores)

        # Simulate two accumulation steps
        grads = [jnp.ones_like(c) * 0.01 for c in simple_cores]
        acc.accumulate(grads)
        should_update = acc.accumulate(grads)
        assert should_update

        new_cores, new_opt_state = acc.apply_update(simple_cores)
        assert len(new_cores) == len(simple_cores)


# ============================================================================
# Sharding and replication
# ============================================================================

class TestShardingReplication:

    def test_replicate_cores_shape(self, simple_cores):
        devices = jax.devices()[:1]  # use first device only
        replicated = replicate_cores(simple_cores, devices)
        assert len(replicated) == len(simple_cores)

    def test_pipeline_partition_2_stages(self, simple_cores):
        stages = pipeline_partition_cores(simple_cores, n_stages=2)
        assert len(stages) == 2
        total = sum(len(s) for s in stages)
        assert total == len(simple_cores)

    def test_pipeline_partition_equal_stages(self):
        cores = [np.ones((1, 4, 4)) for _ in range(4)]
        stages = pipeline_partition_cores(cores, n_stages=2)
        assert len(stages[0]) == 2
        assert len(stages[1]) == 2

    def test_pipeline_partition_more_stages_than_cores(self):
        cores = [np.ones((1, 4, 4)) for _ in range(2)]
        stages = pipeline_partition_cores(cores, n_stages=4)
        total = sum(len(s) for s in stages)
        assert total == 2


# ============================================================================
# vmap operations
# ============================================================================

class TestVmapOperations:

    def test_vmap_tt_contract_shape(self, simple_cores, rng):
        batch = jnp.array(rng.normal(0, 1, (8, 64)).astype(np.float32))
        result = vmap_tt_contract(simple_cores, batch)
        assert result.shape[0] == 8

    def test_vmap_mps_inner_scalar(self, simple_cores):
        result = vmap_mps_inner(simple_cores, simple_cores)
        # Inner product of TT with itself should be positive (|TT|^2 >= 0)
        assert result.shape == () or result.ndim == 0

    def test_pmap_tt_norm_positive(self, simple_cores):
        norm = pmap_tt_norm(simple_cores)
        # Norm should be non-negative
        assert float(norm) >= 0.0


# ============================================================================
# Pipeline parallelism
# ============================================================================

class TestPipelineParallelTT:

    def test_pipeline_forward(self, simple_cores, rng):
        devices = jax.devices()[:1]
        pipeline = PipelineParallelTT(simple_cores, n_stages=1, devices=devices)
        x = jnp.array(rng.normal(0, 1, (4,)).astype(np.float32))
        result = pipeline.forward(x)
        assert result is not None

    def test_pipeline_update_cores(self, simple_cores, rng):
        devices = jax.devices()[:1]
        pipeline = PipelineParallelTT(simple_cores, n_stages=1, devices=devices)
        new_cores = [c * 2.0 for c in simple_cores]
        pipeline.update_cores(new_cores)
        # Should not raise
        assert pipeline.stage_cores is not None


# ============================================================================
# Distributed training
# ============================================================================

class TestDistributedTrainer:

    def test_trainer_init(self, simple_cores):
        def dummy_loss(cores, batch):
            return jnp.mean(batch ** 2)

        config = DistributedTrainingConfig(
            n_epochs=2,
            batch_size=16,
            log_every=1,
            checkpoint_every=0,
        )
        trainer = DistributedTTTrainer(simple_cores, dummy_loss, config)
        assert trainer._step == 0
        assert trainer._epoch == 0

    def test_trainer_train_step(self, simple_cores, small_dataset):
        def dummy_loss(cores, batch):
            return jnp.mean(batch ** 2)

        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=16,
            checkpoint_every=0,
            log_every=0,
        )
        trainer = DistributedTTTrainer(simple_cores, dummy_loss, config)
        batch = jnp.array(small_dataset[:16])
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_epoch(self, simple_cores, small_dataset):
        def dummy_loss(cores, batch):
            return jnp.mean(batch ** 2)

        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=16,
            checkpoint_every=0,
            log_every=0,
        )
        trainer = DistributedTTTrainer(simple_cores, dummy_loss, config)
        mean_loss = trainer.train_epoch(small_dataset)
        assert isinstance(mean_loss, float)
        assert trainer._epoch == 1

    def test_trainer_loss_history(self, simple_cores, small_dataset):
        def dummy_loss(cores, batch):
            return jnp.mean(batch ** 2)

        config = DistributedTrainingConfig(
            n_epochs=2,
            batch_size=16,
            checkpoint_every=0,
            log_every=0,
        )
        trainer = DistributedTTTrainer(simple_cores, dummy_loss, config)
        trainer.train(small_dataset)
        assert len(trainer.loss_history) > 0


# ============================================================================
# Checkpointing
# ============================================================================

class TestCheckpointing:

    def test_save_and_load_npz(self, simple_cores, tmp_path):
        path = str(tmp_path / "cores.npz")
        saved_path = save_cores_checkpoint(simple_cores, path, metadata={"rank": 4})
        loaded_cores, metadata = load_cores_checkpoint(saved_path)
        assert len(loaded_cores) == len(simple_cores)
        for orig, loaded in zip(simple_cores, loaded_cores):
            assert np.allclose(np.array(orig), np.array(loaded), atol=1e-5)

    def test_trainer_checkpoint(self, simple_cores, small_dataset, tmp_path):
        def dummy_loss(cores, batch):
            return jnp.mean(batch ** 2)

        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=16,
            checkpoint_every=1,
            checkpoint_dir=str(tmp_path),
            log_every=0,
        )
        trainer = DistributedTTTrainer(simple_cores, dummy_loss, config)
        trainer.train_epoch(small_dataset)

        # Find saved checkpoint
        ckpts = list(tmp_path.glob("*.pkl"))
        assert len(ckpts) >= 1


# ============================================================================
# Hyperparameter sweep
# ============================================================================

class TestHParamSweep:

    def test_sweep_returns_sorted_results(self, small_dataset, rng):
        def dummy_train(hparams, data):
            r = hparams.get("rank", 4)
            return float(1.0 / r)  # lower rank = worse

        grid = {"rank": [2, 4, 8], "lr": [0.01, 0.001]}
        results = distributed_hparam_sweep(dummy_train, grid, small_dataset)
        assert len(results) == 6
        # Results should be sorted by val_loss ascending
        losses = [r["val_loss"] for r in results]
        assert losses == sorted(losses)

    def test_sweep_handles_failure(self, small_dataset):
        def failing_train(hparams, data):
            if hparams["rank"] == 4:
                raise RuntimeError("Intentional test failure")
            return 0.5

        grid = {"rank": [2, 4, 8]}
        results = distributed_hparam_sweep(failing_train, grid, small_dataset)
        # Should not crash; failure should produce inf val_loss
        assert any(r["val_loss"] == float("inf") for r in results)
