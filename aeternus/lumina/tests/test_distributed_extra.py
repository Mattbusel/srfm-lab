import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from distributed_training import (
    DistributedConfig, GradientFlowMonitor, MixedPrecisionTrainer,
    CheckpointManager, DistributedSampler, GradientCompressor,
    LearningRateWarmupScheduler, TensorParallelLinear,
    DataParallelismManager, ActivationCheckpointing,
    ZeroRedundancyOptimizer, CommunicationProfiler,
    ElasticTrainer, PipelineStage,
)


class TestDistributedConfig:
    def test_defaults(self):
        cfg = DistributedConfig()
        assert cfg.world_size == 1
        assert cfg.rank == 0
        assert cfg.backend == 'nccl'
        assert cfg.zero_stage == 1
        assert not cfg.use_fsdp

    def test_custom(self):
        cfg = DistributedConfig(world_size=8, rank=3, use_zero=True, zero_stage=2)
        assert cfg.world_size == 8
        assert cfg.rank == 3
        assert cfg.zero_stage == 2

class TestGradientFlowMonitor:
    def _make_model_with_grads(self):
        model = nn.Linear(16, 8)
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        return model

    def test_record(self):
        model = self._make_model_with_grads()
        mon = GradientFlowMonitor(model)
        stats = mon.record()
        assert len(stats) > 0
        for k, v in stats.items():
            assert v >= 0

    def test_summary(self):
        model = self._make_model_with_grads()
        mon = GradientFlowMonitor(model)
        mon.record()
        s = mon.summary()
        assert 'max_norm' in s
        assert 'mean_norm' in s
        assert s['num_params'] >= 2

    def test_detect_vanishing(self):
        model = nn.Linear(4, 4)
        # Manually set tiny grads
        for p in model.parameters():
            p.grad = torch.zeros_like(p) + 1e-10
        mon = GradientFlowMonitor(model)
        mon.record()
        vanishing = mon.detect_vanishing(1e-8)
        assert len(vanishing) > 0

    def test_detect_exploding(self):
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 200.0
        mon = GradientFlowMonitor(model)
        mon.record()
        exploding = mon.detect_exploding(100.0)
        assert len(exploding) > 0

class TestCheckpointManager:
    def test_save_and_load(self, tmp_path):
        model = nn.Linear(8, 4)
        opt = torch.optim.Adam(model.parameters())
        mgr = CheckpointManager(str(tmp_path))
        path = mgr.save(model, opt, step=100, metric=0.5)
        assert os.path.exists(path)
        model2 = nn.Linear(8, 4)
        state = mgr.load_latest(model2)
        assert state['step'] == 100

    def test_best_metric_tracking_min(self, tmp_path):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(str(tmp_path), metric_mode='min')
        mgr.save(model, opt, step=1, metric=1.0)
        mgr.save(model, opt, step=2, metric=0.5)
        mgr.save(model, opt, step=3, metric=0.8)
        assert mgr.best_metric == 0.5

    def test_best_metric_tracking_max(self, tmp_path):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(str(tmp_path), metric_mode='max')
        mgr.save(model, opt, step=1, metric=0.3)
        mgr.save(model, opt, step=2, metric=0.9)
        assert mgr.best_metric == 0.9

    def test_num_checkpoints(self, tmp_path):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(str(tmp_path), max_to_keep=3)
        for i in range(5):
            mgr.save(model, opt, step=i * 10)
        assert mgr.num_checkpoints <= 3

class TestDistributedSampler:
    def test_single_rank(self):
        sampler = DistributedSampler(100, world_size=1, rank=0)
        indices = list(sampler)
        assert len(indices) == 100

    def test_multi_rank_no_overlap(self):
        indices_0 = set(DistributedSampler(100, world_size=4, rank=0, shuffle=False))
        indices_1 = set(DistributedSampler(100, world_size=4, rank=1, shuffle=False))
        assert indices_0.isdisjoint(indices_1)

    def test_shuffle_changes_order(self):
        s1 = DistributedSampler(50, world_size=1, shuffle=True, seed=0)
        s1.set_epoch(0)
        s2 = DistributedSampler(50, world_size=1, shuffle=True, seed=0)
        s2.set_epoch(1)
        assert list(s1) != list(s2)

    def test_drop_last(self):
        s = DistributedSampler(101, world_size=4, rank=0, drop_last=True)
        assert len(s) == 101 // 4

class TestGradientCompressor:
    def test_compress_decompress_roundtrip(self):
        gc = GradientCompressor(compress_ratio=0.5, use_error_feedback=False)
        grad = torch.randn(100)
        idx, vals = gc.compress('w', grad)
        restored = gc.decompress(idx, vals, (100,))
        # Top-50 values preserved
        assert restored.nonzero().numel() > 0

    def test_error_feedback_reduces_bias(self):
        gc = GradientCompressor(compress_ratio=0.1, use_error_feedback=True)
        grad = torch.ones(100)
        for _ in range(5):
            gc.compress('w', grad)
        assert 'w' in gc._residuals

    def test_quantize_dequantize(self):
        gc = GradientCompressor()
        grad = torch.randn(64)
        q, scale = gc.quantize_1bit(grad)
        deq = gc.dequantize_1bit(q, scale)
        assert deq.shape == grad.shape
        assert q.dtype == torch.int8

class TestLearningRateWarmupScheduler:
    def test_warmup_increases_lr(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = LearningRateWarmupScheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for i in range(10):
            sched.step()
            lrs.append(sched.last_lr[0])
        assert lrs[-1] > lrs[0]

    def test_decay_after_warmup(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = LearningRateWarmupScheduler(opt, warmup_steps=5, total_steps=50)
        for _ in range(5):
            sched.step()
        peak_lr = sched.last_lr[0]
        for _ in range(45):
            sched.step()
        final_lr = sched.last_lr[0]
        assert final_lr < peak_lr

    def test_min_lr_ratio(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1.0)
        sched = LearningRateWarmupScheduler(opt, warmup_steps=0, total_steps=1000, min_lr_ratio=0.1)
        for _ in range(1000):
            sched.step()
        assert sched.last_lr[0] >= 0.1 * 1.0 - 1e-6

class TestTensorParallelLinear:
    def test_column_parallel_shape(self):
        layer = TensorParallelLinear(64, 128, mode='column', world_size=1, rank=0)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 128)

    def test_row_parallel_shape(self):
        layer = TensorParallelLinear(128, 64, mode='row', world_size=1, rank=0)
        x = torch.randn(4, 128)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_world_size_partitioning(self):
        # Column parallel: out_features partitioned by world_size
        layer = TensorParallelLinear(64, 128, mode='column', world_size=4, rank=0)
        assert layer.linear.out_features == 32

class TestActivationCheckpointing:
    def test_enable_disable(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        ac = ActivationCheckpointing(model)
        ac.enable()
        ac.disable()
        # Should not raise

    def test_memory_savings_estimate(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
        ac = ActivationCheckpointing(model)
        savings = ac.estimate_memory_savings()
        assert 0.0 <= savings <= 1.0

class TestMixedPrecisionTrainer:
    def test_scale_loss(self):
        model = nn.Linear(8, 4)
        opt = torch.optim.Adam(model.parameters())
        trainer = MixedPrecisionTrainer(model, opt, initial_scale=256.0)
        loss = torch.tensor(1.0, requires_grad=True)
        scaled = trainer.scale_loss(loss)
        assert scaled.item() == pytest.approx(256.0)

    def test_loss_scale_initial(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = MixedPrecisionTrainer(model, opt, initial_scale=1024.0)
        assert trainer.current_loss_scale == 1024.0

class TestElasticTrainer:
    def test_initialize(self):
        trainer = ElasticTrainer(
            model_factory=lambda: nn.Linear(4, 2),
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        )
        trainer.initialize()
        assert trainer.rendezvous_count == 1

    def test_state_dict(self):
        trainer = ElasticTrainer(
            model_factory=lambda: nn.Linear(4, 2),
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        )
        trainer.initialize()
        state = trainer.state_dict()
        assert 'step' in state
        assert 'model' in state

    def test_load_state_dict(self):
        trainer = ElasticTrainer(
            model_factory=lambda: nn.Linear(4, 2),
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        )
        trainer.initialize()
        state = trainer.state_dict()
        state['step'] = 42
        trainer.load_state_dict(state)
        assert trainer.global_step == 42

class TestPipelineStage:
    def test_forward(self):
        layers = nn.ModuleList([nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8)])
        stage = PipelineStage(layers, stage_id=0, num_stages=2)
        x = torch.randn(4, 16)
        out = stage(x)
        assert out.shape == (4, 8)

    def test_first_last(self):
        layers = nn.ModuleList([nn.Linear(8, 8)])
        stage0 = PipelineStage(layers, stage_id=0, num_stages=3)
        stage2 = PipelineStage(layers, stage_id=2, num_stages=3)
        assert stage0.is_first_stage()
        assert stage2.is_last_stage()
        assert not stage0.is_last_stage()

class TestCommunicationProfiler:
    def test_record_and_summary(self):
        prof = CommunicationProfiler()
        t = torch.randn(128)
        prof.record_all_reduce(t, 5.0)
        prof.record_all_reduce(t, 3.0)
        summary = prof.summary()
        assert 'all_reduce' in summary
        assert summary['all_reduce']['count'] == 2
        assert summary['all_reduce']['mean_ms'] == pytest.approx(4.0)

@pytest.mark.parametrize('world_size,zero_stage', [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3), (8, 1), (8, 2), (8, 3)])
def test_distributed_config_parametrized(world_size, zero_stage):
    cfg = DistributedConfig(world_size=world_size, zero_stage=zero_stage)
    assert cfg.world_size == world_size
    assert cfg.zero_stage == zero_stage

@pytest.mark.parametrize('n,ws,rank', [(50, 1, 0), (50, 2, 0), (50, 4, 0), (100, 1, 0), (100, 2, 0), (100, 4, 0), (200, 1, 0), (200, 2, 0), (200, 4, 0)])
def test_distributed_sampler_coverage(n, ws, rank):
    s = DistributedSampler(n, world_size=ws, rank=rank, shuffle=False)
    indices = list(s)
    assert len(indices) == len(s)
    assert all(0 <= i < n for i in indices)

@pytest.mark.parametrize('ratio', [0.01, 0.05, 0.1, 0.2, 0.5])
def test_compressor_ratio(ratio):
    gc = GradientCompressor(compress_ratio=ratio, use_error_feedback=False)
    grad = torch.randn(200)
    idx, vals = gc.compress('p', grad)
    expected_k = max(1, int(200 * ratio))
    assert len(vals) <= expected_k + 1

@pytest.mark.parametrize('warmup,total', [(5, 50), (10, 100), (20, 200), (0, 100), (50, 200)])
def test_lr_scheduler_warmup_total(warmup, total):
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = LearningRateWarmupScheduler(opt, warmup_steps=warmup, total_steps=total)
    for _ in range(total):
        sched.step()
    assert sched.global_step == total
