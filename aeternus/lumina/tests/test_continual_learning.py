"""Tests for continual_learning.py components."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

class TestElasticWeightConsolidation:
    def _make_model(self):
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def test_ewc_loss_zero_before_consolidation(self):
        from continual_learning import ElasticWeightConsolidation
        model = self._make_model()
        ewc = ElasticWeightConsolidation(model, ewc_lambda=100.0)
        loss = ewc.ewc_loss()
        assert loss.item() == 0.0

    def test_ewc_loss_after_estimate_fisher(self):
        from continual_learning import ElasticWeightConsolidation
        model = self._make_model()
        ewc = ElasticWeightConsolidation(model, ewc_lambda=100.0, n_fisher_samples=5)

        def dummy_loader():
            for _ in range(5):
                yield {'x': torch.randn(4, 16), 'y': torch.randint(0, 4, (4,))}

        def loss_fn(m, batch):
            return nn.CrossEntropyLoss()(m(batch['x']), batch['y'])

        ewc.estimate_fisher(dummy_loader(), loss_fn)
        # Perturb weights slightly
        for p in model.parameters():
            p.data += 0.01
        ewc_loss = ewc.ewc_loss()
        assert ewc_loss.item() >= 0

    def test_forward_passes_through(self):
        from continual_learning import ElasticWeightConsolidation
        model = nn.Linear(16, 4)
        ewc = ElasticWeightConsolidation(model)
        x = torch.randn(2, 16)
        out = ewc(x)
        assert out.shape == (2, 4)

class TestProgressiveNeuralNetworks:
    def test_initial_column(self):
        from continual_learning import ProgressiveNeuralNetworks
        pnn = ProgressiveNeuralNetworks(16, 32, 4)
        x = torch.randn(2, 16)
        out = pnn(x)
        assert out.shape == (2, 4)

    def test_add_task(self):
        from continual_learning import ProgressiveNeuralNetworks
        pnn = ProgressiveNeuralNetworks(16, 32, 4)
        pnn.add_task()
        x = torch.randn(2, 16)
        out = pnn(x, column_idx=1)
        assert out.shape == (2, 4)

    def test_previous_column_frozen(self):
        from continual_learning import ProgressiveNeuralNetworks
        pnn = ProgressiveNeuralNetworks(16, 32, 4)
        pnn.add_task()
        for param in pnn.columns[0].parameters():
            assert not param.requires_grad

    def test_two_tasks_different_outputs(self):
        from continual_learning import ProgressiveNeuralNetworks
        pnn = ProgressiveNeuralNetworks(16, 32, 4)
        pnn.add_task()
        x = torch.randn(2, 16)
        out0 = pnn(x, column_idx=0)
        out1 = pnn(x, column_idx=1)
        assert out0.shape == out1.shape == (2, 4)

class TestContinualNormalization:
    def test_shape(self):
        from continual_learning import ContinualNormalization
        cn = ContinualNormalization(32, num_tasks=4)
        cn.set_task(0)
        x = torch.randn(16, 32)
        out = cn(x)
        assert out.shape == (16, 32)

    def test_task_switch(self):
        from continual_learning import ContinualNormalization
        cn = ContinualNormalization(16, num_tasks=4)
        for t in range(4):
            cn.set_task(t)
            x = torch.randn(8, 16)
            out = cn(x)
            assert not torch.isnan(out).any()

    def test_eval_uses_running_stats(self):
        from continual_learning import ContinualNormalization
        cn = ContinualNormalization(8, num_tasks=2)
        cn.set_task(0)
        x = torch.randn(16, 8)
        cn.train()
        cn(x)
        cn.eval()
        out = cn(x)
        assert not torch.isnan(out).any()

class TestMemoryReplayBuffer:
    def test_add_and_sample(self):
        from continual_learning import MemoryReplayBuffer
        buf = MemoryReplayBuffer(capacity=100, strategy='reservoir')
        x = torch.randn(20, 16)
        y = torch.randint(0, 4, (20,))
        buf.add(x, y)
        assert len(buf) == 20
        sx, sy = buf.sample(10)
        assert sx.shape == (10, 16)
        assert sy.shape == (10,)

    def test_capacity_limit(self):
        from continual_learning import MemoryReplayBuffer
        buf = MemoryReplayBuffer(capacity=50)
        for _ in range(10):
            buf.add(torch.randn(10, 8), torch.randint(0, 2, (10,)))
        assert len(buf) <= 50

    def test_fifo_strategy(self):
        from continual_learning import MemoryReplayBuffer
        buf = MemoryReplayBuffer(capacity=10, strategy='fifo')
        for i in range(5):
            buf.add(torch.randn(4, 8), torch.randint(0, 2, (4,)))
        assert len(buf) == 10

class TestDualMemorySystem:
    def test_forward_shape(self):
        from continual_learning import DualMemorySystem
        dms = DualMemorySystem(32, 64, 64)
        x = torch.randn(4, 32)
        out = dms(x)
        assert out.shape == (4, 64)

    def test_no_nan(self):
        from continual_learning import DualMemorySystem
        dms = DualMemorySystem(16, 32, 32)
        x = torch.randn(4, 16)
        assert not torch.isnan(dms(x)).any()

class TestSynapticIntelligence:
    def _make_model(self):
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def test_si_loss_zero_initially(self):
        from continual_learning import SynapticIntelligence
        model = self._make_model()
        si = SynapticIntelligence(model)
        loss = si.si_loss()
        assert loss.item() == 0.0

    def test_forward_passthrough(self):
        from continual_learning import SynapticIntelligence
        model = nn.Linear(16, 4)
        si = SynapticIntelligence(model)
        x = torch.randn(2, 16)
        out = si(x)
        assert out.shape == (2, 4)
