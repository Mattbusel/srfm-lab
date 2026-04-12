"""Tests for pretraining.py extended components."""
import pytest
import torch
import torch.nn as nn

def _make_simple_lm(vocab=100, d=32, seq=16):
    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.head = nn.Linear(d, vocab, bias=False)
        def forward(self, ids):
            return self.head(self.emb(ids))
    return LM()

def _make_seq_encoder(in_dim=8, d=32):
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, d)
        def forward(self, x):
            return self.proj(x)
    return Enc()

class TestMaskedLanguageModeling:
    def test_loss_is_scalar(self):
        from pretraining import MaskedLanguageModeling
        lm = _make_simple_lm()
        mlm = MaskedLanguageModeling(lm, vocab_size=100, d_model=32)
        ids = torch.randint(1, 100, (2, 16))
        loss, logits = mlm(ids)
        assert loss.dim() == 0
        assert logits.shape == (2, 16, 100)

    def test_loss_positive(self):
        from pretraining import MaskedLanguageModeling
        lm = _make_simple_lm()
        mlm = MaskedLanguageModeling(lm, vocab_size=100, d_model=32)
        ids = torch.randint(1, 100, (2, 16))
        loss, _ = mlm(ids)
        assert loss.item() > 0

    def test_gradient_flows(self):
        from pretraining import MaskedLanguageModeling
        lm = _make_simple_lm()
        mlm = MaskedLanguageModeling(lm, vocab_size=100, d_model=32)
        ids = torch.randint(1, 100, (2, 8))
        loss, _ = mlm(ids)
        loss.backward()
        has_grad = any(p.grad is not None for p in mlm.parameters())
        assert has_grad

class TestCausalLanguageModeling:
    def test_loss_scalar(self):
        from pretraining import CausalLanguageModeling
        lm = _make_simple_lm()
        clm = CausalLanguageModeling(lm, vocab_size=100, d_model=32)
        ids = torch.randint(0, 100, (2, 16))
        loss, logits = clm(ids)
        assert loss.dim() == 0
        assert logits.shape == (2, 15, 100)

    def test_loss_positive(self):
        from pretraining import CausalLanguageModeling
        lm = _make_simple_lm()
        clm = CausalLanguageModeling(lm, vocab_size=100, d_model=32)
        ids = torch.randint(0, 100, (2, 8))
        loss, _ = clm(ids)
        assert loss.item() > 0

class TestFinancialMaskedModeling:
    def test_loss_scalar(self):
        from pretraining import FinancialMaskedModeling
        enc = _make_seq_encoder(8, 32)
        fmlm = FinancialMaskedModeling(enc, d_model=32, input_dim=8)
        x = torch.randn(2, 16, 8)
        loss, preds = fmlm(x)
        assert loss.dim() == 0
        assert preds.shape == (2, 16, 8)

    def test_loss_non_negative(self):
        from pretraining import FinancialMaskedModeling
        enc = _make_seq_encoder(4, 16)
        fmlm = FinancialMaskedModeling(enc, d_model=16, input_dim=4)
        x = torch.randn(2, 8, 4)
        loss, _ = fmlm(x)
        assert loss.item() >= 0

class TestTimeSeriesContrastiveLearning:
    def test_loss_and_accuracy(self):
        from pretraining import TimeSeriesContrastiveLearning
        enc = _make_seq_encoder(8, 32)
        cl = TimeSeriesContrastiveLearning(enc, d_model=32, projection_dim=16)
        x = torch.randn(4, 16, 8)
        loss, metrics = cl(x)
        assert loss.dim() == 0
        assert 'contrastive_loss' in metrics
        assert 'contrastive_accuracy' in metrics

    def test_loss_positive(self):
        from pretraining import TimeSeriesContrastiveLearning
        enc = _make_seq_encoder(4, 16)
        cl = TimeSeriesContrastiveLearning(enc, d_model=16, projection_dim=8)
        x = torch.randn(4, 8, 4)
        loss, _ = cl(x)
        assert loss.item() >= 0

class TestDataAugmentationPipeline:
    def test_jitter_shape(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline()
        aug.train()
        x = torch.randn(2, 16, 8)
        out = aug(x, ['jitter'])
        assert out.shape == x.shape

    def test_scaling_shape(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline()
        aug.train()
        x = torch.randn(2, 16, 8)
        out = aug(x, ['scaling'])
        assert out.shape == x.shape

    def test_temporal_dropout_shape(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline(dropout_prob=0.2)
        aug.train()
        x = torch.randn(2, 16, 8)
        out = aug(x, ['temporal_dropout'])
        assert out.shape == x.shape

    def test_no_aug_in_eval(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline(jitter_std=10.0)
        aug.eval()
        x = torch.randn(2, 8, 4)
        out = aug(x, ['jitter'])
        assert torch.allclose(out, x)

    def test_magnitude_warp_shape(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline()
        aug.train()
        x = torch.randn(2, 16, 4)
        out = aug(x, ['magnitude_warp'])
        assert out.shape == x.shape

    def test_window_slice_shape(self):
        from pretraining import DataAugmentationPipeline
        aug = DataAugmentationPipeline()
        aug.train()
        x = torch.randn(2, 32, 4)
        out = aug(x, ['window_slice'])
        assert out.shape == x.shape

class TestWarmupCosineDecayScheduler:
    def test_warmup_phase(self):
        from pretraining import WarmupCosineDecayScheduler
        model = nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineDecayScheduler(opt, warmup_steps=100, total_steps=1000)
        # At step 0: LR should be 0
        lrs = sched.get_lr(0)
        assert lrs[0] == 0

    def test_lr_increases_during_warmup(self):
        from pretraining import WarmupCosineDecayScheduler
        model = nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineDecayScheduler(opt, warmup_steps=100, total_steps=1000)
        lr50 = sched.get_lr(50)[0]
        lr100 = sched.get_lr(100)[0]
        assert lr50 < lr100

    def test_state_dict_roundtrip(self):
        from pretraining import WarmupCosineDecayScheduler
        model = nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineDecayScheduler(opt, 100, 1000)
        for _ in range(10):
            sched.step()
        sd = sched.state_dict()
        sched.load_state_dict(sd)
        assert sched._step == 10

class TestOnlineLossMonitor:
    def test_update_and_summary(self):
        from pretraining import OnlineLossMonitor
        monitor = OnlineLossMonitor()
        for i in range(50):
            monitor.update(3.0 - i * 0.01)
        s = monitor.summary()
        assert 'ema_loss' in s
        assert s['ema_loss'] is not None

    def test_plateau_detection(self):
        from pretraining import OnlineLossMonitor
        monitor = OnlineLossMonitor(patience=10)
        for _ in range(20):
            monitor.update(2.5)  # constant loss
        assert monitor.is_plateauing()

    def test_loss_trend_negative_when_improving(self):
        from pretraining import OnlineLossMonitor
        monitor = OnlineLossMonitor()
        for i in range(200):
            monitor.update(3.0 - i * 0.005)
        trend = monitor.loss_trend(100)
        assert trend < 0

class TestGradientAccumulatorWithWarmup:
    def test_accumulation(self):
        from pretraining import GradientAccumulatorWithWarmup
        model = nn.Linear(8, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        accum = GradientAccumulatorWithWarmup(model, opt, accumulation_steps=4)
        stepped = []
        for i in range(8):
            x = torch.randn(2, 8)
            loss = model(x).sum()
            was_stepped = accum.step(loss)
            stepped.append(was_stepped)
        assert sum(stepped) == 2
        assert accum.global_step() == 2

    def test_gradient_norm_clipping(self):
        from pretraining import GradientAccumulatorWithWarmup
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        accum = GradientAccumulatorWithWarmup(model, opt, accumulation_steps=1, max_grad_norm=0.1)
        x = torch.randn(2, 4) * 100
        loss = model(x).sum()
        accum.step(loss)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm() <= 0.2  # loose check

@pytest.mark.parametrize('B,T,D,mask_prob', [
    (2, 8, 4, 0.15), (4, 16, 8, 0.20), (2, 32, 4, 0.10),
    (1, 4, 8, 0.30), (8, 8, 4, 0.15), (2, 16, 16, 0.20),
])
def test_financial_mlm_parametrized(B, T, D, mask_prob):
    from pretraining import FinancialMaskedModeling
    enc = nn.Sequential(nn.Linear(D, 32), nn.ReLU())
    # Wrap to return sequence output
    class SeqEnc(nn.Module):
        def __init__(self): super().__init__(); self.enc = enc
        def forward(self, x): return self.enc(x)
    fmlm = FinancialMaskedModeling(SeqEnc(), d_model=32, input_dim=D, mask_prob=mask_prob)
    x = torch.randn(B, T, D)
    loss, preds = fmlm(x)
    assert not torch.isnan(loss)
    assert preds.shape == (B, T, D)
