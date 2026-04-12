"""Tests for finetuning.py extended components."""
import pytest
import torch
import torch.nn as nn
import copy

def _make_model():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

class TestLayerFreezer:
    def test_freeze_embeddings(self):
        from finetuning import LayerFreezer, FinetuningConfig
        model = nn.Embedding(100, 32)
        cfg = FinetuningConfig()
        freezer = LayerFreezer(model, cfg)
        freezer.freeze_embeddings()
        assert not any(p.requires_grad for p in model.parameters())

    def test_unfreeze_all(self):
        from finetuning import LayerFreezer, FinetuningConfig
        model = nn.Embedding(100, 32)
        cfg = FinetuningConfig()
        freezer = LayerFreezer(model, cfg)
        freezer.freeze_embeddings()
        freezer.unfreeze_all()
        assert all(p.requires_grad for p in model.parameters())

    def test_num_trainable(self):
        from finetuning import LayerFreezer, FinetuningConfig
        model = _make_model()
        cfg = FinetuningConfig()
        freezer = LayerFreezer(model, cfg)
        total = sum(p.numel() for p in model.parameters())
        trainable = freezer.num_trainable()
        assert trainable == total

class TestMixoutRegularizer:
    def test_forward_passthrough(self):
        from finetuning import MixoutRegularizer
        model = _make_model()
        pretrained = copy.deepcopy(model)
        reg = MixoutRegularizer(model, pretrained, p=0.1)
        x = torch.randn(2, 16)
        out = reg(x)
        assert out.shape == (2, 8)

    def test_mixout_changes_weights(self):
        from finetuning import MixoutRegularizer
        model = _make_model()
        pretrained = copy.deepcopy(model)
        # Perturb model weights
        with torch.no_grad():
            for p in model.parameters():
                p.data += 1.0
        reg = MixoutRegularizer(model, pretrained, p=0.5)
        reg.train()
        reg.apply_mixout()
        # Some weights should now be between original and pretrained
        changed = False
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), pretrained.named_parameters()):
            if not torch.allclose(p1, p2 + 1.0):
                changed = True
                break
        assert changed

class TestTaskVectorFinetuner:
    def test_compute_task_vector(self):
        from finetuning import TaskVectorFinetuner
        pretrained = _make_model()
        finetuned = copy.deepcopy(pretrained)
        with torch.no_grad():
            for p in finetuned.parameters():
                p.data += 0.1
        tvf = TaskVectorFinetuner(pretrained)
        tv = tvf.compute_task_vector(finetuned)
        assert len(tv) > 0
        for name, vec in tv.items():
            assert torch.allclose(vec, torch.full_like(vec, 0.1), atol=1e-5)

    def test_apply_task_vector(self):
        from finetuning import TaskVectorFinetuner
        pretrained = _make_model()
        finetuned = copy.deepcopy(pretrained)
        with torch.no_grad():
            for p in finetuned.parameters():
                p.data += 1.0
        tvf = TaskVectorFinetuner(pretrained)
        tv = tvf.compute_task_vector(finetuned)
        target = copy.deepcopy(pretrained)
        tvf.apply_task_vector(target, tv, alpha=0.5)
        for (n1, p1), (n2, p2) in zip(target.named_parameters(), pretrained.named_parameters()):
            expected = p2 + 0.5
            assert torch.allclose(p1, expected, atol=1e-4)

    def test_combine_task_vectors(self):
        from finetuning import TaskVectorFinetuner
        pretrained = _make_model()
        tvf = TaskVectorFinetuner(pretrained)
        tv1 = {n: torch.ones_like(p) for n, p in pretrained.named_parameters()}
        tv2 = {n: torch.ones_like(p) * 2 for n, p in pretrained.named_parameters()}
        combined = tvf.combine_task_vectors([tv1, tv2], weights=[0.5, 0.5])
        for n, v in combined.items():
            assert torch.allclose(v, torch.full_like(v, 1.5), atol=1e-5)

class TestWiSEFT:
    def test_forward_shape(self):
        from finetuning import WiSEFT
        pretrained = nn.Linear(8, 4)
        finetuned = copy.deepcopy(pretrained)
        with torch.no_grad():
            for p in finetuned.parameters():
                p.data += 0.5
        wise = WiSEFT(pretrained, finetuned, alpha=0.5)
        x = torch.randn(2, 8)
        out = wise(x)
        assert out.shape == (2, 4)

    def test_alpha_zero_equals_pretrained(self):
        from finetuning import WiSEFT
        pretrained = nn.Linear(4, 2)
        finetuned = copy.deepcopy(pretrained)
        with torch.no_grad():
            for p in finetuned.parameters():
                p.data += 1.0
        wise = WiSEFT(pretrained, finetuned, alpha=0.0)
        x = torch.randn(2, 4)
        with torch.no_grad():
            out_wise = wise(x)
            out_pre = pretrained(x)
        assert torch.allclose(out_wise, out_pre, atol=1e-5)

    def test_set_alpha(self):
        from finetuning import WiSEFT
        pretrained = nn.Linear(4, 2)
        finetuned = copy.deepcopy(pretrained)
        wise = WiSEFT(pretrained, finetuned, alpha=0.5)
        wise.set_alpha(0.8)
        assert wise.alpha == 0.8

class TestRegExFinetuner:
    def test_l2_regularization(self):
        from finetuning import RegExFinetuner
        model = nn.Linear(8, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        regularizer = RegExFinetuner(model, opt, l2_lambda=1e-4)
        loss = regularizer.regularization_loss()
        assert loss.item() >= 0

    def test_train_step(self):
        from finetuning import RegExFinetuner
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        regularizer = RegExFinetuner(model, opt, l2_lambda=1e-4)
        x = torch.randn(4, 4)
        batch_loss = model(x).sum()
        total_loss = regularizer.train_step(batch_loss)
        assert isinstance(total_loss, float)
        assert regularizer.step_count == 1

@pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_wise_ft_interpolation(alpha):
    from finetuning import WiSEFT
    pretrained = nn.Linear(8, 4)
    finetuned = copy.deepcopy(pretrained)
    with torch.no_grad():
        for p in finetuned.parameters():
            p.data += 1.0
    wise = WiSEFT(pretrained, finetuned, alpha=alpha)
    x = torch.randn(2, 8)
    with torch.no_grad():
        out = wise(x)
    assert out.shape == (2, 4)
    assert not torch.isnan(out).any()
