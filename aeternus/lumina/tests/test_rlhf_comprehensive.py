"""Comprehensive tests for rlhf.py components."""
import pytest
import torch
import torch.nn as nn

def _make_lm(vocab=100, d=32, layers=2, seq=32):
    embed = nn.Embedding(vocab, d)
    lm_head = nn.Linear(d, vocab)
    class SimpleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = embed
            self.lm_head = lm_head
        def forward(self, ids):
            return self.lm_head(self.embed(ids))
    return SimpleLM()

class TestRewardModel:
    def test_scalar_output_shape(self):
        from rlhf import RewardModel
        lm = _make_lm()
        rm = RewardModel(lm, d_model=32, pooling='last_token')
        ids = torch.randint(0, 100, (2, 16))
        reward = rm(ids)
        assert reward.shape == (2,)

    def test_mean_pooling(self):
        from rlhf import RewardModel
        lm = _make_lm()
        rm = RewardModel(lm, d_model=32, pooling='mean')
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16)
        reward = rm(ids, mask)
        assert reward.shape == (2,)

    def test_no_nan(self):
        from rlhf import RewardModel
        lm = _make_lm()
        rm = RewardModel(lm, d_model=32)
        ids = torch.randint(0, 100, (4, 8))
        assert not torch.isnan(rm(ids)).any()

    def test_gradient_flows(self):
        from rlhf import RewardModel
        lm = _make_lm()
        rm = RewardModel(lm, d_model=32)
        ids = torch.randint(0, 100, (2, 8))
        rm(ids).sum().backward()
        for p in rm.value_head.parameters():
            if p.requires_grad:
                assert p.grad is not None

class TestBradleyTerryLoss:
    def test_chosen_better_gives_lower_loss(self):
        from rlhf import BradleyTerryLoss
        loss_fn = BradleyTerryLoss()
        # Large margin: chosen >> rejected
        r_c = torch.tensor([5.0, 4.0, 3.0])
        r_r = torch.tensor([-1.0, -2.0, -3.0])
        loss_good = loss_fn(r_c, r_r).item()
        # Equal reward -> higher loss
        r_eq = torch.zeros(3)
        loss_bad = loss_fn(r_eq, r_eq).item()
        assert loss_good < loss_bad

    def test_positive_loss(self):
        from rlhf import BradleyTerryLoss
        loss_fn = BradleyTerryLoss()
        r_c = torch.randn(8)
        r_r = torch.randn(8)
        loss = loss_fn(r_c, r_r)
        assert loss.item() > 0

class TestDirectPreferenceOptimization:
    def test_loss_scalar(self):
        from rlhf import DirectPreferenceOptimization
        policy = _make_lm(vocab=100, d=32)
        ref = _make_lm(vocab=100, d=32)
        dpo = DirectPreferenceOptimization(policy, ref, beta=0.1)
        prompt = torch.randint(0, 100, (2, 4))
        chosen = torch.randint(0, 100, (2, 8))
        rejected = torch.randint(0, 100, (2, 8))
        loss, metrics = dpo(prompt, chosen, rejected)
        assert loss.dim() == 0
        assert 'accuracy' in metrics

    def test_gradient_flows(self):
        from rlhf import DirectPreferenceOptimization
        policy = _make_lm(vocab=50, d=16)
        ref = _make_lm(vocab=50, d=16)
        dpo = DirectPreferenceOptimization(policy, ref, beta=0.1)
        prompt = torch.randint(0, 50, (2, 4))
        chosen = torch.randint(0, 50, (2, 4))
        rejected = torch.randint(0, 50, (2, 4))
        loss, _ = dpo(prompt, chosen, rejected)
        loss.backward()
        has_grad = any(p.grad is not None for p in policy.parameters())
        assert has_grad

    def test_ref_model_frozen(self):
        from rlhf import DirectPreferenceOptimization
        policy = _make_lm(vocab=50, d=16)
        ref = _make_lm(vocab=50, d=16)
        dpo = DirectPreferenceOptimization(policy, ref)
        for p in dpo.ref.parameters():
            assert not p.requires_grad

class TestIdentityPreferenceOptimization:
    def test_loss_shape(self):
        from rlhf import IdentityPreferenceOptimization
        policy = _make_lm(vocab=50, d=16)
        ref = _make_lm(vocab=50, d=16)
        ipo = IdentityPreferenceOptimization(policy, ref, tau=0.1)
        prompt = torch.randint(0, 50, (2, 4))
        chosen = torch.randint(0, 50, (2, 4))
        rejected = torch.randint(0, 50, (2, 4))
        loss, metrics = ipo(prompt, chosen, rejected)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

class TestRewardModelTrainer:
    def test_train_step(self):
        from rlhf import RewardModel, RewardModelTrainer
        lm = _make_lm(vocab=50, d=16)
        rm = RewardModel(lm, d_model=16)
        trainer = RewardModelTrainer(rm, learning_rate=1e-4)
        prompt = torch.randint(0, 50, (2, 4))
        chosen = torch.randint(0, 50, (2, 8))
        rejected = torch.randint(0, 50, (2, 8))
        loss = trainer.train_step(prompt, chosen, rejected)
        assert isinstance(loss, float)
        assert loss > 0

class TestPreferenceDataset:
    def test_add_and_getitem(self):
        from rlhf import PreferenceDataset
        ds = PreferenceDataset()
        p = torch.randint(0, 100, (10,))
        c = torch.randint(0, 100, (20,))
        r = torch.randint(0, 100, (20,))
        ds.add(p, c, r)
        assert len(ds) == 1
        item = ds[0]
        assert 'prompt' in item and 'chosen' in item and 'rejected' in item

class TestGeneralizedAdvantageEstimation:
    def test_shapes(self):
        from rlhf import GeneralizedAdvantagEstimation
        T = 16
        rewards = torch.randn(T)
        values = torch.randn(T + 1)
        dones = torch.zeros(T)
        returns, adv = GeneralizedAdvantagEstimation.compute(rewards, values, dones)
        assert returns.shape == (T,)
        assert adv.shape == (T,)

    def test_no_nan(self):
        from rlhf import GeneralizedAdvantagEstimation
        T = 8
        rewards = torch.randn(T)
        values = torch.randn(T + 1)
        dones = torch.zeros(T)
        returns, adv = GeneralizedAdvantagEstimation.compute(rewards, values, dones)
        assert not torch.isnan(returns).any()
        assert not torch.isnan(adv).any()

@pytest.mark.parametrize('beta,prompt_len,resp_len', [
    (0.05, 4, 4), (0.1, 4, 8), (0.2, 8, 8), (0.5, 8, 16),
    (0.1, 4, 4), (0.2, 8, 4),
])
def test_dpo_parametrized(beta, prompt_len, resp_len):
    from rlhf import DirectPreferenceOptimization
    policy = _make_lm(vocab=50, d=16)
    ref = _make_lm(vocab=50, d=16)
    dpo = DirectPreferenceOptimization(policy, ref, beta=beta)
    prompt = torch.randint(0, 50, (2, prompt_len))
    chosen = torch.randint(0, 50, (2, resp_len))
    rejected = torch.randint(0, 50, (2, resp_len))
    loss, metrics = dpo(prompt, chosen, rejected)
    assert not torch.isnan(loss)
    assert 0.0 <= metrics['accuracy'] <= 1.0
