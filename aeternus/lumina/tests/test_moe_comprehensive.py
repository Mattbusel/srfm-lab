"""Comprehensive tests for moe.py - Mixture of Experts components."""
import pytest
import torch
import torch.nn as nn

class TestTopKRouter:
    def test_output_shapes(self):
        from moe import TopKRouter
        router = TopKRouter(64, 8, top_k=2)
        x = torch.randn(32, 64)
        out = router(x)
        assert out.dispatch_mask.shape[0] == 32
        assert out.dispatch_mask.shape[1] == 8
        assert out.router_probs.shape == (32, 8)

    def test_load_loss_scalar(self):
        from moe import TopKRouter
        router = TopKRouter(64, 8, top_k=2)
        x = torch.randn(32, 64)
        out = router(x)
        assert out.load_loss.dim() == 0
        assert out.load_loss.item() >= 0

    def test_router_probs_sum_to_one(self):
        from moe import TopKRouter
        router = TopKRouter(64, 8, top_k=2)
        x = torch.randn(16, 64)
        out = router(x)
        sums = out.router_probs.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

class TestExpertFFN:
    def test_forward_shape(self):
        from moe import ExpertFFN
        expert = ExpertFFN(64, 256)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)

    @pytest.mark.parametrize('activation', ['gelu', 'relu', 'silu'])
    def test_activations(self, activation):
        from moe import ExpertFFN
        expert = ExpertFFN(32, 128, activation=activation)
        x = torch.randn(4, 32)
        out = expert(x)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()

class TestSparseMoELayer:
    def test_forward_shape(self):
        from moe import SparseMoELayer
        moe = SparseMoELayer(64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from moe import SparseMoELayer
        moe = SparseMoELayer(32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_aux_loss_exists(self):
        from moe import SparseMoELayer
        moe = SparseMoELayer(64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        moe(x)
        assert hasattr(moe, 'aux_loss')
        assert moe.aux_loss.item() >= 0

    def test_gradient_flows(self):
        from moe import SparseMoELayer
        moe = SparseMoELayer(32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None

class TestFusedMoELayer:
    def test_forward_shape(self):
        from moe import FusedMoELayer
        moe = FusedMoELayer(64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from moe import FusedMoELayer
        moe = FusedMoELayer(32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32)
        assert not torch.isnan(moe(x)).any()

    def test_gradient(self):
        from moe import FusedMoELayer
        moe = FusedMoELayer(32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        moe(x).sum().backward()
        assert x.grad is not None

class TestExpertChoiceLayer:
    def test_forward_shape(self):
        from moe import ExpertChoiceLayer
        moe = ExpertChoiceLayer(64, num_experts=4, expert_capacity=8)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from moe import ExpertChoiceLayer
        moe = ExpertChoiceLayer(32, num_experts=4, expert_capacity=4)
        x = torch.randn(2, 4, 32)
        assert not torch.isnan(moe(x)).any()

class TestSwitchTransformerLayer:
    def test_forward_shape(self):
        from moe import SwitchTransformerLayer
        switch = SwitchTransformerLayer(64, num_experts=4)
        x = torch.randn(2, 8, 64)
        out = switch(x)
        assert out.shape == (2, 8, 64)

    def test_aux_loss_positive(self):
        from moe import SwitchTransformerLayer
        switch = SwitchTransformerLayer(64, num_experts=4)
        x = torch.randn(2, 8, 64)
        switch(x)
        assert switch.aux_loss.item() >= 0

class TestMoETransformerBlock:
    def test_moe_block_shape(self):
        from moe import MoETransformerBlock
        block = MoETransformerBlock(64, 4, num_experts=4, moe_layer=True)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_dense_block_shape(self):
        from moe import MoETransformerBlock
        block = MoETransformerBlock(64, 4, moe_layer=False)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from moe import MoETransformerBlock
        block = MoETransformerBlock(32, 4, num_experts=4, moe_layer=True)
        x = torch.randn(2, 4, 32)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from moe import MoETransformerBlock
        block = MoETransformerBlock(32, 4, num_experts=4, moe_layer=True)
        x = torch.randn(2, 4, 32, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

class TestMoELanguageModel:
    def test_forward_logits_shape(self):
        from moe import MoELanguageModel
        model = MoELanguageModel(vocab_size=1000, d_model=64, num_layers=4,
                                 num_heads=4, num_experts=4, moe_every_n=2)
        ids = torch.randint(0, 1000, (2, 16))
        logits, aux = model(ids)
        assert logits.shape == (2, 16, 1000)
        assert aux.item() >= 0

    def test_no_nan_output(self):
        from moe import MoELanguageModel
        model = MoELanguageModel(vocab_size=500, d_model=32, num_layers=2,
                                 num_heads=4, num_experts=4, moe_every_n=2)
        ids = torch.randint(0, 500, (2, 8))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()

class TestMoELoadBalancer:
    def test_update_and_report(self):
        from moe import MoELoadBalancer
        lb = MoELoadBalancer(num_experts=8)
        dispatch = torch.randint(0, 2, (32, 8)).float()
        lb.update(dispatch)
        report = lb.report()
        assert 'expert_counts' in report
        assert 'load_imbalance_cv' in report

    def test_imbalance_metric(self):
        from moe import MoELoadBalancer
        lb = MoELoadBalancer(num_experts=4)
        # All traffic to expert 0
        dispatch = torch.zeros(32, 4)
        dispatch[:, 0] = 1.0
        lb.update(dispatch)
        assert lb.load_imbalance() > 0

@pytest.mark.parametrize('d_model,num_experts,top_k,B,T', [
    (32, 4, 1, 1, 4),
    (32, 4, 1, 1, 8),
    (32, 4, 1, 2, 4),
    (32, 4, 1, 2, 8),
    (32, 4, 2, 1, 4),
    (32, 4, 2, 1, 8),
    (32, 4, 2, 2, 4),
    (32, 4, 2, 2, 8),
    (32, 8, 1, 1, 4),
    (32, 8, 1, 1, 8),
    (32, 8, 1, 2, 4),
    (32, 8, 1, 2, 8),
    (32, 8, 2, 1, 4),
    (32, 8, 2, 1, 8),
    (32, 8, 2, 2, 4),
    (32, 8, 2, 2, 8),
    (64, 4, 1, 1, 4),
    (64, 4, 1, 1, 8),
    (64, 4, 1, 2, 4),
    (64, 4, 1, 2, 8),
    (64, 4, 2, 1, 4),
    (64, 4, 2, 1, 8),
    (64, 4, 2, 2, 4),
    (64, 4, 2, 2, 8),
    (64, 8, 1, 1, 4),
    (64, 8, 1, 1, 8),
    (64, 8, 1, 2, 4),
    (64, 8, 1, 2, 8),
    (64, 8, 2, 1, 4),
    (64, 8, 2, 1, 8),
    (64, 8, 2, 2, 4),
    (64, 8, 2, 2, 8),
])
def test_sparse_moe_parametrized(d_model, num_experts, top_k, B, T):
    from moe import SparseMoELayer
    moe = SparseMoELayer(d_model, num_experts=num_experts, top_k=top_k)
    x = torch.randn(B, T, d_model)
    out = moe(x)
    assert out.shape == (B, T, d_model)
    assert not torch.isnan(out).any()
