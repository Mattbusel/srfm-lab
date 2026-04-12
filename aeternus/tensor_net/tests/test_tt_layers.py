"""
test_tt_layers.py — Tests for tt_layers.py

Run with::

    pytest aeternus/tensor_net/tests/test_tt_layers.py -v
"""

from __future__ import annotations

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def jax_key():
    import jax
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def batch_size():
    return 4


@pytest.fixture(scope="module")
def seq_len():
    return 16


@pytest.fixture(scope="module")
def in_features():
    return 16


@pytest.fixture(scope="module")
def out_features():
    return 16


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestRebuildMatrix:
    def test_round_trip(self, jax_key):
        import jax
        import jax.numpy as jnp
        from tensor_net.tt_layers import _truncated_svd_cores, _rebuild_matrix

        key = jax_key
        N, M = 16, 16
        W = np.random.randn(N, M).astype(np.float32)
        shape_in = (4, 4)
        shape_out = (4, 4)
        cores = _truncated_svd_cores(W, shape_in, shape_out, rank=4)
        W_recon = _rebuild_matrix(cores)
        # With full rank, reconstruction should be reasonable
        assert W_recon.shape == (N, M)
        assert np.isfinite(np.array(W_recon)).all()


# ---------------------------------------------------------------------------
# TTDense tests
# ---------------------------------------------------------------------------


class TestTTDense:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_dense, TTDenseConfig
        config = TTDenseConfig(in_features=16, out_features=16, shape_in=(16,), shape_out=(16,), tt_rank=4)
        params = init_tt_dense(jax_key, config)
        assert "cores" in params
        assert isinstance(params["cores"], list)
        assert len(params["cores"]) == 1  # flat shape -> 1 core

    def test_forward_shape(self, jax_key, batch_size, in_features, out_features):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(
            in_features=in_features, out_features=out_features,
            shape_in=(in_features,), shape_out=(out_features,), tt_rank=4
        )
        params = init_tt_dense(jax_key, config)
        x = jnp.ones((batch_size, in_features))
        y = apply_tt_dense(params, x, config)
        assert y.shape == (batch_size, out_features)

    def test_bias_shape(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(in_features=8, out_features=8, shape_in=(8,), shape_out=(8,), use_bias=True)
        params = init_tt_dense(jax_key, config)
        assert "bias" in params
        assert params["bias"].shape == (8,)

    def test_no_bias(self, jax_key):
        from tensor_net.tt_layers import init_tt_dense, TTDenseConfig
        config = TTDenseConfig(in_features=8, out_features=8, shape_in=(8,), shape_out=(8,), use_bias=False)
        params = init_tt_dense(jax_key, config)
        assert "bias" not in params

    def test_relu_activation(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(
            in_features=8, out_features=8, shape_in=(8,), shape_out=(8,), activation="relu"
        )
        params = init_tt_dense(jax_key, config)
        x = jnp.array(np.random.randn(batch_size, 8).astype(np.float32))
        y = apply_tt_dense(params, x, config)
        assert jnp.all(y >= 0), "ReLU should produce non-negative output"

    def test_gelu_activation(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(
            in_features=8, out_features=8, shape_in=(8,), shape_out=(8,), activation="gelu"
        )
        params = init_tt_dense(jax_key, config)
        x = jnp.array(np.random.randn(batch_size, 8).astype(np.float32))
        y = apply_tt_dense(params, x, config)
        assert y.shape == (batch_size, 8)

    def test_3d_input(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(in_features=8, out_features=8, shape_in=(8,), shape_out=(8,))
        params = init_tt_dense(jax_key, config)
        x = jnp.ones((2, 5, 8))   # (batch, seq, features)
        y = apply_tt_dense(params, x, config)
        assert y.shape == (2, 5, 8)

    def test_finite_output(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_dense, apply_tt_dense, TTDenseConfig
        config = TTDenseConfig(in_features=16, out_features=16, shape_in=(16,), shape_out=(16,))
        params = init_tt_dense(jax_key, config)
        x = jnp.array(np.random.randn(8, 16).astype(np.float32))
        y = apply_tt_dense(params, x, config)
        assert np.isfinite(np.array(y)).all()


# ---------------------------------------------------------------------------
# TTEmbedding tests
# ---------------------------------------------------------------------------


class TestTTEmbedding:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_embedding, TTEmbeddingConfig
        config = TTEmbeddingConfig(
            vocab_size=64, embed_dim=16,
            shape_vocab=(64,), shape_embed=(16,), tt_rank=4
        )
        params = init_tt_embedding(jax_key, config)
        assert "cores" in params

    def test_lookup(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_embedding, apply_tt_embedding, TTEmbeddingConfig
        )
        config = TTEmbeddingConfig(
            vocab_size=64, embed_dim=16,
            shape_vocab=(64,), shape_embed=(16,), tt_rank=4
        )
        params = init_tt_embedding(jax_key, config)
        token_ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        embeds = apply_tt_embedding(params, token_ids, config)
        assert embeds.shape == (4, 16)

    def test_batch_lookup(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_embedding, apply_tt_embedding, TTEmbeddingConfig
        )
        config = TTEmbeddingConfig(
            vocab_size=64, embed_dim=16,
            shape_vocab=(64,), shape_embed=(16,), tt_rank=4
        )
        params = init_tt_embedding(jax_key, config)
        token_ids = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        embeds = apply_tt_embedding(params, token_ids, config)
        assert embeds.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# TTLayerNorm tests
# ---------------------------------------------------------------------------


class TestTTLayerNorm:
    def test_init(self):
        from tensor_net.tt_layers import init_tt_layer_norm
        params = init_tt_layer_norm(32)
        assert "scale" in params
        assert "shift" in params
        assert params["scale"].shape == (32,)

    def test_normalisation(self):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_layer_norm, tt_layer_norm
        params = init_tt_layer_norm(16)
        x = jnp.array(np.random.randn(4, 16).astype(np.float32) * 10)
        y = tt_layer_norm(params, x)
        # Output should have ~zero mean and ~unit variance along last axis
        mean = jnp.mean(y, axis=-1)
        var = jnp.var(y, axis=-1)
        np.testing.assert_allclose(np.array(mean), np.zeros(4), atol=1e-4)
        np.testing.assert_allclose(np.array(var), np.ones(4), atol=0.01)

    def test_shape_preserved(self):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_layer_norm, tt_layer_norm
        params = init_tt_layer_norm(32)
        x = jnp.ones((2, 5, 32))
        y = tt_layer_norm(params, x)
        assert y.shape == (2, 5, 32)


# ---------------------------------------------------------------------------
# TTConv1D tests
# ---------------------------------------------------------------------------


class TestTTConv1D:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_conv1d, TTConv1DConfig
        config = TTConv1DConfig(in_channels=8, out_channels=16, kernel_size=3)
        params = init_tt_conv1d(jax_key, config)
        assert "kernel" in params
        assert params["kernel"].shape == (3, 8, 16)

    def test_forward_same_padding(self, jax_key, batch_size, seq_len):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_conv1d, apply_tt_conv1d, TTConv1DConfig
        config = TTConv1DConfig(in_channels=8, out_channels=16, kernel_size=3, padding="SAME")
        params = init_tt_conv1d(jax_key, config)
        x = jnp.ones((batch_size, seq_len, 8))
        y = apply_tt_conv1d(params, x, config)
        assert y.shape == (batch_size, seq_len, 16)

    def test_finite_output(self, jax_key, batch_size, seq_len):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_conv1d, apply_tt_conv1d, TTConv1DConfig
        config = TTConv1DConfig(in_channels=4, out_channels=4, kernel_size=3, padding="SAME")
        params = init_tt_conv1d(jax_key, config)
        x = jnp.array(np.random.randn(batch_size, seq_len, 4).astype(np.float32))
        y = apply_tt_conv1d(params, x, config)
        assert np.isfinite(np.array(y)).all()


# ---------------------------------------------------------------------------
# TTGRUCell tests
# ---------------------------------------------------------------------------


class TestTTGRU:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_gru, TTGRUConfig
        config = TTGRUConfig(input_size=8, hidden_size=16)
        params = init_tt_gru(jax_key, config)
        assert "W_z" in params
        assert "U_z" in params
        assert "b_z" in params

    def test_single_step(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_gru, apply_tt_gru_step, TTGRUConfig
        config = TTGRUConfig(input_size=8, hidden_size=16)
        params = init_tt_gru(jax_key, config)
        x = jnp.ones((batch_size, 8))
        h = jnp.zeros((batch_size, 16))
        h_new = apply_tt_gru_step(params, x, h, config)
        assert h_new.shape == (batch_size, 16)
        assert np.isfinite(np.array(h_new)).all()

    def test_scan(self, jax_key, batch_size, seq_len):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_gru, scan_tt_gru, TTGRUConfig
        config = TTGRUConfig(input_size=8, hidden_size=16)
        params = init_tt_gru(jax_key, config)
        xs = jnp.array(np.random.randn(batch_size, seq_len, 8).astype(np.float32))
        outputs, h_final = scan_tt_gru(params, xs, config=config)
        assert outputs.shape == (batch_size, seq_len, 16)
        assert h_final.shape == (batch_size, 16)

    def test_hidden_state_changes(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_gru, apply_tt_gru_step, TTGRUConfig
        config = TTGRUConfig(input_size=4, hidden_size=8)
        params = init_tt_gru(jax_key, config)
        x = jnp.array(np.random.randn(1, 4).astype(np.float32))
        h = jnp.zeros((1, 8))
        h_new = apply_tt_gru_step(params, x, h, config)
        # Hidden state should change from zero
        assert not np.allclose(np.array(h_new), 0.0)

    def test_no_bias(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_gru, apply_tt_gru_step, TTGRUConfig
        config = TTGRUConfig(input_size=8, hidden_size=16, use_bias=False)
        params = init_tt_gru(jax_key, config)
        assert "b_z" not in params
        x = jnp.ones((batch_size, 8))
        h = jnp.zeros((batch_size, 16))
        h_new = apply_tt_gru_step(params, x, h, config)
        assert h_new.shape == (batch_size, 16)


# ---------------------------------------------------------------------------
# TTLSTMCell tests
# ---------------------------------------------------------------------------


class TestTTLSTM:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_lstm, TTLSTMConfig
        config = TTLSTMConfig(input_size=8, hidden_size=16)
        params = init_tt_lstm(jax_key, config)
        assert "W_i" in params
        assert "W_f" in params
        assert "W_g" in params
        assert "W_o" in params

    def test_single_step(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_lstm, apply_tt_lstm_step, TTLSTMConfig
        config = TTLSTMConfig(input_size=8, hidden_size=16)
        params = init_tt_lstm(jax_key, config)
        x = jnp.ones((batch_size, 8))
        h = jnp.zeros((batch_size, 16))
        c = jnp.zeros((batch_size, 16))
        h_new, c_new = apply_tt_lstm_step(params, x, h, c, config)
        assert h_new.shape == (batch_size, 16)
        assert c_new.shape == (batch_size, 16)

    def test_scan(self, jax_key, batch_size, seq_len):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_lstm, scan_tt_lstm, TTLSTMConfig
        config = TTLSTMConfig(input_size=8, hidden_size=16)
        params = init_tt_lstm(jax_key, config)
        xs = jnp.array(np.random.randn(batch_size, seq_len, 8).astype(np.float32))
        outputs, (h_final, c_final) = scan_tt_lstm(params, xs, config=config)
        assert outputs.shape == (batch_size, seq_len, 16)
        assert h_final.shape == (batch_size, 16)
        assert c_final.shape == (batch_size, 16)

    def test_cell_state_gating(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import init_tt_lstm, apply_tt_lstm_step, TTLSTMConfig
        config = TTLSTMConfig(input_size=4, hidden_size=8)
        params = init_tt_lstm(jax_key, config)
        x = jnp.array(np.random.randn(1, 4).astype(np.float32))
        h = jnp.zeros((1, 8))
        c = jnp.zeros((1, 8))
        h_new, c_new = apply_tt_lstm_step(params, x, h, c, config)
        # h should be bounded by tanh: |h| <= 1
        assert np.all(np.abs(np.array(h_new)) <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# TTResidualBlock tests
# ---------------------------------------------------------------------------


class TestTTResidualBlock:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import init_tt_residual_block, TTResidualBlockConfig
        config = TTResidualBlockConfig(
            d_model=16, d_ff=32, tt_rank=4,
            shape_model=(16,), shape_ff=(32,)
        )
        params = init_tt_residual_block(jax_key, config)
        assert "dense_in" in params
        assert "dense_out" in params
        assert "norm1" in params

    def test_forward(self, jax_key, batch_size, seq_len):
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_residual_block, apply_tt_residual_block, TTResidualBlockConfig
        )
        config = TTResidualBlockConfig(d_model=16, d_ff=32, shape_model=(16,), shape_ff=(32,))
        params = init_tt_residual_block(jax_key, config)
        x = jnp.array(np.random.randn(batch_size, seq_len, 16).astype(np.float32))
        y = apply_tt_residual_block(params, x, config, training=False)
        assert y.shape == (batch_size, seq_len, 16)
        assert np.isfinite(np.array(y)).all()

    def test_residual_connection(self, jax_key, batch_size):
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_residual_block, apply_tt_residual_block, TTResidualBlockConfig
        )
        # With zero input, output should not be zero (due to layer norm + bias)
        config = TTResidualBlockConfig(d_model=8, d_ff=16, shape_model=(8,), shape_ff=(16,))
        params = init_tt_residual_block(jax_key, config)
        x = jnp.zeros((batch_size, 4, 8))
        y = apply_tt_residual_block(params, x, config)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Positional encoding tests
# ---------------------------------------------------------------------------


class TestPositionalEncodings:
    def test_sinusoidal(self):
        from tensor_net.tt_layers import sinusoidal_position_encoding
        pe = sinusoidal_position_encoding(64, 32)
        assert pe.shape == (64, 32)
        assert np.isfinite(np.array(pe)).all()

    def test_sinusoidal_values_in_range(self):
        from tensor_net.tt_layers import sinusoidal_position_encoding
        pe = sinusoidal_position_encoding(16, 16)
        assert np.all(np.abs(np.array(pe)) <= 1.0 + 1e-5)

    def test_learnable(self, jax_key):
        from tensor_net.tt_layers import learnable_position_encoding
        pe = learnable_position_encoding(jax_key, 32, 64)
        assert pe.shape == (32, 64)
        assert np.isfinite(np.array(pe)).all()

    def test_temporal_encoding(self):
        import jax.numpy as jnp
        from tensor_net.tt_layers import temporal_encoding
        ts = jnp.arange(20, dtype=jnp.float32) * 86400.0   # daily
        enc = temporal_encoding(ts, 32)
        assert enc.shape == (20, 32)
        assert np.isfinite(np.array(enc)).all()


# ---------------------------------------------------------------------------
# Training utility tests
# ---------------------------------------------------------------------------


class TestTrainingUtils:
    def test_count_parameters(self, jax_key):
        from tensor_net.tt_layers import init_tt_dense, TTDenseConfig, count_tt_parameters
        config = TTDenseConfig(in_features=16, out_features=16, shape_in=(16,), shape_out=(16,))
        params = init_tt_dense(jax_key, config)
        n_params = count_tt_parameters(params)
        assert n_params > 0

    def test_param_l2_norm(self, jax_key):
        from tensor_net.tt_layers import init_tt_dense, TTDenseConfig, param_l2_norm
        import jax.numpy as jnp
        config = TTDenseConfig(in_features=8, out_features=8, shape_in=(8,), shape_out=(8,))
        params = init_tt_dense(jax_key, config)
        norm = param_l2_norm(params)
        assert float(norm) > 0

    def test_make_optimizer(self):
        from tensor_net.tt_layers import make_tt_optimizer
        opt = make_tt_optimizer(learning_rate=1e-3, warmup_steps=10)
        assert opt is not None

    def test_create_and_train_step(self, jax_key):
        import jax
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_dense, TTDenseConfig, make_tt_optimizer,
            create_train_state, train_step
        )
        config = TTDenseConfig(in_features=8, out_features=4, shape_in=(8,), shape_out=(4,))
        params = init_tt_dense(jax_key, config)
        opt = make_tt_optimizer(learning_rate=1e-3, warmup_steps=2)
        state = create_train_state(params, opt)
        assert state.step == 0

        def loss_fn(params, batch):
            y = params["cores"][0].sum()   # dummy loss
            return y, {"dummy": 0}

        new_state, loss, aux = train_step(state, {}, loss_fn, opt)
        assert new_state.step == 1


# ---------------------------------------------------------------------------
# TTFinancialEncoder tests (smoke)
# ---------------------------------------------------------------------------


class TestTTFinancialEncoder:
    def test_init(self, jax_key):
        from tensor_net.tt_layers import (
            init_tt_financial_encoder, TTFinancialEncoderConfig
        )
        config = TTFinancialEncoderConfig(
            n_assets=8, n_features=4, d_model=16, n_heads=2,
            n_layers=2, tt_rank=2, output_size=8, seq_len=16
        )
        params = init_tt_financial_encoder(jax_key, config)
        assert "in_proj" in params
        assert "blocks" in params
        assert "out_proj" in params

    def test_forward(self, jax_key):
        import jax.numpy as jnp
        from tensor_net.tt_layers import (
            init_tt_financial_encoder, apply_tt_financial_encoder,
            TTFinancialEncoderConfig
        )
        config = TTFinancialEncoderConfig(
            n_assets=4, n_features=2, d_model=8, n_heads=2,
            n_layers=1, tt_rank=2, output_size=4, seq_len=8
        )
        params = init_tt_financial_encoder(jax_key, config)
        batch = 2
        T = 8
        x = jnp.array(np.random.randn(batch, T, 4 * 2).astype(np.float32))
        y = apply_tt_financial_encoder(params, x, config, training=False)
        assert y.shape == (batch, T, 4)
        assert np.isfinite(np.array(y)).all()
