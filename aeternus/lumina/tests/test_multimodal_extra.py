import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from multimodal import (
    TextEncoder, VisionEncoder, AudioEncoder,
    CrossModalAttention, MultimodalFusion,
    FinancialNewsClassifier, ChartPatternRecognizer,
    EarningsCallAnalyzer, DocumentEmbedder,
    KnowledgeGraphEmbedder, AlternativeDataFusion,
    TemporalGraphNetwork, MultimodalFinancialModel,
)


class TestTextEncoder:
    def test_forward_shape(self):
        enc = TextEncoder(vocab_size=100, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128)
        ids = torch.randint(0, 100, (2, 16))
        seq, pooled = enc(ids)
        assert seq.shape == (2, 16, 64)
        assert pooled.shape == (2, 64)

    def test_with_attention_mask(self):
        enc = TextEncoder(vocab_size=100, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16)
        mask[0, 10:] = 0
        _, pooled = enc(ids, mask)
        assert pooled.shape == (2, 64)

    def test_no_nan(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64)
        ids = torch.randint(0, 200, (3, 20))
        _, pooled = enc(ids)
        assert not torch.isnan(pooled).any()

class TestVisionEncoder:
    def test_forward_shape(self):
        enc = VisionEncoder(image_size=64, patch_size=8, embed_dim=64, num_heads=4, num_layers=2)
        imgs = torch.randn(2, 3, 64, 64)
        all_patches, cls = enc(imgs)
        expected_patches = (64 // 8) ** 2 + 1
        assert all_patches.shape == (2, expected_patches, 64)
        assert cls.shape == (2, 64)

    def test_grayscale(self):
        enc = VisionEncoder(image_size=32, patch_size=8, in_channels=1, embed_dim=32, num_heads=4, num_layers=2)
        imgs = torch.randn(2, 1, 32, 32)
        _, cls = enc(imgs)
        assert cls.shape == (2, 32)

class TestAudioEncoder:
    def test_forward_shape(self):
        enc = AudioEncoder(embed_dim=64, num_heads=4, num_layers=2)
        audio = torch.randn(2, 1, 4000)
        seq, pooled = enc(audio)
        assert pooled.shape == (2, 64)
        assert seq.dim() == 3

class TestCrossModalAttention:
    def test_forward_shape(self):
        attn = CrossModalAttention(embed_dim=64, num_heads=4)
        query = torch.randn(2, 8, 64)
        kv = torch.randn(2, 12, 64)
        out = attn(query, kv)
        assert out.shape == (2, 8, 64)

    def test_with_key_padding_mask(self):
        attn = CrossModalAttention(embed_dim=64, num_heads=4)
        query = torch.randn(2, 8, 64)
        kv = torch.randn(2, 12, 64)
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, 8:] = True
        out = attn(query, kv, mask)
        assert out.shape == (2, 8, 64)

class TestMultimodalFusion_Concat:
    def test_forward(self):
        fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='concat')
        t = torch.randn(2, 64)
        v = torch.randn(2, 64)
        s = torch.randn(2, 64)
        out = fusion(t, v, s)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()

class TestMultimodalFusion_Attention:
    def test_forward(self):
        fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='attention')
        t = torch.randn(2, 64)
        v = torch.randn(2, 64)
        s = torch.randn(2, 64)
        out = fusion(t, v, s)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()

class TestMultimodalFusion_Gated:
    def test_forward(self):
        fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='gated')
        t = torch.randn(2, 64)
        v = torch.randn(2, 64)
        s = torch.randn(2, 64)
        out = fusion(t, v, s)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()

class TestFinancialNewsClassifier:
    def test_forward_shape(self):
        model = FinancialNewsClassifier(vocab_size=100, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128, num_classes=3)
        ids = torch.randint(0, 100, (4, 32))
        logits = model(ids)
        assert logits.shape == (4, 3)

    def test_gradient_flow(self):
        model = FinancialNewsClassifier(vocab_size=100, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, num_classes=5)
        ids = torch.randint(0, 100, (2, 16))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

class TestChartPatternRecognizer:
    def test_forward_shape(self):
        model = ChartPatternRecognizer(image_size=32, patch_size=8, embed_dim=64, num_heads=4, num_layers=2, num_patterns=10)
        imgs = torch.randn(2, 3, 32, 32)
        out = model(imgs)
        assert out.shape == (2, 10)

class TestKnowledgeGraphEmbedder:
    def test_transe_score(self):
        kg = KnowledgeGraphEmbedder(num_entities=100, num_relations=10, embed_dim=32, scoring_fn='transe')
        h = torch.tensor([0, 1])
        r = torch.tensor([0, 0])
        t = torch.tensor([2, 3])
        scores = kg.score(h, r, t)
        assert scores.shape == (2,)

    def test_rotate_score(self):
        kg = KnowledgeGraphEmbedder(num_entities=100, num_relations=10, embed_dim=32, scoring_fn='rotate')
        h = torch.tensor([0, 1])
        r = torch.tensor([0, 0])
        t = torch.tensor([2, 3])
        scores = kg.score(h, r, t)
        assert scores.shape == (2,)

    def test_forward_loss(self):
        kg = KnowledgeGraphEmbedder(num_entities=100, num_relations=10, embed_dim=32)
        h = torch.tensor([0, 1, 2])
        r = torch.tensor([0, 1, 2])
        t = torch.tensor([3, 4, 5])
        neg_t = torch.tensor([6, 7, 8])
        loss = kg(h, r, t, neg_t)
        assert loss.dim() == 0
        assert loss.item() >= 0

class TestAlternativeDataFusion:
    def test_forward_shape(self):
        model = AlternativeDataFusion(satellite_dim=64, credit_dim=32, sentiment_dim=64, jobs_dim=32, esg_dim=16, fused_dim=128)
        B = 4
        sat = torch.randn(B, 64)
        cred = torch.randn(B, 32)
        sent = torch.randn(B, 64)
        jobs = torch.randn(B, 32)
        esg = torch.randn(B, 16)
        out = model(sat, cred, sent, jobs, esg)
        assert out.shape == (B, 128)

class TestTemporalGraphNetwork:
    def test_forward_shape(self):
        tgn = TemporalGraphNetwork(num_nodes=20, node_feat_dim=8, edge_feat_dim=4, embed_dim=32, num_heads=4, num_layers=2)
        node_feats = torch.randn(2, 10, 8)
        edge_index = torch.randint(0, 10, (2, 20))
        out = tgn(node_feats, edge_index)
        assert out.shape == (2, 10, 32)

class TestMultimodalFinancialModel:
    def test_forward_outputs(self):
        model = MultimodalFinancialModel(vocab_size=100, text_dim=64, ts_dim=64, kg_dim=32, alt_fused_dim=64, output_dim=64, num_classes=3)
        B = 2
        ids = torch.randint(0, 100, (B, 16))
        ts = torch.randn(B, 64)
        kg = torch.randn(B, 32)
        alt = torch.randn(B, 64)
        out = model(ids, ts, kg, alt)
        assert 'logits' in out
        assert 'prediction' in out
        assert 'embedding' in out
        assert out['logits'].shape == (B, 3)
        assert out['prediction'].shape == (B,)
        assert out['embedding'].shape == (B, 64)

def test_fusion_concat_B1_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='concat')
    t = torch.randn(1, 32)
    v = torch.randn(1, 32)
    s = torch.randn(1, 32)
    out = fusion(t, v, s)
    assert out.shape == (1, 64)

def test_fusion_concat_B1_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='concat')
    t = torch.randn(1, 64)
    v = torch.randn(1, 64)
    s = torch.randn(1, 64)
    out = fusion(t, v, s)
    assert out.shape == (1, 128)

def test_fusion_concat_B1_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='concat')
    t = torch.randn(1, 128)
    v = torch.randn(1, 128)
    s = torch.randn(1, 128)
    out = fusion(t, v, s)
    assert out.shape == (1, 256)

def test_fusion_concat_B2_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='concat')
    t = torch.randn(2, 32)
    v = torch.randn(2, 32)
    s = torch.randn(2, 32)
    out = fusion(t, v, s)
    assert out.shape == (2, 64)

def test_fusion_concat_B2_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='concat')
    t = torch.randn(2, 64)
    v = torch.randn(2, 64)
    s = torch.randn(2, 64)
    out = fusion(t, v, s)
    assert out.shape == (2, 128)

def test_fusion_concat_B2_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='concat')
    t = torch.randn(2, 128)
    v = torch.randn(2, 128)
    s = torch.randn(2, 128)
    out = fusion(t, v, s)
    assert out.shape == (2, 256)

def test_fusion_concat_B4_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='concat')
    t = torch.randn(4, 32)
    v = torch.randn(4, 32)
    s = torch.randn(4, 32)
    out = fusion(t, v, s)
    assert out.shape == (4, 64)

def test_fusion_concat_B4_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='concat')
    t = torch.randn(4, 64)
    v = torch.randn(4, 64)
    s = torch.randn(4, 64)
    out = fusion(t, v, s)
    assert out.shape == (4, 128)

def test_fusion_concat_B4_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='concat')
    t = torch.randn(4, 128)
    v = torch.randn(4, 128)
    s = torch.randn(4, 128)
    out = fusion(t, v, s)
    assert out.shape == (4, 256)

def test_fusion_attention_B1_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='attention')
    t = torch.randn(1, 32)
    v = torch.randn(1, 32)
    s = torch.randn(1, 32)
    out = fusion(t, v, s)
    assert out.shape == (1, 64)

def test_fusion_attention_B1_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='attention')
    t = torch.randn(1, 64)
    v = torch.randn(1, 64)
    s = torch.randn(1, 64)
    out = fusion(t, v, s)
    assert out.shape == (1, 128)

def test_fusion_attention_B1_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='attention')
    t = torch.randn(1, 128)
    v = torch.randn(1, 128)
    s = torch.randn(1, 128)
    out = fusion(t, v, s)
    assert out.shape == (1, 256)

def test_fusion_attention_B2_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='attention')
    t = torch.randn(2, 32)
    v = torch.randn(2, 32)
    s = torch.randn(2, 32)
    out = fusion(t, v, s)
    assert out.shape == (2, 64)

def test_fusion_attention_B2_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='attention')
    t = torch.randn(2, 64)
    v = torch.randn(2, 64)
    s = torch.randn(2, 64)
    out = fusion(t, v, s)
    assert out.shape == (2, 128)

def test_fusion_attention_B2_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='attention')
    t = torch.randn(2, 128)
    v = torch.randn(2, 128)
    s = torch.randn(2, 128)
    out = fusion(t, v, s)
    assert out.shape == (2, 256)

def test_fusion_attention_B4_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='attention')
    t = torch.randn(4, 32)
    v = torch.randn(4, 32)
    s = torch.randn(4, 32)
    out = fusion(t, v, s)
    assert out.shape == (4, 64)

def test_fusion_attention_B4_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='attention')
    t = torch.randn(4, 64)
    v = torch.randn(4, 64)
    s = torch.randn(4, 64)
    out = fusion(t, v, s)
    assert out.shape == (4, 128)

def test_fusion_attention_B4_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='attention')
    t = torch.randn(4, 128)
    v = torch.randn(4, 128)
    s = torch.randn(4, 128)
    out = fusion(t, v, s)
    assert out.shape == (4, 256)

def test_fusion_gated_B1_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='gated')
    t = torch.randn(1, 32)
    v = torch.randn(1, 32)
    s = torch.randn(1, 32)
    out = fusion(t, v, s)
    assert out.shape == (1, 64)

def test_fusion_gated_B1_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='gated')
    t = torch.randn(1, 64)
    v = torch.randn(1, 64)
    s = torch.randn(1, 64)
    out = fusion(t, v, s)
    assert out.shape == (1, 128)

def test_fusion_gated_B1_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='gated')
    t = torch.randn(1, 128)
    v = torch.randn(1, 128)
    s = torch.randn(1, 128)
    out = fusion(t, v, s)
    assert out.shape == (1, 256)

def test_fusion_gated_B2_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='gated')
    t = torch.randn(2, 32)
    v = torch.randn(2, 32)
    s = torch.randn(2, 32)
    out = fusion(t, v, s)
    assert out.shape == (2, 64)

def test_fusion_gated_B2_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='gated')
    t = torch.randn(2, 64)
    v = torch.randn(2, 64)
    s = torch.randn(2, 64)
    out = fusion(t, v, s)
    assert out.shape == (2, 128)

def test_fusion_gated_B2_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='gated')
    t = torch.randn(2, 128)
    v = torch.randn(2, 128)
    s = torch.randn(2, 128)
    out = fusion(t, v, s)
    assert out.shape == (2, 256)

def test_fusion_gated_B4_D32():
    fusion = MultimodalFusion(text_dim=32, vision_dim=32, ts_dim=32, fused_dim=64, fusion_type='gated')
    t = torch.randn(4, 32)
    v = torch.randn(4, 32)
    s = torch.randn(4, 32)
    out = fusion(t, v, s)
    assert out.shape == (4, 64)

def test_fusion_gated_B4_D64():
    fusion = MultimodalFusion(text_dim=64, vision_dim=64, ts_dim=64, fused_dim=128, fusion_type='gated')
    t = torch.randn(4, 64)
    v = torch.randn(4, 64)
    s = torch.randn(4, 64)
    out = fusion(t, v, s)
    assert out.shape == (4, 128)

def test_fusion_gated_B4_D128():
    fusion = MultimodalFusion(text_dim=128, vision_dim=128, ts_dim=128, fused_dim=256, fusion_type='gated')
    t = torch.randn(4, 128)
    v = torch.randn(4, 128)
    s = torch.randn(4, 128)
    out = fusion(t, v, s)
    assert out.shape == (4, 256)
