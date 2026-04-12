

# ============================================================
# Additional Lumina Model Variants
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math


class LuminaNano(nn.Module):
    """LuminaNano - ultra-small model for edge deployment (~100K params).

    Designed for real-time tick-level inference on resource-constrained devices.
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 64,
        num_classes: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.input_proj(x) + self.pos_embed(pos))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class LuminaAlpha(nn.Module):
    """LuminaAlpha - alpha generation model with multi-horizon forecasting.

    Predicts returns across multiple time horizons simultaneously.
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        horizons: List[int] = None,
        max_seq_len: int = 252,
        dropout: float = 0.1,
    ):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 21, 63]  # 1d, 1w, 1m, 1q
        self.horizons = horizons
        self.d_model = d_model

        self.feature_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Per-horizon prediction heads
        self.horizon_heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for h in horizons
        })

        # Uncertainty head (log-variance)
        self.uncertainty_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model // 2, 1)
            for h in horizons
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.feature_embed(x) + self.pos_embed(pos))

        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)

        pooled = h[:, -1, :]  # use last timestep

        results = {}
        for horizon in self.horizons:
            key = f"h{horizon}"
            # Get intermediate representation
            intermediate = F.gelu(self.horizon_heads[key][0](pooled))
            intermediate = self.dropout(intermediate)
            pred = self.horizon_heads[key][2](intermediate).squeeze(-1)
            log_var = self.uncertainty_heads[key](intermediate).squeeze(-1)
            results[f"pred_{horizon}d"] = pred
            results[f"logvar_{horizon}d"] = log_var

        return results


class LuminaRiskModel(nn.Module):
    """LuminaRiskModel - factor-based risk estimation model.

    Estimates portfolio risk via learned factor exposures and covariance.
    """

    def __init__(
        self,
        n_assets: int,
        n_factors: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.d_model = d_model

        # Asset embedding
        self.asset_embed = nn.Embedding(n_assets, d_model)
        self.feature_proj = nn.Linear(1, d_model)

        # Transformer for cross-asset attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Factor loading head: maps each asset to factor exposures
        self.factor_head = nn.Linear(d_model, n_factors)

        # Specific variance head (idiosyncratic risk)
        self.spec_var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        # Factor covariance (learnable, constrained to be PSD)
        self.factor_cov_tril = nn.Parameter(torch.eye(n_factors))

        self.dropout = nn.Dropout(dropout)

    @property
    def factor_cov(self) -> torch.Tensor:
        L = torch.tril(self.factor_cov_tril)
        return L @ L.T + 1e-6 * torch.eye(self.n_factors, device=L.device)

    def forward(
        self,
        asset_ids: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N = asset_ids.shape
        h = self.asset_embed(asset_ids)  # (B, N, d)

        if returns is not None:
            h = h + self.feature_proj(returns.unsqueeze(-1))

        h = self.dropout(h)
        h = self.encoder(h)
        h = self.norm(h)

        # Factor loadings
        beta = self.factor_head(h)  # (B, N, K)

        # Specific variances
        spec_var = self.spec_var_head(h).squeeze(-1)  # (B, N)

        # Portfolio covariance: Sigma = B * F * B^T + diag(spec_var)
        F_cov = self.factor_cov  # (K, K)
        sigma = torch.bmm(beta, F_cov.unsqueeze(0).expand(B, -1, -1))  # (B, N, K)
        sigma = torch.bmm(sigma, beta.transpose(-2, -1))  # (B, N, N)
        sigma = sigma + torch.diag_embed(spec_var)

        return {
            "beta": beta,
            "spec_var": spec_var,
            "cov_matrix": sigma,
            "factor_cov": F_cov,
        }


class LuminaSentimentModel(nn.Module):
    """LuminaSentimentModel - financial sentiment analysis from text features.

    Maps text embeddings to signed sentiment scores with uncertainty.
    """

    def __init__(
        self,
        text_embed_dim: int = 768,
        market_dim: int = 64,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Modality projections
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        self.market_proj = nn.Linear(market_dim, d_model)

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )
        self.impact_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,
        market_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = text_emb.shape[0]

        h_text = self.text_proj(text_emb)  # (B, d)
        tokens = h_text.unsqueeze(1)        # (B, 1, d)

        if market_feat is not None:
            h_market = self.market_proj(market_feat).unsqueeze(1)  # (B, 1, d)
            tokens = torch.cat([tokens, h_market], dim=1)

        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)

        return {
            "sentiment": self.sentiment_head(pooled).squeeze(-1),
            "impact": self.impact_head(pooled).squeeze(-1),
            "duration_days": self.duration_head(pooled).squeeze(-1),
        }


class LuminaTrendFollower(nn.Module):
    """LuminaTrendFollower - momentum/mean-reversion signal model.

    Detects trend regime and generates position sizing signals.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        seq_len: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Regime: trend | mean-revert | volatile | neutral
        self.regime_head = nn.Linear(d_model, 4)
        # Position: continuous in [-1, 1]
        self.position_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        # Stop-loss level
        self.stop_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.input_proj(x) + self.pos_embed(pos))
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1, :]

        return {
            "regime_logits": self.regime_head(pooled),
            "position": torch.tanh(self.position_head(pooled)).squeeze(-1),
            "stop_loss": self.stop_head(pooled).squeeze(-1),
        }


class LuminaMarketMaker(nn.Module):
    """LuminaMarketMaker - bid-ask spread and inventory model for market making.

    Learns optimal spread and inventory thresholds from order book features.
    """

    def __init__(
        self,
        orderbook_levels: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # Order book: prices + sizes at each level, bid and ask -> 4 * levels features
        ob_dim = 4 * orderbook_levels

        self.ob_proj = nn.Linear(ob_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Quote outputs: bid spread, ask spread, quote size
        self.bid_spread = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.ask_spread = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.quote_size = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.inventory_risk = nn.Sequential(nn.Linear(d_model, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, orderbook: torch.Tensor, seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """orderbook: (B, T, 4*levels) time series of order book snapshots"""
        B, T, D = orderbook.shape
        h = self.dropout(self.ob_proj(orderbook))
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1, :]

        return {
            "bid_spread_bps": self.bid_spread(pooled).squeeze(-1),
            "ask_spread_bps": self.ask_spread(pooled).squeeze(-1),
            "quote_size": self.quote_size(pooled).squeeze(-1),
            "inventory_risk": self.inventory_risk(pooled).squeeze(-1),
        }
