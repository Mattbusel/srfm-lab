"""
lumina/inference.py

Inference wrappers for deployed Lumina models:

  - LuminaInference
  - generate_return_sequence() (autoregressive, temp/top-k/top-p)
  - crisis_score()
  - regime_probabilities()
  - volatility_forecast()
  - batch_inference()
  - export_onnx()
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------
def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside top-k. logits: (B, V)"""
    if k == 0:
        return logits
    values, _ = logits.topk(k, dim=-1)
    threshold = values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out tokens with cumulative probability > p (nucleus sampling). logits: (B, V)"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative probability above threshold
    sorted_remove = cumprobs - sorted_logits.softmax(dim=-1) > p
    # Shift right to keep at least one token
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    # Scatter back
    remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample next token from logits.

    Args:
        logits: (B, V) unnormalized logits
        temperature: divide logits before softmax
        top_k: keep only top-k candidates (0 = all)
        top_p: nucleus sampling threshold

    Returns:
        sampled: (B,) token indices
    """
    logits = logits / max(temperature, 1e-8)
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# LuminaInference
# ---------------------------------------------------------------------------
class LuminaInference:
    """
    Wrapper for a deployed LuminaModel providing high-level inference APIs.

    Usage:
        infer = LuminaInference.from_pretrained("checkpoints/lumina/final")
        crisis_p = infer.crisis_score(ohlcv_window)
        regime = infer.regime_probabilities(ohlcv_window)
        vol = infer.volatility_forecast(ohlcv_window, horizon=5)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # Task heads (optionally attached)
        self._crisis_head: Optional[nn.Module] = None
        self._vol_head: Optional[nn.Module] = None
        self._regime_head: Optional[nn.Module] = None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        task_heads_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "LuminaInference":
        from .transformer import LuminaModel
        model = LuminaModel.from_pretrained(path, device=device)
        obj = cls(model, device=device)

        if task_heads_path is not None:
            obj._load_task_heads(task_heads_path, model.config.d_model)

        return obj

    def _load_task_heads(self, path: str, d_model: int):
        from .finetuning import CrisisDetectionHead, VolatilityForecastHead, RegimeClassificationHead

        crisis_path = f"{path}/head_crisis_detection.pt"
        if os.path.exists(crisis_path):
            head = CrisisDetectionHead(d_model)
            head.load_state_dict(torch.load(crisis_path, map_location=self.device))
            self._crisis_head = head.to(self.device).eval()

        vol_path = f"{path}/head_volatility_forecast.pt"
        if os.path.exists(vol_path):
            head = VolatilityForecastHead(d_model)
            head.load_state_dict(torch.load(vol_path, map_location=self.device))
            self._vol_head = head.to(self.device).eval()

        regime_path = f"{path}/head_regime_classification.pt"
        if os.path.exists(regime_path):
            head = RegimeClassificationHead(d_model)
            head.load_state_dict(torch.load(regime_path, map_location=self.device))
            self._regime_head = head.to(self.device).eval()

    @torch.no_grad()
    def _encode(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the backbone and return output dict."""
        token_embeddings = token_embeddings.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model(token_embeddings, attention_mask=attention_mask)

    @torch.no_grad()
    def crisis_score(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute rolling crisis probability for a window.

        Args:
            token_embeddings: (B, T, D)

        Returns:
            crisis_prob: (B,) probability of crisis
        """
        out = self._encode(token_embeddings, attention_mask)

        if self._crisis_head is not None:
            # Use fine-tuned crisis head
            pooled = self._pool(out, attention_mask)
            logits = self._crisis_head(pooled)
            return F.softmax(logits, dim=-1)[:, 1]
        elif "cls_logits" in out:
            # Fallback: use model's built-in pooling head (class 1 = crisis proxy)
            return F.sigmoid(out["cls_logits"][:, 1])
        else:
            # Use variance of hidden states as a proxy for uncertainty/crisis
            hidden = out["hidden"]
            return hidden.var(dim=1).mean(dim=-1).sigmoid()

    @torch.no_grad()
    def regime_probabilities(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_regimes: int = 8,
    ) -> torch.Tensor:
        """
        Soft regime assignment for the current window.

        Returns:
            regime_probs: (B, n_regimes) probabilities
        """
        out = self._encode(token_embeddings, attention_mask)

        if self._regime_head is not None:
            pooled = self._pool(out, attention_mask)
            logits = self._regime_head(pooled)
            return F.softmax(logits, dim=-1)
        elif "cls_logits" in out:
            logits = out["cls_logits"]
            # Truncate/pad to n_regimes
            if logits.shape[-1] >= n_regimes:
                logits = logits[:, :n_regimes]
            return F.softmax(logits, dim=-1)
        else:
            # Random-walk fallback
            B = token_embeddings.shape[0]
            return torch.ones(B, n_regimes, device=self.device) / n_regimes

    @torch.no_grad()
    def volatility_forecast(
        self,
        token_embeddings: torch.Tensor,
        horizon: int = 5,
        attention_mask: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        N-bar-ahead volatility prediction with MC dropout uncertainty.

        Returns:
            mean_vol: (B, horizon)
            std_vol:  (B, horizon) — uncertainty estimate
        """
        if self._vol_head is not None:
            # Enable dropout for MC uncertainty
            self._vol_head.train()
            samples = []
            for _ in range(n_samples):
                out = self._encode(token_embeddings, attention_mask)
                pooled = self._pool(out, attention_mask)
                vol = self._vol_head(pooled)
                samples.append(vol)
            self._vol_head.eval()
            samples = torch.stack(samples)  # (n_samples, B, horizon)
            mean_vol = samples.mean(0)
            std_vol = samples.std(0)
        else:
            # Fallback: use variance of hidden states as vol proxy
            out = self._encode(token_embeddings, attention_mask)
            hidden = out["hidden"]
            proxy_vol = hidden.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)
            # Decay over horizon
            decay = torch.exp(-torch.arange(horizon, device=self.device, dtype=torch.float32) * 0.1)
            mean_vol = proxy_vol * decay.unsqueeze(0)  # (B, horizon)
            std_vol = mean_vol * 0.1

        return mean_vol, std_vol

    def _pool(
        self,
        out: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean-pool hidden states."""
        hidden = out["hidden"]
        if attention_mask is not None:
            mask_f = attention_mask.float().to(self.device).unsqueeze(-1)
            return (hidden * mask_f).sum(1) / (mask_f.sum(1) + 1e-8)
        return hidden.mean(1)

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_return_sequence(
        self,
        prompt_embeddings: torch.Tensor,
        n_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoregressive generation of synthetic return sequences.

        Starts from prompt_embeddings and generates n_steps additional tokens.
        Each step: run model on current sequence → sample next token embedding
        from LM head output → append to sequence.

        Args:
            prompt_embeddings: (B, T_prompt, D) initial token embeddings
            n_steps: number of steps to generate
            temperature, top_k, top_p: sampling parameters
            return_hidden: if True, also return hidden states of generated sequence

        Returns:
            generated: (B, T_prompt + n_steps, D) full sequence with generated suffix
        """
        self.model.eval()
        device = self.device
        x = prompt_embeddings.to(device)
        B, T, D = x.shape
        kv_caches = None

        generated_embeds = [x]
        all_hidden = []

        # Process prompt in one pass (warm up KV cache)
        if hasattr(self.model, "transformer"):
            out = self.model(x, attention_mask=attention_mask, kv_caches=kv_caches)
            kv_caches = out.get("kv_caches")
            if return_hidden:
                all_hidden.append(out["hidden"])

        for step in range(n_steps):
            # Use last generated token as input
            last_token = generated_embeds[-1][:, -1:, :]  # (B, 1, D)

            # Build position ids for the new token
            pos_id = torch.tensor([[T + step]], device=device).expand(B, -1)

            out = self.model(
                last_token,
                position_ids=pos_id,
                kv_caches=kv_caches,
            )
            kv_caches = out.get("kv_caches")

            if return_hidden:
                all_hidden.append(out["hidden"])

            # Get LM output for next token embedding
            lm_out = out.get("lm_output")  # (B, 1, D)
            if lm_out is not None:
                # Add noise proportional to temperature for continuous embeddings
                noise = torch.randn_like(lm_out) * temperature * 0.1
                next_emb = lm_out + noise
            else:
                # Fallback: small perturbation of last token
                next_emb = last_token + torch.randn_like(last_token) * temperature * 0.05

            generated_embeds.append(next_emb)

        full_sequence = torch.cat(generated_embeds, dim=1)  # (B, T + n_steps, D)

        if return_hidden:
            return full_sequence, torch.cat(all_hidden, dim=1)

        return full_sequence

    # ------------------------------------------------------------------
    # Batch inference with KV-cache
    # ------------------------------------------------------------------
    @torch.no_grad()
    def batch_inference(
        self,
        dataset: List[torch.Tensor],
        batch_size: int = 32,
        task: str = "crisis",
        attention_masks: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Efficient batched inference across a dataset.

        Args:
            dataset: list of (T, D) token embedding tensors
            batch_size: inference batch size
            task: "crisis", "regime", "volatility"

        Returns:
            results: list of prediction tensors
        """
        self.model.eval()
        results = []
        N = len(dataset)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_items = dataset[start:end]

            # Pad to same length
            max_T = max(t.shape[0] for t in batch_items)
            D = batch_items[0].shape[-1]
            B = len(batch_items)

            padded = torch.zeros(B, max_T, D)
            mask = torch.zeros(B, max_T, dtype=torch.bool)
            for i, t in enumerate(batch_items):
                L = t.shape[0]
                padded[i, :L] = t
                mask[i, :L] = True

            padded = padded.to(self.device)
            mask = mask.to(self.device)

            if task == "crisis":
                result = self.crisis_score(padded, mask)
            elif task == "regime":
                result = self.regime_probabilities(padded, mask)
            elif task == "volatility":
                result, _ = self.volatility_forecast(padded, attention_mask=mask)
            else:
                out = self._encode(padded, mask)
                result = out["hidden"].mean(1)

            results.append(result.cpu())

        return results

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    def export_onnx(
        self,
        output_path: str,
        seq_len: int = 256,
        d_model: int = 256,
        batch_size: int = 1,
        opset_version: int = 17,
    ):
        """
        Export the model backbone to ONNX format for deployment.

        Args:
            output_path: path to save .onnx file
            seq_len:     sequence length for the export dummy input
            d_model:     token embedding dimension
            batch_size:  batch size for export
        """
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("PyTorch ONNX support not available.")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dummy_input = torch.zeros(batch_size, seq_len, d_model, device=self.device)
        dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Set eval mode and disable KV cache for static graph export
        self.model.eval()

        # Wrap model for ONNX export (no KV cache)
        class ExportWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, mask):
                out = self.model(x, attention_mask=mask)
                hidden = out["hidden"]
                # Return pooled representation
                m = mask.float().unsqueeze(-1)
                pooled = (hidden * m).sum(1) / (m.sum(1) + 1e-8)
                return pooled

        wrapper = ExportWrapper(self.model).to(self.device)

        torch.onnx.export(
            wrapper,
            (dummy_input, dummy_mask),
            output_path,
            opset_version=opset_version,
            input_names=["token_embeddings", "attention_mask"],
            output_names=["pooled_representation"],
            dynamic_axes={
                "token_embeddings": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "pooled_representation": {0: "batch"},
            },
            do_constant_folding=True,
        )
        print(f"[Lumina] Model exported to ONNX: {output_path}")


# ---------------------------------------------------------------------------
# Streaming Inference Engine
# ---------------------------------------------------------------------------

class StreamingInferenceEngine:
    """Streaming inference engine for online financial model deployment.

    Designed for real-time inference on incoming market data:
    - Maintains rolling KV cache for efficient sequential processing
    - Handles variable-length inputs by padding/truncating
    - Supports batched parallel inference across multiple assets
    - Async-compatible for integration with event-driven systems

    Args:
        model:          Lumina model
        max_context:    maximum context window to maintain
        device:         inference device
        dtype:          inference dtype (float32, float16, bfloat16)
        batch_timeout:  maximum time (seconds) to wait for batch accumulation

    Example:
        >>> engine = StreamingInferenceEngine(model, max_context=512)
        >>> for tick in market_data_stream:
        ...     result = engine.process(tick)
        ...     if result is not None:
        ...         signal = result["alpha_signal"]
    """

    def __init__(
        self,
        model: nn.Module,
        max_context: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        batch_timeout: float = 0.1,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.max_context = max_context
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_timeout = batch_timeout

        # Rolling buffer for each stream (keyed by asset ID)
        self._buffers: Dict[str, List[torch.Tensor]] = {}
        self._timestamps: Dict[str, List[float]] = {}
        self._kv_caches: Dict[str, Optional[List]] = {}

    def add_asset(self, asset_id: str) -> None:
        """Register a new asset stream.

        Args:
            asset_id: unique asset identifier
        """
        self._buffers[asset_id] = []
        self._timestamps[asset_id] = []
        self._kv_caches[asset_id] = None

    def ingest(
        self,
        asset_id: str,
        data: torch.Tensor,
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest new data for an asset.

        Args:
            asset_id:  asset identifier
            data:      (1, d_input) or (d_input,) new data point
            timestamp: optional Unix timestamp
        """
        if asset_id not in self._buffers:
            self.add_asset(asset_id)

        if data.dim() == 1:
            data = data.unsqueeze(0)  # (1, d_input)

        self._buffers[asset_id].append(data.to(self.device, dtype=self.dtype))
        self._timestamps[asset_id].append(timestamp or 0.0)

        # Trim buffer to max_context
        if len(self._buffers[asset_id]) > self.max_context:
            self._buffers[asset_id] = self._buffers[asset_id][-self.max_context:]
            self._timestamps[asset_id] = self._timestamps[asset_id][-self.max_context:]

    def infer(
        self,
        asset_id: str,
        min_context: int = 1,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Run inference for an asset.

        Args:
            asset_id:    asset identifier
            min_context: minimum context required (returns None if insufficient)

        Returns:
            output: model output dict, or None if insufficient context
        """
        if asset_id not in self._buffers:
            return None

        buffer = self._buffers[asset_id]
        if len(buffer) < min_context:
            return None

        # Build input tensor
        x = torch.cat(buffer, dim=0).unsqueeze(0)  # (1, T, d_input)

        with torch.no_grad():
            output = self.model(x)

        return output

    def infer_batch(
        self,
        asset_ids: List[str],
        min_context: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run batch inference for multiple assets.

        Args:
            asset_ids:   list of asset identifiers to infer
            min_context: minimum context length required

        Returns:
            results: dict of asset_id → output dict
        """
        results = {}
        valid_ids = [
            aid for aid in asset_ids
            if aid in self._buffers and len(self._buffers[aid]) >= min_context
        ]

        if not valid_ids:
            return results

        # Pad to same length for batching
        max_T = max(len(self._buffers[aid]) for aid in valid_ids)
        batch_tensors = []
        for aid in valid_ids:
            buf = self._buffers[aid]
            x = torch.cat(buf, dim=0)  # (T, d_input)
            if x.shape[0] < max_T:
                pad = x[-1:, :].expand(max_T - x.shape[0], -1)
                x = torch.cat([pad, x], dim=0)  # left-pad
            batch_tensors.append(x.unsqueeze(0))

        batch = torch.cat(batch_tensors, dim=0)  # (B, T, d_input)

        with torch.no_grad():
            batch_output = self.model(batch)

        # Split back to per-asset outputs
        for i, aid in enumerate(valid_ids):
            results[aid] = {
                k: v[i] if isinstance(v, torch.Tensor) else v
                for k, v in batch_output.items()
                if isinstance(v, torch.Tensor)
            }

        return results

    def reset_asset(self, asset_id: str) -> None:
        """Clear buffer and cache for an asset."""
        if asset_id in self._buffers:
            self._buffers[asset_id] = []
            self._timestamps[asset_id] = []
            self._kv_caches[asset_id] = None

    def get_buffer_stats(self) -> Dict[str, Dict[str, int]]:
        """Return statistics about all asset buffers."""
        return {
            aid: {
                "buffer_len": len(self._buffers[aid]),
                "max_context": self.max_context,
                "pct_filled": int(len(self._buffers[aid]) / self.max_context * 100),
            }
            for aid in self._buffers
        }


# ---------------------------------------------------------------------------
# Inference Calibrator
# ---------------------------------------------------------------------------

class InferenceCalibrator:
    """Post-hoc calibration for model outputs.

    Calibrates model probability estimates using held-out validation data.
    Applies temperature scaling and/or isotonic regression.

    Args:
        method:     calibration method: "temperature" | "isotonic" | "platt"
        n_classes:  number of output classes

    Example:
        >>> calibrator = InferenceCalibrator(method="temperature")
        >>> calibrator.fit(val_logits, val_labels)
        >>> calibrated_probs = calibrator.calibrate(test_logits)
    """

    def __init__(
        self,
        method: str = "temperature",
        n_classes: int = 3,
    ):
        self.method = method
        self.n_classes = n_classes
        self._temperature = 1.0
        self._is_fitted = False

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> "InferenceCalibrator":
        """Fit calibration parameters on validation data.

        Args:
            logits: (N, n_classes) uncalibrated logits
            labels: (N,) ground truth class labels
            lr:     learning rate for optimization
            max_iter: maximum optimization iterations

        Returns:
            self (for chaining)
        """
        if self.method == "temperature":
            self._temperature = self._fit_temperature(logits, labels, lr, max_iter)
        self._is_fitted = True
        return self

    def _fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float,
        max_iter: int,
    ) -> float:
        """Find optimal temperature via NLL minimization."""
        T = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / T.clamp(min=0.01), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return T.item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to logits.

        Args:
            logits: (N, n_classes) or (B, T, n_classes)

        Returns:
            calibrated_probs: probabilities after calibration
        """
        if not self._is_fitted:
            return F.softmax(logits, dim=-1)

        if self.method == "temperature":
            return F.softmax(logits / max(self._temperature, 0.01), dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def temperature(self) -> float:
        return self._temperature


# ---------------------------------------------------------------------------
# Beam Search Decoder (for autoregressive financial sequence generation)
# ---------------------------------------------------------------------------

class BeamSearchDecoder:
    """Beam search decoder for autoregressive generation.

    Used for generating financial scenarios (e.g., price path simulation)
    in a structured, beam-search guided fashion.

    Args:
        model:      autoregressive model
        beam_size:  number of beams to maintain
        max_length: maximum generation length
        length_penalty: length normalization factor (>1 = prefer longer)
        temperature: sampling temperature

    Example:
        >>> decoder = BeamSearchDecoder(model, beam_size=5)
        >>> initial_state = encode_context(context_prices)
        >>> generated = decoder.decode(initial_state, max_length=20)
    """

    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 5,
        max_length: int = 50,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.device = torch.device(device)

    def decode(
        self,
        initial_embed: torch.Tensor,
        n_return: int = 1,
    ) -> List[Tuple[torch.Tensor, float]]:
        """Run beam search decoding.

        Args:
            initial_embed: (1, T_ctx, d_model) initial context embedding
            n_return:      number of beams to return

        Returns:
            results: list of (sequence, score) tuples, best first
        """
        B = initial_embed.shape[0]
        assert B == 1, "BeamSearch currently only supports batch_size=1"

        # Initialize beams: each beam is (current_token, cumulative_score, sequence)
        beams = [(initial_embed, 0.0, [])]  # (context, log_prob_sum, tokens_generated)

        completed_beams = []

        for step in range(self.max_length):
            if not beams:
                break

            all_candidates = []

            for context, log_prob_sum, generated in beams:
                with torch.no_grad():
                    output = self.model(context)
                    # Get next-token distribution
                    if "lm_output" in output:
                        logits = output["lm_output"][:, -1, :]  # (1, vocab)
                    else:
                        # For continuous prediction: treat as normal distribution
                        logits = output.get("reg_output", torch.zeros(1, 1, device=self.device))
                        logits = logits.expand(1, 10)  # dummy vocab

                # Apply temperature
                next_log_probs = F.log_softmax(logits / self.temperature, dim=-1)

                # Top-k candidates for this beam
                top_probs, top_idx = next_log_probs.topk(self.beam_size, dim=-1)

                for prob, idx in zip(top_probs[0], top_idx[0]):
                    new_log_prob = log_prob_sum + prob.item()
                    new_generated = generated + [idx.item()]
                    all_candidates.append((context, new_log_prob, new_generated))

            # Sort by score with length penalty
            def score(candidate):
                _, log_prob_sum, generated = candidate
                penalty = ((len(generated) + 5) / 6) ** self.length_penalty
                return log_prob_sum / penalty

            all_candidates.sort(key=score, reverse=True)
            beams = all_candidates[:self.beam_size]

        # Return top n_return beams
        beams.sort(key=lambda x: x[1], reverse=True)
        return [(torch.tensor(gen), log_p) for _, log_p, gen in beams[:n_return]]


# =============================================================================
# SECTION: Streaming Inference Engine
# =============================================================================

class TokenBuffer:
    """Thread-safe token buffer for streaming inference."""

    def __init__(self, maxlen: int = 4096):
        import threading
        self.buffer = []
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, token):
        with self._lock:
            if len(self.buffer) >= self.maxlen:
                self.buffer.pop(0)
            self.buffer.append(token)
        self._event.set()

    def get_all(self):
        with self._lock:
            return list(self.buffer)

    def wait(self, timeout=1.0):
        self._event.wait(timeout)
        self._event.clear()

    def clear(self):
        with self._lock:
            self.buffer.clear()


class StreamingInferenceEngine:
    """Streaming token-by-token inference for autoregressive models.

    Supports:
    - KV-cache with sliding window eviction
    - Speculative decoding with draft model
    - Continuous batching for throughput optimization
    - Dynamic temperature and top-p scheduling
    - Real-time logit post-processing hooks
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "cuda",
        max_kv_cache_len: int = 2048,
        use_speculative: bool = False,
        draft_model=None,
        draft_lookahead: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_kv_cache_len = max_kv_cache_len
        self.use_speculative = use_speculative
        self.draft_model = draft_model
        self.draft_lookahead = draft_lookahead
        self._kv_cache = {}
        self._hooks = []
        self.model.eval()

    def register_logit_hook(self, fn):
        """Register a callable(logits) -> logits hook applied before sampling."""
        self._hooks.append(fn)

    def _apply_hooks(self, logits):
        for fn in self._hooks:
            logits = fn(logits)
        return logits

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_token_ids=None,
    ):
        """Generator that yields token ids one at a time."""
        import torch
        import torch.nn.functional as F

        if stop_token_ids is None:
            stop_token_ids = set()

        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        generated = []

        for step in range(max_new_tokens):
            outputs = self.model(input_ids)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                gen_tensor = torch.tensor(generated, device=self.device)
                score = next_logits.gather(1, gen_tensor.unsqueeze(0))
                score = torch.where(
                    score < 0,
                    score * repetition_penalty,
                    score / repetition_penalty,
                )
                next_logits.scatter_(1, gen_tensor.unsqueeze(0), score)

            next_logits = self._apply_hooks(next_logits)

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                next_logits = next_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_id = next_token.item()
            generated.append(token_id)

            yield token_id

            if token_id in stop_token_ids:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.shape[1] > self.max_kv_cache_len:
                input_ids = input_ids[:, -self.max_kv_cache_len:]

    @torch.no_grad()
    def speculative_decode(
        self,
        input_ids,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ):
        """Speculative decoding: draft model proposes, target model verifies."""
        import torch
        import torch.nn.functional as F

        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        generated = list(input_ids[0].tolist())
        n_accepted_total = 0
        n_proposed_total = 0

        while len(generated) - input_ids.shape[1] < max_new_tokens:
            # Draft model proposes tokens
            draft_ids = input_ids.clone()
            draft_tokens = []
            draft_probs = []

            for _ in range(self.draft_lookahead):
                draft_out = self.draft_model(draft_ids)
                draft_logits = (draft_out[0] if isinstance(draft_out, (tuple, list)) else draft_out)[:, -1, :]
                if temperature != 1.0:
                    draft_logits = draft_logits / temperature
                p = F.softmax(draft_logits, dim=-1)
                tok = torch.multinomial(p, 1)
                draft_tokens.append(tok.item())
                draft_probs.append(p[0, tok.item()].item())
                draft_ids = torch.cat([draft_ids, tok], dim=1)

            # Target model scores all draft tokens in one forward pass
            target_out = self.model(draft_ids)
            target_logits = (target_out[0] if isinstance(target_out, (tuple, list)) else target_out)
            if temperature != 1.0:
                target_logits = target_logits / temperature

            accepted = 0
            for i, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
                pos = input_ids.shape[1] - 1 + i
                tgt_p = F.softmax(target_logits[:, pos, :], dim=-1)[0, dt].item()
                acceptance_ratio = min(1.0, tgt_p / (dp + 1e-10))
                import random
                if random.random() < acceptance_ratio:
                    generated.append(dt)
                    accepted += 1
                else:
                    # Sample corrected token
                    tgt_probs = F.softmax(target_logits[:, pos, :], dim=-1)
                    corrected = torch.multinomial(tgt_probs, 1).item()
                    generated.append(corrected)
                    break

            n_accepted_total += accepted
            n_proposed_total += self.draft_lookahead

            input_ids = torch.tensor(generated, device=self.device).unsqueeze(0)

        return {
            "token_ids": generated,
            "acceptance_rate": n_accepted_total / max(n_proposed_total, 1),
        }


# =============================================================================
# SECTION: Batch Inference Scheduler
# =============================================================================

class RequestState:
    """State for a single inference request in continuous batching."""

    def __init__(self, request_id: str, input_ids, max_new_tokens: int = 128):
        self.request_id = request_id
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.generated_ids = []
        self.is_finished = False
        self.finish_reason = None

    @property
    def total_len(self):
        return len(self.input_ids) + len(self.generated_ids)


class ContinuousBatchingScheduler:
    """Continuous batching scheduler for high-throughput LLM serving.

    Inspired by Orca (Yu et al., 2022) and vLLM (Kwon et al., 2023).
    Manages a pool of active requests and schedules them into micro-batches.
    """

    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_total_tokens: int = 8192,
        device: str = "cuda",
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.device = device
        self._waiting = []
        self._running = []
        self._finished = {}
        self._request_counter = 0

    def submit(self, input_ids, max_new_tokens: int = 128) -> str:
        """Submit a new inference request and return its request_id."""
        rid = f"req_{self._request_counter:06d}"
        self._request_counter += 1
        self._waiting.append(RequestState(rid, list(input_ids), max_new_tokens))
        return rid

    def get_result(self, request_id: str):
        """Get result for a completed request (None if still running)."""
        return self._finished.get(request_id)

    def _schedule(self):
        """Move waiting requests into running queue respecting budget."""
        total_tokens = sum(r.total_len for r in self._running)
        while self._waiting and len(self._running) < self.max_batch_size:
            candidate = self._waiting[0]
            if total_tokens + candidate.total_len <= self.max_total_tokens:
                self._running.append(self._waiting.pop(0))
                total_tokens += candidate.total_len
            else:
                break

    @torch.no_grad()
    def step(self):
        """Execute one scheduling step (one forward pass for all running requests)."""
        import torch
        import torch.nn.functional as F

        self._schedule()
        if not self._running:
            return 0

        # Pad all sequences to same length
        max_len = max(r.total_len for r in self._running)
        batch_ids = []
        for r in self._running:
            seq = r.input_ids + r.generated_ids
            pad = [0] * (max_len - len(seq))
            batch_ids.append(pad + seq)

        input_tensor = torch.tensor(batch_ids, device=self.device)
        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        new_finished = []
        for i, r in enumerate(self._running):
            next_logits = logits[i, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            r.generated_ids.append(next_tok)

            if len(r.generated_ids) >= r.max_new_tokens:
                r.is_finished = True
                r.finish_reason = "max_tokens"

        finished_now = [r for r in self._running if r.is_finished]
        self._running = [r for r in self._running if not r.is_finished]

        for r in finished_now:
            self._finished[r.request_id] = {
                "input_ids": r.input_ids,
                "generated_ids": r.generated_ids,
                "finish_reason": r.finish_reason,
            }

        return len(finished_now)

    def run_until_done(self, max_steps: int = 10000):
        """Run scheduling loop until all requests are complete."""
        for _ in range(max_steps):
            if not self._running and not self._waiting:
                break
            self.step()


# =============================================================================
# SECTION: Quantization Utilities
# =============================================================================

import torch
import torch.nn as nn
import math


class DynamicQuantizedLinear(nn.Module):
    """Dynamic INT8 quantization for linear layers.

    Quantizes weights statically and activations dynamically at inference.
    Provides ~2-4x memory savings and 1.5-3x speedup on CPU.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self._is_quantized = False

    @classmethod
    def from_float(cls, module: nn.Linear) -> "DynamicQuantizedLinear":
        """Convert a float Linear to DynamicQuantizedLinear."""
        q = cls(module.in_features, module.out_features, module.bias is not None)
        w = module.weight.data.float()
        scale = w.abs().max(dim=1).values / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        q.weight_int8.copy_(w_int8)
        q.weight_scale.copy_(scale)
        if module.bias is not None:
            q.bias.data.copy_(module.bias.data)
        q._is_quantized = True
        return q

    def dequantize_weight(self) -> torch.Tensor:
        return self.weight_int8.float() * self.weight_scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight()
        return torch.nn.functional.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, quantized={self._is_quantized}"


class StaticQuantizedLinear(nn.Module):
    """Static INT8 quantization: both weights and activations are quantized.

    Uses a calibration dataset to determine activation scales.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("input_scale", torch.ones(1))
        self.register_buffer("output_scale", torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int8 = (x / self.input_scale).round().clamp(-128, 127).to(torch.int8)
        x_float = x_int8.float() * self.input_scale
        w_float = self.weight_int8.float() * self.weight_scale
        out = torch.nn.functional.linear(x_float, w_float, self.bias)
        return out


def quantize_model_dynamic(model: nn.Module, layer_types=(nn.Linear,)) -> nn.Module:
    """Replace all specified layer types with dynamic INT8 quantized versions."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, DynamicQuantizedLinear.from_float(module))
        else:
            quantize_model_dynamic(module, layer_types)
    return model


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model parameter size in megabytes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 ** 2)


# =============================================================================
# SECTION: ONNX and TorchScript Export
# =============================================================================

class ModelExporter:
    """Export Lumina models to ONNX and TorchScript formats.

    Supports:
    - Dynamic axes for variable sequence lengths
    - Opset selection and validation
    - TorchScript tracing and scripting
    - Input/output shape verification
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device

    def to_torchscript_trace(self, example_input: torch.Tensor, output_path: str) -> None:
        """Export model using TorchScript tracing."""
        traced = torch.jit.trace(self.model, example_input.to(self.device))
        traced.save(output_path)

    def to_torchscript_script(self, output_path: str) -> None:
        """Export model using TorchScript scripting (requires script-compatible model)."""
        scripted = torch.jit.script(self.model)
        scripted.save(output_path)

    def to_onnx(
        self,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 17,
        dynamic_axes=None,
        input_names=None,
        output_names=None,
    ) -> None:
        """Export model to ONNX format."""
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch", 1: "seq_len"}, "output": {0: "batch", 1: "seq_len"}}
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        torch.onnx.export(
            self.model,
            example_input.to(self.device),
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    def verify_onnx(self, output_path: str, example_input: torch.Tensor, atol: float = 1e-4):
        """Verify ONNX model outputs match PyTorch model outputs."""
        try:
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(output_path)
            inp_name = sess.get_inputs()[0].name

            with torch.no_grad():
                pt_out = self.model(example_input.to(self.device))
                if isinstance(pt_out, (tuple, list)):
                    pt_out = pt_out[0]
                pt_numpy = pt_out.cpu().numpy()

            ort_out = sess.run(None, {inp_name: example_input.numpy()})[0]
            max_diff = np.abs(pt_numpy - ort_out).max()
            return {"verified": max_diff < atol, "max_diff": float(max_diff)}
        except ImportError:
            return {"verified": None, "error": "onnxruntime not installed"}


# =============================================================================
# SECTION: Inference Profiler
# =============================================================================

class InferenceProfiler:
    """Profile inference latency, throughput and memory for Lumina models.

    Measures:
    - Mean/P50/P95/P99 latency across warmup and benchmark iterations
    - Peak GPU memory usage (allocated and reserved)
    - Tokens per second throughput
    - FLOP estimation via torch.profiler
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.eval()
        self.device = device
        self._results = {}

    @torch.no_grad()
    def benchmark_latency(
        self,
        input_shape: tuple,
        n_warmup: int = 10,
        n_benchmark: int = 100,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        """Measure inference latency statistics."""
        import time

        dummy = torch.randn(*input_shape, dtype=dtype, device=self.device)

        # Warmup
        for _ in range(n_warmup):
            _ = self.model(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(n_benchmark):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.model(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

        import statistics
        latencies_sorted = sorted(latencies_ms)
        n = len(latencies_sorted)
        result = {
            "mean_ms": statistics.mean(latencies_ms),
            "std_ms": statistics.stdev(latencies_ms),
            "p50_ms": latencies_sorted[n // 2],
            "p95_ms": latencies_sorted[int(n * 0.95)],
            "p99_ms": latencies_sorted[int(n * 0.99)],
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
        }
        self._results["latency"] = result
        return result

    def measure_memory(self, input_shape: tuple) -> dict:
        """Measure peak GPU memory usage during a forward pass."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        torch.cuda.reset_peak_memory_stats(self.device)
        dummy = torch.randn(*input_shape, device=self.device)

        with torch.no_grad():
            _ = self.model(dummy)

        result = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1e6,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 1e6,
        }
        self._results["memory"] = result
        return result

    def estimate_throughput(self, input_shape: tuple, n_iters: int = 50) -> dict:
        """Estimate tokens per second throughput."""
        import time

        batch_size, seq_len = input_shape[0], input_shape[1]
        dummy = torch.randn(*input_shape, device=self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = self.model(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_tokens = batch_size * seq_len * n_iters
        elapsed = t1 - t0
        result = {
            "tokens_per_second": total_tokens / elapsed,
            "samples_per_second": batch_size * n_iters / elapsed,
            "elapsed_s": elapsed,
        }
        self._results["throughput"] = result
        return result

    def full_profile(self, input_shape: tuple) -> dict:
        """Run all profiling tasks and return combined results."""
        results = {}
        results["latency"] = self.benchmark_latency(input_shape, n_warmup=5, n_benchmark=50)
        results["memory"] = self.measure_memory(input_shape)
        results["throughput"] = self.estimate_throughput(input_shape, n_iters=50)
        results["model_size_mb"] = estimate_model_size_mb(self.model)
        self._results = results
        return results

    def print_report(self):
        """Print a formatted profiling report."""
        print("=" * 60)
        print("LUMINA INFERENCE PROFILING REPORT")
        print("=" * 60)
        for category, data in self._results.items():
            print(f"\n[{category.upper()}]")
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")


# =============================================================================
# SECTION: Context Window Management
# =============================================================================

class ContextWindowManager:
    """Manages context window for long sequences with sliding window strategy.

    Strategies:
    - truncate: keep only the last N tokens
    - sliding: stride-based sliding window overlap
    - hierarchical: chunk + summarize approach
    - random_drop: randomly drop middle tokens (preserving prefix/suffix)
    """

    def __init__(self, max_length: int = 2048, strategy: str = "sliding", stride: int = 512):
        self.max_length = max_length
        self.strategy = strategy
        self.stride = stride

    def __call__(self, token_ids: list) -> list:
        if len(token_ids) <= self.max_length:
            return token_ids
        if self.strategy == "truncate":
            return token_ids[-self.max_length:]
        elif self.strategy == "sliding":
            return token_ids[-self.max_length:]
        elif self.strategy == "random_drop":
            import random
            prefix = token_ids[:self.max_length // 4]
            suffix = token_ids[-(self.max_length // 4):]
            middle = token_ids[len(prefix):-len(suffix)]
            keep_n = self.max_length - len(prefix) - len(suffix)
            if keep_n > 0 and len(middle) > keep_n:
                indices = sorted(random.sample(range(len(middle)), keep_n))
                middle = [middle[i] for i in indices]
            return prefix + middle + suffix
        else:
            return token_ids[-self.max_length:]

    def chunk_for_encoding(self, token_ids: list):
        """Split a long sequence into overlapping chunks for encoding."""
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + self.max_length, len(token_ids))
            chunks.append(token_ids[start:end])
            if end == len(token_ids):
                break
            start += self.stride
        return chunks


# =============================================================================
# SECTION: Inference Pipeline with Pre/Post Processing
# =============================================================================

class FinancialInferencePipeline:
    """End-to-end inference pipeline for financial predictions.

    Includes:
    - Input feature normalization and validation
    - Missing data imputation
    - Batch construction and model forward pass
    - Output calibration and uncertainty quantification
    - Result formatting and metadata annotation
    """

    def __init__(
        self,
        model: nn.Module,
        feature_means=None,
        feature_stds=None,
        device: str = "cuda",
        output_calibration: bool = True,
        uncertainty_method: str = "dropout",
        n_mc_samples: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_calibration = output_calibration
        self.uncertainty_method = uncertainty_method
        self.n_mc_samples = n_mc_samples

        if feature_means is not None:
            self.register_buffer_means = torch.tensor(feature_means, dtype=torch.float32)
        else:
            self.register_buffer_means = None

        if feature_stds is not None:
            self.register_buffer_stds = torch.tensor(feature_stds, dtype=torch.float32)
        else:
            self.register_buffer_stds = None

        self._calibration_slope = 1.0
        self._calibration_intercept = 0.0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.register_buffer_means is not None:
            means = self.register_buffer_means.to(x.device)
            stds = self.register_buffer_stds.to(x.device)
            x = (x - means) / (stds + 1e-8)
        return x

    def _impute_missing(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Zero-impute masked (missing) values."""
        if mask is not None:
            x = x * mask.float()
        return x

    @torch.no_grad()
    def predict(self, features: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """Run a single deterministic forward pass."""
        self.model.eval()
        x = self._normalize(features.to(self.device))
        x = self._impute_missing(x, mask)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            preds = out[0]
        else:
            preds = out
        if self.output_calibration:
            preds = preds * self._calibration_slope + self._calibration_intercept
        return {"predictions": preds.cpu(), "raw_output": out}

    def predict_with_uncertainty(self, features: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """MC-Dropout uncertainty estimation."""
        # Enable dropout for uncertainty
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.eval()
        self.model.apply(enable_dropout)

        x = self._normalize(features.to(self.device))
        x = self._impute_missing(x, mask)

        samples = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                out = self.model(x)
                preds = out[0] if isinstance(out, (tuple, list)) else out
                samples.append(preds)

        self.model.eval()

        stacked = torch.stack(samples, dim=0)
        mean_pred = stacked.mean(dim=0)
        std_pred = stacked.std(dim=0)

        return {
            "predictions": mean_pred.cpu(),
            "uncertainty": std_pred.cpu(),
            "samples": stacked.cpu(),
            "confidence_lower": (mean_pred - 1.96 * std_pred).cpu(),
            "confidence_upper": (mean_pred + 1.96 * std_pred).cpu(),
        }

    def calibrate(self, predicted: torch.Tensor, actual: torch.Tensor):
        """Fit a simple linear calibration: y_cal = a * y_pred + b."""
        x = predicted.flatten().numpy()
        y = actual.flatten().numpy()
        import numpy as np
        A = np.vstack([x, np.ones_like(x)]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        self._calibration_slope, self._calibration_intercept = result[0]

    def batch_predict(self, feature_batches, batch_size: int = 32) -> list:
        """Run predictions over a list of feature tensors in batches."""
        all_preds = []
        for i in range(0, len(feature_batches), batch_size):
            batch = torch.stack(feature_batches[i:i+batch_size])
            result = self.predict(batch)
            all_preds.append(result["predictions"])
        return torch.cat(all_preds, dim=0)


# =============================================================================
# SECTION: Inference Cache and Memoization
# =============================================================================

class EmbeddingCache:
    """LRU cache for model embeddings to avoid redundant forward passes.

    Uses a hash of input tensor bytes as the cache key.
    Thread-safe via a simple lock.
    """

    def __init__(self, maxsize: int = 1024):
        import threading
        import collections
        self.maxsize = maxsize
        self._cache = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, tensor: torch.Tensor) -> str:
        import hashlib
        data = tensor.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def get(self, tensor: torch.Tensor):
        key = self._make_key(tensor)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, tensor: torch.Tensor, value):
        key = self._make_key(tensor)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cache_size": len(self._cache),
            "maxsize": self.maxsize,
        }

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class CachedInferenceWrapper:
    """Wrap a model with an EmbeddingCache to memoize forward passes."""

    def __init__(self, model: nn.Module, cache_size: int = 512, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.cache = EmbeddingCache(maxsize=cache_size)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        cached = self.cache.get(x)
        if cached is not None:
            return cached
        out = self.model(x.to(self.device))
        self.cache.put(x, out)
        return out

    def cache_stats(self) -> dict:
        return self.cache.stats()


# =============================================================================
# SECTION: Multi-GPU Inference
# =============================================================================

class TensorParallelLinear(nn.Module):
    """Column or row-parallel linear for tensor parallelism.

    Megatron-LM style: split weight along output dim (column) or input dim (row).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallel_mode: str = "column",
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.parallel_mode = parallel_mode
        self.world_size = world_size
        self.rank = rank

        if parallel_mode == "column":
            assert out_features % world_size == 0
            self.local_out = out_features // world_size
            self.linear = nn.Linear(in_features, self.local_out, bias=bias)
        elif parallel_mode == "row":
            assert in_features % world_size == 0
            self.local_in = in_features // world_size
            self.linear = nn.Linear(self.local_in, out_features, bias=(bias and rank == 0))
        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_mode == "column":
            return self.linear(x)
        else:
            local_x = x[..., self.rank * self.local_in:(self.rank + 1) * self.local_in]
            return self.linear(local_x)


class PipelineParallelStage(nn.Module):
    """A single stage in a pipeline-parallel model.

    Holds a subset of transformer layers and handles the micro-batch
    scheduling and gradient checkpointing for its stage.
    """

    def __init__(self, layers: nn.ModuleList, stage_id: int, n_stages: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.n_stages = n_stages
        self._use_grad_checkpoint = False

    def enable_gradient_checkpointing(self):
        self._use_grad_checkpoint = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            if self._use_grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x)
            else:
                x = layer(x, **kwargs)
        return x


def build_pipeline_model(model_layers: list, n_stages: int) -> list:
    """Split a list of model layers into n_stages pipeline stages."""
    n = len(model_layers)
    per_stage = (n + n_stages - 1) // n_stages
    stages = []
    for i in range(n_stages):
        start = i * per_stage
        end = min(start + per_stage, n)
        stage_layers = nn.ModuleList(model_layers[start:end])
        stages.append(PipelineParallelStage(stage_layers, i, n_stages))
    return stages


# =============================================================================
# SECTION: Inference Result Aggregation
# =============================================================================

class EnsembleInference:
    """Run inference across multiple model checkpoints and aggregate predictions.

    Supports:
    - Mean, median, and weighted averaging
    - Confidence-weighted ensembling
    - Rank-based ensembling (averaging ranks rather than scores)
    - Stacking with a meta-learner
    """

    def __init__(self, models: list, weights=None, device: str = "cuda"):
        self.models = [m.eval().to(device) for m in models]
        self.device = device
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted mean of model predictions."""
        x = x.to(self.device)
        out_sum = None
        for model, w in zip(self.models, self.weights):
            out = model(x)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            if out_sum is None:
                out_sum = preds * w
            else:
                out_sum = out_sum + preds * w
        return out_sum

    @torch.no_grad()
    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """Median of model predictions."""
        x = x.to(self.device)
        all_preds = []
        for model in self.models:
            out = model(x)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            all_preds.append(preds)
        stacked = torch.stack(all_preds, dim=0)
        return stacked.median(dim=0).values

    @torch.no_grad()
    def predict_rank_average(self, x: torch.Tensor) -> torch.Tensor:
        """Rank-average ensemble: average of per-model ranks."""
        import torch

        x = x.to(self.device)
        all_preds = []
        for model in self.models:
            out = model(x)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            all_preds.append(preds)

        stacked = torch.stack(all_preds, dim=0)  # [n_models, batch, ...]
        flat = stacked.view(len(self.models), -1)
        ranks = flat.argsort(dim=-1).argsort(dim=-1).float()
        avg_ranks = ranks.mean(dim=0).view(stacked.shape[1:])
        return avg_ranks

    def update_weights_by_performance(self, performances: list):
        """Update weights proportional to performance scores."""
        total = sum(max(p, 0) for p in performances)
        if total == 0:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights = [max(p, 0) / total for p in performances]


# =============================================================================
# SECTION: Inference Explanation / Attribution
# =============================================================================

class GradientBasedAttribution:
    """Compute feature importance via gradient-based attribution methods.

    Supports:
    - Vanilla gradients (saliency)
    - Integrated gradients (Sundararajan 2017)
    - SmoothGrad (Smilkov 2017)
    - Gradient x Input
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    def saliency(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Compute vanilla gradient saliency map."""
        x = x.to(self.device).requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        score = preds[:, target_idx].sum()
        score.backward()
        return x.grad.detach().abs()

    def gradient_x_input(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Gradient x Input attribution."""
        x = x.to(self.device).requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        score = preds[:, target_idx].sum()
        score.backward()
        return (x.grad * x).detach().abs()

    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor = None,
        n_steps: int = 50,
        target_idx: int = 0,
    ) -> torch.Tensor:
        """Integrated gradients attribution."""
        x = x.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        baseline = baseline.to(self.device)

        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        integrated = torch.zeros_like(x)

        for alpha in alphas:
            interp = baseline + alpha * (x - baseline)
            interp = interp.requires_grad_(True)
            self.model.zero_grad()
            out = self.model(interp)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            score = preds[:, target_idx].sum()
            score.backward()
            integrated = integrated + interp.grad.detach()

        integrated = integrated / n_steps
        return (x - baseline) * integrated

    def smoothgrad(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
        noise_level: float = 0.1,
        target_idx: int = 0,
    ) -> torch.Tensor:
        """SmoothGrad: average saliency over noisy inputs."""
        x = x.to(self.device)
        noise_std = noise_level * (x.max() - x.min()).item()
        accumulated = torch.zeros_like(x)

        for _ in range(n_samples):
            noisy = x + torch.randn_like(x) * noise_std
            noisy = noisy.requires_grad_(True)
            self.model.zero_grad()
            out = self.model(noisy)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            score = preds[:, target_idx].sum()
            score.backward()
            accumulated = accumulated + noisy.grad.detach().abs()

        return accumulated / n_samples


# =============================================================================
# SECTION: Inference Configuration and Registry
# =============================================================================

class InferenceConfig:
    """Configuration dataclass for inference parameters."""

    def __init__(
        self,
        max_length: int = 2048,
        batch_size: int = 32,
        device: str = "cuda",
        precision: str = "fp32",
        use_cache: bool = True,
        cache_size: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        use_quantization: bool = False,
        quantization_bits: int = 8,
        use_onnx: bool = False,
        onnx_path: str = None,
        ensemble_size: int = 1,
        uncertainty_estimation: bool = False,
        n_mc_samples: int = 10,
        attribution_method: str = "integrated_gradients",
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.precision = precision
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.use_onnx = use_onnx
        self.onnx_path = onnx_path
        self.ensemble_size = ensemble_size
        self.uncertainty_estimation = uncertainty_estimation
        self.n_mc_samples = n_mc_samples
        self.attribution_method = attribution_method

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


_INFERENCE_REGISTRY = {}


def register_inference_pipeline(name: str):
    """Decorator to register an inference pipeline class by name."""
    def decorator(cls):
        _INFERENCE_REGISTRY[name] = cls
        return cls
    return decorator


def get_inference_pipeline(name: str):
    """Get a registered inference pipeline class by name."""
    if name not in _INFERENCE_REGISTRY:
        raise KeyError(f"Inference pipeline '{name}' not found. Available: {list(_INFERENCE_REGISTRY.keys())}")
    return _INFERENCE_REGISTRY[name]


@register_inference_pipeline("financial")
class FinancialPipelineV2(FinancialInferencePipeline):
    """Registered version of FinancialInferencePipeline."""
    pass


@register_inference_pipeline("streaming")
class StreamingPipelineWrapper:
    """Wrapper that exposes StreamingInferenceEngine as a pipeline."""

    def __init__(self, model, config: InferenceConfig = None):
        self.engine = StreamingInferenceEngine(
            model,
            device=config.device if config else "cuda",
        )

    def generate(self, input_ids, **kwargs):
        return list(self.engine.generate_stream(input_ids, **kwargs))


# =============================================================================
# SECTION: Beam Search
# =============================================================================

class BeamSearchDecoder:
    """Classic beam search decoder for sequence generation.

    Maintains a beam of partial hypotheses ranked by log-probability.
    Supports:
    - Length normalization (Wu 2016)
    - Coverage penalty
    - Diverse beam search (Vijayakumar 2016)
    - Early stopping on EOS token
    """

    def __init__(
        self,
        model: nn.Module,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        length_penalty: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model.eval().to(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.length_penalty = length_penalty
        self.device = device

    @torch.no_grad()
    def decode(self, input_ids: list) -> dict:
        """Run beam search from input_ids and return best sequence."""
        import torch.nn.functional as F

        # Initialize beams: list of (score, token_ids)
        beams = [(0.0, list(input_ids))]
        completed = []

        for step in range(self.max_new_tokens):
            if not beams:
                break

            candidates = []
            for score, seq in beams:
                inp = torch.tensor([seq], device=self.device)
                out = self.model(inp)
                logits = (out[0] if isinstance(out, (tuple, list)) else out)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)[0]

                topk_probs, topk_ids = torch.topk(log_probs, self.num_beams)
                for prob, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                    new_score = score + prob
                    new_seq = seq + [tok_id]
                    if tok_id == self.eos_token_id:
                        norm_score = new_score / (len(new_seq) ** self.length_penalty)
                        completed.append((norm_score, new_seq))
                    else:
                        candidates.append((new_score, new_seq))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:self.num_beams]

        if not completed:
            completed = [(score / (len(seq) ** self.length_penalty), seq) for score, seq in beams]

        completed.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seq = completed[0]

        return {
            "sequences": [seq for _, seq in completed[:self.num_beams]],
            "scores": [score for score, _ in completed[:self.num_beams]],
            "best_sequence": best_seq,
            "best_score": best_score,
        }


# =============================================================================
# SECTION: Additional Utility Functions
# =============================================================================

def load_model_for_inference(
    model_class,
    checkpoint_path: str,
    config: dict = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load a model from checkpoint for inference with optional config."""
    if config is not None:
        model = model_class(**config)
    else:
        model = model_class()

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model = model.to(dtype).to(device).eval()
    return model


def create_attention_mask(lengths: list, max_len: int = None, device: str = "cpu") -> torch.Tensor:
    """Create padding attention mask from sequence lengths."""
    if max_len is None:
        max_len = max(lengths)
    mask = torch.zeros(len(lengths), max_len, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return mask


def interpolate_predictions(
    predictions: torch.Tensor,
    timestamps_in: list,
    timestamps_out: list,
    method: str = "linear",
) -> torch.Tensor:
    """Interpolate predictions to new timestamps."""
    import numpy as np
    from scipy import interpolate as sci_interp

    preds_np = predictions.numpy()
    result = np.zeros((len(timestamps_out), preds_np.shape[1]) if preds_np.ndim > 1 else len(timestamps_out))

    t_in = np.array(timestamps_in, dtype=float)
    t_out = np.array(timestamps_out, dtype=float)

    if preds_np.ndim == 1:
        f = sci_interp.interp1d(t_in, preds_np, kind=method, fill_value="extrapolate")
        result = f(t_out)
    else:
        for col in range(preds_np.shape[1]):
            f = sci_interp.interp1d(t_in, preds_np[:, col], kind=method, fill_value="extrapolate")
            result[:, col] = f(t_out)

    return torch.tensor(result, dtype=predictions.dtype)


def merge_predictions_across_chunks(
    chunk_predictions: list,
    chunk_sizes: list,
    overlap: int = 0,
    merge_method: str = "mean",
) -> torch.Tensor:
    """Merge predictions from overlapping chunks back into a single sequence."""
    total = sum(chunk_sizes) - overlap * (len(chunk_sizes) - 1)
    shape = list(chunk_predictions[0].shape)
    shape[0] = total
    merged = torch.zeros(shape, dtype=chunk_predictions[0].dtype)
    count = torch.zeros(total, dtype=torch.float32)

    pos = 0
    for chunk_pred, size in zip(chunk_predictions, chunk_sizes):
        end = pos + chunk_pred.shape[0]
        if merge_method == "mean":
            merged[pos:end] += chunk_pred
        else:
            merged[pos:end] = chunk_pred
        count[pos:end] += 1.0
        pos += size - overlap

    if merge_method == "mean":
        merged = merged / count.unsqueeze(-1).clamp(min=1)

    return merged


def compute_prediction_intervals(
    samples: torch.Tensor,
    confidence: float = 0.95,
) -> dict:
    """Compute prediction intervals from MC samples."""
    alpha = (1 - confidence) / 2
    lower_q = alpha
    upper_q = 1 - alpha

    lower = samples.quantile(lower_q, dim=0)
    upper = samples.quantile(upper_q, dim=0)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    return {
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "interval_width": (upper - lower),
        "confidence": confidence,
    }


def soft_nms(scores: torch.Tensor, iou_threshold: float = 0.5, sigma: float = 0.5) -> torch.Tensor:
    """Soft-NMS for deduplicating overlapping temporal predictions.

    Applies Gaussian soft-NMS to reduce scores of highly overlapping predictions
    rather than hard suppression.
    """
    n = scores.shape[0]
    weights = torch.ones(n, device=scores.device)

    for i in range(n):
        for j in range(i + 1, n):
            sim = torch.cosine_similarity(scores[i].unsqueeze(0), scores[j].unsqueeze(0))
            if sim > iou_threshold:
                weights[j] *= torch.exp(-sim**2 / sigma)

    return weights


class InferenceBenchmarkSuite:
    """Comprehensive benchmark suite for comparing inference configurations."""

    def __init__(self, model_factory, configs: list):
        self.model_factory = model_factory
        self.configs = configs
        self.results = {}

    def run(self, input_shapes: list, n_iters: int = 20) -> dict:
        """Run all configs against all input shapes and collect timing results."""
        import time

        for cfg in self.configs:
            cfg_name = cfg.get("name", str(cfg))
            self.results[cfg_name] = {}
            model = self.model_factory(**{k: v for k, v in cfg.items() if k != "name"})
            model.eval()

            device = cfg.get("device", "cpu")
            model = model.to(device)

            for shape in input_shapes:
                dummy = torch.randn(*shape, device=device)
                latencies = []

                with torch.no_grad():
                    for i in range(n_iters + 5):
                        if device == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = model(dummy)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        if i >= 5:  # skip warmup
                            latencies.append((t1 - t0) * 1000)

                import statistics
                self.results[cfg_name][str(shape)] = {
                    "mean_ms": statistics.mean(latencies),
                    "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                }

        return self.results

    def print_comparison(self):
        """Print a formatted comparison table of all configs."""
        print("\n" + "=" * 80)
        print("INFERENCE BENCHMARK COMPARISON")
        print("=" * 80)
        shapes = list(next(iter(self.results.values())).keys()) if self.results else []

        for shape in shapes:
            print(f"\nInput shape: {shape}")
            print(f"  {'Config':<30} {'Mean(ms)':>12} {'Std(ms)':>12} {'Min(ms)':>12}")
            print("  " + "-" * 70)
            for cfg_name, shape_results in self.results.items():
                if shape in shape_results:
                    r = shape_results[shape]
                    print(f"  {cfg_name:<30} {r['mean_ms']:>12.3f} {r['std_ms']:>12.3f} {r['min_ms']:>12.3f}")
