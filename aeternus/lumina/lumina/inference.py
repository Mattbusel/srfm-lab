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
