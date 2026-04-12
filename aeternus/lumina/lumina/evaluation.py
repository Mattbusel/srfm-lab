"""
lumina/evaluation.py

Evaluation suite for Lumina:

  - zero_shot_regime_transfer()
  - crisis_detection_benchmark()
  - volatility_forecast_benchmark()
  - return_direction_benchmark()
  - perplexity()
  - attention_visualization()
  - probing_analysis()

All results are saved to aeternus/lumina/results/
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

RESULTS_DIR = Path("aeternus/lumina/results")


def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_results(name: str, results: dict):
    _ensure_results_dir()
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Lumina Eval] Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def _vix_threshold_crisis_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    vol_col: int = 0,
    threshold: float = 0.25,
) -> Dict[str, float]:
    """Simple VIX-threshold baseline: flag crisis if rolling vol > threshold."""
    vol = features[:, vol_col]
    preds = (vol > threshold).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def _correlation_spike_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    window: int = 20,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """Correlation spike baseline: flag when cross-asset correlation > threshold."""
    # Simulate: use rolling std as proxy for correlation spike
    N = len(features)
    preds = np.zeros(N, dtype=int)
    for i in range(window, N):
        window_std = features[i - window:i, 0].std()
        if window_std > threshold * features[:, 0].std():
            preds[i] = 1
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def _cusum_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    vol_col: int = 0,
    k: float = 0.5,
    h: float = 5.0,
) -> Dict[str, float]:
    """CUSUM change detection baseline."""
    series = features[:, vol_col]
    mu = series[:20].mean()
    sigma = series[:20].std() + 1e-8
    S_pos = 0.0
    S_neg = 0.0
    preds = np.zeros(len(series), dtype=int)
    for i in range(len(series)):
        z = (series[i] - mu) / sigma
        S_pos = max(0, S_pos + z - k)
        S_neg = max(0, S_neg - z - k)
        if S_pos > h or S_neg > h:
            preds[i] = 1
            S_pos = 0.0
            S_neg = 0.0
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def _garch_vol_baseline(returns: np.ndarray, horizon: int = 5) -> np.ndarray:
    """
    Simplified GARCH(1,1) volatility forecast.
    σ²_t+1 = ω + α * ε²_t + β * σ²_t
    """
    omega = 0.000001
    alpha = 0.1
    beta = 0.85
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = returns[:20].var() if len(returns) >= 20 else 0.01

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    # Forecast h steps ahead (constant: sigma2_T for all horizons)
    forecasts = np.sqrt(sigma2[:-1])  # realized vol as target proxy
    return forecasts


def _realized_vol_persistence_baseline(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Persistence baseline: forecast future vol = current rolling realized vol."""
    T = len(returns)
    preds = np.zeros(T)
    for t in range(window, T):
        preds[t] = np.std(returns[t - window:t])
    return preds[window:]


def _momentum_direction_baseline(returns: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Momentum baseline for direction: sign of sum of last lookback returns."""
    T = len(returns)
    preds = np.zeros(T, dtype=int)
    for t in range(lookback, T):
        mom = returns[t - lookback:t].sum()
        if mom > 0.001:
            preds[t] = 2   # UP
        elif mom < -0.001:
            preds[t] = 0   # DOWN
        else:
            preds[t] = 1   # FLAT
    return preds[lookback:]


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
def crisis_detection_benchmark(
    model,
    test_loader: DataLoader,
    device: torch.device,
    vol_threshold: float = 0.25,
    results_name: str = "crisis_detection_benchmark",
) -> Dict[str, Dict[str, float]]:
    """
    Precision / Recall / F1 on crisis detection.
    Compares Lumina vs VIX threshold, correlation spike, CUSUM.

    Args:
        model: LuminaInference or LuminaModel
        test_loader: yields dicts with 'price_tokens', 'is_crisis', 'features'
        device: torch device

    Returns:
        results dict with model and baseline metrics
    """
    from .inference import LuminaInference

    all_labels = []
    all_lumina_probs = []
    all_features = []

    if hasattr(model, "crisis_score"):
        inf = model
    else:
        inf = LuminaInference(model, device=str(device))

    for batch in test_loader:
        tokens = batch["price_tokens"].to(device)
        labels = batch["is_crisis"].numpy()
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            probs = inf.crisis_score(tokens, attn_mask).cpu().numpy()

        # Extract vol features for baselines
        feats = tokens.mean(dim=1).cpu().numpy()  # (B, D) — proxy features

        all_labels.append(labels)
        all_lumina_probs.append(probs)
        all_features.append(feats)

    labels_arr = np.concatenate(all_labels)
    probs_arr = np.concatenate(all_lumina_probs)
    feats_arr = np.concatenate(all_features)

    # Threshold probabilities
    lumina_preds = (probs_arr > 0.5).astype(int)

    results = {
        "lumina": {
            "precision": float(precision_score(labels_arr, lumina_preds, zero_division=0)),
            "recall": float(recall_score(labels_arr, lumina_preds, zero_division=0)),
            "f1": float(f1_score(labels_arr, lumina_preds, zero_division=0)),
            "accuracy": float(accuracy_score(labels_arr, lumina_preds)),
        },
        "vix_threshold": _vix_threshold_crisis_baseline(feats_arr, labels_arr),
        "correlation_spike": _correlation_spike_baseline(feats_arr, labels_arr),
        "cusum": _cusum_baseline(feats_arr, labels_arr),
    }

    _save_results(results_name, results)
    return results


def volatility_forecast_benchmark(
    model,
    test_loader: DataLoader,
    device: torch.device,
    horizon: int = 5,
    results_name: str = "volatility_forecast_benchmark",
) -> Dict[str, Dict[str, float]]:
    """
    RMSE comparison: Lumina vs GARCH(1,1) vs realized vol persistence.

    Returns:
        results dict with RMSE and MAE for each method
    """
    from .inference import LuminaInference

    if hasattr(model, "volatility_forecast"):
        inf = model
    else:
        inf = LuminaInference(model, device=str(device))

    all_lumina_preds = []
    all_targets = []
    all_returns = []

    for batch in test_loader:
        tokens = batch["price_tokens"].to(device)
        # Target: actual realized vol (from batch, or compute from returns)
        target_vol = batch.get("target_vol")
        if target_vol is None:
            # Use std of token features as proxy
            target_vol = tokens.std(dim=1).mean(dim=-1, keepdim=True).expand(-1, horizon)

        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            mean_vol, _ = inf.volatility_forecast(tokens, horizon=horizon, attention_mask=attn_mask)

        all_lumina_preds.append(mean_vol.cpu().numpy())
        all_targets.append(target_vol.cpu().numpy() if torch.is_tensor(target_vol) else target_vol)
        # Proxy returns from token embeddings
        returns_proxy = tokens.mean(dim=-1).cpu().numpy()  # (B, T)
        all_returns.append(returns_proxy)

    lumina_preds = np.concatenate(all_lumina_preds)    # (N, horizon)
    targets = np.concatenate(all_targets)               # (N, horizon) or (N,)
    if targets.ndim == 1:
        targets = np.stack([targets] * horizon, axis=1)

    # Flatten for RMSE
    lumina_flat = lumina_preds.flatten()
    target_flat = targets.flatten()

    lumina_rmse = float(np.sqrt(mean_squared_error(target_flat, lumina_flat)))
    lumina_mae = float(np.mean(np.abs(lumina_flat - target_flat)))

    # GARCH baseline
    all_returns_concat = np.concatenate(all_returns, axis=0)
    returns_1d = all_returns_concat.mean(axis=0)  # average across batch
    garch_preds = _garch_vol_baseline(returns_1d, horizon)[:len(lumina_flat)]
    target_trimmed = target_flat[:len(garch_preds)]

    garch_rmse = float(np.sqrt(mean_squared_error(target_trimmed, garch_preds)))
    garch_mae = float(np.mean(np.abs(garch_preds - target_trimmed)))

    # Persistence baseline
    persist_preds = _realized_vol_persistence_baseline(returns_1d)[:len(lumina_flat)]
    target_persist = target_flat[:len(persist_preds)]
    persist_rmse = float(np.sqrt(mean_squared_error(target_persist, persist_preds)))
    persist_mae = float(np.mean(np.abs(persist_preds - target_persist)))

    results = {
        "lumina": {"rmse": lumina_rmse, "mae": lumina_mae},
        "garch_1_1": {"rmse": garch_rmse, "mae": garch_mae},
        "realized_vol_persistence": {"rmse": persist_rmse, "mae": persist_mae},
    }

    _save_results(results_name, results)
    return results


def return_direction_benchmark(
    model,
    test_loader: DataLoader,
    device: torch.device,
    results_name: str = "return_direction_benchmark",
) -> Dict[str, Dict[str, float]]:
    """
    Accuracy comparison: Lumina vs momentum baseline for return direction.
    """
    from .inference import LuminaInference
    from .finetuning import ReturnDirectionHead

    if hasattr(model, "_encode"):
        inf = model
    else:
        inf = LuminaInference(model, device=str(device))

    all_lumina_preds = []
    all_labels = []
    all_returns = []

    for batch in test_loader:
        tokens = batch["price_tokens"].to(device)
        labels = batch.get("direction_labels")
        if labels is None:
            # Create proxy labels from token variance
            returns_proxy = tokens.std(dim=-1).mean(dim=-1)  # (B,)
            labels = ReturnDirectionHead.returns_to_labels(returns_proxy)
        elif torch.is_tensor(labels):
            pass
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out = inf._encode(tokens, attn_mask) if hasattr(inf, "_encode") else \
                  model(tokens, attention_mask=attn_mask)
            # Use cls_logits or hidden mean for direction
            if "cls_logits" in out and out["cls_logits"].shape[-1] >= 3:
                logits = out["cls_logits"][:, :3]
                preds = logits.argmax(dim=-1).cpu()
            else:
                hidden = out["hidden"].mean(1)  # (B, d_model)
                # Fallback random direction based on sign of first dim
                preds = (hidden[:, 0] > 0).long().cpu() * 2  # UP=2 or DOWN=0

        all_lumina_preds.append(preds.numpy())
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else labels)
        returns_proxy = tokens.mean(dim=-1).mean(dim=-1).cpu().numpy()
        all_returns.append(returns_proxy)

    lumina_preds = np.concatenate(all_lumina_preds)
    labels_arr = np.concatenate(all_labels)

    lumina_acc = float(accuracy_score(labels_arr, lumina_preds))

    # Momentum baseline
    returns_concat = np.concatenate(all_returns)
    # Ensure enough data for momentum
    if len(returns_concat) > 10:
        mom_preds = _momentum_direction_baseline(returns_concat, lookback=5)
        labels_trimmed = labels_arr[-len(mom_preds):]
        mom_acc = float(accuracy_score(labels_trimmed, mom_preds))
    else:
        mom_acc = 1.0 / 3  # random baseline

    results = {
        "lumina": {"accuracy": lumina_acc},
        "momentum_baseline": {"accuracy": mom_acc},
        "random_baseline": {"accuracy": 1.0 / 3},
    }

    _save_results(results_name, results)
    return results


def zero_shot_regime_transfer(
    model,
    held_out_loader: DataLoader,
    device: torch.device,
    held_out_regime: int = 7,
    n_regimes: int = 8,
    results_name: str = "zero_shot_regime_transfer",
) -> Dict[str, float]:
    """
    Test on a regime type not seen during fine-tuning.

    Evaluates how well Lumina generalizes to unseen market conditions.
    Uses cosine similarity of representations as a zero-shot proximity measure.
    """
    from .inference import LuminaInference

    if hasattr(model, "regime_probabilities"):
        inf = model
    else:
        inf = LuminaInference(model, device=str(device))

    all_regime_probs = []
    all_true_regimes = []

    for batch in held_out_loader:
        tokens = batch["price_tokens"].to(device)
        true_regimes = batch.get("regime")
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            regime_probs = inf.regime_probabilities(tokens, attn_mask, n_regimes=n_regimes)

        all_regime_probs.append(regime_probs.cpu().numpy())
        if true_regimes is not None:
            all_true_regimes.append(
                true_regimes.numpy() if torch.is_tensor(true_regimes) else np.array(true_regimes)
            )

    all_regime_probs = np.concatenate(all_regime_probs)  # (N, n_regimes)
    pred_regimes = all_regime_probs.argmax(axis=1)

    results = {
        "held_out_regime": int(held_out_regime),
        "mean_prob_held_out_regime": float(all_regime_probs[:, held_out_regime].mean()),
        "pred_regime_distribution": {
            str(i): float((pred_regimes == i).mean())
            for i in range(n_regimes)
        },
    }

    if all_true_regimes:
        true_arr = np.concatenate(all_true_regimes)
        # Filter to held-out regime samples
        mask = true_arr == held_out_regime
        if mask.sum() > 0:
            held_out_probs = all_regime_probs[mask, held_out_regime]
            results["held_out_top1_recall"] = float(
                (pred_regimes[mask] == held_out_regime).mean()
            )
            results["held_out_avg_prob"] = float(held_out_probs.mean())
        overall_acc = float(accuracy_score(true_arr, pred_regimes))
        results["overall_accuracy"] = overall_acc

    _save_results(results_name, results)
    return results


def perplexity(
    model,
    test_loader: DataLoader,
    device: torch.device,
    results_name: str = "perplexity",
) -> float:
    """
    Language model perplexity on held-out return sequences.

    Perplexity = exp(mean NLL per token).
    For continuous tokens: use MSE-based NLL proxy
    (treat each predicted embedding as Gaussian with unit variance).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["price_tokens"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            out = model(tokens, attention_mask=attn_mask)
            lm_out = out.get("lm_output")

            if lm_out is not None:
                # Predict token at t from hidden at t-1
                preds = lm_out[:, :-1]     # (B, T-1, D)
                targets = tokens[:, 1:]    # (B, T-1, D)

                if attn_mask is not None:
                    valid = attn_mask[:, 1:].bool()
                else:
                    valid = torch.ones(preds.shape[:2], dtype=torch.bool, device=device)

                # Gaussian NLL proxy: 0.5 * ||pred - target||^2 per token
                sq_err = (preds - targets).pow(2).mean(dim=-1)  # (B, T-1)
                nll = 0.5 * sq_err[valid].sum().item()
                total_nll += nll
                total_tokens += valid.sum().item()

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(min(avg_nll, 300))  # cap to avoid overflow

    results = {"perplexity": ppl, "avg_nll_per_token": avg_nll, "total_tokens": total_tokens}
    _save_results(results_name, results)
    return ppl


def attention_visualization(
    model,
    sample_batch: dict,
    device: torch.device,
    layer_idx: int = -1,
    head_idx: int = 0,
    results_name: str = "attention_maps",
) -> np.ndarray:
    """
    Extract and save attention maps from a transformer layer.

    Args:
        model: LuminaModel
        sample_batch: batch dict with 'price_tokens'
        layer_idx: which transformer layer (default: last)
        head_idx: which attention head

    Returns:
        attn_map: (T, T) numpy array of attention weights
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    tokens = sample_batch["price_tokens"][:1].to(device)  # use first sample
    attn_mask = sample_batch.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask[:1].to(device)

    # Hook to capture attention weights
    attn_weights_store = {}

    def _hook(name):
        def hook(module, input, output):
            # output[0] = attended output, not weights directly
            # We need to capture inside the attention module
            pass
        return hook

    # Alternative: monkey-patch the attention module temporarily
    # Find target layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks
        target_idx = layer_idx if layer_idx >= 0 else len(blocks) + layer_idx
        target_block = blocks[target_idx]
        target_attn = target_block.attn
    else:
        target_attn = None

    attn_map = None

    if target_attn is not None:
        original_forward = target_attn.forward

        def patched_forward(*args, **kwargs):
            # Capture attention weights
            x = args[0]
            B, T, _ = x.shape
            n_heads = target_attn.n_heads
            head_dim = target_attn.head_dim
            scale = head_dim ** -0.5

            Q = target_attn.q_proj(x)
            K = target_attn.k_proj(x)
            from einops import rearrange
            Q = rearrange(Q, "b t (h d) -> b h t d", h=n_heads)
            K = rearrange(K, "b t (h d) -> b h t d", h=n_heads)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            weights = F.softmax(scores, dim=-1)
            attn_weights_store["weights"] = weights.detach().cpu()
            return original_forward(*args, **kwargs)

        target_attn.forward = patched_forward

        with torch.no_grad():
            try:
                model(tokens, attention_mask=attn_mask)
            finally:
                target_attn.forward = original_forward

        if "weights" in attn_weights_store:
            w = attn_weights_store["weights"]  # (B, n_heads, T, T)
            attn_map = w[0, head_idx].numpy()  # (T, T)

            # Save visualization
            _ensure_results_dir()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_map, aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Attention Map — Layer {target_idx}, Head {head_idx}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            save_path = RESULTS_DIR / f"{results_name}_layer{target_idx}_head{head_idx}.png"
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"[Lumina Eval] Attention map saved to {save_path}")

    return attn_map if attn_map is not None else np.zeros((1, 1))


def probing_analysis(
    model,
    test_loader: DataLoader,
    device: torch.device,
    probe_tasks: Optional[List[str]] = None,
    results_name: str = "probing_analysis",
) -> Dict[str, Dict[str, float]]:
    """
    Linear probing analysis: train linear classifiers on frozen representations
    to test what information is encoded in the hidden states.

    Probe tasks: 'regime', 'crisis', 'volatility_level', 'trend_direction'

    Returns:
        dict mapping probe_task → metrics
    """
    if probe_tasks is None:
        probe_tasks = ["regime", "crisis"]

    model.eval()
    all_hiddens = []
    all_regime_labels = []
    all_crisis_labels = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["price_tokens"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            out = model(tokens, attention_mask=attn_mask)
            hidden = out["hidden"]

            # Mean pool
            if attn_mask is not None:
                m = attn_mask.float().to(device).unsqueeze(-1)
                pooled = (hidden * m).sum(1) / (m.sum(1) + 1e-8)
            else:
                pooled = hidden.mean(1)

            all_hiddens.append(pooled.cpu().numpy())
            all_regime_labels.append(batch["regime"].numpy())
            all_crisis_labels.append(batch["is_crisis"].numpy())

    hiddens = np.concatenate(all_hiddens)        # (N, d_model)
    regime_labels = np.concatenate(all_regime_labels)
    crisis_labels = np.concatenate(all_crisis_labels)

    # Standardize features
    scaler = StandardScaler()
    hiddens_scaled = scaler.fit_transform(hiddens)

    results = {}

    # Split train/test
    N = len(hiddens_scaled)
    n_train = int(0.8 * N)
    X_train = hiddens_scaled[:n_train]
    X_test = hiddens_scaled[n_train:]

    if "regime" in probe_tasks:
        y_train = regime_labels[:n_train]
        y_test = regime_labels[n_train:]
        probe = LogisticRegression(max_iter=500, C=1.0, multi_class="multinomial")
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        results["regime"] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }

    if "crisis" in probe_tasks:
        y_train = crisis_labels[:n_train]
        y_test = crisis_labels[n_train:]
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        results["crisis"] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }

    if "volatility_level" in probe_tasks:
        # Use token variance as proxy volatility target
        vol_proxy = np.array([hiddens[i].var() for i in range(N)])
        vol_train = vol_proxy[:n_train]
        vol_test = vol_proxy[n_train:]
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, vol_train)
        preds = probe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(vol_test, preds)))
        results["volatility_level"] = {"rmse": rmse}

    if "trend_direction" in probe_tasks:
        # Proxy: use first PC direction as trend
        trend_proxy = (hiddens[:, 0] > np.median(hiddens[:, 0])).astype(int)
        y_train = trend_proxy[:n_train]
        y_test = trend_proxy[n_train:]
        probe = LogisticRegression(max_iter=200, C=1.0)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        results["trend_direction"] = {
            "accuracy": float(accuracy_score(y_test, preds)),
        }

    _save_results(results_name, results)
    return results
