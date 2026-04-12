

# ============================================================
# Extended Interpretability Components - Part 2
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np


class LIMEExplainer:
    """LIME (Ribeiro et al. 2016) for time series financial model explanations.

    Fits a local linear model around each prediction to explain it.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        noise_std: float = 0.1,
        kernel_width: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.kernel_width = kernel_width
        self.device = device
        self.model.eval()

    def _kernel(self, distances: torch.Tensor) -> torch.Tensor:
        """Exponential kernel for distance weighting."""
        return torch.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

    def explain(
        self,
        x: torch.Tensor,
        target_class: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for input x.

        Returns feature importances via local linear approximation.
        """
        B, T, D = x.shape

        # Generate perturbed samples
        perturbations = torch.randn(self.n_samples, T, D, device=x.device) * self.noise_std
        x_perturbed = x + perturbations  # broadcasting over batch

        # Get model predictions on perturbed inputs
        with torch.no_grad():
            perturbed_out = []
            for i in range(self.n_samples):
                xi = x_perturbed[i:i+1].to(self.device)
                out = self.model(xi)
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() > 1:
                    out = out[:, target_class] if out.shape[-1] > target_class else out.mean(-1)
                perturbed_out.append(out.cpu())
            y_perturbed = torch.cat(perturbed_out)  # (n_samples,)

        # Compute distances from original
        dist = perturbations.reshape(self.n_samples, -1).norm(dim=-1)
        weights = self._kernel(dist)

        # Fit weighted linear regression
        X_lime = perturbations.reshape(self.n_samples, T * D)
        W = torch.diag(weights)

        # Ridge regression: beta = (X^T W X + lambda I)^{-1} X^T W y
        lam = 1e-3
        XtW = X_lime.T @ W
        A = XtW @ X_lime + lam * torch.eye(T * D, device=X_lime.device)
        b = XtW @ y_perturbed

        try:
            beta = torch.linalg.solve(A, b)
        except Exception:
            beta = torch.zeros(T * D)

        # Reshape back to (T, D) importance
        feature_importance = beta.reshape(T, D).abs()

        # Time importance (averaged over features)
        time_importance = feature_importance.mean(dim=-1)

        # Feature importance (averaged over time)
        dim_importance = feature_importance.mean(dim=0)

        return {
            "feature_importance": feature_importance,
            "time_importance": time_importance,
            "dim_importance": dim_importance,
            "local_model_weights": beta,
            "local_predictions": y_perturbed,
        }


class ShapleyValueEstimator:
    """Shapley values estimation for financial model interpretability.

    Implements kernel SHAP approximation (Lundberg & Lee 2017).
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        n_samples: int = 50,
        device: str = "cpu",
    ):
        self.model = model
        self.background = background.to(device)
        self.n_samples = n_samples
        self.device = device
        self.model.eval()

    def _model_output(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() > 1:
                out = out.mean(-1)
        return out

    def shap_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate SHAP values for input x.

        x: (B, T, D)
        Returns: (B, T, D) SHAP values
        """
        B, T, D = x.shape
        n_features = T * D

        x_flat = x.reshape(B, n_features)
        bg_flat = self.background.reshape(-1, n_features)
        n_bg = bg_flat.shape[0]

        shap_vals = torch.zeros(B, n_features, device=x.device)

        for b in range(B):
            xi = x_flat[b]  # (n_features,)
            shap_b = torch.zeros(n_features, device=x.device)

            for _ in range(self.n_samples):
                # Random permutation of features
                perm = torch.randperm(n_features)
                # Random background sample
                bg_idx = torch.randint(0, n_bg, (1,)).item()
                bg = bg_flat[bg_idx]

                # For each feature, compute marginal contribution
                x_with = xi.clone()
                x_without = xi.clone()

                split_idx = torch.randint(1, n_features, (1,)).item()
                subset = perm[:split_idx]

                # Without feature: background values for features in subset
                x_without[subset] = bg[subset]

                # With feature: original values for features in subset
                # (already set in x_with)

                # Difference in predictions
                xi_batch = x_with.unsqueeze(0).reshape(1, T, D).to(self.device)
                xj_batch = x_without.unsqueeze(0).reshape(1, T, D).to(self.device)

                out_with = self._model_output(xi_batch).item()
                out_without = self._model_output(xj_batch).item()
                delta = out_with - out_without

                shap_b[subset] += delta / max(split_idx, 1)

            shap_vals[b] = shap_b / self.n_samples

        return shap_vals.reshape(B, T, D)


class CausalAttentionAnalyzer:
    """Analyzes causal structure in attention patterns to identify key time points."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_maps: Dict[str, List[torch.Tensor]] = {}
        self._hooks = []

    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        if name not in self._attention_maps:
                            self._attention_maps[name] = []
                        self._attention_maps[name].append(attn_weights.detach().cpu())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def analyze_attention(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run forward pass and analyze captured attention patterns."""
        self._attention_maps.clear()
        self.register_hooks()

        with torch.no_grad():
            _ = self.model(x)

        self.remove_hooks()

        analysis = {}
        for layer_name, attn_list in self._attention_maps.items():
            if not attn_list:
                continue
            attn = attn_list[-1]  # Last batch's attention: (B, H, T, T) or (B*H, T, T)

            if attn.dim() == 3:
                attn = attn.unsqueeze(1)  # (B, 1, T, T)

            B, H, T, _ = attn.shape

            # Attention entropy (lower = more focused)
            entropy = -(attn * (attn + 1e-10).log()).sum(-1).mean()

            # Average attention to last token (recency bias)
            last_token_attn = attn[:, :, -1, :].mean()

            # Attention to earlier tokens (long-range dependencies)
            if T > 1:
                early_attn = attn[:, :, :, :T//4].mean()
            else:
                early_attn = torch.tensor(0.0)

            analysis[layer_name] = {
                "entropy": entropy.item(),
                "last_token_attention": last_token_attn.item(),
                "early_token_attention": early_attn.item(),
                "n_heads": H,
                "seq_len": T,
            }

        return analysis

    def compute_attention_rollout(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute attention rollout (Abnar & Zuidema 2020) for full attribution."""
        self._attention_maps.clear()
        self.register_hooks()
        with torch.no_grad():
            _ = self.model(x)
        self.remove_hooks()

        all_attns = list(self._attention_maps.values())
        if not all_attns:
            return None

        processed = []
        for attn_list in all_attns:
            if not attn_list:
                continue
            attn = attn_list[-1]
            if attn.dim() == 3:
                attn = attn.unsqueeze(1)
            # Average over heads
            attn_avg = attn.mean(1)  # (B, T, T)
            # Add residual: 0.5 * attn + 0.5 * I
            B, T, _ = attn_avg.shape
            I = torch.eye(T, device=attn_avg.device).unsqueeze(0).expand(B, -1, -1)
            rollout = 0.5 * attn_avg + 0.5 * I
            rollout = rollout / rollout.sum(-1, keepdim=True)
            processed.append(rollout)

        # Chain multiplication of all attention matrices
        if not processed:
            return None
        result = processed[0]
        for p in processed[1:]:
            if result.shape == p.shape:
                result = torch.bmm(result, p)

        return result


class ModelCardGenerator:
    """Generates model cards documenting Lumina model properties."""

    def __init__(self, model: nn.Module, model_name: str, version: str = "1.0"):
        self.model = model
        self.model_name = model_name
        self.version = version

    def generate_card(
        self,
        task_description: str,
        training_data: str,
        evaluation_results: Optional[Dict[str, float]] = None,
        limitations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a model card dictionary."""
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        card = {
            "model_name": self.model_name,
            "version": self.version,
            "task_description": task_description,
            "training_data": training_data,
            "model_architecture": {
                "class": type(self.model).__name__,
                "total_parameters": n_params,
                "trainable_parameters": n_trainable,
                "frozen_parameters": n_params - n_trainable,
                "memory_fp32_mb": n_params * 4 / 1e6,
                "memory_fp16_mb": n_params * 2 / 1e6,
            },
            "evaluation_results": evaluation_results or {},
            "limitations": limitations or [
                "Financial forecasts are probabilistic and not guaranteed",
                "Model performance may degrade in novel market regimes",
                "Backtested results may not reflect future performance",
            ],
            "intended_use": "Financial research and quantitative investment",
            "out_of_scope_use": "Direct investment advice, retail investor decision-making",
        }
        return card

    def format_markdown(self, card: Dict[str, Any]) -> str:
        """Format model card as markdown."""
        lines = [
            f"# {card['model_name']} v{card['version']}",
            "",
            "## Model Description",
            card.get("task_description", ""),
            "",
            "## Model Architecture",
        ]
        arch = card.get("model_architecture", {})
        for k, v in arch.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        if card.get("evaluation_results"):
            lines.append("## Evaluation Results")
            for metric, val in card["evaluation_results"].items():
                lines.append(f"- **{metric}**: {val:.4f}" if isinstance(val, float) else f"- **{metric}**: {val}")
            lines.append("")

        lines.append("## Limitations")
        for lim in card.get("limitations", []):
            lines.append(f"- {lim}")

        return "
".join(lines)


class FinancialInterpretabilityReport:
    """Generates comprehensive interpretability reports for financial models."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def _run_gradient_attribution(self, x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """Simple gradient-based attribution."""
        x_req = x.clone().requires_grad_(True)
        out = self.model(x_req)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() > 1:
            out = out[:, target_idx]
        out.sum().backward()
        return x_req.grad.abs().detach()

    def feature_importance_report(
        self,
        x: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        method: str = "gradient",
    ) -> Dict[str, Any]:
        """Generate feature importance report."""
        B, T, D = x.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(D)]

        if method == "gradient":
            try:
                importance = self._run_gradient_attribution(x)  # (B, T, D)
            except Exception:
                importance = torch.ones(B, T, D)
        else:
            importance = torch.ones(B, T, D)

        # Aggregate over batch and time
        dim_importance = importance.mean(dim=[0, 1])  # (D,)
        time_importance = importance.mean(dim=[0, 2])  # (T,)

        top_k = min(10, D)
        top_indices = dim_importance.topk(top_k).indices.tolist()
        top_names = [feature_names[i] for i in top_indices]
        top_values = dim_importance[top_indices].tolist()

        return {
            "method": method,
            "feature_names": feature_names,
            "dim_importance": dim_importance.tolist(),
            "time_importance": time_importance.tolist(),
            "top_features": list(zip(top_names, top_values)),
            "most_important_timestep": int(time_importance.argmax()),
            "least_important_timestep": int(time_importance.argmin()),
        }

    def generate_full_report(
        self,
        x: torch.Tensor,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate full interpretability report."""
        report = {
            "input_shape": list(x.shape),
            "feature_importance": self.feature_importance_report(x, feature_names),
        }

        # Model complexity
        n_params = sum(p.numel() for p in self.model.parameters())
        report["model_complexity"] = {
            "n_params": n_params,
            "n_layers": len(list(self.model.modules())),
        }

        return report


class WhatIfAnalyzer:
    """What-if analysis: perturbs specific features to understand model sensitivity."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            if isinstance(out, tuple):
                out = out[0]
        return out

    def feature_ablation(
        self,
        x: torch.Tensor,
        baseline_val: float = 0.0,
    ) -> Dict[str, Any]:
        """Ablate each feature dimension and measure prediction change."""
        B, T, D = x.shape
        base_pred = self._predict(x)

        importances = []
        for d in range(D):
            x_ablated = x.clone()
            x_ablated[:, :, d] = baseline_val
            ablated_pred = self._predict(x_ablated)
            delta = (base_pred - ablated_pred).abs().mean().item()
            importances.append(delta)

        return {
            "feature_ablation_scores": importances,
            "most_important_feature_idx": int(np.argmax(importances)),
            "least_important_feature_idx": int(np.argmin(importances)),
        }

    def timestep_ablation(
        self,
        x: torch.Tensor,
        baseline_val: float = 0.0,
    ) -> Dict[str, Any]:
        """Ablate each timestep and measure prediction change."""
        B, T, D = x.shape
        base_pred = self._predict(x)

        importances = []
        for t in range(T):
            x_ablated = x.clone()
            x_ablated[:, t, :] = baseline_val
            ablated_pred = self._predict(x_ablated)
            delta = (base_pred - ablated_pred).abs().mean().item()
            importances.append(delta)

        return {
            "timestep_ablation_scores": importances,
            "most_important_timestep": int(np.argmax(importances)),
            "recent_bias": float(np.mean(importances[-T//4:])) / (float(np.mean(importances)) + 1e-10),
        }

    def sensitivity_analysis(
        self,
        x: torch.Tensor,
        feature_idx: int,
        perturbation_range: Tuple[float, float] = (-3.0, 3.0),
        n_points: int = 20,
    ) -> Dict[str, Any]:
        """Analyze model sensitivity to a specific feature across a range of values."""
        lo, hi = perturbation_range
        values = np.linspace(lo, hi, n_points)
        predictions = []

        for v in values:
            x_pert = x.clone()
            x_pert[:, :, feature_idx] = float(v)
            pred = self._predict(x_pert).mean().item()
            predictions.append(pred)

        return {
            "feature_idx": feature_idx,
            "input_values": values.tolist(),
            "predictions": predictions,
            "monotonicity": float(np.corrcoef(values, predictions)[0, 1]),
            "range": float(max(predictions) - min(predictions)),
        }


class QuantileRegressionExplainer:
    """Explains prediction intervals for quantile regression models."""

    def __init__(self, model: nn.Module, quantiles: List[float] = None):
        self.model = model
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    def prediction_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """Get prediction interval for given confidence level."""
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, dict):
                return out

        # If model returns raw predictions, use them as point estimates
        lo_q = (1 - confidence) / 2
        hi_q = 1 - lo_q

        return {
            "point_estimate": out,
            "lower_bound": out * (1 - confidence / 2),
            "upper_bound": out * (1 + confidence / 2),
            "confidence": confidence,
        }

    def coverage_check(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        confidence: float = 0.9,
    ) -> Dict[str, float]:
        """Check empirical coverage of prediction intervals."""
        interval = self.prediction_interval(x, confidence)
        lo = interval.get("lower_bound")
        hi = interval.get("upper_bound")

        if lo is None or hi is None:
            return {"empirical_coverage": float("nan")}

        covered = ((y_true >= lo) & (y_true <= hi)).float()
        empirical_coverage = covered.mean().item()
        interval_width = (hi - lo).abs().mean().item()

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": confidence,
            "coverage_gap": empirical_coverage - confidence,
            "avg_interval_width": interval_width,
        }
