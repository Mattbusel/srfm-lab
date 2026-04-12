"""
lumina/evaluation.py

Evaluation suite for Lumina Financial Foundation Model:

  - InformationCoefficient (IC)   : Spearman rank correlation of forecasts
  - DirectionalAccuracy           : hit rate on direction predictions
  - SharpeAttribution             : Sharpe ratio decomposition
  - CalibrationMetrics            : reliability / ECE / MCE
  - BenchmarkComparison           : compare vs baselines
  - FinancialMetrics              : Sharpe, Sortino, Calmar, max drawdown
  - ModelEvaluation               : unified evaluation framework
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Information Coefficient (IC)
# ---------------------------------------------------------------------------

def spearman_rank_ic(
    forecasts: np.ndarray,   # (T, N) T time steps, N assets
    returns:   np.ndarray,   # (T, N) realized returns
) -> np.ndarray:
    """
    Compute Spearman rank IC cross-sectionally at each time step.
    Returns array of shape (T,) with IC at each step.
    """
    from scipy.stats import spearmanr
    T = forecasts.shape[0]
    ics = np.zeros(T)
    for t in range(T):
        fc = forecasts[t]
        rt = returns[t]
        valid = ~np.isnan(fc) & ~np.isnan(rt)
        if valid.sum() < 3:
            ics[t] = np.nan
            continue
        ic, _ = spearmanr(fc[valid], rt[valid])
        ics[t] = ic
    return ics


def pearson_ic(
    forecasts: np.ndarray,
    returns:   np.ndarray,
) -> np.ndarray:
    """Pearson correlation-based IC."""
    T = forecasts.shape[0]
    ics = np.zeros(T)
    for t in range(T):
        fc = forecasts[t]
        rt = returns[t]
        valid = ~np.isnan(fc) & ~np.isnan(rt)
        if valid.sum() < 3:
            ics[t] = np.nan
            continue
        fc_v, rt_v = fc[valid], rt[valid]
        if fc_v.std() < 1e-8 or rt_v.std() < 1e-8:
            ics[t] = 0.0
            continue
        ics[t] = np.corrcoef(fc_v, rt_v)[0, 1]
    return ics


def icir(ics: np.ndarray) -> float:
    """Information Coefficient Information Ratio: mean(IC) / std(IC)."""
    valid = ics[~np.isnan(ics)]
    if len(valid) < 2:
        return 0.0
    return float(valid.mean() / (valid.std() + 1e-8))


def compute_ic_metrics(
    forecasts: np.ndarray,
    returns:   np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive IC metrics."""
    rank_ics  = spearman_rank_ic(forecasts, returns)
    pear_ics  = pearson_ic(forecasts, returns)
    valid_r   = rank_ics[~np.isnan(rank_ics)]
    valid_p   = pear_ics[~np.isnan(pear_ics)]

    return {
        "rank_ic_mean":    float(np.nanmean(rank_ics)),
        "rank_ic_std":     float(np.nanstd(rank_ics)),
        "rank_icir":       icir(rank_ics),
        "rank_ic_positive_frac": float((valid_r > 0).mean()) if len(valid_r) > 0 else 0.0,
        "pearson_ic_mean": float(np.nanmean(pear_ics)),
        "pearson_icir":    icir(pear_ics),
    }


# ---------------------------------------------------------------------------
# Directional Accuracy
# ---------------------------------------------------------------------------

def directional_accuracy(
    forecasts:  np.ndarray,   # (T,) or (T, N) point forecasts
    returns:    np.ndarray,   # (T,) or (T, N) realized returns
    threshold:  float = 0.0,  # minimum return magnitude to count
) -> float:
    """
    Directional accuracy: fraction of times the forecast got the direction right.
    Optionally filters out near-zero returns below threshold.
    """
    fc, rt = np.asarray(forecasts), np.asarray(returns)
    mask   = np.abs(rt) > threshold

    if mask.sum() == 0:
        return 0.5  # no signal

    fc_sign = np.sign(fc[mask])
    rt_sign = np.sign(rt[mask])
    return float((fc_sign == rt_sign).mean())


def hit_rate_by_confidence(
    forecasts:    np.ndarray,   # (T,) or (T, N)
    returns:      np.ndarray,
    n_bins:       int = 10,
) -> Dict[str, np.ndarray]:
    """
    Hit rate stratified by forecast confidence (magnitude).
    Returns dict with 'confidence_bins' and 'hit_rates'.
    """
    fc = np.asarray(forecasts).flatten()
    rt = np.asarray(returns).flatten()

    confidence = np.abs(fc)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges   = np.percentile(confidence, percentiles)

    hit_rates = np.zeros(n_bins)
    counts    = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi    = bin_edges[i], bin_edges[i + 1]
        in_bin    = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 else (confidence >= lo)
        if in_bin.sum() > 0:
            hit_rates[i] = (np.sign(fc[in_bin]) == np.sign(rt[in_bin])).mean()
            counts[i]    = in_bin.sum()

    return {
        "bin_edges":   bin_edges,
        "hit_rates":   hit_rates,
        "counts":      counts,
    }


# ---------------------------------------------------------------------------
# Financial Performance Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns:    np.ndarray,
    freq:       int   = 252,   # annualization factor
    risk_free:  float = 0.0,
) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / freq
    if excess.std() < 1e-8:
        return 0.0
    return float(excess.mean() / excess.std() * math.sqrt(freq))


def sortino_ratio(
    returns:    np.ndarray,
    freq:       int   = 252,
    target:     float = 0.0,
) -> float:
    """Sortino ratio: Sharpe using only downside deviation."""
    excess    = returns - target / freq
    downside  = excess[excess < 0]
    if len(downside) < 2 or downside.std() < 1e-8:
        return 0.0
    downside_std = downside.std()
    return float(excess.mean() / downside_std * math.sqrt(freq))


def calmar_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """Calmar ratio: annualized return / maximum drawdown."""
    ann_ret = returns.mean() * freq
    cum     = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdown    = (rolling_max - cum) / rolling_max
    max_dd      = drawdown.max()
    if max_dd < 1e-8:
        return 0.0
    return float(ann_ret / max_dd)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of a return series."""
    cum         = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdown    = (rolling_max - cum) / rolling_max
    return float(drawdown.max())


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Return the full drawdown time series."""
    cum         = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    return (rolling_max - cum) / rolling_max


def var_cvar(
    returns:    np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Value-at-Risk and Conditional Value-at-Risk (Expected Shortfall)."""
    alpha = 1 - confidence
    var   = float(np.percentile(returns, alpha * 100))
    cvar  = float(returns[returns <= var].mean())
    return var, cvar


def omega_ratio(
    returns:   np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Omega ratio: ratio of gains to losses above/below threshold."""
    above = np.maximum(returns - threshold, 0).sum()
    below = np.maximum(threshold - returns, 0).sum()
    if below < 1e-8:
        return float('inf')
    return float(above / below)


def compute_financial_metrics(
    returns:    np.ndarray,
    freq:       int = 252,
) -> Dict[str, float]:
    """Comprehensive set of financial performance metrics."""
    var_95, cvar_95 = var_cvar(returns, 0.95)
    return {
        "annualized_return": float(returns.mean() * freq),
        "annualized_vol":    float(returns.std() * math.sqrt(freq)),
        "sharpe_ratio":      sharpe_ratio(returns, freq),
        "sortino_ratio":     sortino_ratio(returns, freq),
        "calmar_ratio":      calmar_ratio(returns, freq),
        "max_drawdown":      max_drawdown(returns),
        "var_95":            var_95,
        "cvar_95":           cvar_95,
        "omega_ratio":       omega_ratio(returns),
        "hit_rate":          float((returns > 0).mean()),
        "avg_win":           float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0,
        "avg_loss":          float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0,
        "win_loss_ratio":    float(abs(returns[returns > 0].mean() / (returns[returns < 0].mean() + 1e-8)))
                             if (returns > 0).any() and (returns < 0).any() else 0.0,
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs:    np.ndarray,   # (N, C) predicted probabilities
    labels:   np.ndarray,   # (N,) true class labels
    n_bins:   int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE measures whether predicted confidence matches empirical accuracy.
    """
    confidences  = probs.max(-1)
    predictions  = probs.argmax(-1)
    accuracies   = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce  = 0.0, 0.0

    bin_stats = []
    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        in_bin   = (confidences >= lo) & (confidences < hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bin_stats.append({"n": 0, "acc": 0.0, "conf": 0.0, "weight": 0.0})
            continue
        avg_acc  = accuracies[in_bin].mean()
        avg_conf = confidences[in_bin].mean()
        cal_err  = abs(avg_acc - avg_conf)
        weight   = n_in_bin / len(labels)
        ece     += weight * cal_err
        mce      = max(mce, cal_err)
        bin_stats.append({"n": n_in_bin, "acc": avg_acc, "conf": avg_conf, "weight": weight})

    # Brier score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    brier = ((probs - one_hot) ** 2).sum(-1).mean()

    return {
        "ece":         ece,
        "mce":         mce,
        "brier_score": float(brier),
        "bin_stats":   bin_stats,
    }


def reliability_diagram_data(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Return data needed for plotting a reliability diagram."""
    cal = expected_calibration_error(probs, labels, n_bins)
    stats = cal["bin_stats"]

    confs   = np.array([s["conf"] for s in stats])
    accs    = np.array([s["acc"]  for s in stats])
    weights = np.array([s["weight"] for s in stats])

    return {"confidences": confs, "accuracies": accs, "weights": weights, "ece": cal["ece"]}


# ---------------------------------------------------------------------------
# Sharpe Attribution
# ---------------------------------------------------------------------------

class SharpeAttribution:
    """
    Decompose portfolio Sharpe ratio into factor contributions.
    Uses Brinson-Hood-Beebower style attribution.
    """

    @staticmethod
    def factor_attribution(
        portfolio_returns: np.ndarray,    # (T,)
        factor_returns:    np.ndarray,    # (T, K) K factors
        factor_names:      Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Regress portfolio returns on factor returns.
        Returns factor betas, R-squared, and Sharpe attribution per factor.
        """
        from numpy.linalg import lstsq

        T, K   = factor_returns.shape
        X      = np.hstack([np.ones((T, 1)), factor_returns])
        y      = portfolio_returns

        betas, _, _, _ = lstsq(X, y, rcond=None)
        alpha          = betas[0]
        factor_betas   = betas[1:]

        # Fitted returns
        y_hat = X @ betas
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2     = 1 - ss_res / (ss_tot + 1e-8)

        # Factor Sharpe attribution
        factor_names = factor_names or [f"factor_{i}" for i in range(K)]
        attribution  = {}
        for i, (name, beta) in enumerate(zip(factor_names, factor_betas)):
            factor_contribution = beta * factor_returns[:, i]
            attribution[name]   = {
                "beta":   float(beta),
                "sharpe": sharpe_ratio(factor_contribution),
                "t_stat": float(beta / (factor_returns[:, i].std() + 1e-8) * math.sqrt(T)),
            }

        return {
            "alpha":          float(alpha),
            "r_squared":      float(r2),
            "factor_betas":   factor_betas.tolist(),
            "attribution":    attribution,
            "residual_sharpe": sharpe_ratio(y - y_hat),
        }

    @staticmethod
    def time_varying_sharpe(
        returns:     np.ndarray,
        window:      int = 63,
        freq:        int = 252,
    ) -> np.ndarray:
        """Rolling Sharpe ratio over a window."""
        T       = len(returns)
        rolling = np.full(T, np.nan)
        for t in range(window, T):
            seg        = returns[t - window:t]
            rolling[t] = sharpe_ratio(seg, freq)
        return rolling


# ---------------------------------------------------------------------------
# Benchmark comparisons
# ---------------------------------------------------------------------------

class BenchmarkComparison:
    """Compare model predictions against standard baselines."""

    @staticmethod
    def momentum_baseline(
        prices:     np.ndarray,   # (T,)
        lookback:   int   = 20,
        horizon:    int   = 1,
    ) -> np.ndarray:
        """Simple momentum: buy if past `lookback` return is positive."""
        T       = len(prices)
        signals = np.zeros(T)
        for t in range(lookback, T - horizon):
            past_ret = (prices[t] - prices[t - lookback]) / prices[t - lookback]
            signals[t] = 1.0 if past_ret > 0 else -1.0
        return signals

    @staticmethod
    def mean_reversion_baseline(
        prices:   np.ndarray,
        window:   int = 20,
        n_std:    float = 1.0,
    ) -> np.ndarray:
        """Mean reversion: buy below lower Bollinger band, sell above upper."""
        T       = len(prices)
        signals = np.zeros(T)
        for t in range(window, T):
            seg   = prices[t - window:t]
            mean  = seg.mean()
            std   = seg.std()
            if prices[t] < mean - n_std * std:
                signals[t] = 1.0
            elif prices[t] > mean + n_std * std:
                signals[t] = -1.0
        return signals

    @staticmethod
    def buy_and_hold_baseline(T: int) -> np.ndarray:
        """Always long."""
        return np.ones(T)

    @staticmethod
    def compare(
        model_signals:    np.ndarray,
        realized_returns: np.ndarray,
        baselines:        Dict[str, np.ndarray],
        freq:             int = 252,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model against baselines.
        Signals are in {-1, 0, 1}; returns are daily returns.
        """
        results = {}

        # Model strategy
        model_rets = model_signals * realized_returns
        results["model"] = compute_financial_metrics(model_rets, freq)
        results["model"]["directional_acc"] = directional_accuracy(model_signals, realized_returns)

        for name, signals in baselines.items():
            strat_rets = signals * realized_returns
            results[name] = compute_financial_metrics(strat_rets, freq)
            results[name]["directional_acc"] = directional_accuracy(signals, realized_returns)

        return results


# ---------------------------------------------------------------------------
# Benchmark tasks (from __init__.py imports)
# ---------------------------------------------------------------------------

def crisis_detection_benchmark(
    model:    nn.Module,
    data:     Dict[str, Any],
    device:   str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model on financial crisis detection.
    Returns precision, recall, F1 for crisis vs. normal regime.
    """
    model.eval()
    results = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if "ohlcv" not in data or "crisis_labels" not in data:
        return results

    ohlcv  = torch.from_numpy(data["ohlcv"]).float().to(device)
    labels = data["crisis_labels"]

    with torch.no_grad():
        enc  = model(ohlcv)
        if isinstance(enc, dict):
            emb = enc.get("cls_emb", enc["hidden"][:, 0])
        else:
            emb = enc

    # Simple thresholding on embedding norm as crisis signal
    scores = emb.norm(dim=-1).cpu().numpy()
    threshold = np.percentile(scores, 90)
    preds  = (scores > threshold).astype(int)

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels).astype(int).flatten()
    preds  = preds.flatten()

    min_len = min(len(preds), len(labels))
    preds   = preds[:min_len]
    labels  = labels[:min_len]

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def volatility_forecast_benchmark(
    model:     nn.Module,
    data:      Dict[str, Any],
    device:    str = "cpu",
    vol_head:  Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate volatility forecasting performance."""
    model.eval()
    results = {"mae": 0.0, "rmse": 0.0, "qlike": 0.0}

    if "ohlcv" not in data or "vol_targets" not in data:
        return results

    ohlcv      = torch.from_numpy(data["ohlcv"]).float().to(device)
    vol_targets = np.asarray(data["vol_targets"])

    with torch.no_grad():
        enc = model(ohlcv)
        if isinstance(enc, dict):
            emb = enc.get("cls_emb", enc["hidden"][:, 0])
        else:
            emb = enc

        if vol_head is not None:
            preds = vol_head(emb).cpu().numpy().flatten()
        else:
            # Proxy: use embedding norm as vol signal
            preds = emb.norm(dim=-1).cpu().numpy() * 0.01

    min_len    = min(len(preds), len(vol_targets))
    preds      = preds[:min_len]
    vol_targets = vol_targets[:min_len]

    mae   = float(np.abs(preds - vol_targets).mean())
    rmse  = float(np.sqrt(((preds - vol_targets) ** 2).mean()))
    qlike = float((vol_targets / (preds + 1e-8) - np.log(vol_targets / (preds + 1e-8)) - 1).mean())

    return {"mae": mae, "rmse": rmse, "qlike": qlike}


def return_direction_benchmark(
    model:       nn.Module,
    data:        Dict[str, Any],
    device:      str = "cpu",
    class_head:  Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate direction prediction accuracy."""
    model.eval()
    results = {"accuracy": 0.0, "directional_acc": 0.0}

    if "ohlcv" not in data or "direction_labels" not in data:
        return results

    ohlcv  = torch.from_numpy(data["ohlcv"]).float().to(device)
    labels = np.asarray(data["direction_labels"])

    with torch.no_grad():
        enc = model(ohlcv)
        if isinstance(enc, dict):
            emb = enc.get("cls_emb", enc["hidden"][:, 0])
        else:
            emb = enc

        if class_head is not None:
            logits = class_head(emb)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(-1).cpu().numpy()
        else:
            # Default: use sign of first embedding dim
            preds = (emb[:, 0] > 0).long().cpu().numpy()
            probs = None

    min_len = min(len(preds), len(labels))
    preds   = preds[:min_len]
    labels  = labels[:min_len]

    acc = float((preds == labels).mean())

    # Direction acc ignoring "flat" class (label = 2)
    not_flat = labels != 2
    if not_flat.sum() > 0:
        dir_acc = float((preds[not_flat] == labels[not_flat]).mean())
    else:
        dir_acc = acc

    result = {"accuracy": acc, "directional_acc": dir_acc}

    if probs is not None:
        cal = expected_calibration_error(probs[:min_len], labels)
        result["ece"]         = cal["ece"]
        result["brier_score"] = cal["brier_score"]

    return result


def perplexity(
    model:   nn.Module,
    loader:  Any,
    device:  str = "cpu",
) -> float:
    """
    Compute perplexity for a causal language / next-patch-prediction model.
    Perplexity = exp(mean NLL per token).
    """
    model.eval()
    total_nll  = 0.0
    total_toks = 0

    with torch.no_grad():
        for batch in loader:
            ohlcv = batch["ohlcv"].to(device) if isinstance(batch, dict) else batch.to(device)

            out = model(ohlcv)
            if isinstance(out, dict) and "logits" in out:
                logits = out["logits"][:, :-1]
                targets = ohlcv[:, 1:, 3].long()  # close price as proxy
                nll     = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                          targets.reshape(-1), reduction="sum")
                total_nll  += nll.item()
                total_toks += targets.numel()
            elif isinstance(out, dict) and "hidden" in out:
                # NPP-style: use reconstruction loss
                hidden  = out["hidden"]
                loss    = (hidden ** 2).mean()
                total_nll  += loss.item()
                total_toks += 1

    if total_toks == 0:
        return float('inf')
    return float(math.exp(total_nll / total_toks))


# ---------------------------------------------------------------------------
# Unified evaluation framework
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Unified evaluator for Lumina models.
    Runs all benchmark tasks and returns comprehensive metrics.
    """

    def __init__(
        self,
        model:   nn.Module,
        device:  str = "cpu",
        heads:   Optional[Dict[str, nn.Module]] = None,
    ):
        self.model  = model.to(device)
        self.device = device
        self.heads  = heads or {}

    def evaluate_all(
        self,
        test_data: Dict[str, Any],
        freq:      int = 252,
    ) -> Dict[str, Any]:
        """Run all evaluation benchmarks."""
        results = {}

        # Crisis detection
        results["crisis"] = crisis_detection_benchmark(self.model, test_data, self.device)

        # Volatility forecasting
        results["volatility"] = volatility_forecast_benchmark(
            self.model, test_data, self.device,
            vol_head=self.heads.get("volatility"),
        )

        # Direction prediction
        results["direction"] = return_direction_benchmark(
            self.model, test_data, self.device,
            class_head=self.heads.get("direction"),
        )

        # IC metrics if forecasts available
        if "forecasts" in test_data and "returns" in test_data:
            results["ic"] = compute_ic_metrics(
                test_data["forecasts"], test_data["returns"]
            )

        # Financial metrics if strategy returns available
        if "strategy_returns" in test_data:
            results["financial"] = compute_financial_metrics(
                test_data["strategy_returns"], freq
            )

        return results

    def evaluate_ic(
        self,
        ohlcv:    np.ndarray,   # (T, window, 5)
        returns:  np.ndarray,   # (T, N) cross-sectional returns
    ) -> Dict[str, float]:
        """Evaluate information coefficient."""
        self.model.eval()
        all_forecasts = []

        with torch.no_grad():
            for i in range(len(ohlcv)):
                x   = torch.from_numpy(ohlcv[i]).float().unsqueeze(0).to(self.device)
                enc = self.model(x)
                if isinstance(enc, dict):
                    emb = enc.get("cls_emb", enc["hidden"][:, 0])
                else:
                    emb = enc
                all_forecasts.append(emb.cpu().numpy())

        forecasts = np.array(all_forecasts).squeeze(1)[:, 0]   # use first dim as signal

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        if forecasts.ndim == 1:
            forecasts = forecasts.reshape(-1, 1)

        return compute_ic_metrics(forecasts, returns)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "spearman_rank_ic",
    "pearson_ic",
    "icir",
    "compute_ic_metrics",
    "directional_accuracy",
    "hit_rate_by_confidence",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "drawdown_series",
    "var_cvar",
    "omega_ratio",
    "compute_financial_metrics",
    "expected_calibration_error",
    "reliability_diagram_data",
    "SharpeAttribution",
    "BenchmarkComparison",
    "ModelEvaluator",
    "PortfolioBacktester",
    "RiskMetrics",
    "FactorEvaluator",
    "StatisticalTests",
    "crisis_detection_benchmark",
    "volatility_forecast_benchmark",
    "return_direction_benchmark",
    "perplexity",
]


# ---------------------------------------------------------------------------
# Portfolio Backtester
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """Backtest a portfolio strategy using model-generated signals.

    Simulates portfolio performance given:
    - Predicted signal/alpha scores (cross-sectional)
    - Actual realized returns
    - Transaction costs
    - Portfolio constraints (long-only, max weight, etc.)

    Supports:
    - Equal-weight top-N long-only strategy
    - Signal-weighted long-short strategy
    - Risk-parity weighting
    - Maximum Sharpe portfolio (simplified)

    Args:
        strategy:         "top_n" | "signal_weighted" | "risk_parity"
        top_n:            for "top_n": number of stocks to hold long
        transaction_cost: per-trade cost as fraction (e.g., 0.001 = 10bps)
        rebalance_freq:   rebalance every N periods
        max_weight:       maximum weight per asset (for diversification)
        min_weight:       minimum weight per asset

    Example:
        >>> backtester = PortfolioBacktester(strategy="top_n", top_n=20)
        >>> signals = np.random.randn(252, 100)  # (T, N) signal matrix
        >>> returns = np.random.randn(252, 100) * 0.01
        >>> metrics = backtester.run(signals, returns)
        >>> print(metrics["sharpe_ratio"])
    """

    def __init__(
        self,
        strategy: str = "signal_weighted",
        top_n: int = 20,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 1,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
        annualization: float = 252.0,
    ):
        self.strategy = strategy
        self.top_n = top_n
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.annualization = annualization

    def _compute_weights(
        self,
        signal: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute portfolio weights from signal vector.

        Args:
            signal:       (N,) signal for one time period
            prev_weights: (N,) previous period weights

        Returns:
            weights: (N,) portfolio weights
        """
        N = len(signal)

        if self.strategy == "top_n":
            # Equal-weight top-N long
            top_idx = np.argsort(signal)[-self.top_n:]
            weights = np.zeros(N)
            weights[top_idx] = 1.0 / self.top_n

        elif self.strategy == "signal_weighted":
            # Long-short, signal proportional weighting
            weights = signal.copy()
            # Normalize: long leg and short leg separately
            pos_mask = weights > 0
            neg_mask = weights < 0
            if pos_mask.any():
                weights[pos_mask] /= weights[pos_mask].sum()
            if neg_mask.any():
                weights[neg_mask] /= -weights[neg_mask].sum()
            weights *= 0.5  # 50% long, 50% short

        elif self.strategy == "risk_parity":
            # Naive risk parity: equal weight (simplified, no vol estimates)
            weights = np.ones(N) / N

        else:
            weights = np.ones(N) / N

        # Apply constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize long side
        pos_sum = weights[weights > 0].sum()
        if pos_sum > 0:
            weights[weights > 0] /= pos_sum

        return weights

    def _compute_turnover(
        self,
        new_weights: np.ndarray,
        old_weights: np.ndarray,
    ) -> float:
        """Compute portfolio turnover (sum of absolute weight changes / 2)."""
        return np.abs(new_weights - old_weights).sum() / 2.0

    def run(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Run full backtest.

        Args:
            signals: (T, N) signal matrix (one row per time period)
            returns: (T, N) realized return matrix

        Returns:
            results: dict with performance metrics and time series
        """
        T, N = signals.shape
        assert returns.shape == (T, N), f"Shape mismatch: signals={signals.shape}, returns={returns.shape}"

        portfolio_returns = np.zeros(T)
        weights = np.zeros(N)
        all_weights = np.zeros((T, N))
        turnovers = np.zeros(T)

        for t in range(T):
            if t % self.rebalance_freq == 0:
                new_weights = self._compute_weights(signals[t], weights)
                turnover = self._compute_turnover(new_weights, weights)
                turnovers[t] = turnover
                weights = new_weights

            all_weights[t] = weights
            gross_return = (weights * returns[t]).sum()
            tc = self.transaction_cost * turnovers[t]
            portfolio_returns[t] = gross_return - tc

        # Compute metrics
        cumulative = np.cumprod(1 + portfolio_returns)
        total_return = cumulative[-1] - 1.0
        ann_return = (cumulative[-1]) ** (self.annualization / T) - 1.0
        ann_vol = portfolio_returns.std() * np.sqrt(self.annualization)
        sr = ann_return / (ann_vol + 1e-10)
        dd_series = 1 - cumulative / np.maximum.accumulate(cumulative)
        max_dd = dd_series.max()

        return {
            "portfolio_returns": portfolio_returns,
            "cumulative_returns": cumulative,
            "weights": all_weights,
            "turnovers": turnovers,
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sr,
            "max_drawdown": max_dd,
            "avg_turnover": turnovers[turnovers > 0].mean() if (turnovers > 0).any() else 0.0,
            "total_transaction_costs": (turnovers * self.transaction_cost).sum(),
        }


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

class RiskMetrics:
    """Comprehensive risk metrics for financial portfolios.

    Computes both standard and advanced risk measures.

    Standard metrics:
    - Volatility (annualized)
    - Value at Risk (VaR) at multiple confidence levels
    - Conditional VaR (CVaR) / Expected Shortfall
    - Maximum Drawdown and Recovery Time

    Advanced metrics:
    - Downside Deviation and Sortino Ratio
    - Ulcer Index
    - Sterling Ratio
    - Pain Index
    - Tail Ratio (95th / 5th percentile)
    - Gain-to-Pain Ratio

    Args:
        annualization: periods per year (252 for daily, 52 for weekly)
        risk_free_rate: annualized risk-free rate

    Example:
        >>> rm = RiskMetrics(annualization=252)
        >>> returns = np.random.randn(252) * 0.01
        >>> metrics = rm.compute_all(returns)
    """

    def __init__(
        self,
        annualization: float = 252.0,
        risk_free_rate: float = 0.05,
    ):
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / annualization

    def volatility(self, returns: np.ndarray) -> float:
        """Annualized volatility (standard deviation of returns)."""
        return returns.std() * np.sqrt(self.annualization)

    def downside_deviation(
        self,
        returns: np.ndarray,
        threshold: Optional[float] = None,
    ) -> float:
        """Downside deviation (volatility of negative returns only).

        Args:
            returns:   (T,) return series
            threshold: minimum acceptable return (defaults to daily risk-free)

        Returns:
            downside_dev: annualized downside deviation
        """
        if threshold is None:
            threshold = self._daily_rf
        downside = np.minimum(returns - threshold, 0.0)
        return np.sqrt((downside ** 2).mean()) * np.sqrt(self.annualization)

    def ulcer_index(self, returns: np.ndarray) -> float:
        """Ulcer Index: RMS of drawdowns.

        Measures 'pain' by RMS of all drawdowns (not just maximum).
        """
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        drawdown_pct = (cum - peak) / (peak + 1e-10) * 100
        return np.sqrt((drawdown_pct ** 2).mean())

    def pain_index(self, returns: np.ndarray) -> float:
        """Pain Index: mean of all drawdowns."""
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        drawdown_pct = np.abs((cum - peak) / (peak + 1e-10)) * 100
        return drawdown_pct.mean()

    def tail_ratio(self, returns: np.ndarray, percentile: float = 95.0) -> float:
        """Tail ratio: upper tail / lower tail return magnitude ratio."""
        upper = np.percentile(returns, percentile)
        lower = np.percentile(returns, 100 - percentile)
        return abs(upper / (lower + 1e-10))

    def gain_to_pain(self, returns: np.ndarray) -> float:
        """Gain-to-Pain ratio: sum of gains / sum of |losses|."""
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        return gains / (losses + 1e-10)

    def var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """Value at Risk.

        Args:
            returns:          (T,) return series
            confidence_level: e.g., 0.95 for 95% VaR
            method:           "historical" | "parametric"

        Returns:
            var: VaR at given confidence level (positive number)
        """
        if method == "historical":
            return -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            import scipy.stats as stats
            mu = returns.mean()
            sigma = returns.std()
            z = stats.norm.ppf(1 - confidence_level)
            return -(mu + z * sigma)
        else:
            return -np.percentile(returns, (1 - confidence_level) * 100)

    def cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> float:
        """Conditional VaR (Expected Shortfall).

        Average of returns below VaR threshold.
        """
        var = self.var(returns, confidence_level, "historical")
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return var
        return -tail.mean()

    def sterling_ratio(self, returns: np.ndarray, period_years: float = 1.0) -> float:
        """Sterling Ratio: annualized return / average annual max drawdown."""
        ann_ret = returns.mean() * self.annualization
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        mdd = ((peak - cum) / peak).max()
        return ann_ret / (mdd + 0.10 + 1e-10)  # +10% penalty per Sterling's original formula

    def compute_all(self, returns: np.ndarray) -> Dict[str, float]:
        """Compute all risk metrics.

        Args:
            returns: (T,) return series

        Returns:
            metrics: dict of metric name → value
        """
        ann_ret = returns.mean() * self.annualization
        ann_vol = self.volatility(returns)
        sortino = (ann_ret - self.risk_free_rate) / (self.downside_deviation(returns) + 1e-10)
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        mdd = ((peak - cum) / peak).max()

        return {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": (ann_ret - self.risk_free_rate) / (ann_vol + 1e-10),
            "sortino_ratio": sortino,
            "max_drawdown": mdd,
            "calmar_ratio": ann_ret / (mdd + 1e-10),
            "var_95": self.var(returns, 0.95),
            "var_99": self.var(returns, 0.99),
            "cvar_95": self.cvar(returns, 0.95),
            "cvar_99": self.cvar(returns, 0.99),
            "downside_deviation": self.downside_deviation(returns),
            "ulcer_index": self.ulcer_index(returns),
            "pain_index": self.pain_index(returns),
            "tail_ratio": self.tail_ratio(returns),
            "gain_to_pain": self.gain_to_pain(returns),
            "sterling_ratio": self.sterling_ratio(returns),
            "n_periods": len(returns),
            "n_positive": (returns > 0).sum(),
            "hit_rate": (returns > 0).mean(),
            "skewness": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 3)),
            "kurtosis": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 4) - 3),
        }


# ---------------------------------------------------------------------------
# Factor Evaluator
# ---------------------------------------------------------------------------

class FactorEvaluator:
    """Evaluate a financial factor/alpha signal.

    Computes standard quantitative finance factor evaluation metrics:
    - Information Coefficient (IC) and IC IR
    - Quantile analysis (factor sorted returns by quintile)
    - Monotonicity tests
    - Turnover and capacity estimates
    - Factor decay analysis (IC at multiple horizons)

    Args:
        n_quantiles:   number of quantile buckets for sorting
        annualization: periods per year

    Example:
        >>> evaluator = FactorEvaluator(n_quantiles=5)
        >>> T, N = 252, 100
        >>> factors = np.random.randn(T, N)   # (time, assets) factor scores
        >>> returns = np.random.randn(T, N) * 0.01
        >>> results = evaluator.evaluate(factors, returns)
    """

    def __init__(self, n_quantiles: int = 5, annualization: float = 252.0):
        self.n_quantiles = n_quantiles
        self.annualization = annualization

    def ic_series(
        self, factors: np.ndarray, returns: np.ndarray, method: str = "spearman"
    ) -> np.ndarray:
        """Compute IC time series.

        Args:
            factors: (T, N) factor matrix
            returns: (T, N) forward return matrix
            method:  "spearman" or "pearson"

        Returns:
            ic: (T,) IC series (NaN where computation not possible)
        """
        from scipy import stats
        T, N = factors.shape
        ic = np.full(T, np.nan)
        for t in range(T):
            f = factors[t]
            r = returns[t]
            valid = ~np.isnan(f) & ~np.isnan(r)
            if valid.sum() < 3:
                continue
            if method == "spearman":
                ic[t] = stats.spearmanr(f[valid], r[valid])[0]
            else:
                ic[t] = stats.pearsonr(f[valid], r[valid])[0]
        return ic

    def quantile_returns(
        self, factors: np.ndarray, returns: np.ndarray
    ) -> np.ndarray:
        """Compute mean returns by factor quantile.

        Args:
            factors: (T, N) factor matrix
            returns: (T, N) return matrix

        Returns:
            quantile_returns: (T, n_quantiles) returns per quantile per period
        """
        T, N = factors.shape
        q_returns = np.full((T, self.n_quantiles), np.nan)

        for t in range(T):
            f = factors[t]
            r = returns[t]
            valid = ~np.isnan(f) & ~np.isnan(r)
            if valid.sum() < self.n_quantiles:
                continue
            f_v = f[valid]
            r_v = r[valid]
            # Sort by factor and split into quantiles
            sort_idx = np.argsort(f_v)
            q_size = len(sort_idx) // self.n_quantiles
            for q in range(self.n_quantiles):
                start = q * q_size
                end = (q + 1) * q_size if q < self.n_quantiles - 1 else len(sort_idx)
                q_returns[t, q] = r_v[sort_idx[start:end]].mean()

        return q_returns

    def evaluate(
        self,
        factors: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Comprehensive factor evaluation.

        Args:
            factors: (T, N) factor scores (pre-normalized)
            returns: (T, N) one-period-ahead returns

        Returns:
            results: dict with all evaluation metrics
        """
        T, N = factors.shape

        # IC metrics
        ic = self.ic_series(factors, returns, method="spearman")
        ic_valid = ic[~np.isnan(ic)]

        ic_mean = ic_valid.mean() if len(ic_valid) > 0 else np.nan
        ic_std = ic_valid.std() if len(ic_valid) > 0 else np.nan
        ic_ir = ic_mean / (ic_std + 1e-10)
        ic_positive_frac = (ic_valid > 0).mean() if len(ic_valid) > 0 else np.nan

        # Quantile returns
        q_rets = self.quantile_returns(factors, returns)
        q_means = np.nanmean(q_rets, axis=0)
        q_ann = q_means * self.annualization

        # Monotonicity: correlation of quantile index with mean return
        if not np.any(np.isnan(q_means)):
            from scipy.stats import spearmanr
            mono_corr = spearmanr(np.arange(self.n_quantiles), q_means)[0]
        else:
            mono_corr = np.nan

        # Long-short spread (top - bottom quantile)
        ls_spread = q_ann[-1] - q_ann[0]

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_positive_fraction": ic_positive_frac,
            "ic_series": ic,
            "quantile_annualized_returns": q_ann.tolist(),
            "long_short_spread": ls_spread,
            "monotonicity_correlation": mono_corr,
            "n_periods": T,
            "n_assets": N,
        }


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

class StatisticalTests:
    """Statistical significance tests for financial model evaluation.

    Tests:
    - t-test on IC series (H0: mean IC = 0)
    - Diebold-Mariano test (compare forecast accuracy)
    - White's Reality Check (data snooping correction)
    - Walk-forward statistical significance

    Example:
        >>> st = StatisticalTests()
        >>> ic_series = np.random.randn(252) * 0.05 + 0.02
        >>> result = st.ic_ttest(ic_series)
        >>> print(f"t-stat: {result['t_stat']:.3f}, p-value: {result['p_value']:.4f}")
    """

    @staticmethod
    def ic_ttest(
        ic_series: np.ndarray,
        null_ic: float = 0.0,
    ) -> Dict[str, float]:
        """One-sample t-test on IC series.

        Tests H0: mean(IC) = null_ic against H1: mean(IC) != null_ic.

        Args:
            ic_series: (T,) IC series
            null_ic:   null hypothesis IC value (default 0)

        Returns:
            result: dict with t_stat, p_value, mean_ic, std_ic, n
        """
        from scipy import stats
        valid = ic_series[~np.isnan(ic_series)]
        if len(valid) < 2:
            return {"t_stat": np.nan, "p_value": np.nan, "mean_ic": np.nan}

        t_stat, p_value = stats.ttest_1samp(valid, null_ic)
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_ic": valid.mean(),
            "std_ic": valid.std(),
            "n": len(valid),
            "significant_5pct": p_value < 0.05,
            "significant_1pct": p_value < 0.01,
        }

    @staticmethod
    def sharpe_significance(
        returns: np.ndarray,
        target_sharpe: float = 0.0,
        annualization: float = 252.0,
    ) -> Dict[str, float]:
        """Test if Sharpe ratio is significantly different from target.

        Uses the modified t-test accounting for non-normality
        (Lo 2002 approach).

        Args:
            returns:        (T,) return series
            target_sharpe: null hypothesis Sharpe ratio
            annualization:  periods per year

        Returns:
            result: dict with t_stat, p_value, sharpe_ratio
        """
        from scipy import stats

        T = len(returns)
        mu = returns.mean()
        sigma = returns.std()
        sr = mu / (sigma + 1e-10) * np.sqrt(annualization)

        # Lo (2002) asymptotic distribution accounting for non-normality
        skew = float(np.mean(((returns - mu) / (sigma + 1e-10)) ** 3))
        kurt = float(np.mean(((returns - mu) / (sigma + 1e-10)) ** 4))
        sr_var = (1 / T) * (1 - skew * sr / np.sqrt(annualization) + (kurt - 1) / 4 * (sr / np.sqrt(annualization)) ** 2)
        t_stat = (sr - target_sharpe) / (np.sqrt(sr_var) + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "sharpe_ratio": sr,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05,
            "n_periods": T,
        }

    @staticmethod
    def diebold_mariano(
        errors1: np.ndarray,
        errors2: np.ndarray,
        loss_fn: str = "mse",
        h: int = 1,
    ) -> Dict[str, float]:
        """Diebold-Mariano test for equal predictive accuracy.

        Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t).
        H0 says both models have equal predictive accuracy.
        H1: model 2 is significantly better than model 1.

        Args:
            errors1: (T,) forecast errors from model 1
            errors2: (T,) forecast errors from model 2
            loss_fn: "mse" | "mae" | "absolute"
            h:       forecast horizon

        Returns:
            result: dict with DM statistic and p-value
        """
        from scipy import stats

        if loss_fn == "mse":
            d = errors1 ** 2 - errors2 ** 2
        elif loss_fn == "mae":
            d = np.abs(errors1) - np.abs(errors2)
        else:
            d = np.abs(errors1) - np.abs(errors2)

        T = len(d)
        d_mean = d.mean()

        # HAC variance (Newey-West with h-1 lags)
        gamma0 = np.var(d)
        lags = h - 1
        long_run_var = gamma0
        for lag in range(1, lags + 1):
            gamma_lag = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
            long_run_var += 2 * (1 - lag / h) * gamma_lag

        dm_stat = d_mean / np.sqrt(max(long_run_var / T, 1e-10))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return {
            "dm_statistic": dm_stat,
            "p_value": p_value,
            "mean_loss_diff": d_mean,
            "model1_better": dm_stat < 0,
            "model2_better": dm_stat > 0,
            "significant_5pct": p_value < 0.05,
        }


# =============================================================================
# SECTION: Advanced Portfolio Analytics
# =============================================================================

class AttributionAnalyzer:
    """Brinson-Hood-Beebower (BHB) performance attribution.

    Decomposes portfolio excess return into:
    - Allocation effect: over/underweighting sector vs benchmark
    - Selection effect: stock selection within sectors
    - Interaction effect: joint allocation and selection

    Reference: Brinson & Beebower, "Determinants of Portfolio Performance"
    Financial Analysts Journal 1986.

    Args:
        frequency: Data frequency ('daily', 'weekly', 'monthly')
    """

    def __init__(self, frequency: str = "daily") -> None:
        self.frequency = frequency
        self._annualization = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 252)

    def compute_bhb(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        asset_returns: np.ndarray,
        sector_map: Dict[int, str],
    ) -> Dict[str, float]:
        """Compute BHB attribution decomposition.

        Args:
            portfolio_weights: (N,) portfolio asset weights
            benchmark_weights: (N,) benchmark asset weights
            asset_returns: (N,) realized asset returns this period
            sector_map: Dict mapping asset index to sector name
        Returns:
            Dict with allocation, selection, interaction, total effects
        """
        sectors = list(set(sector_map.values()))
        alloc, select, interact = 0.0, 0.0, 0.0

        for sector in sectors:
            mask = np.array([sector_map.get(i, "") == sector for i in range(len(asset_returns))])
            if mask.sum() == 0:
                continue

            wp = portfolio_weights[mask].sum()  # Portfolio sector weight
            wb = benchmark_weights[mask].sum()   # Benchmark sector weight

            # Sector returns
            rp = (portfolio_weights[mask] * asset_returns[mask]).sum() / (wp + 1e-10)
            rb = (benchmark_weights[mask] * asset_returns[mask]).sum() / (wb + 1e-10)
            rb_total = (benchmark_weights * asset_returns).sum()  # Total benchmark return

            # BHB effects
            alloc += (wp - wb) * (rb - rb_total)
            select += wb * (rp - rb)
            interact += (wp - wb) * (rp - rb)

        total = alloc + select + interact
        return {
            "allocation_effect": alloc,
            "selection_effect": select,
            "interaction_effect": interact,
            "total_active_return": total,
        }

    def rolling_attribution(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        asset_returns: np.ndarray,
        sector_map: Dict[int, str],
        window: int = 21,
    ) -> Dict[str, np.ndarray]:
        """Rolling BHB attribution over time.

        Args:
            portfolio_weights: (T, N)
            benchmark_weights: (T, N)
            asset_returns: (T, N)
            sector_map: Dict
            window: Rolling window
        Returns:
            Dict of (T,) attribution time series
        """
        T = len(asset_returns)
        allocs = np.zeros(T)
        selects = np.zeros(T)
        interacts = np.zeros(T)

        for t in range(window, T):
            w_start = max(0, t - window)
            pw_avg = portfolio_weights[w_start:t].mean(axis=0)
            bw_avg = benchmark_weights[w_start:t].mean(axis=0)
            r_avg = asset_returns[w_start:t].mean(axis=0)
            result = self.compute_bhb(pw_avg, bw_avg, r_avg, sector_map)
            allocs[t] = result["allocation_effect"]
            selects[t] = result["selection_effect"]
            interacts[t] = result["interaction_effect"]

        return {
            "allocation": allocs,
            "selection": selects,
            "interaction": interacts,
            "total": allocs + selects + interacts,
        }


class FactorExposureAnalyzer:
    """Analyze factor exposures and alpha decomposition.

    Regresses portfolio returns on systematic risk factors
    to identify alpha (unexplained return) and beta exposures.

    Supported factor models:
    - CAPM: single market factor
    - Fama-French 3-factor
    - Carhart 4-factor (FF3 + momentum)
    - Custom factor models

    Args:
        factors: (T, F) factor return matrix
        factor_names: List of factor names
    """

    def __init__(
        self,
        factors: np.ndarray,
        factor_names: Optional[List[str]] = None,
    ) -> None:
        self.factors = factors
        T, F = factors.shape
        self.factor_names = factor_names or [f"factor_{i}" for i in range(F)]

    def estimate_betas(
        self,
        portfolio_returns: np.ndarray,
    ) -> Dict[str, float]:
        """OLS regression of portfolio on factors.

        Args:
            portfolio_returns: (T,) portfolio return series
        Returns:
            Dict with alpha, betas, r_squared, t_stats
        """
        T = len(portfolio_returns)
        # Add intercept
        X = np.column_stack([np.ones(T), self.factors])  # (T, 1+F)
        y = portfolio_returns

        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
        except np.linalg.LinAlgError:
            return {"error": "singular matrix"}

        # Residuals and stats
        y_hat = X @ beta
        resid = y - y_hat
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Standard errors
        sigma2 = ss_res / max(1, T - len(beta))
        try:
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
        except Exception:
            se = np.zeros(len(beta))

        t_stats = beta / (se + 1e-10)

        result = {
            "alpha": beta[0],
            "alpha_t_stat": t_stats[0],
            "r_squared": r2,
            "annualized_alpha": beta[0] * 252,
        }
        for i, name in enumerate(self.factor_names):
            result[f"beta_{name}"] = beta[i + 1]
            result[f"t_{name}"] = t_stats[i + 1]

        return result

    def tracking_error(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, float]:
        """Compute tracking error and information ratio.

        Args:
            portfolio_returns: (T,) portfolio returns
            benchmark_returns: (T,) benchmark returns
        Returns:
            Dict with tracking_error, information_ratio, active_return
        """
        active_returns = portfolio_returns - benchmark_returns
        te = active_returns.std() * np.sqrt(252)
        ar = active_returns.mean() * 252
        ir = ar / (te + 1e-10)
        return {
            "tracking_error": te,
            "annualized_active_return": ar,
            "information_ratio": ir,
            "active_return_t_stat": ar / (te / np.sqrt(max(1, len(active_returns))) + 1e-10),
        }


class MarketImpactModel:
    """Market impact model for realistic transaction cost estimation.

    Implements multiple market impact models:
    - Linear: impact proportional to trade size
    - Square root (Almgren): impact ~ sigma * sqrt(ADV * participation_rate)
    - Power law: impact ~ (order_size / ADV)^alpha

    Reference: Almgren et al., "Direct Estimation of Equity Market Impact"
    (2005)

    Args:
        model_type: 'linear', 'sqrt', or 'power'
        eta: Linear impact coefficient
        sigma: Volatility scaling
    """

    def __init__(
        self,
        model_type: str = "sqrt",
        eta: float = 0.1,
        sigma: float = 0.02,
        alpha: float = 0.6,
    ) -> None:
        self.model_type = model_type
        self.eta = eta
        self.sigma = sigma
        self.alpha = alpha

    def estimate_impact(
        self,
        order_size: float,
        adv: float,
        price: float,
        volatility: float,
        side: str = "buy",
    ) -> Dict[str, float]:
        """Estimate market impact for a single trade.

        Args:
            order_size: Trade size in shares
            adv: Average daily volume in shares
            price: Current price per share
            volatility: Daily return volatility
            side: 'buy' or 'sell'
        Returns:
            Dict with impact_bps, total_cost_usd, effective_spread
        """
        participation = order_size / max(1, adv)

        if self.model_type == "linear":
            impact = self.eta * participation
        elif self.model_type == "sqrt":
            # Almgren square root model
            impact = self.sigma * np.sqrt(participation) * np.sign(1 if side == "buy" else -1)
            impact = abs(impact)
        elif self.model_type == "power":
            impact = self.sigma * (participation ** self.alpha)
        else:
            impact = self.eta * participation

        impact_bps = impact * 10000
        total_cost = impact * price * order_size

        return {
            "impact_fraction": impact,
            "impact_bps": impact_bps,
            "total_cost_usd": total_cost,
            "participation_rate": participation,
        }


class RegimeDetectionBacktest:
    """Backtesting framework with regime-conditional analysis.

    Splits backtest periods by detected market regime and reports
    performance metrics separately for each regime. Enables
    understanding of strategy behavior across market conditions.

    Args:
        regime_labels: (T,) integer regime labels
        regime_names: Optional mapping from label to name
    """

    def __init__(
        self,
        regime_labels: np.ndarray,
        regime_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.regime_labels = regime_labels
        unique_regimes = sorted(set(regime_labels))
        if regime_names is None:
            regime_names = {r: f"regime_{r}" for r in unique_regimes}
        self.regime_names = regime_names

    def compute_regime_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics within each regime.

        Args:
            returns: (T,) portfolio return series
            benchmark_returns: (T,) optional benchmark returns
        Returns:
            Dict of {regime_name: {metric: value}}
        """
        results = {}
        for regime_id, regime_name in self.regime_names.items():
            mask = self.regime_labels == regime_id
            if mask.sum() < 5:
                continue
            r = returns[mask]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / (ann_vol + 1e-10)
            max_dd = self._max_drawdown(r)
            calmar = ann_ret / (abs(max_dd) + 1e-10)
            results[regime_name] = {
                "num_days": int(mask.sum()),
                "ann_return": ann_ret,
                "ann_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "calmar_ratio": calmar,
                "hit_rate": float((r > 0).mean()),
                "avg_return": float(r.mean()),
            }
            if benchmark_returns is not None:
                bm = benchmark_returns[mask]
                active = r - bm
                results[regime_name]["active_return"] = active.mean() * 252
                te = active.std() * np.sqrt(252)
                results[regime_name]["tracking_error"] = te
                results[regime_name]["information_ratio"] = results[regime_name]["active_return"] / (te + 1e-10)

        return results

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cum = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum)
        drawdowns = (cum - rolling_max) / (rolling_max + 1e-10)
        return float(drawdowns.min())


class TransactionCostModel:
    """Realistic transaction cost model for backtesting.

    Components:
    - Bid-ask spread (proportional to volatility and market cap)
    - Commission: fixed per-share or percentage
    - Market impact: function of order size relative to ADV
    - Short borrow rate: for short positions

    Args:
        commission_rate: Fixed commission as fraction of trade value
        spread_model: 'fixed', 'vol_proportional', or 'cap_based'
        fixed_spread_bps: Fixed spread if spread_model='fixed'
        vol_spread_mult: Multiplier for vol-proportional spread
        short_borrow_rate: Annual short borrow rate (e.g., 0.01 = 1%)
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,
        spread_model: str = "fixed",
        fixed_spread_bps: float = 10.0,
        vol_spread_mult: float = 0.5,
        short_borrow_rate: float = 0.02,
    ) -> None:
        self.commission_rate = commission_rate
        self.spread_model = spread_model
        self.fixed_spread_bps = fixed_spread_bps
        self.vol_spread_mult = vol_spread_mult
        self.short_borrow_rate = short_borrow_rate / 252  # Daily

    def estimate_spread(
        self,
        volatility: float,
        market_cap: Optional[float] = None,
    ) -> float:
        """Estimate bid-ask spread in basis points.

        Args:
            volatility: Daily return volatility
            market_cap: Optional market cap for size-based spread
        Returns:
            Estimated spread in bps
        """
        if self.spread_model == "fixed":
            return self.fixed_spread_bps
        elif self.spread_model == "vol_proportional":
            return volatility * self.vol_spread_mult * 10000
        elif self.spread_model == "cap_based" and market_cap is not None:
            # Smaller caps have wider spreads
            if market_cap > 10e9:
                return 3.0
            elif market_cap > 1e9:
                return 8.0
            else:
                return 20.0
        return self.fixed_spread_bps

    def compute_round_trip_cost(
        self,
        trade_value: float,
        volatility: float = 0.02,
        market_cap: Optional[float] = None,
        is_short: bool = False,
        holding_days: int = 1,
    ) -> Dict[str, float]:
        """Compute total round-trip transaction cost.

        Args:
            trade_value: Absolute trade value in dollars
            volatility: Daily return volatility
            market_cap: Optional for spread model
            is_short: Whether this is a short position
            holding_days: Days held (for short borrow cost)
        Returns:
            Dict with component costs and total
        """
        spread_bps = self.estimate_spread(volatility, market_cap)
        spread_cost = spread_bps / 10000 * trade_value * 2  # 2-way
        commission = self.commission_rate * trade_value * 2
        borrow_cost = 0.0
        if is_short:
            borrow_cost = self.short_borrow_rate * holding_days * trade_value
        total = spread_cost + commission + borrow_cost
        return {
            "spread_cost": spread_cost,
            "commission": commission,
            "borrow_cost": borrow_cost,
            "total": total,
            "total_bps": total / (trade_value + 1e-10) * 10000,
        }


class BootstrapMetricCalculator:
    """Bootstrap resampling for robust statistical inference on metrics.

    Computes bootstrap confidence intervals for performance metrics
    (Sharpe, Sortino, IC, etc.) to assess statistical significance.

    Args:
        num_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (e.g., 0.95)
        random_seed: Reproducibility seed
        block_size: Block size for block bootstrap (preserves autocorrelation)
    """

    def __init__(
        self,
        num_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        block_size: int = 10,
    ) -> None:
        self.num_bootstrap = num_bootstrap
        self.confidence_level = confidence_level
        self.block_size = block_size
        np.random.seed(random_seed)

    def _block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Generate block bootstrap sample."""
        T = len(data)
        num_blocks = T // self.block_size + 1
        block_starts = np.random.randint(0, max(1, T - self.block_size), size=num_blocks)
        blocks = [data[s:s + self.block_size] for s in block_starts]
        sample = np.concatenate(blocks)[:T]
        return sample

    def bootstrap_metric(
        self,
        returns: np.ndarray,
        metric_fn: Callable,
        use_block: bool = True,
    ) -> Dict[str, float]:
        """Bootstrap a scalar metric.

        Args:
            returns: (T,) return series
            metric_fn: Function that takes (T,) returns and returns scalar
            use_block: Use block bootstrap (True) or iid (False)
        Returns:
            Dict with mean, std, ci_lower, ci_upper, p_value_positive
        """
        bootstrap_values = []
        for _ in range(self.num_bootstrap):
            if use_block:
                sample = self._block_bootstrap(returns)
            else:
                sample = returns[np.random.randint(0, len(returns), len(returns))]
            try:
                val = metric_fn(sample)
                bootstrap_values.append(val)
            except Exception:
                continue

        if not bootstrap_values:
            return {"error": "metric computation failed"}

        bv = np.array(bootstrap_values)
        alpha = (1 - self.confidence_level) / 2
        ci_lower = np.quantile(bv, alpha)
        ci_upper = np.quantile(bv, 1 - alpha)
        p_positive = float((bv > 0).mean())

        return {
            "mean": float(bv.mean()),
            "std": float(bv.std()),
            f"ci_{int(self.confidence_level*100)}_lower": ci_lower,
            f"ci_{int(self.confidence_level*100)}_upper": ci_upper,
            "p_value_positive": p_positive,
            "original_value": float(metric_fn(returns)),
        }

    def bootstrap_sharpe(self, returns: np.ndarray) -> Dict[str, float]:
        """Bootstrap Sharpe ratio."""
        def sharpe_fn(r):
            return r.mean() * 252 / (r.std() * np.sqrt(252) + 1e-10)
        return self.bootstrap_metric(returns, sharpe_fn)

    def bootstrap_ic(
        self,
        predictions: np.ndarray,
        realized: np.ndarray,
    ) -> Dict[str, float]:
        """Bootstrap Information Coefficient (rank correlation)."""
        from scipy import stats

        def ic_fn_inner(idx):
            p, r = predictions[idx], realized[idx]
            return stats.spearmanr(p, r).correlation

        def ic_bootstrap(dummy_r):
            # Hack: resample by index
            return ic_fn_inner(slice(None))  # Just return the full IC

        # Manual bootstrap
        T = len(predictions)
        bootstrap_values = []
        for _ in range(self.num_bootstrap):
            if self.block_size > 1:
                num_blocks = T // self.block_size + 1
                starts = np.random.randint(0, max(1, T - self.block_size), size=num_blocks)
                idx = np.concatenate([np.arange(s, min(s + self.block_size, T)) for s in starts])[:T]
            else:
                idx = np.random.randint(0, T, T)
            try:
                from scipy import stats as scipy_stats
                ic = scipy_stats.spearmanr(predictions[idx], realized[idx]).correlation
                bootstrap_values.append(float(ic))
            except Exception:
                pass

        if not bootstrap_values:
            return {}
        bv = np.array(bootstrap_values)
        alpha = (1 - self.confidence_level) / 2
        return {
            "ic_mean": float(bv.mean()),
            "ic_std": float(bv.std()),
            "ic_ci_lower": float(np.quantile(bv, alpha)),
            "ic_ci_upper": float(np.quantile(bv, 1 - alpha)),
            "ic_t_stat": float(bv.mean() / (bv.std() / np.sqrt(len(bv)) + 1e-10)),
        }


class RollingSharpeAnalysis:
    """Rolling Sharpe ratio and drawdown analysis utilities.

    Provides time series of risk-adjusted performance metrics
    to identify periods of strategy degradation or improvement.

    Args:
        window: Rolling window in trading days
        annualization: Trading days per year for annualization
    """

    def __init__(self, window: int = 63, annualization: int = 252) -> None:
        self.window = window
        self.annualization = annualization

    def rolling_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling Sharpe ratio."""
        T = len(returns)
        sharpes = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization
            ann_vol = r.std() * np.sqrt(self.annualization)
            sharpes[t] = ann_ret / (ann_vol + 1e-10)
        return sharpes

    def rolling_sortino(self, returns: np.ndarray, mar: float = 0.0) -> np.ndarray:
        """Compute rolling Sortino ratio."""
        T = len(returns)
        sortinos = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization - mar
            downside = r[r < mar / self.annualization] - mar / self.annualization
            dd_std = np.sqrt((downside ** 2).mean()) * np.sqrt(self.annualization)
            sortinos[t] = ann_ret / (dd_std + 1e-10)
        return sortinos

    def rolling_max_drawdown(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling maximum drawdown."""
        T = len(returns)
        mdd = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            cum = (1 + r).cumprod()
            roll_max = np.maximum.accumulate(cum)
            dd = (cum - roll_max) / (roll_max + 1e-10)
            mdd[t] = dd.min()
        return mdd

    def rolling_calmar(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling Calmar ratio."""
        T = len(returns)
        calmar = np.full(T, np.nan)
        sharpes = self.rolling_sharpe(returns)
        mdd = self.rolling_max_drawdown(returns)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization
            dd = abs(mdd[t])
            calmar[t] = ann_ret / (dd + 1e-10)
        return calmar

    def regime_performance_summary(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Summary statistics per regime."""
        regimes = {}
        for r_id in sorted(set(regime_labels)):
            mask = regime_labels == r_id
            r = returns[mask]
            if len(r) < 2:
                continue
            regimes[f"regime_{r_id}"] = {
                "n": int(len(r)),
                "mean": float(r.mean() * self.annualization),
                "vol": float(r.std() * np.sqrt(self.annualization)),
                "sharpe": float(r.mean() * self.annualization / (r.std() * np.sqrt(self.annualization) + 1e-10)),
                "hit_rate": float((r > 0).mean()),
                "skew": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-10)),
                "kurt": float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-10) - 3),
            }
        return regimes


_NEW_EVALUATION_EXPORTS = [
    "AttributionAnalyzer", "FactorExposureAnalyzer", "MarketImpactModel",
    "RegimeDetectionBacktest", "TransactionCostModel",
    "BootstrapMetricCalculator", "RollingSharpeAnalysis",
]


# =============================================================================
# SECTION: Advanced Evaluation and Backtesting (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
import math


class MonteCarloBacktest:
    """Monte Carlo simulation-based backtesting.

    Runs multiple market scenarios to stress test strategy performance.
    Computes distribution of outcomes and tail risk metrics.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        horizon_days: int = 252,
        confidence_levels: List[float] = None,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10, 0.25]
        self.seed = seed

    def simulate_returns(
        self,
        mean: float,
        std: float,
        skew: float = 0.0,
        kurt: float = 3.0,
    ) -> np.ndarray:
        """Simulate return paths with given moments."""
        np.random.seed(self.seed)
        # Use moment matching: normal if skew=0, kurt=3
        if abs(skew) < 0.01 and abs(kurt - 3.0) < 0.01:
            returns = np.random.normal(mean, std, (self.n_simulations, self.horizon_days))
        else:
            # Simple skew-t approximation
            t_df = 6.0 / (kurt - 3.0 + 1e-6) + 4
            t_df = max(4.01, min(t_df, 1000))
            from scipy import stats
            returns = stats.t.rvs(t_df, loc=mean, scale=std, size=(self.n_simulations, self.horizon_days))
        return returns

    def run(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray = None,
    ) -> dict:
        """Run Monte Carlo backtest and compute risk metrics."""
        # Fit moments from historical returns
        mu = strategy_returns.mean()
        sigma = strategy_returns.std()
        skew = float(((strategy_returns - mu) ** 3).mean() / (sigma ** 3 + 1e-8))
        kurt = float(((strategy_returns - mu) ** 4).mean() / (sigma ** 4 + 1e-8))

        sim_returns = self.simulate_returns(mu, sigma, skew, kurt)

        # Compute cumulative returns for each path
        cum_returns = (1 + sim_returns).cumprod(axis=1) - 1

        # Final wealth distribution
        final_wealth = cum_returns[:, -1]

        results = {
            "mean_final_wealth": final_wealth.mean(),
            "std_final_wealth": final_wealth.std(),
            "median_final_wealth": np.median(final_wealth),
            "skewness": skew,
            "excess_kurtosis": kurt - 3,
        }

        for cl in self.confidence_levels:
            var = np.percentile(final_wealth, cl * 100)
            cvar = final_wealth[final_wealth <= var].mean() if (final_wealth <= var).any() else var
            results[f"VaR_{int(cl*100)}pct"] = var
            results[f"CVaR_{int(cl*100)}pct"] = cvar

        # Probability of loss
        results["prob_loss"] = (final_wealth < 0).mean()
        results["prob_loss_10pct"] = (final_wealth < -0.10).mean()
        results["prob_loss_20pct"] = (final_wealth < -0.20).mean()

        # Max drawdown distribution
        def max_drawdown(path):
            cummax = np.maximum.accumulate(path + 1)
            drawdown = (path + 1) / cummax - 1
            return drawdown.min()

        mdd_dist = np.array([max_drawdown(sim_returns[i]) for i in range(min(self.n_simulations, 100))])
        results["expected_max_drawdown"] = mdd_dist.mean()
        results["worst_10pct_max_drawdown"] = np.percentile(mdd_dist, 10)

        return results


class FactorModelEvaluator:
    """Evaluate factor model performance: IC, ICIR, factor returns, etc.

    Implements standard quantitative equity research metrics.
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        holding_period: int = 1,
        transaction_cost_bps: float = 5.0,
    ):
        self.n_quantiles = n_quantiles
        self.holding_period = holding_period
        self.tc_bps = transaction_cost_bps / 10000

    def compute_ic(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
        method: str = "spearman",
    ) -> float:
        """Compute Information Coefficient (IC)."""
        from scipy import stats
        valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
        if valid.sum() < 5:
            return float("nan")

        f = factor_scores[valid]
        r = forward_returns[valid]

        if method == "spearman":
            ic, _ = stats.spearmanr(f, r)
        elif method == "pearson":
            ic, _ = stats.pearsonr(f, r)
        else:
            ic, _ = stats.spearmanr(f, r)

        return float(ic)

    def compute_ic_series(
        self,
        factor_panel: np.ndarray,
        return_panel: np.ndarray,
    ) -> Dict[str, float]:
        """Compute IC series metrics over time."""
        T = factor_panel.shape[0]
        ic_series = []

        for t in range(T):
            ic = self.compute_ic(factor_panel[t], return_panel[t])
            if not np.isnan(ic):
                ic_series.append(ic)

        ic_arr = np.array(ic_series)
        if len(ic_arr) == 0:
            return {}

        return {
            "mean_ic": float(ic_arr.mean()),
            "icir": float(ic_arr.mean() / (ic_arr.std() + 1e-8)),
            "ic_positive_pct": float((ic_arr > 0).mean()),
            "ic_greater_02": float((ic_arr > 0.02).mean()),
            "ic_t_stat": float(ic_arr.mean() / (ic_arr.std() / (len(ic_arr) ** 0.5) + 1e-8)),
        }

    def quantile_returns(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute mean forward return by factor quantile."""
        valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
        f = factor_scores[valid]
        r = forward_returns[valid]

        quantile_edges = np.linspace(0, 100, self.n_quantiles + 1)
        quantile_returns = []

        for i in range(self.n_quantiles):
            lo = np.percentile(f, quantile_edges[i])
            hi = np.percentile(f, quantile_edges[i + 1])
            mask = (f >= lo) & (f < hi)
            if mask.any():
                quantile_returns.append(r[mask].mean())
            else:
                quantile_returns.append(float("nan"))

        return np.array(quantile_returns)

    def long_short_return(
        self,
        factor_scores: np.ndarray,
        forward_returns: np.ndarray,
    ) -> float:
        """Compute long-short return: top quintile minus bottom quintile."""
        q_returns = self.quantile_returns(factor_scores, forward_returns)
        if np.isnan(q_returns[0]) or np.isnan(q_returns[-1]):
            return float("nan")
        return float(q_returns[-1] - q_returns[0]) - 2 * self.tc_bps

    def factor_decay(
        self,
        factor_scores: np.ndarray,
        returns_multi_horizon: np.ndarray,
    ) -> np.ndarray:
        """Compute IC decay across multiple forward horizons."""
        n_horizons = returns_multi_horizon.shape[1]
        ics = []
        for h in range(n_horizons):
            ic = self.compute_ic(factor_scores, returns_multi_horizon[:, h])
            ics.append(ic)
        return np.array(ics)


class WalkForwardValidator:
    """Walk-forward validation for time series models.

    Simulates real deployment:
    - Train on past N years
    - Validate on next M months
    - Roll forward M months at a time
    """

    def __init__(
        self,
        train_window: int = 252 * 3,
        val_window: int = 21,
        step_size: int = 21,
        min_train_samples: int = 252,
    ):
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples

    def split(self, T: int) -> List[Tuple[range, range]]:
        """Generate (train_idx, val_idx) splits for walk-forward validation."""
        splits = []
        pos = 0

        while pos + self.train_window + self.val_window <= T:
            train_start = max(0, pos)
            train_end = pos + self.train_window
            val_start = train_end
            val_end = min(T, val_start + self.val_window)

            if train_end - train_start >= self.min_train_samples:
                splits.append((
                    range(train_start, train_end),
                    range(val_start, val_end),
                ))

            pos += self.step_size

        return splits

    def evaluate(
        self,
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn,
    ) -> Dict[str, np.ndarray]:
        """Run walk-forward evaluation.

        Args:
            model_fn: callable(X_train, y_train, X_val) -> predictions
            X: [T, n_features] features
            y: [T] targets
            metric_fn: callable(y_true, y_pred) -> float
        """
        splits = self.split(len(X))
        metrics = []
        train_sizes = []

        for train_idx, val_idx in splits:
            X_train = X[list(train_idx)]
            y_train = y[list(train_idx)]
            X_val = X[list(val_idx)]
            y_val = y[list(val_idx)]

            y_pred = model_fn(X_train, y_train, X_val)
            m = metric_fn(y_val, y_pred)
            metrics.append(m)
            train_sizes.append(len(train_idx))

        return {
            "metrics": np.array(metrics),
            "mean_metric": np.nanmean(metrics),
            "std_metric": np.nanstd(metrics),
            "n_splits": len(splits),
            "train_sizes": np.array(train_sizes),
        }


class PerformanceAttributionSuite:
    """Comprehensive performance attribution for portfolio strategies.

    Includes:
    - Brinson-Hood-Beebower (BHB) attribution
    - Factor-based attribution (Fama-French, BARRA)
    - Risk-adjusted return attribution
    - Style box analysis
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def brinson_attribution(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        sector_map: Dict[int, str] = None,
    ) -> dict:
        """Compute BHB attribution: allocation + selection + interaction effects."""
        n_assets = len(portfolio_weights)
        total_port_return = (portfolio_weights * portfolio_returns).sum()
        total_bench_return = (benchmark_weights * benchmark_returns).sum()
        excess = total_port_return - total_bench_return

        allocation = (portfolio_weights - benchmark_weights) * (benchmark_returns - total_bench_return)
        selection = benchmark_weights * (portfolio_returns - benchmark_returns)
        interaction = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)

        result = {
            "total_excess": float(excess),
            "allocation_effect": float(allocation.sum()),
            "selection_effect": float(selection.sum()),
            "interaction_effect": float(interaction.sum()),
            "attribution_sum": float(allocation.sum() + selection.sum() + interaction.sum()),
        }

        if sector_map:
            sector_attribution = {}
            for asset_idx, sector in sector_map.items():
                if sector not in sector_attribution:
                    sector_attribution[sector] = {"allocation": 0.0, "selection": 0.0}
                if asset_idx < n_assets:
                    sector_attribution[sector]["allocation"] += float(allocation[asset_idx])
                    sector_attribution[sector]["selection"] += float(selection[asset_idx])
            result["sector_attribution"] = sector_attribution

        return result

    def factor_attribution(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
        risk_free_rate: float = None,
    ) -> dict:
        """Attribute portfolio returns to risk factors via OLS regression."""
        rf = risk_free_rate or self.risk_free_rate / 252
        excess_returns = portfolio_returns - rf

        factor_matrix = np.column_stack(list(factor_returns.values()))
        factor_names = list(factor_returns.keys())

        A = np.hstack([factor_matrix, np.ones((len(excess_returns), 1))])
        try:
            betas, _, _, _ = np.linalg.lstsq(A, excess_returns, rcond=None)
        except np.linalg.LinAlgError:
            return {"error": "Regression failed"}

        factor_betas = {name: float(b) for name, b in zip(factor_names, betas[:-1])}
        alpha = float(betas[-1])

        # Factor contributions
        contributions = {
            name: float(betas[i] * factor_returns[name].mean())
            for i, name in enumerate(factor_names)
        }

        pred = A @ betas
        ss_res = ((excess_returns - pred) ** 2).sum()
        ss_tot = ((excess_returns - excess_returns.mean()) ** 2).sum()
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))

        return {
            "alpha_annualized": alpha * 252,
            "factor_betas": factor_betas,
            "factor_contributions": contributions,
            "r_squared": r2,
            "specific_return": float(excess_returns.mean() - sum(contributions.values())),
        }

    def style_box_analysis(
        self,
        portfolio_weights: np.ndarray,
        market_caps: np.ndarray,
        pb_ratios: np.ndarray,
    ) -> dict:
        """Compute Morningstar-style style box position (size x value/growth)."""
        # Size dimension: small/mid/large
        weighted_mcap = (portfolio_weights * market_caps).sum() / portfolio_weights.sum()
        total_mcap = market_caps.sum()
        relative_size = weighted_mcap / (total_mcap / len(market_caps))

        if relative_size > 2.0:
            size_style = "large"
        elif relative_size > 0.5:
            size_style = "mid"
        else:
            size_style = "small"

        # Value/growth dimension: P/B ratio
        weighted_pb = (portfolio_weights * pb_ratios).sum() / portfolio_weights.sum()
        universe_pb = pb_ratios.mean()

        if weighted_pb < universe_pb * 0.8:
            value_style = "value"
        elif weighted_pb > universe_pb * 1.2:
            value_style = "growth"
        else:
            value_style = "blend"

        return {
            "size_style": size_style,
            "value_style": value_style,
            "relative_size": float(relative_size),
            "portfolio_pb": float(weighted_pb),
            "universe_pb": float(universe_pb),
        }


class TailRiskMetrics:
    """Compute tail risk metrics for financial strategies.

    Includes:
    - Value at Risk (VaR) - parametric and historical
    - Expected Shortfall (ES / CVaR)
    - Conditional Drawdown at Risk (CDaR)
    - Omega ratio
    - Upside/Downside capture ratios
    """

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]

    def historical_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Historical simulation VaR."""
        return float(np.percentile(returns, confidence * 100))

    def parametric_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Parametric (normal) VaR."""
        from scipy import stats
        mu = returns.mean()
        sigma = returns.std()
        return float(stats.norm.ppf(confidence, mu, sigma))

    def expected_shortfall(
        self,
        returns: np.ndarray,
        confidence: float = 0.05,
    ) -> float:
        """Expected Shortfall (CVaR): mean loss beyond VaR."""
        var = self.historical_var(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """Omega ratio: probability-weighted gain/loss ratio above/below threshold."""
        gains = (returns[returns > threshold] - threshold).sum()
        losses = (threshold - returns[returns <= threshold]).sum()
        return float(gains / max(losses, 1e-10))

    def capture_ratios(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> dict:
        """Upside and downside capture ratios."""
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0

        if up_mask.any():
            upside_capture = float(portfolio_returns[up_mask].mean() / (benchmark_returns[up_mask].mean() + 1e-10))
        else:
            upside_capture = float("nan")

        if down_mask.any():
            downside_capture = float(portfolio_returns[down_mask].mean() / (benchmark_returns[down_mask].mean() + 1e-10))
        else:
            downside_capture = float("nan")

        return {
            "upside_capture": upside_capture,
            "downside_capture": downside_capture,
            "capture_ratio": upside_capture / max(abs(downside_capture), 1e-10) if not (
                math.isnan(upside_capture) or math.isnan(downside_capture)
            ) else float("nan"),
        }

    def full_tail_risk_report(self, returns: np.ndarray) -> dict:
        """Compute all tail risk metrics."""
        result = {}
        for cl in self.confidence_levels:
            result[f"hist_VaR_{int(cl*100)}"] = self.historical_var(returns, cl)
            result[f"param_VaR_{int(cl*100)}"] = self.parametric_var(returns, cl)
            result[f"ES_{int(cl*100)}"] = self.expected_shortfall(returns, cl)

        result["omega_ratio"] = self.omega_ratio(returns)
        result["skewness"] = float(((returns - returns.mean()) ** 3).mean() / (returns.std() ** 3 + 1e-8))
        result["excess_kurtosis"] = float(
            ((returns - returns.mean()) ** 4).mean() / (returns.std() ** 4 + 1e-8) - 3
        )

        return result
