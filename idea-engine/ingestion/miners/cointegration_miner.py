"""
Cointegration miner — discovers pairs and basket mean-reversion opportunities.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


@dataclass
class CointegrationMinerConfig:
    min_half_life: float = 3.0
    max_half_life: float = 60.0
    adf_alpha: float = 0.05
    min_abs_zscore: float = 1.5
    zscore_window: int = 30
    max_pairs: int = 20


def mine_cointegrated_pairs(
    prices: np.ndarray,
    symbols: list[str],
    config: Optional[CointegrationMinerConfig] = None,
) -> list[dict]:
    """
    Scan all pairs for cointegration and extract active mean reversion opportunities.
    prices: (T, N) price matrix.
    symbols: list of N symbol names.
    """
    from lib.math.time_series_models import engle_granger_test

    if config is None:
        config = CointegrationMinerConfig()

    T, N = prices.shape
    log_prices = np.log(prices + 1e-10)
    findings = []

    for i, j in combinations(range(N), 2):
        if len(findings) >= config.max_pairs:
            break

        y = log_prices[:, i]
        x = log_prices[:, j]

        result = engle_granger_test(y, x)

        if not result["is_cointegrated"]:
            continue

        hl = result.get("half_life", np.inf)
        if not (config.min_half_life <= hl <= config.max_half_life):
            continue

        # Current z-score
        spread = result["spread"]
        if len(spread) >= config.zscore_window:
            recent = spread[-config.zscore_window:]
            z = float((spread[-1] - recent.mean()) / (recent.std() + 1e-10))
        else:
            z = float((spread[-1] - spread.mean()) / (spread.std() + 1e-10))

        if abs(z) < config.min_abs_zscore:
            continue

        findings.append({
            "type": "cointegrated_pair",
            "pair": (symbols[i], symbols[j]),
            "beta": float(result["beta"]),
            "alpha": float(result["alpha"]),
            "adf_stat": float(result["adf_stat"]),
            "half_life": float(hl),
            "z_score": float(z),
            "signal": float(-np.sign(z)),
            "confidence": float(min(abs(z) / 3.0, 1.0)),
            "template": "ou_zscore_entry" if hl < 20 else "kalman_spread_fade",
            "description": f"{symbols[i]}/{symbols[j]}: z={z:.2f}, HL={hl:.1f} periods",
        })

    return sorted(findings, key=lambda x: abs(x["z_score"]), reverse=True)


def mine_basket_opportunities(
    prices: np.ndarray,
    symbols: list[str],
    max_basket_size: int = 5,
    config: Optional[CointegrationMinerConfig] = None,
) -> list[dict]:
    """
    Mine multi-asset basket mean reversion opportunities via Johansen cointegration.
    """
    from lib.signals.mean_reversion_signals import johansen_cointegration, ou_zscore

    if config is None:
        config = CointegrationMinerConfig()

    T, N = prices.shape
    if N < 3:
        return []

    log_prices = np.log(prices + 1e-10)
    result = johansen_cointegration(log_prices)

    if result["n_cointegrated"] < 1:
        return []

    findings = []
    vectors = result["vectors"]

    for k in range(vectors.shape[1]):
        vec = vectors[:, k]
        spread = log_prices @ vec

        hl_result = _ou_half_life_simple(spread)
        if not (config.min_half_life <= hl_result <= config.max_half_life):
            continue

        z = ou_zscore(spread, window=config.zscore_window)
        current_z = float(z[-1])

        if abs(current_z) < config.min_abs_zscore:
            continue

        findings.append({
            "type": "basket_cointegration",
            "vector_index": k,
            "weights": dict(zip(symbols, vec.tolist())),
            "z_score": current_z,
            "half_life": float(hl_result),
            "n_coint_vectors": result["n_cointegrated"],
            "signal": float(-np.sign(current_z)),
            "confidence": float(min(abs(current_z) / 3.0, 0.9)),
            "template": "basket_residual_trade",
            "description": f"Basket coint vector {k}: z={current_z:.2f}, HL={hl_result:.1f}",
        })

    return findings


def _ou_half_life_simple(spread: np.ndarray) -> float:
    """Simple OU half-life estimate."""
    import math
    dx = np.diff(spread)
    x = spread[:-1]
    beta = np.cov(dx, x)[0, 1] / max(x.var(), 1e-10)
    if beta >= 0:
        return float("inf")
    return float(-math.log(2) / beta)
