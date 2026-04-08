"""
Market microstructure models.

Implements:
  - Bid-ask spread decomposition (Roll model)
  - Kyle's lambda (price impact from order flow)
  - Easley-Lopez de Prado-O'Hara VPIN model
  - Price impact decay (Bouchaud power law)
  - Order book imbalance models
  - Toxic flow detection (alpha, adverse selection)
  - Trade classification (tick rule, Lee-Ready, bulk classification)
  - Realized spread and effective spread
  - PIN (probability of informed trading) estimation
  - Market depth and resilience
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Trade Classification ──────────────────────────────────────────────────────

def tick_rule(prices: np.ndarray) -> np.ndarray:
    """
    Tick rule trade classification: +1 (uptick = buy), -1 (downtick = sell).
    Ties: carry forward previous classification.
    """
    T = len(prices)
    direction = np.zeros(T)
    direction[0] = 1
    for t in range(1, T):
        diff = prices[t] - prices[t - 1]
        if diff > 0:
            direction[t] = 1
        elif diff < 0:
            direction[t] = -1
        else:
            direction[t] = direction[t - 1]
    return direction


def lee_ready(
    prices: np.ndarray,
    midpoints: np.ndarray,
) -> np.ndarray:
    """
    Lee-Ready trade classification.
    If price > midpoint → buy, price < midpoint → sell, else tick rule.
    """
    direction = np.zeros(len(prices))
    tick = tick_rule(prices)
    for t in range(len(prices)):
        if prices[t] > midpoints[t]:
            direction[t] = 1
        elif prices[t] < midpoints[t]:
            direction[t] = -1
        else:
            direction[t] = tick[t]
    return direction


def bulk_volume_classification(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_buckets: int = 50,
) -> np.ndarray:
    """
    Bulk Volume Classification (Easley et al.): classifies each price bar's volume.
    Returns buy volume fraction per bar.
    """
    T = len(prices)
    buy_frac = np.zeros(T)

    for t in range(1, T):
        dp = prices[t] - prices[t - 1]
        sigma = abs(dp)
        if sigma < 1e-10:
            buy_frac[t] = 0.5
        else:
            # Z = dp/sigma buckets from -0.5 to +0.5 → buy fraction via normal CDF
            z = dp / sigma
            from scipy.stats import norm
            buy_frac[t] = float(norm.cdf(z))

    return buy_frac


# ── Roll Spread Model ─────────────────────────────────────────────────────────

def roll_spread(prices: np.ndarray) -> dict:
    """
    Roll (1984) spread estimator from transaction prices.
    Spread = 2 * sqrt(max(-cov(dp_t, dp_{t-1}), 0))
    """
    dp = np.diff(prices)
    n = len(dp)
    if n < 2:
        return {"spread": 0.0, "cov": 0.0}

    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    spread = 2 * math.sqrt(max(-cov, 0))

    return {
        "spread": float(spread),
        "serial_covariance": float(cov),
        "midpoint_var": float(dp.var() + 2 * cov),
        "adverse_selection_frac": float(1 - min((-cov) / (dp.var() / 4 + 1e-10), 1)) if cov < 0 else 0.5,
    }


def realized_spread(
    prices: np.ndarray,
    directions: np.ndarray,
    midpoints: np.ndarray,
    lag: int = 5,
) -> dict:
    """
    Realized spread and price impact decomposition.
    Effective half-spread = direction * (price - midpoint)
    Realized half-spread = direction * (price - midpoint_at_t+lag)
    Price impact = effective - realized
    """
    n = len(prices)
    if n <= lag:
        return {"effective_spread": 0.0, "realized_spread": 0.0, "price_impact": 0.0}

    eff_half = directions[:n-lag] * (prices[:n-lag] - midpoints[:n-lag])
    real_half = directions[:n-lag] * (prices[:n-lag] - midpoints[lag:n])

    return {
        "effective_spread": float(2 * eff_half.mean()),
        "realized_spread": float(2 * real_half.mean()),
        "price_impact": float(2 * (eff_half - real_half).mean()),
        "adverse_selection_frac": float((eff_half - real_half).mean() / (eff_half.mean() + 1e-10)),
    }


# ── Kyle Lambda ───────────────────────────────────────────────────────────────

def kyle_lambda_estimate(
    price_changes: np.ndarray,
    order_flow_imbalance: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """
    Rolling Kyle's lambda: dp = lambda * OFI + noise.
    Higher lambda = less liquid, more price impact per unit flow.
    """
    T = len(price_changes)
    lambdas = np.zeros(T)

    for t in range(window, T):
        dp = price_changes[t - window: t]
        ofi = order_flow_imbalance[t - window: t]

        # OLS: dp ~ lambda * ofi
        if ofi.var() < 1e-10:
            lambdas[t] = 0.0
            continue
        lam = float(np.cov(dp, ofi)[0, 1] / ofi.var())
        lambdas[t] = lam

    return lambdas


# ── VPIN ──────────────────────────────────────────────────────────────────────

def vpin_detailed(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: Optional[float] = None,
    n_buckets_window: int = 50,
) -> dict:
    """
    Volume-Synchronized PIN (VPIN) — Easley, Lopez de Prado, O'Hara (2012).
    Measures toxicity of order flow. High VPIN → informed trading present.
    """
    T = len(prices)
    if bucket_size is None:
        bucket_size = float(volumes.mean())

    buy_frac = bulk_volume_classification(prices, volumes)

    # Build volume buckets
    buy_vol = volumes * buy_frac
    sell_vol = volumes * (1 - buy_frac)

    # VPIN: |buy - sell| / bucket_size averaged over window
    abs_imbalance = np.abs(buy_vol - sell_vol)
    vpin_series = np.zeros(T)

    for t in range(n_buckets_window, T):
        window_imbalance = abs_imbalance[t - n_buckets_window: t]
        window_vol = volumes[t - n_buckets_window: t]
        vpin_series[t] = float(window_imbalance.sum() / max(window_vol.sum(), 1))

    return {
        "vpin": vpin_series,
        "current_vpin": float(vpin_series[-1]),
        "vpin_90pct": float(np.percentile(vpin_series[n_buckets_window:], 90)),
        "is_toxic": bool(vpin_series[-1] > np.percentile(vpin_series[n_buckets_window:], 75)),
        "toxicity_percentile": float(
            100 * np.mean(vpin_series[n_buckets_window:] <= vpin_series[-1])
        ),
    }


# ── Order Book Imbalance ──────────────────────────────────────────────────────

def order_book_imbalance(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    levels: int = 5,
    depth_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Order book imbalance across levels.
    OBI = sum(w_i * bid_i - w_i * ask_i) / sum(w_i * (bid_i + ask_i))

    Returns OBI series: +1 = buy pressure, -1 = sell pressure.
    """
    if depth_weights is None:
        depth_weights = 1.0 / np.arange(1, levels + 1)  # decay by depth

    T = len(bid_sizes) // levels
    obi = np.zeros(T)

    for t in range(T):
        bids = bid_sizes[t * levels: (t + 1) * levels]
        asks = ask_sizes[t * levels: (t + 1) * levels]
        w = depth_weights[:len(bids)]
        num = float(np.sum(w * bids - w * asks))
        denom = float(np.sum(w * (bids + asks)))
        obi[t] = num / max(denom, 1e-10)

    return obi


# ── Price Impact Decay (Bouchaud) ─────────────────────────────────────────────

def price_impact_decay(
    order_size: float,
    daily_volume: float,
    sigma_daily: float,
    decay_exponent: float = 0.5,
    decay_time: float = 1.0,    # time units for impact to decay
    t: float = 0.1,             # time since order
) -> dict:
    """
    Bouchaud et al. power-law impact decay model.
    Impact(t) = Impact_0 * (1 - (t/tau)^alpha)
    where alpha = decay_exponent, tau = decay_time.
    """
    participation = order_size / max(daily_volume, 1)
    # Initial impact (square-root model)
    impact_0 = sigma_daily * math.sqrt(participation)

    # Decay
    decay = max(1 - (min(t, decay_time) / decay_time)**decay_exponent, 0)
    current_impact = impact_0 * decay

    # Transient vs permanent
    permanent_frac = 0.5  # roughly half of impact is permanent (Bouchaud)
    permanent_impact = impact_0 * permanent_frac
    transient_impact = impact_0 * (1 - permanent_frac) * decay

    return {
        "initial_impact": float(impact_0),
        "current_impact": float(current_impact),
        "permanent_impact": float(permanent_impact),
        "transient_impact": float(transient_impact),
        "decay_fraction": float(1 - decay),
        "impact_bps": float(impact_0 * 10000),
    }


# ── PIN Model (Glosten-Milgrom) ───────────────────────────────────────────────

def pin_estimate(
    buy_counts: np.ndarray,
    sell_counts: np.ndarray,
    n_iter: int = 50,
) -> dict:
    """
    PIN (Probability of Informed Trading) via EM estimation.
    Model: alpha = prob informed day, delta = prob bad news, mu = informed arrival rate, eps = uninformed rate.
    Returns estimated (alpha, delta, mu, epsilon, PIN).
    """
    # Initialize
    alpha = 0.2
    delta = 0.5
    mu = float(max(buy_counts.mean(), sell_counts.mean()) * 0.1)
    eps = float(min(buy_counts.mean(), sell_counts.mean()) * 0.5)

    n = len(buy_counts)

    for _ in range(n_iter):
        # E-step: classify each day as informed or uninformed
        # P(day_t | good news) = P(B_t | mu+eps) * P(S_t | eps)
        # P(day_t | bad news)  = P(B_t | eps) * P(S_t | mu+eps)
        # P(day_t | no info)   = P(B_t | eps) * P(S_t | eps)

        def log_poisson(k, lam):
            if lam <= 0:
                return -1e10
            return k * math.log(lam + 1e-10) - lam - math.lgamma(k + 1)

        ll_good = np.array([log_poisson(buy_counts[t], mu + eps) + log_poisson(sell_counts[t], eps)
                            for t in range(n)])
        ll_bad  = np.array([log_poisson(buy_counts[t], eps) + log_poisson(sell_counts[t], mu + eps)
                            for t in range(n)])
        ll_none = np.array([log_poisson(buy_counts[t], eps) + log_poisson(sell_counts[t], eps)
                            for t in range(n)])

        log_prior_good = math.log(alpha * (1 - delta) + 1e-10)
        log_prior_bad  = math.log(alpha * delta + 1e-10)
        log_prior_none = math.log(1 - alpha + 1e-10)

        # Posterior (unnormalized log)
        log_post_good = ll_good + log_prior_good
        log_post_bad  = ll_bad + log_prior_bad
        log_post_none = ll_none + log_prior_none

        # Softmax
        log_posts = np.column_stack([log_post_good, log_post_bad, log_post_none])
        log_posts -= log_posts.max(axis=1, keepdims=True)
        posts = np.exp(log_posts)
        posts /= posts.sum(axis=1, keepdims=True)

        P_good = posts[:, 0]
        P_bad  = posts[:, 1]
        P_none = posts[:, 2]

        # M-step: update parameters
        alpha_new = float((P_good + P_bad).mean())
        delta_new = float(P_bad.sum() / max((P_good + P_bad).sum(), 1e-10))
        mu_new = float(
            (P_good * buy_counts + P_bad * sell_counts).sum() /
            max((P_good + P_bad).sum(), 1e-10)
        )
        eps_new = float(
            (P_good * sell_counts + P_bad * buy_counts + P_none * (buy_counts + sell_counts)).sum() /
            max(2 * n, 1)
        )

        alpha = max(min(alpha_new, 0.99), 0.01)
        delta = max(min(delta_new, 0.99), 0.01)
        mu    = max(mu_new, 0.01)
        eps   = max(eps_new, 0.01)

    pin = alpha * mu / (alpha * mu + 2 * eps)

    return {
        "alpha": float(alpha),
        "delta": float(delta),
        "mu": float(mu),
        "epsilon": float(eps),
        "pin": float(pin),
        "interpretation": (
            "high_pin_informed_trading" if pin > 0.25
            else "moderate_informed" if pin > 0.15
            else "low_pin_uninformed"
        ),
    }


# ── Market Depth and Resilience ───────────────────────────────────────────────

def market_depth_score(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    mid: float,
    depth_bps: float = 50,
) -> dict:
    """
    Measure market depth within a price range of depth_bps from mid.
    Returns total size and weighted-average depth.
    """
    bid_range = mid * (1 - depth_bps / 10000)
    ask_range = mid * (1 + depth_bps / 10000)

    bid_mask = bid_prices >= bid_range
    ask_mask = ask_prices <= ask_range

    bid_depth = float(bid_sizes[bid_mask].sum())
    ask_depth = float(ask_sizes[ask_mask].sum())
    total_depth = bid_depth + ask_depth

    # Resilience: how quickly does depth recover after a trade
    # Proxy: depth distribution - more at midpoint = more resilient
    if bid_sizes.sum() > 0:
        avg_bid_dist = float(np.average(np.abs(bid_prices - mid), weights=bid_sizes + 1e-10))
    else:
        avg_bid_dist = depth_bps * mid / 10000

    resilience = float(1.0 / max(avg_bid_dist / mid * 10000, 1))

    return {
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
        "total_depth_usd": total_depth,
        "depth_imbalance": float((bid_depth - ask_depth) / max(total_depth, 1)),
        "resilience_score": float(min(resilience, 1.0)),
        "depth_bps_range": float(depth_bps),
    }
