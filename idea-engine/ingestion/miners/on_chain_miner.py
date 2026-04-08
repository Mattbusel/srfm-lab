"""
On-chain data miner — discovers opportunities from blockchain metrics.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class OnChainMetrics:
    exchange_inflow: np.ndarray          # coins flowing to exchanges (daily)
    exchange_outflow: np.ndarray         # coins flowing from exchanges
    miner_outflow: np.ndarray            # miner selling
    active_addresses: np.ndarray         # daily active addresses
    transaction_count: np.ndarray        # daily tx count
    nvt_ratio: Optional[np.ndarray] = None   # NVT = market_cap / tx_volume
    sopr: Optional[np.ndarray] = None        # Spent Output Profit Ratio
    nupl: Optional[np.ndarray] = None        # Net Unrealized Profit/Loss
    hash_rate: Optional[np.ndarray] = None   # mining hash rate
    coin_days_destroyed: Optional[np.ndarray] = None


def mine_on_chain_signals(
    metrics: OnChainMetrics,
    prices: np.ndarray,
    window: int = 14,
) -> list[dict]:
    """
    Mine all available on-chain signals from provided metrics.
    Returns ranked list of hypothesis candidates.
    """
    findings = []
    T = len(prices)

    # ── Exchange Flow Signals ─────────────────────────────────────────────────

    net_flow = metrics.exchange_outflow - metrics.exchange_inflow
    n = min(len(net_flow), T)

    if n >= window:
        net_flow_ma = _rolling_mean(net_flow[:n], window)
        net_flow_z = _zscore(net_flow_ma[-window:])

        # Sustained outflows = accumulation
        if net_flow_z[-1] < -1.5:
            findings.append({
                "type": "exchange_outflow_accumulation",
                "score": float(min(abs(net_flow_z[-1]) / 3.0, 1.0)),
                "z_score": float(net_flow_z[-1]),
                "action": "enter_long",
                "template": "exchange_outflow_accumulation",
                "confidence": float(min(abs(net_flow_z[-1]) / 3.0, 0.85)),
                "description": f"Exchange net outflow z={net_flow_z[-1]:.2f} — sustained accumulation",
            })

        # Inflows spike = distribution / sell pressure
        if net_flow_z[-1] > 2.0:
            findings.append({
                "type": "exchange_inflow_distribution",
                "score": float(min(net_flow_z[-1] / 3.0, 1.0)),
                "z_score": float(net_flow_z[-1]),
                "action": "reduce_long",
                "template": "on_chain_distribution",
                "confidence": float(min(net_flow_z[-1] / 3.0, 0.80)),
                "description": f"Exchange inflow spike z={net_flow_z[-1]:.2f} — distribution pressure",
            })

    # ── Miner Capitulation ────────────────────────────────────────────────────

    if len(metrics.miner_outflow) >= window and metrics.hash_rate is not None:
        mo = metrics.miner_outflow
        hr = metrics.hash_rate
        n_m = min(len(mo), len(hr), T)

        mo_ratio = float(mo[-1] / (mo[-window:].mean() + 1e-10))
        hr_change = float((hr[-1] - hr[-window]) / (abs(hr[-window]) + 1e-10))

        if mo_ratio > 2.5 and hr_change < -0.08:
            findings.append({
                "type": "miner_capitulation",
                "score": float(min(mo_ratio / 5.0, 1.0)),
                "miner_outflow_ratio": mo_ratio,
                "hash_rate_change": hr_change,
                "action": "enter_long_contrarian",
                "template": "miner_capitulation",
                "confidence": 0.70,
                "description": f"Miner capitulation: outflow {mo_ratio:.1f}x avg, hash rate {hr_change:.1%}",
            })

    # ── NUPL Extremes ─────────────────────────────────────────────────────────

    if metrics.nupl is not None and len(metrics.nupl) >= 2:
        nupl_val = float(metrics.nupl[-1])

        if nupl_val > 0.75:
            findings.append({
                "type": "nupl_euphoria",
                "score": float(min((nupl_val - 0.75) * 4, 1.0)),
                "nupl": nupl_val,
                "action": "reduce_long_cycle_top",
                "template": "nupl_extreme",
                "confidence": 0.75,
                "description": f"NUPL={nupl_val:.2f} in euphoria zone — cycle top risk",
            })

        elif nupl_val < -0.25:
            findings.append({
                "type": "nupl_capitulation",
                "score": float(min(abs(nupl_val + 0.25) * 4, 1.0)),
                "nupl": nupl_val,
                "action": "enter_long_cycle_bottom",
                "template": "nupl_extreme",
                "confidence": 0.72,
                "description": f"NUPL={nupl_val:.2f} in capitulation zone — cycle bottom signal",
            })

    # ── SOPR Retest ───────────────────────────────────────────────────────────

    if metrics.sopr is not None and len(metrics.sopr) >= window:
        sopr = metrics.sopr
        sopr_val = float(sopr[-1])
        sopr_prev = float(sopr[-2])
        price_trend = float(np.sign(prices[-1] - prices[-min(window, T):][0]))

        # SOPR bouncing off 1.0 in uptrend
        if sopr_prev < 1.0 and sopr_val > 1.0 and price_trend > 0:
            findings.append({
                "type": "sopr_bull_retest",
                "score": 0.65,
                "sopr": sopr_val,
                "action": "enter_long",
                "template": "sopr_retest",
                "confidence": 0.65,
                "description": "SOPR bounced through 1.0 in uptrend — bullish continuation",
            })

        # SOPR failing at 1.0 in downtrend
        elif sopr_prev > 1.0 and sopr_val < 1.0 and price_trend < 0:
            findings.append({
                "type": "sopr_bear_retest",
                "score": 0.60,
                "sopr": sopr_val,
                "action": "enter_short",
                "template": "sopr_retest",
                "confidence": 0.60,
                "description": "SOPR failed 1.0 in downtrend — bearish continuation",
            })

    # ── NVT Signal ────────────────────────────────────────────────────────────

    if metrics.nvt_ratio is not None and len(metrics.nvt_ratio) >= window * 2:
        nvt = metrics.nvt_ratio
        nvt_z = float(_zscore(nvt[-window * 2:])[-1])

        if nvt_z > 2.0:
            findings.append({
                "type": "nvt_overvalued",
                "score": float(min(nvt_z / 3.0, 1.0)),
                "nvt_z": nvt_z,
                "action": "reduce_exposure",
                "template": "on_chain_valuation",
                "confidence": float(min(nvt_z / 3.0, 0.70)),
                "description": f"NVT ratio z={nvt_z:.2f} — network potentially overvalued relative to usage",
            })

        elif nvt_z < -1.5:
            findings.append({
                "type": "nvt_undervalued",
                "score": float(min(abs(nvt_z) / 3.0, 1.0)),
                "nvt_z": nvt_z,
                "action": "add_exposure",
                "template": "on_chain_valuation",
                "confidence": float(min(abs(nvt_z) / 3.0, 0.65)),
                "description": f"NVT ratio z={nvt_z:.2f} — network potentially undervalued",
            })

    # ── Coin Days Destroyed ───────────────────────────────────────────────────

    if metrics.coin_days_destroyed is not None and len(metrics.coin_days_destroyed) >= 90:
        cdd = metrics.coin_days_destroyed
        cdd_90d_avg = float(cdd[-90:].mean())
        cdd_today = float(cdd[-1])
        cdd_multiple = cdd_today / max(cdd_90d_avg, 1e-10)

        if cdd_multiple > 5.0:
            findings.append({
                "type": "old_coins_moving",
                "score": float(min(cdd_multiple / 10.0, 1.0)),
                "cdd_multiple": float(cdd_multiple),
                "action": "caution_distribution",
                "template": "dormant_coins_moving",
                "confidence": 0.65,
                "description": f"CDD spike {cdd_multiple:.1f}x 90d avg — old hands distributing",
            })

    # ── Active Address Divergence ─────────────────────────────────────────────

    if len(metrics.active_addresses) >= window and T >= window:
        aa = metrics.active_addresses[:min(len(metrics.active_addresses), T)]
        n_aa = len(aa)
        if n_aa >= window:
            aa_trend = float(np.polyfit(np.arange(window), aa[-window:], 1)[0] / aa[-window:].mean())
            price_trend_5d = float((prices[-1] - prices[-min(5, T)]) / prices[-min(5, T)])

            # Price rising, addresses declining = potential top
            if price_trend_5d > 0.05 and aa_trend < -0.01:
                findings.append({
                    "type": "address_price_divergence",
                    "score": 0.55,
                    "price_trend_5d": float(price_trend_5d),
                    "address_trend": float(aa_trend),
                    "action": "caution_reduce",
                    "template": "on_chain_divergence",
                    "confidence": 0.55,
                    "description": f"Price up {price_trend_5d:.1%} but active addresses declining",
                })

    return sorted(findings, key=lambda x: x["confidence"], reverse=True)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = x[max(0, i - w + 1): i + 1].mean()
    return result


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    return (x - mu) / max(sigma, 1e-10)
