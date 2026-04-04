"""
mc.py — Regime-aware Monte Carlo engine for Spacetime Arena.

Features:
  - Classify trades by regime at entry (BULL/BEAR/SIDEWAYS/HIGH_VOL)
  - Separate return distributions per regime
  - AR(1) serial correlation in losses (configurable, default 0.3)
  - 10K paths default
  - Kelly-optimal sizing
  - Portfolio MC with configurable cross-instrument correlation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOL")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    n_sims: int = 10_000
    months: int = 12
    serial_corr: float = 0.3        # AR(1) loss correlation coefficient
    blowup_threshold: float = 0.10  # equity < 10% of start = blowup
    regime_aware: bool = True


@dataclass
class MCPathStats:
    final_equity: float
    max_drawdown: float
    blowup: bool


@dataclass
class MCResult:
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    blowup_rate: float
    median_equity: float
    mean_equity: float
    pct_5:  float
    pct_25: float
    pct_75: float
    pct_95: float
    trades_per_month: float
    kelly_fraction: float
    regime_stats: Dict[str, Dict[str, float]]  # per-regime win_rate, avg_return, count


# ---------------------------------------------------------------------------
# Trade record duck-typing
# ---------------------------------------------------------------------------

def _trade_return(t: object) -> float:
    """Extract normalized return from a trade record (dict or dataclass)."""
    if isinstance(t, dict):
        pnl     = t.get("pnl", 0.0)
        pos_val = t.get("dollar_pos") or t.get("entry_value") or 0.0
        if pos_val and pos_val != 0:
            return pnl / pos_val
        return pnl
    pnl = getattr(t, "pnl", 0.0)
    ep  = getattr(t, "entry_price", None)
    frac = getattr(t, "pos_frac", None) or getattr(t, "position_frac", None) or 1.0
    if ep and ep > 0:
        return pnl / (ep * frac) if frac else pnl
    return float(pnl)


def _trade_regime(t: object) -> str:
    """Extract regime string from trade record."""
    if isinstance(t, dict):
        return str(t.get("regime", "SIDEWAYS"))
    return str(getattr(t, "regime", "SIDEWAYS"))


def _trade_exit_time(t: object) -> object:
    if isinstance(t, dict):
        return t.get("exit_time") or t.get("entry_time")
    return getattr(t, "exit_time", None) or getattr(t, "entry_time", None)


# ---------------------------------------------------------------------------
# Kelly optimal fraction
# ---------------------------------------------------------------------------

def compute_kelly(returns: np.ndarray) -> float:
    """
    Kelly-optimal fraction: find f* that maximizes E[log(1 + f*r)] over all returns.
    Uses scipy minimize_scalar on the bounded interval [0, 2].
    """
    if len(returns) == 0:
        return 0.0

    from scipy.optimize import minimize_scalar  # type: ignore

    def neg_log_growth(f: float) -> float:
        vals = 1.0 + f * returns
        vals = np.where(vals <= 0, 1e-10, vals)
        return -float(np.mean(np.log(vals)))

    try:
        result = minimize_scalar(neg_log_growth, bounds=(0.0, 2.0), method="bounded")
        return float(np.clip(result.x, 0.0, 1.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Regime classifier from trade list
# ---------------------------------------------------------------------------

def classify_by_regime(trades: List[object]) -> Dict[str, List[float]]:
    """Group trade returns by entry regime."""
    buckets: Dict[str, List[float]] = {r: [] for r in REGIMES}
    for t in trades:
        regime = _trade_regime(t).upper()
        # Map HIGH_VOLATILITY → HIGH_VOL etc.
        if "HIGH" in regime or "VOL" in regime:
            regime = "HIGH_VOL"
        elif regime not in REGIMES:
            regime = "SIDEWAYS"
        ret = _trade_return(t)
        buckets[regime].append(float(ret))
    return buckets


# ---------------------------------------------------------------------------
# Single-instrument MC
# ---------------------------------------------------------------------------

def run_mc(
    trades: List[object],
    starting_equity: float = 1_000_000.0,
    cfg: Optional[MCConfig] = None,
) -> MCResult:
    """
    Run Monte Carlo simulation on a list of trade records.

    Parameters
    ----------
    trades           : list of TradeRecord, dicts, or any object with pnl/regime attrs
    starting_equity  : starting portfolio value
    cfg              : MCConfig (defaults if None)
    """
    if cfg is None:
        cfg = MCConfig()

    if len(trades) < 5:
        raise ValueError(f"Too few trades for MC: {len(trades)}")

    # Estimate trades per month
    import pandas as pd
    times = [_trade_exit_time(t) for t in trades]
    try:
        ts = pd.to_datetime([str(x)[:19] for x in times if x is not None])
        span_months = max(1, (ts.max() - ts.min()).days / 30)
    except Exception:
        span_months = max(1, len(trades) / 10)
    trades_per_month = len(trades) / span_months

    # Build regime-stratified return distributions
    regime_buckets = classify_by_regime(trades)
    all_returns = np.array([_trade_return(t) for t in trades], dtype=float)

    # Kelly fraction from all returns
    kelly = compute_kelly(all_returns)

    # Regime stats
    regime_stats: Dict[str, Dict[str, float]] = {}
    for r, rets in regime_buckets.items():
        if rets:
            arr = np.array(rets)
            regime_stats[r] = {
                "count": float(len(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "avg_return": float(np.mean(arr)),
                "std_return": float(np.std(arr)),
            }

    rng = np.random.default_rng(42)

    final_equities = np.zeros(cfg.n_sims)
    max_drawdowns  = np.zeros(cfg.n_sims)
    blowups = 0

    # Regime transition matrix (simplified Markov)
    regime_list = [_trade_regime(t).upper() for t in trades]
    regime_list = ["HIGH_VOL" if ("HIGH" in r or "VOL" in r) else r if r in REGIMES else "SIDEWAYS"
                   for r in regime_list]
    regime_seq_for_sim = regime_list if len(regime_list) > 10 else None

    for sim_idx in range(cfg.n_sims):
        eq   = starting_equity
        peak = eq
        max_dd = 0.0

        # Determine regime order for this sim
        n_trades = max(1, int(rng.normal(trades_per_month, math.sqrt(trades_per_month + 1e-9))))
        n_total  = int(n_trades * cfg.months)

        # AR(1) loss state
        prev_loss = False
        prev_loss_val = 0.0

        for _ in range(n_total):
            if eq <= 0:
                blowups += 1
                break

            # Sample regime: use actual sequence if available, else random
            if regime_seq_for_sim and cfg.regime_aware:
                regime = rng.choice(regime_seq_for_sim)
            else:
                regime = "SIDEWAYS"

            bucket = regime_buckets.get(regime, [])
            if not bucket:
                bucket = list(all_returns)

            # AR(1): if previous trade was a loss, bias toward drawing another loss
            if cfg.regime_aware and prev_loss and len(bucket) > 1:
                losses_arr = np.array([r for r in bucket if r < 0])
                gains_arr  = np.array([r for r in bucket if r >= 0])
                if len(losses_arr) > 0 and len(gains_arr) > 0:
                    p_loss = min(0.99, (sum(1 for r in bucket if r < 0) / len(bucket))
                                 + cfg.serial_corr * prev_loss_val)
                    if rng.random() < p_loss:
                        ret = float(rng.choice(losses_arr))
                    else:
                        ret = float(rng.choice(gains_arr))
                else:
                    ret = float(rng.choice(bucket))
            else:
                ret = float(rng.choice(bucket))

            pos_frac = min(kelly, 1.0) if kelly > 0 else 0.25
            eq += pos_frac * eq * ret

            if ret < 0:
                prev_loss     = True
                prev_loss_val = abs(ret)
            else:
                prev_loss     = False
                prev_loss_val = 0.0

            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        final_equities[sim_idx] = max(0.0, eq)
        max_drawdowns[sim_idx]  = max_dd
        if eq < starting_equity * cfg.blowup_threshold:
            blowups += 1

    blowup_rate = blowups / cfg.n_sims

    return MCResult(
        final_equities=final_equities,
        max_drawdowns=max_drawdowns,
        blowup_rate=blowup_rate,
        median_equity=float(np.median(final_equities)),
        mean_equity=float(np.mean(final_equities)),
        pct_5=float(np.percentile(final_equities, 5)),
        pct_25=float(np.percentile(final_equities, 25)),
        pct_75=float(np.percentile(final_equities, 75)),
        pct_95=float(np.percentile(final_equities, 95)),
        trades_per_month=trades_per_month,
        kelly_fraction=kelly,
        regime_stats=regime_stats,
    )


# ---------------------------------------------------------------------------
# Portfolio MC
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMCResult:
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    blowup_rate: float
    median_equity: float
    pct_5: float
    pct_95: float
    per_instrument: Dict[str, MCResult]


def run_portfolio_mc(
    trade_lists: Dict[str, List[object]],
    starting_equity: float = 1_000_000.0,
    n_sims: int = 10_000,
    months: int = 12,
    cross_corr: float = 0.3,
) -> PortfolioMCResult:
    """
    Portfolio Monte Carlo: simulate across multiple instruments simultaneously.

    Parameters
    ----------
    trade_lists  : {sym: [trades]} dict
    starting_equity  : total portfolio starting equity
    n_sims           : number of simulations
    months           : forward projection window
    cross_corr       : correlation between instrument returns (0=independent)
    """
    syms = list(trade_lists.keys())
    n = len(syms)
    if n == 0:
        raise ValueError("No instruments provided")

    # Per-instrument single MC for stats
    per_inst: Dict[str, MCResult] = {}
    for sym, tl in trade_lists.items():
        if len(tl) >= 5:
            try:
                per_inst[sym] = run_mc(tl, starting_equity / n,
                                        MCConfig(n_sims=min(1000, n_sims), months=months))
            except Exception as e:
                logger.warning("MC failed for %s: %s", sym, e)

    # Build return distributions
    returns_per_sym = {}
    for sym, tl in trade_lists.items():
        arr = np.array([_trade_return(t) for t in tl], dtype=float)
        if len(arr) > 0:
            returns_per_sym[sym] = arr

    if not returns_per_sym:
        raise ValueError("No valid trade returns found")

    # Cholesky for correlated sampling
    active_syms = list(returns_per_sym.keys())
    n_active = len(active_syms)
    corr_matrix = np.full((n_active, n_active), cross_corr)
    np.fill_diagonal(corr_matrix, 1.0)

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(n_active)

    rng = np.random.default_rng(42)
    alloc = 1.0 / n_active

    final_equities = np.zeros(n_sims)
    max_drawdowns  = np.zeros(n_sims)
    blowups = 0

    import pandas as pd
    trades_per_month_vals = []
    for sym in active_syms:
        tl = trade_lists[sym]
        times = [_trade_exit_time(t) for t in tl]
        try:
            ts = pd.to_datetime([str(x)[:19] for x in times if x is not None])
            span = max(1, (ts.max() - ts.min()).days / 30)
        except Exception:
            span = max(1, len(tl) / 10)
        trades_per_month_vals.append(len(tl) / span)
    avg_tpm = float(np.mean(trades_per_month_vals)) if trades_per_month_vals else 10.0

    for sim_idx in range(n_sims):
        eq   = starting_equity
        peak = eq
        max_dd = 0.0

        n_steps = int(avg_tpm * months)
        for _ in range(n_steps):
            if eq <= 0:
                blowups += 1
                break

            # Correlated sampling
            z = rng.standard_normal(n_active)
            z_corr = L @ z

            eq_delta = 0.0
            for i, sym in enumerate(active_syms):
                arr = returns_per_sym[sym]
                # Transform uniform via empirical quantile
                u = float(min(0.9999, max(0.0001, 0.5 + 0.5 * math.tanh(z_corr[i] / 1.5))))
                idx = int(u * len(arr))
                ret = float(arr[np.argsort(arr)[idx]])
                eq_delta += alloc * eq * ret * 0.25  # modest position sizing

            eq += eq_delta
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        final_equities[sim_idx] = max(0.0, eq)
        max_drawdowns[sim_idx]  = max_dd
        if eq < starting_equity * 0.10:
            blowups += 1

    blowup_rate = blowups / n_sims

    return PortfolioMCResult(
        final_equities=final_equities,
        max_drawdowns=max_drawdowns,
        blowup_rate=blowup_rate,
        median_equity=float(np.median(final_equities)),
        pct_5=float(np.percentile(final_equities, 5)),
        pct_95=float(np.percentile(final_equities, 95)),
        per_instrument=per_inst,
    )
