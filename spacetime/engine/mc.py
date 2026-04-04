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
    Kelly-optimal fraction using the binary Kelly formula.

    Estimates win_rate p and average win/loss ratio b from the return distribution,
    then applies: f* = p - (1-p) / b

    For raw dollar P&L arrays, normalizes by median absolute return first.
    Returns a value in [0, 1].
    """
    if len(returns) == 0:
        return 0.0

    # Normalize raw dollar amounts to fractional returns in [-1, 1]
    max_abs = float(np.max(np.abs(returns)))
    if max_abs > 10.0:
        returns = returns / max_abs

    wins   = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    p   = float(len(wins)) / len(returns)
    q   = 1.0 - p
    b   = float(np.mean(wins)) / float(np.mean(np.abs(losses)))  # avg win / avg loss

    kelly = p - q / b
    return float(np.clip(kelly, 0.0, 1.0))


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

    # Detect if returns are raw dollar P&L (not fractional).
    # When dollar-mode, the simulation adds P&L arithmetically scaled by Kelly;
    # when fractional-mode, it compounds geometrically.
    _dollar_mode = len(all_returns) > 0 and float(np.max(np.abs(all_returns))) > 10.0

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

    # Vectorized simulation: generate all paths at once for speed.
    # AR(1) serial correlation is applied via a post-sampling adjustment
    # on the full matrix using a Markov-style loss-state transition.
    _all_arr = np.array(list(all_returns), dtype=float)
    pos_frac = min(kelly, 1.0) if kelly > 0 else 0.25
    n_total  = max(1, int(trades_per_month * cfg.months))

    # Draw returns: shape (n_sims, n_total)
    draws = rng.choice(_all_arr, size=(cfg.n_sims, n_total), replace=True)

    # Apply AR(1) serial correlation if enabled
    if cfg.regime_aware and cfg.serial_corr > 0 and len(_all_arr) > 1:
        _losses = _all_arr[_all_arr < 0]
        _gains  = _all_arr[_all_arr >= 0]
        _base_p_loss = float(np.mean(_all_arr < 0))
        _all_max_abs = float(np.max(np.abs(_all_arr))) if len(_all_arr) > 0 else 1.0
        if len(_losses) > 0 and len(_gains) > 0:
            for t in range(1, n_total):
                prev_col = draws[:, t - 1]
                loss_mask = prev_col < 0
                if loss_mask.any():
                    # For sims where previous trade was a loss, increase p_loss
                    p_loss = np.where(
                        loss_mask,
                        np.minimum(0.99, _base_p_loss + cfg.serial_corr * np.abs(prev_col) / _all_max_abs),
                        _base_p_loss,
                    )
                    rand_u = rng.random(cfg.n_sims)
                    should_redraw = loss_mask & (rand_u < p_loss - _base_p_loss)
                    if should_redraw.any():
                        n_redraw = int(should_redraw.sum())
                        draws[should_redraw, t] = rng.choice(_losses, size=n_redraw, replace=True)

    # Compute equity paths
    scaled = pos_frac * draws  # (n_sims, n_total)

    if _dollar_mode:
        # Arithmetic: cumulative sum of dollar P&L
        paths = starting_equity + np.cumsum(scaled, axis=1)
    else:
        # Geometric: cumulative product of (1 + fractional return)
        factors = np.clip(1.0 + scaled, 1e-10, None)
        paths = starting_equity * np.cumprod(factors, axis=1)

    # Clamp negative equity to 0
    paths = np.maximum(paths, 0.0)

    # Final equities and per-path max drawdown
    final_equities = paths[:, -1]

    # Max drawdown: running max then (peak - equity) / peak
    running_max = np.maximum.accumulate(paths, axis=1)
    # Prepend starting_equity as the initial peak
    running_max = np.maximum(running_max, starting_equity)
    dd_matrix = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
    max_drawdowns = dd_matrix.max(axis=1)

    # Blowup: any path crossing 0 mid-sim OR final equity below threshold
    blowup_mid_any = (paths == 0.0).any(axis=1)
    blowup_end = final_equities < starting_equity * cfg.blowup_threshold
    blowup_mask = blowup_mid_any | blowup_end
    blowups = int(blowup_mask.sum())

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

    # Detect dollar-mode per instrument
    _dollar_modes = {
        sym: float(np.max(np.abs(arr))) > 10.0
        for sym, arr in returns_per_sym.items()
    }

    for sim_idx in range(n_sims):
        eq   = starting_equity
        peak = eq
        max_dd = 0.0
        blowup_mid = False

        n_steps = int(avg_tpm * months)
        for _ in range(n_steps):
            if eq <= 0:
                blowup_mid = True
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
                if _dollar_modes.get(sym, False):
                    eq_delta += alloc * ret * 0.25  # arithmetic for dollar P&L
                else:
                    eq_delta += alloc * eq * ret * 0.25  # geometric for fractional

            eq += eq_delta
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        final_equities[sim_idx] = max(0.0, eq)
        max_drawdowns[sim_idx]  = max_dd
        if blowup_mid or eq < starting_equity * 0.10:
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
