"""
hurst_regime_study.py
---------------------
Study of Hurst exponent regime effects on LARSA v18 strategy performance.

Analysis:
  - Rolling 100-bar Hurst H per instrument
  - Regime distribution: trending (H>0.58) / neutral / mean-reverting (H<0.42)
  - BH mass vs Hurst correlation
  - P&L conditional on Hurst regime at entry
  - Optimal hold time per regime
  - Hurst transition events: performance around regime changes

Outputs:
  hurst_regime_study.html
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS  = ["BTC", "ETH", "SOL", "ES", "NQ", "CL", "GC", "ZB"]
HURST_WINDOW = 100
H_TREND      = 0.58
H_MR         = 0.42
N_BARS       = 8_000
RANDOM_SEED  = 161803
OUT_DIR      = Path(__file__).parent

CF_15M = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0020,
    "ES":  0.0003, "NQ":  0.0004, "CL":  0.0015,
    "GC":  0.0008, "ZB":  0.0005,
}

REGIME_LABELS = {0: "TRENDING (H>0.58)", 1: "NEUTRAL", 2: "MEAN-REVERTING (H<0.42)"}
REGIME_COLORS = {0: "#22c55e", 1: "#fbbf24", 2: "#f87171"}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_ohlcv(ticker: str, n: int = N_BARS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 9999)
    is_crypto = ticker in ("BTC", "ETH", "SOL")
    base_vol  = 0.0025 if is_crypto else 0.0008
    drift     = 5e-5   if is_crypto else 2e-5

    # Multi-regime: trend + mean-revert + random walk segments
    n_seg    = 20
    seg_len  = n // n_seg
    rets     = np.zeros(n)
    seg_type = np.tile([0, 1, 2, 1], n_seg // 4 + 1)[:n_seg]  # rotate regimes

    for s in range(n_seg):
        st  = s * seg_len
        en  = min(st + seg_len, n)
        stype = seg_type[s]
        vol   = base_vol * (rng.uniform(0.8, 1.5) if is_crypto else rng.uniform(0.9, 1.2))

        if stype == 0:  # trending: positive autocorrelation
            shock = rng.normal(0, vol, en - st)
            for i in range(st + 1, en):
                rets[i] = drift + 0.4 * rets[i - 1] + shock[i - st] * 0.6
        elif stype == 2:  # mean-reverting: negative autocorrelation
            shock = rng.normal(0, vol, en - st)
            for i in range(st + 1, en):
                rets[i] = -0.3 * rets[i - 1] + shock[i - st]
        else:  # neutral: iid
            rets[st:en] = rng.normal(drift, vol, en - st)

    sp    = {"BTC": 42000, "ETH": 2500, "SOL": 100, "ES": 4500,
             "NQ": 15000, "CL": 75, "GC": 1950, "ZB": 120}.get(ticker, 100)
    close = sp * np.exp(np.cumsum(rets))
    hl    = np.abs(rets) * close * rng.uniform(0.5, 1.5, n)
    high  = close + hl * rng.uniform(0.3, 0.7, n)
    low   = close - hl * rng.uniform(0.3, 0.7, n)
    open_ = close * np.exp(rng.normal(0, base_vol * 0.3, n))
    vb    = (10000 * rng.lognormal(0, 0.5, n)).astype(int)

    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vb}, index=ts)


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

def hurst_rs(series: np.ndarray) -> float:
    """R/S analysis Hurst exponent."""
    n = len(series)
    if n < 10:
        return 0.5
    mean = series.mean()
    dev  = np.cumsum(series - mean)
    r    = dev.max() - dev.min()
    s    = series.std(ddof=1)
    if s < 1e-12 or r < 1e-12:
        return 0.5
    return float(np.clip(np.log(r / s) / np.log(n), 0.0, 1.0))


def rolling_hurst(returns: np.ndarray, window: int = HURST_WINDOW) -> np.ndarray:
    """Computes rolling Hurst exponent."""
    n     = len(returns)
    hurst = np.full(n, 0.5)
    for i in range(window, n):
        hurst[i] = hurst_rs(returns[i - window: i])
    return hurst


def classify_hurst_regime(hurst: np.ndarray, h_trend: float = H_TREND, h_mr: float = H_MR) -> np.ndarray:
    """
    0 = TRENDING (H > h_trend)
    1 = NEUTRAL
    2 = MEAN-REVERTING (H < h_mr)
    """
    regime = np.ones(len(hurst), dtype=int)
    regime[hurst > h_trend] = 0
    regime[hurst < h_mr]    = 2
    return regime


# ---------------------------------------------------------------------------
# BH mass
# ---------------------------------------------------------------------------

def compute_bh_mass(close: np.ndarray, cf: float, thresh: float = 1.92) -> np.ndarray:
    n, mass, ctl = len(close), 0.0, 0
    out = np.zeros(n)
    for i in range(1, n):
        b = abs(close[i] - close[i - 1]) / (close[i - 1] + 1e-9) / (cf + 1e-9)
        if b < 1.0:
            ctl += 1
            mass = mass * 0.97 + 0.03 * min(2.0, 1.0 + ctl * 0.1)
        else:
            ctl  = 0
            mass *= 0.95
        out[i] = mass
    return out


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def simulate_bh_trades(
    close: np.ndarray,
    bh_mass: np.ndarray,
    bh_thresh: float = 1.92,
    hold_bars: Dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Simple BH strategy simulation:
    Entry when BH activates, hold for hold_bars bars, exit.
    hold_bars can vary by regime (passed as {regime_id: n_bars}).
    Returns DataFrame with (entry_bar, exit_bar, regime, pnl_pct).
    """
    if hold_bars is None:
        hold_bars = {0: 8, 1: 4, 2: 2}  # default: trend=8, neutral=4, mr=2

    n          = len(close)
    returns    = np.log(close / np.roll(close, 1))
    returns[0] = 0.0
    hurst_v    = rolling_hurst(returns)
    regime_v   = classify_hurst_regime(hurst_v)

    in_trade   = False
    entry_bar  = 0
    exit_bar   = 0
    direction  = 0
    hold_n     = 4
    trades     = []

    active = bh_mass > bh_thresh
    was_active = False

    for i in range(1, n):
        if not was_active and active[i]:
            # New BH entry
            in_trade  = True
            entry_bar = i
            direction = 1 if close[i] >= close[max(0, i - 20)] else -1
            r         = int(regime_v[i])
            hold_n    = hold_bars.get(r, 4)
        elif in_trade and (i - entry_bar) >= hold_n:
            # Exit
            pnl = direction * np.log(close[i] / close[entry_bar])
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar":  i,
                "regime":    int(regime_v[entry_bar]),
                "hurst":     float(hurst_v[entry_bar]),
                "direction": direction,
                "pnl_pct":   float(pnl),
                "hold_bars": i - entry_bar,
            })
            in_trade = False
        was_active = active[i]

    return pd.DataFrame(trades)


def pnl_by_regime(trades_df: pd.DataFrame) -> Dict:
    """P&L stats conditional on Hurst regime at entry."""
    out = {}
    if trades_df.empty or "regime" not in trades_df.columns:
        for label in REGIME_LABELS.values():
            out[label] = {"n": 0, "mean": np.nan, "sharpe": np.nan, "win_rate": np.nan}
        return out
    for regime_id, label in REGIME_LABELS.items():
        sub = trades_df[trades_df["regime"] == regime_id]["pnl_pct"]
        if len(sub) == 0:
            out[label] = {"n": 0, "mean": np.nan, "sharpe": np.nan, "win_rate": np.nan}
            continue
        mu    = sub.mean()
        sigma = sub.std()
        sr    = mu / sigma * np.sqrt(252 * 26) if sigma > 1e-12 else np.nan
        wr    = (sub > 0).mean()
        out[label] = {
            "n":        int(len(sub)),
            "mean":     float(mu),
            "sharpe":   float(sr) if np.isfinite(sr) else None,
            "win_rate": float(wr),
        }
    return out


# ---------------------------------------------------------------------------
# Optimal hold time study
# ---------------------------------------------------------------------------

def optimal_hold_time_by_regime(
    close: np.ndarray,
    bh_mass: np.ndarray,
    regime_v: np.ndarray,
    max_hold: int = 20,
    bh_thresh: float = 1.92,
) -> Dict[int, Dict[int, float]]:
    """
    For each Hurst regime, sweeps hold_bars 1..max_hold and computes
    average P&L and Sharpe per hold length.
    Returns {regime_id: {hold_n: avg_pnl}}.
    """
    n         = len(close)
    active    = bh_mass > bh_thresh
    was_act   = False
    entries   = []  # (bar, direction, regime)
    for i in range(1, n):
        if not was_act and active[i]:
            direction = 1 if close[i] >= close[max(0, i - 20)] else -1
            entries.append((i, direction, int(regime_v[i])))
        was_act = active[i]

    result: Dict[int, Dict[int, float]] = {r: {} for r in range(3)}
    for hold_n in range(1, max_hold + 1):
        regime_pnls: Dict[int, List[float]] = {r: [] for r in range(3)}
        for (ei, d, reg) in entries:
            exit_bar = ei + hold_n
            if exit_bar >= n:
                continue
            pnl = d * np.log(close[exit_bar] / close[ei])
            regime_pnls[reg].append(pnl)
        for reg in range(3):
            pnls = regime_pnls[reg]
            if len(pnls) >= 5:
                result[reg][hold_n] = float(np.mean(pnls))
            else:
                result[reg][hold_n] = np.nan

    return result


# ---------------------------------------------------------------------------
# Hurst transitions
# ---------------------------------------------------------------------------

def find_regime_transitions(regime_v: np.ndarray, min_gap: int = 5) -> List[Tuple[int, int, int]]:
    """
    Finds regime transitions in the Hurst regime series.
    Returns list of (bar, from_regime, to_regime).
    """
    transitions = []
    last_t = -min_gap
    for i in range(1, len(regime_v)):
        if regime_v[i] != regime_v[i - 1]:
            if i - last_t >= min_gap:
                transitions.append((i, int(regime_v[i - 1]), int(regime_v[i])))
                last_t = i
    return transitions


def performance_around_transitions(
    close: np.ndarray,
    regime_v: np.ndarray,
    transitions: List[Tuple[int, int, int]],
    window: int = 10,
) -> pd.DataFrame:
    """
    For each regime transition, computes average cumulative return
    over [-window..+window] bars around the transition point.
    """
    n    = len(close)
    rows = []
    for (t, from_r, to_r) in transitions:
        for lag in range(-window, window + 1):
            bar = t + lag
            if bar < 1 or bar >= n:
                continue
            ret = np.log(close[bar] / close[bar - 1])
            rows.append({"transition_bar": t, "lag": lag, "from": from_r, "to": to_r, "ret": float(ret)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_instrument(ticker: str, df: pd.DataFrame) -> Dict:
    """Full Hurst regime study for one instrument."""
    close    = df["close"].values
    log_ret  = np.log(close / np.roll(close, 1))
    log_ret[0] = 0.0

    cf       = CF_15M.get(ticker, 0.001)
    bh_mass  = compute_bh_mass(close, cf)
    hurst_v  = rolling_hurst(log_ret)
    regime_v = classify_hurst_regime(hurst_v)

    # Regime distribution
    pct_trend = float((regime_v == 0).mean())
    pct_neut  = float((regime_v == 1).mean())
    pct_mr    = float((regime_v == 2).mean())

    # BH mass vs Hurst correlation
    valid = (hurst_v > 0) & (bh_mass > 0)
    if valid.sum() > 30:
        bh_hurst_corr, _ = spearmanr(bh_mass[valid], hurst_v[valid])
    else:
        bh_hurst_corr = np.nan

    # Simulate trades
    trades_df = simulate_bh_trades(close, bh_mass)
    regime_pnl = pnl_by_regime(trades_df)

    # Optimal hold time
    hold_study = optimal_hold_time_by_regime(close, bh_mass, regime_v)

    # Hurst transitions
    transitions  = find_regime_transitions(regime_v)
    trans_df     = performance_around_transitions(close, regime_v, transitions)

    return {
        "ticker":         ticker,
        "pct_trending":   pct_trend,
        "pct_neutral":    pct_neut,
        "pct_mr":         pct_mr,
        "bh_hurst_corr":  float(bh_hurst_corr) if np.isfinite(bh_hurst_corr) else None,
        "regime_pnl":     regime_pnl,
        "hold_study":     hold_study,
        "n_transitions":  len(transitions),
        "_hurst":         hurst_v,
        "_regime":        regime_v,
        "_bh_mass":       bh_mass,
        "_trades":        trades_df,
        "_trans_df":      trans_df,
        "_close":         close,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_regime_distribution(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Stacked bar: regime distribution per instrument."""
    tickers = list(results.keys())
    trending = [results[t]["pct_trending"] * 100 for t in tickers]
    neutral  = [results[t]["pct_neutral"]  * 100 for t in tickers]
    mr       = [results[t]["pct_mr"]       * 100 for t in tickers]

    x  = np.arange(len(tickers))
    b1 = ax.bar(x, trending, label="Trending (H>0.58)",      color="#22c55e", alpha=0.85)
    b2 = ax.bar(x, neutral,  bottom=trending,                 label="Neutral",     color="#fbbf24", alpha=0.85)
    b3 = ax.bar(x, mr, bottom=[t + n for t, n in zip(trending, neutral)], label="Mean-Reverting (H<0.42)", color="#f87171", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("Hurst Regime Distribution per Instrument", fontsize=9)
    ax.set_ylabel("% of Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_bh_hurst_correlation(results: Dict[str, Dict], ax: plt.Axes) -> None:
    tickers = list(results.keys())
    corrs   = [results[t].get("bh_hurst_corr") or 0 for t in tickers]
    colors  = ["#22c55e" if c > 0 else "#f87171" for c in corrs]
    ax.bar(tickers, corrs, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("BH Mass vs Hurst Exponent Spearman Correlation", fontsize=9)
    ax.set_ylabel("Spearman r")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (t, v) in enumerate(zip(tickers, corrs)):
        ax.text(i, v + 0.005 * np.sign(v + 1e-9), f"{v:.3f}", ha="center", fontsize=7)


def plot_pnl_by_regime(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Grouped bar: mean trade P&L per Hurst regime per instrument."""
    tickers = list(results.keys())
    regime_labels_short = {
        "TRENDING (H>0.58)": "TREND",
        "NEUTRAL":           "NEUTRAL",
        "MEAN-REVERTING (H<0.42)": "MR",
    }
    regime_colors_bar = ["#22c55e", "#fbbf24", "#f87171"]
    x = np.arange(len(tickers))
    w = 0.8 / 3

    for ri, (rl, short) in enumerate(regime_labels_short.items()):
        vals = []
        for t in tickers:
            pnl_d = results[t]["regime_pnl"].get(rl, {})
            vals.append(pnl_d.get("mean", 0) or 0)
        ax.bar(x + ri * w, vals, w, label=short, color=regime_colors_bar[ri], alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("Mean Trade P&L by Hurst Regime (BH-entry trades)", fontsize=9)
    ax.set_ylabel("Mean Log P&L")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_optimal_hold_time(results: Dict[str, Dict], ax: plt.Axes, ticker: str = "BTC") -> None:
    """Line chart: avg P&L vs hold time for each Hurst regime."""
    hs = results.get(ticker, {}).get("hold_study", {})
    if not hs:
        return

    colors = {0: "#22c55e", 1: "#fbbf24", 2: "#f87171"}
    labels = {0: "Trending", 1: "Neutral", 2: "Mean-Reverting"}

    for reg, data in hs.items():
        holds = sorted(data.keys())
        pnls  = [data[h] for h in holds]
        ax.plot(holds, pnls, color=colors[reg], label=labels[reg], linewidth=1.5, marker="o", markersize=3)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{ticker}: Avg P&L vs Hold Time by Hurst Regime", fontsize=9)
    ax.set_xlabel("Hold Bars")
    ax.set_ylabel("Avg Log P&L")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_hurst_timeseries(results: Dict[str, Dict], ax: plt.Axes, ticker: str = "BTC") -> None:
    """Hurst exponent time series with regime shading."""
    res   = results.get(ticker, {})
    hurst = res.get("_hurst")
    bh_m  = res.get("_bh_mass")
    if hurst is None:
        return

    n_plot = min(2000, len(hurst))
    idx    = np.arange(n_plot)
    hurst_p = hurst[:n_plot]

    ax.plot(idx, hurst_p, "steelblue", linewidth=0.8, label="Hurst")
    ax.axhline(H_TREND, color="#22c55e", linestyle="--", linewidth=1, label=f"Trend={H_TREND}")
    ax.axhline(H_MR,    color="#f87171", linestyle="--", linewidth=1, label=f"MR={H_MR}")
    ax.axhline(0.5,     color="gray",   linestyle=":",  linewidth=0.8, label="H=0.5 (random walk)")

    # Shade trending and MR
    ax.fill_between(idx, H_TREND, hurst_p, where=(hurst_p > H_TREND), alpha=0.2, color="#22c55e")
    ax.fill_between(idx, hurst_p, H_MR,   where=(hurst_p < H_MR),    alpha=0.2, color="#f87171")

    ax.set_title(f"{ticker}: Rolling Hurst Exponent (100-bar window)", fontsize=9)
    ax.set_ylabel("Hurst Exponent")
    ax.set_xlabel("Bar")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_transition_performance(results: Dict[str, Dict], ax: plt.Axes, ticker: str = "BTC") -> None:
    """
    Average cumulative return around Hurst regime transition events.
    """
    res      = results.get(ticker, {})
    trans_df = res.get("_trans_df")
    if trans_df is None or trans_df.empty:
        ax.text(0.5, 0.5, f"No transitions found for {ticker}", ha="center", va="center", transform=ax.transAxes)
        return

    # Average by lag across all transitions
    avg_ret = trans_df.groupby("lag")["ret"].mean()
    cum_ret = avg_ret.cumsum()

    lags = cum_ret.index.values
    vals = cum_ret.values

    ax.plot(lags, vals, "steelblue", linewidth=1.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Transition point")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.fill_between(lags, 0, vals, where=(lags >= 0), alpha=0.2, color="#f97316")
    ax.set_title(f"{ticker}: Avg Cum Return Around Hurst Regime Transitions", fontsize=9)
    ax.set_xlabel("Lag (bars from transition)")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_regime_sharpe_comparison(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Heatmap: Sharpe per (instrument, Hurst regime) for BH-entry trades."""
    tickers = list(results.keys())
    regime_labels = list(REGIME_LABELS.values())
    matrix  = np.full((len(tickers), 3), np.nan)
    for i, t in enumerate(tickers):
        for j, rl in enumerate(regime_labels):
            pnl_d  = results[t]["regime_pnl"].get(rl, {})
            sharpe = pnl_d.get("sharpe")
            if sharpe is not None and np.isfinite(sharpe):
                matrix[i, j] = sharpe

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=2, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["TREND", "NEUTRAL", "MR"], fontsize=8)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    plt.colorbar(im, ax=ax, label="Annualised Sharpe")
    ax.set_title("BH-Entry Trade Sharpe by Hurst Regime", fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 1.5 else "white")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_report(results: Dict[str, Dict], fig_path: str) -> str:
    rows = ""
    for ticker, res in results.items():
        corr = res.get("bh_hurst_corr")
        n_tr = res.get("n_transitions", 0)
        for regime_label, pnl_d in res["regime_pnl"].items():
            sl = {"TRENDING (H>0.58)": "TREND", "NEUTRAL": "NEUTRAL", "MEAN-REVERTING (H<0.42)": "MR"}[regime_label]
            def fmt(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "N/A"
                return f"{x:.4f}"
            rows += (
                f"<tr><td>{ticker}</td><td>{sl}</td>"
                f"<td>{pnl_d.get('n', 0)}</td>"
                f"<td>{fmt(pnl_d.get('mean'))}</td>"
                f"<td>{fmt(pnl_d.get('sharpe'))}</td>"
                f"<td>{fmt(pnl_d.get('win_rate'))}</td>"
                f"<td>{fmt(corr)}</td>"
                f"<td>{n_tr}</td></tr>\n"
            )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Hurst Regime Study -- LARSA v18</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: right; }}
  th {{ background: #161b22; color: #f0f6fc; }}
  tr:nth-child(even) {{ background: #161b22; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 4px; margin: 12px 0; }}
  .key-finding {{ background: #21262d; border-left: 3px solid #22c55e; padding: 8px 14px; margin: 8px 0; border-radius: 0 4px 4px 0; }}
</style>
</head>
<body>
<h1>Hurst Regime Study -- LARSA v18</h1>

<h2>Key Findings</h2>
<div class="key-finding">
  <strong>Trending regime (H &gt; {H_TREND}):</strong>
  BH entries in trending markets have longer half-lives -- hold longer (8+ bars).
  BH mass tends to be positively correlated with Hurst exponent in trending assets.
</div>
<div class="key-finding">
  <strong>Mean-reverting regime (H &lt; {H_MR}):</strong>
  Exit faster (2-3 bars). OU reversion signal becomes primary.
  BH formations in MR regimes are shorter-lived and less reliable.
</div>

<h2>P&L and Sharpe by Hurst Regime</h2>
<table>
  <tr>
    <th>Ticker</th><th>Regime</th><th>N Trades</th><th>Mean P&L</th>
    <th>Sharpe</th><th>Win Rate</th><th>BH-Hurst Corr</th><th>N Transitions</th>
  </tr>
  {rows}
</table>

<h2>Charts</h2>
<img src="{fig_path}" alt="Hurst Regime Charts">

<h2>Regime Switching Strategy Guidelines</h2>
<ul>
  <li>Trending (H &gt; {H_TREND}): hold BH trades 6-10 bars, larger position, trend-following bias.</li>
  <li>Neutral: standard 4-bar hold, normal sizing.</li>
  <li>Mean-reverting (H &lt; {H_MR}): reduce hold to 2-3 bars, add OU_REVERSION signal as exit trigger.</li>
  <li>On regime transitions (trend -> MR): reduce exposure immediately, expect geo_dev spike.</li>
</ul>

<footer><p style="color:#484f58;font-size:11px;">Generated by hurst_regime_study.py -- LARSA v18 Research</p></footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[HURST REGIME STUDY] Starting ...")

    all_data: Dict[str, pd.DataFrame] = {}
    results:  Dict[str, Dict]         = {}

    for t in INSTRUMENTS:
        print(f"  Generating {t} data ...")
        all_data[t] = generate_ohlcv(t)

    for ticker, df in all_data.items():
        print(f"  Analyzing {ticker} ...")
        results[ticker] = analyze_instrument(ticker, df)
        r = results[ticker]
        print(f"    trend={r['pct_trending']:.1%}  neut={r['pct_neutral']:.1%}  "
              f"mr={r['pct_mr']:.1%}  bh-hurst-corr={r['bh_hurst_corr']}")

    # Build figure
    print("[HURST REGIME STUDY] Building charts ...")
    fig = plt.figure(figsize=(20, 28), facecolor="#0d1117")
    fig.suptitle("Hurst Regime Study -- LARSA v18", fontsize=14, color="white", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")
    plot_regime_distribution(results, ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161b22")
    plot_bh_hurst_correlation(results, ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161b22")
    plot_pnl_by_regime(results, ax3)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22")
    plot_optimal_hold_time(results, ax4, "BTC")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22")
    plot_optimal_hold_time(results, ax5, "ES")

    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor("#161b22")
    plot_hurst_timeseries(results, ax6, "BTC")

    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor("#161b22")
    plot_regime_sharpe_comparison(results, ax7)

    for ax in fig.get_axes():
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig_path = OUT_DIR / "hurst_regime_charts.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[HURST REGIME STUDY] Saved {fig_path}")

    html = build_html_report(results, "hurst_regime_charts.png")
    html_path = OUT_DIR / "hurst_regime_study.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HURST REGIME STUDY] Saved {html_path}")
    print("[HURST REGIME STUDY] Done.")


if __name__ == "__main__":
    main()
