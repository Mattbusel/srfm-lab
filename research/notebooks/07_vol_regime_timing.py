"""
07_vol_regime_timing.py — Volatility Regime Timing Analysis

- Compute realized volatility (21d rolling) for all instruments
- Classify vol regime: LOW, MID, HIGH, EXTREME
- For each regime: BH formation rate, success rate, avg trade return
- Hawkes process fit: self-excitation in BH formation events
- Outputs: research/outputs/vol_regime_timing.png

Run: python research/notebooks/07_vol_regime_timing.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))

from srfm_core import MinkowskiClassifier, BlackHoleDetector

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "ES":  {"annual_vol": 0.15, "cf": 0.001},
    "NQ":  {"annual_vol": 0.20, "cf": 0.0012},
    "BTC": {"annual_vol": 0.80, "cf": 0.005},
    "GC":  {"annual_vol": 0.12, "cf": 0.008},
    "CL":  {"annual_vol": 0.35, "cf": 0.015},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_with_regimes(sym: str, n: int = 1260, seed: int = 42) -> pd.DataFrame:
    """Generate daily OHLCV with realistic volatility clustering."""
    rng = np.random.default_rng(hash(sym) % 2**32 + seed)
    annual_vol = ASSETS.get(sym, {}).get("annual_vol", 0.15)
    daily_vol  = annual_vol / math.sqrt(252)

    # GARCH(1,1) volatility
    omega = 1e-5; alpha_g = 0.10; beta_g = 0.85
    h = daily_vol**2
    closes = np.empty(n)
    closes[0] = 4500.0
    vols = np.empty(n)
    for i in range(n):
        vols[i] = math.sqrt(h)
        ret = 0.0001 + vols[i] * rng.standard_normal()
        closes[i] = closes[max(0, i-1)] * max(0.01, 1.0 + np.clip(ret, -0.15, 0.15))
        h = omega + alpha_g * (ret**2) + beta_g * h
        h = max(h, 1e-8)

    idx  = pd.date_range("2019-01-02", periods=n, freq="B")
    noise = vols * 0.3 * np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        "open":   closes * (1 - noise/2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility regime classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_vol_regime(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Classify each day into vol regime based on 21d rolling realized vol.
    Percentile thresholds: LOW < 25th < MID < 75th < HIGH < 90th < EXTREME.
    """
    log_rets = np.log(df["close"] / df["close"].shift(1))
    rvol = log_rets.rolling(window).std() * math.sqrt(252)
    rvol = rvol.reindex(df.index)

    # Only compute thresholds on non-NaN values
    valid = rvol.dropna()
    p25 = float(valid.quantile(0.25))
    p75 = float(valid.quantile(0.75))
    p90 = float(valid.quantile(0.90))

    regimes = []
    for v in rvol:
        if pd.isna(v):
            regimes.append("MID")
        elif v < p25:
            regimes.append("LOW")
        elif v < p75:
            regimes.append("MID")
        elif v < p90:
            regimes.append("HIGH")
        else:
            regimes.append("EXTREME")

    return pd.Series(regimes, index=df.index), rvol


# ─────────────────────────────────────────────────────────────────────────────
# BH analysis per vol regime
# ─────────────────────────────────────────────────────────────────────────────

def compute_bh_series_with_trades(
    df: pd.DataFrame,
    cf: float,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Compute BH state series and extract simple 'trades' (long when BH active).
    Returns: (bh_df, trades_list)
    """
    mc  = MinkowskiClassifier(cf=cf)
    bh  = BlackHoleDetector(1.5, 1.0, 0.95)
    closes = df["close"].values
    n = len(closes)

    bh_active = np.zeros(n, dtype=bool)
    bh_mass   = np.zeros(n)
    bh_dir    = np.zeros(n, dtype=int)
    bh_events = [""] * n

    mc.update(float(closes[0]))
    prev_active = False
    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    trade_high = 0.0
    trade_low  = 0.0

    for i in range(1, n):
        bit = mc.update(float(closes[i]))
        act = bh.update(bit, float(closes[i]), float(closes[i-1]))
        bh_active[i] = act
        bh_mass[i]   = bh.bh_mass
        bh_dir[i]    = bh.bh_dir

        if act and not prev_active:
            bh_events[i] = "formed"
            if not in_trade and bh.bh_dir > 0:
                in_trade     = True
                entry_idx    = i
                entry_price  = closes[i]
                trade_high   = closes[i]
                trade_low    = closes[i]

        elif not act and prev_active:
            bh_events[i] = "collapsed"
            if in_trade:
                exit_price = closes[i]
                ret = (exit_price - entry_price) / (entry_price + 1e-9)
                mfe = (trade_high - entry_price) / (entry_price + 1e-9)
                mae = (entry_price - trade_low)  / (entry_price + 1e-9)
                trades.append({
                    "entry_idx":   entry_idx,
                    "exit_idx":    i,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "ret":         ret,
                    "mfe":         max(0.0, mfe),
                    "mae":         max(0.0, mae),
                    "win":         ret > 0,
                    "hold_bars":   i - entry_idx,
                })
                in_trade = False

        if in_trade:
            trade_high = max(trade_high, closes[i])
            trade_low  = min(trade_low, closes[i])

        prev_active = act

    bh_df = pd.DataFrame({
        "close":     closes,
        "bh_active": bh_active,
        "bh_mass":   bh_mass,
        "bh_dir":    bh_dir,
        "bh_event":  bh_events,
    }, index=df.index)

    return bh_df, trades


def analyze_bh_by_vol_regime(
    df: pd.DataFrame,
    bh_df: pd.DataFrame,
    vol_regime: pd.Series,
    trades: List[dict],
) -> pd.DataFrame:
    """
    For each vol regime: compute BH formation rate, win rate, avg return.
    """
    regimes = ["LOW", "MID", "HIGH", "EXTREME"]
    rows = []

    for regime in regimes:
        mask = vol_regime == regime
        n_bars = int(mask.sum())
        if n_bars == 0:
            continue

        # BH formation rate: formations per 100 bars
        bh_idx = bh_df[mask & (bh_df["bh_event"] == "formed")].index
        formation_rate = len(bh_idx) / max(1, n_bars) * 100

        # Trades that started in this vol regime
        regime_trades = [t for t in trades
                         if t["entry_idx"] < len(df.index) and
                         mask.iloc[t["entry_idx"]] if t["entry_idx"] < len(mask) else False]

        n_trades = len(regime_trades)
        if n_trades > 0:
            win_rate = float(np.mean([t["win"] for t in regime_trades]))
            avg_ret  = float(np.mean([t["ret"] for t in regime_trades]))
            avg_mfe  = float(np.mean([t["mfe"] for t in regime_trades]))
            avg_mae  = float(np.mean([t["mae"] for t in regime_trades]))
        else:
            win_rate = avg_ret = avg_mfe = avg_mae = 0.0

        # Activation fraction within this regime
        bh_in_regime = bh_df[mask]["bh_active"]
        activation_frac = float(bh_in_regime.mean())

        rows.append({
            "vol_regime":      regime,
            "n_bars":          n_bars,
            "n_trades":        n_trades,
            "formation_rate":  round(formation_rate, 3),
            "activation_frac": round(activation_frac, 4),
            "win_rate":        round(win_rate, 4),
            "avg_return":      round(avg_ret, 5),
            "avg_mfe":         round(avg_mfe, 5),
            "avg_mae":         round(avg_mae, 5),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Hawkes process
# ─────────────────────────────────────────────────────────────────────────────

def fit_hawkes_process(event_times: np.ndarray) -> dict:
    """
    Fit a Hawkes process to BH formation event times.
    Model: λ(t) = μ + α * Σ_{t_i < t} exp(-β * (t - t_i))
    Uses maximum likelihood estimation.
    Returns: {mu, alpha, beta, branching_ratio}
    """
    if len(event_times) < 5:
        return {"mu": 0.0, "alpha": 0.0, "beta": 1.0, "branching_ratio": 0.0}

    T = float(event_times[-1] - event_times[0]) + 1e-9
    n = len(event_times)

    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0:
            return 1e10

        # Recursive computation of intensities
        log_lik = 0.0
        A = 0.0  # recursive excitation term

        for i in range(n):
            t_i = event_times[i]
            intensity = mu + alpha * A
            if intensity <= 0:
                return 1e10
            log_lik += math.log(intensity + 1e-10)
            # Update A for next event
            if i > 0:
                dt = t_i - event_times[i-1]
                A = math.exp(-beta * dt) * (1 + A)
            else:
                A = 0.0

        # Integral of intensity over [0, T]
        integral = mu * T
        for t_i in event_times:
            integral += (alpha / beta) * (1.0 - math.exp(-beta * (T - t_i)))

        return -(log_lik - integral)

    try:
        from scipy.optimize import minimize
        res = minimize(
            neg_log_likelihood,
            x0=np.array([n / T, 0.5, 1.0]),
            bounds=[(1e-6, None), (0.0, 0.999), (1e-3, None)],
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        mu, alpha, beta = res.x
        return {
            "mu":              float(mu),
            "alpha":           float(alpha),
            "beta":            float(beta),
            "branching_ratio": float(alpha / beta),
            "n_events":        n,
            "time_span":       float(T),
            "converged":       bool(res.success),
        }
    except Exception as e:
        return {"mu": float(n/T), "alpha": 0.0, "beta": 1.0, "branching_ratio": 0.0, "error": str(e)}


def get_formation_event_times(bh_df: pd.DataFrame) -> np.ndarray:
    """Extract event times (as bar indices) for BH formation events."""
    formed = bh_df["bh_event"] == "formed"
    indices = np.where(formed.values)[0]
    return indices.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Vol decile analysis
# ─────────────────────────────────────────────────────────────────────────────

def bh_success_by_vol_decile(
    df: pd.DataFrame,
    bh_df: pd.DataFrame,
    trades: List[dict],
    rvol: pd.Series,
) -> pd.DataFrame:
    """
    Group by realized vol decile and compute BH activation/success rate.
    """
    decile_labels = pd.qcut(rvol.dropna(), q=10, labels=False, duplicates="drop")
    decile_labels = decile_labels.reindex(df.index)

    rows = []
    for d in range(10):
        mask = decile_labels == d
        n_bars = int(mask.sum())
        if n_bars == 0:
            continue
        avg_rvol = float(rvol[mask].mean())
        act_frac = float(bh_df[mask]["bh_active"].mean()) if n_bars > 0 else 0.0

        trade_rets = [t["ret"] for t in trades
                      if t["entry_idx"] < len(mask) and mask.iloc[t["entry_idx"]]]
        win_rate = float(np.mean([r > 0 for r in trade_rets])) if trade_rets else 0.0
        avg_ret  = float(np.mean(trade_rets)) if trade_rets else 0.0

        rows.append({
            "decile":      d + 1,
            "avg_rvol":    round(avg_rvol, 4),
            "n_bars":      n_bars,
            "activation_frac": round(act_frac, 4),
            "n_trades":    len(trade_rets),
            "win_rate":    round(win_rate, 4),
            "avg_return":  round(avg_ret, 5),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("07_vol_regime_timing.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    all_stats = {}

    for sym, cfg in ASSETS.items():
        print(f"\n[{sym}]")
        df = generate_daily_with_regimes(sym, n=1260)
        print(f"  {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

        vol_regime, rvol = classify_vol_regime(df)
        print(f"  Vol regime distribution:\n"
              f"    {vol_regime.value_counts().to_dict()}")

        bh_df, trades = compute_bh_series_with_trades(df, cf=cfg["cf"])
        n_formations = int((bh_df["bh_event"] == "formed").sum())
        print(f"  BH formations: {n_formations}  |  Trades: {len(trades)}")

        # By vol regime
        vol_analysis = analyze_bh_by_vol_regime(df, bh_df, vol_regime, trades)
        print("\n  Edge by vol regime:")
        print(vol_analysis.to_string(index=False))

        # By decile
        decile_analysis = bh_success_by_vol_decile(df, bh_df, trades, rvol)

        # Hawkes process
        event_times = get_formation_event_times(bh_df)
        print(f"\n  Fitting Hawkes process to {len(event_times)} events...")
        hawkes = fit_hawkes_process(event_times)
        print(f"    μ={hawkes['mu']:.4f}  α={hawkes['alpha']:.4f}  "
              f"β={hawkes['beta']:.4f}  "
              f"branching={hawkes['branching_ratio']:.4f}")
        if hawkes["branching_ratio"] > 0.5:
            print(f"    → SELF-EXCITING: BH formations cluster together!")

        all_stats[sym] = {
            "vol_regime_analysis": vol_analysis.to_dict("records"),
            "decile_analysis": decile_analysis.to_dict("records"),
            "hawkes": hawkes,
        }

    # Plotting
    if HAS_PLOT:
        n_syms = len(ASSETS)
        fig, axes = plt.subplots(n_syms, 4, figsize=(20, n_syms * 4))
        if n_syms == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle("Vol Regime Timing Analysis", fontsize=13)

        for row, (sym, cfg) in enumerate(ASSETS.items()):
            df = generate_daily_with_regimes(sym)
            vol_regime, rvol = classify_vol_regime(df)
            bh_df, trades = compute_bh_series_with_trades(df, cf=cfg["cf"])

            # 1. Price + vol regime coloring
            ax = axes[row, 0]
            COLORS = {"LOW": "lime", "MID": "yellow", "HIGH": "orange", "EXTREME": "red"}
            times = np.arange(len(df))
            ax.plot(times, df["close"].values / df["close"].iloc[0], "k-", linewidth=0.6)
            reg_arr = vol_regime.values
            for t in range(len(times)):
                c = COLORS.get(str(reg_arr[t]), "gray")
                ax.axvspan(t, t+1, alpha=0.15, color=c)
            ax.set_title(f"{sym} Price + Vol Regime", fontsize=8)
            ax.set_ylabel("Norm. Price"); ax.grid(alpha=0.2)

            # 2. Realized vol timeseries
            ax2 = axes[row, 1]
            rvol_plot = rvol.values
            ax2.plot(times, rvol_plot, "b-", linewidth=0.7)
            ax2.axhline(float(rvol.quantile(0.75)), color="orange", linestyle="--", linewidth=0.7)
            ax2.axhline(float(rvol.quantile(0.90)), color="red",    linestyle="--", linewidth=0.7)
            ax2.axhline(float(rvol.quantile(0.25)), color="lime",   linestyle="--", linewidth=0.7)
            ax2.set_title(f"{sym} Realized Vol (21d)", fontsize=8)
            ax2.set_ylabel("Ann. Vol"); ax2.grid(alpha=0.2)

            # 3. BH formation overlay
            ax3 = axes[row, 2]
            ax3.plot(times, bh_df["bh_mass"].values, "steelblue", linewidth=0.7, alpha=0.8)
            formed = bh_df["bh_event"] == "formed"
            ax3.scatter(np.where(formed.values)[0], bh_df["bh_mass"][formed].values,
                       color="red", s=15, zorder=5, label="Formation")
            ax3.axhline(1.5, color="orange", linestyle="--", linewidth=0.7, label="bh_form")
            ax3.set_title(f"{sym} BH Mass + Formations", fontsize=8)
            ax3.set_ylabel("BH Mass"); ax3.legend(fontsize=6); ax3.grid(alpha=0.2)

            # 4. Win rate by vol decile
            ax4 = axes[row, 3]
            dec = bh_success_by_vol_decile(df, bh_df, trades, rvol)
            if not dec.empty:
                colors = ["green" if w > 0.5 else "red" for w in dec["win_rate"]]
                ax4.bar(dec["decile"], dec["win_rate"], color=colors, alpha=0.7)
                ax4.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
                ax4.set_title(f"{sym} Win Rate by Vol Decile", fontsize=8)
                ax4.set_xlabel("Vol Decile (1=low)"); ax4.set_ylabel("Win Rate")
                ax4.set_ylim(0, 1); ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = OUTPUTS / "vol_regime_timing.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\nPlot → {out}")

    # Save stats
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    out_json = OUTPUTS / "vol_regime_timing.json"
    with open(out_json, "w") as f:
        json.dump(_clean(all_stats), f, indent=2)
    print(f"Stats → {out_json}")


if __name__ == "__main__":
    main()
