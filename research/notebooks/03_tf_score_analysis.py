"""
03_tf_score_analysis.py — TF Score Edge Analysis

Loads historical trades (from spacetime/db/trades.db if it exists, else generates
synthetic trade history via BH backtest). Groups by tf_score (0-7) and computes:
  - win_rate, avg_return, profit_factor, avg_mae, avg_mfe
  - Time-of-day analysis: does tf=7 outperform at all hours?
  - Instrument analysis: which assets have best edge at high tf_score?
  - Heatmap: tf_score × regime

Outputs: research/outputs/tf_score_analysis.png, tf_score_stats.json

Run: python research/notebooks/03_tf_score_analysis.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DB_PATH = _ROOT / "spacetime" / "db" / "trades.db"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading / generation
# ─────────────────────────────────────────────────────────────────────────────

def load_trades_from_db(db_path: Path) -> pd.DataFrame:
    """Load trades from SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time", conn)
    except Exception as e:
        print(f"  DB error: {e}")
        df = pd.DataFrame()
    conn.close()
    return df


def generate_synthetic_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic trade history covering all tf_scores.
    Higher tf_scores have better edge (higher win rate, better return).
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-01-04 09:30")
    regimes = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]
    syms = ["ES", "NQ", "BTC", "GC", "CL"]

    # Edge by tf_score: win_rate and avg_return improve with score
    tf_edge = {
        0: (0.35, -0.002, 0.008),   # (win_rate, avg_win, avg_loss)
        1: (0.40, 0.003, 0.007),
        2: (0.45, 0.005, 0.007),
        3: (0.50, 0.006, 0.007),
        4: (0.55, 0.007, 0.006),
        5: (0.58, 0.009, 0.006),
        6: (0.62, 0.011, 0.005),
        7: (0.68, 0.014, 0.005),
    }

    rows = []
    current_time = base

    for i in range(n):
        tf_score = int(rng.choice([0,1,2,3,4,5,6,7], p=[0.05,0.05,0.10,0.10,0.15,0.15,0.20,0.20]))
        wr, avg_win, avg_loss = tf_edge[tf_score]
        win  = rng.random() < wr
        sym  = str(rng.choice(syms))
        reg  = str(rng.choice(regimes))
        hold = int(rng.integers(1, 48))

        entry_price = 4500.0 * (1.0 + 0.5 * rng.standard_normal())
        if win:
            ret = float(rng.exponential(avg_win))
            exit_price = entry_price * (1.0 + ret)
            pnl = 1_000_000.0 * 0.25 * ret
            mfe = float(rng.uniform(ret, ret * 2.0))
            mae = float(rng.uniform(0, ret * 0.3))
        else:
            ret = float(rng.exponential(avg_loss))
            exit_price = entry_price * (1.0 - ret)
            pnl = -1_000_000.0 * 0.25 * ret
            mfe = float(rng.uniform(0, ret * 0.5))
            mae = float(rng.uniform(ret, ret * 1.5))

        entry_t = current_time + pd.Timedelta(hours=int(rng.integers(1, 6)))
        exit_t  = entry_t + pd.Timedelta(hours=hold)
        current_time = exit_t

        rows.append({
            "entry_time":        str(entry_t),
            "exit_time":         str(exit_t),
            "sym":               sym,
            "entry_price":       entry_price,
            "exit_price":        exit_price,
            "pnl":               pnl,
            "hold_bars":         hold,
            "mfe":               mfe,
            "mae":               mae,
            "tf_score":          tf_score,
            "regime":            reg,
            "bh_mass_at_entry":  float(rng.uniform(0.5, 3.5)),
            "hour_of_day":       entry_t.hour,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_tf_score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by tf_score and compute edge metrics.
    Returns DataFrame indexed by tf_score with columns:
      count, win_rate, avg_pnl, avg_return_pct, profit_factor, avg_mfe, avg_mae,
      mfe_mae_ratio, expectancy
    """
    rows = []
    for score in sorted(df["tf_score"].unique()):
        grp = df[df["tf_score"] == score]
        wins   = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]

        n    = len(grp)
        wr   = float(len(wins) / n) if n > 0 else 0.0
        wpnl = float(wins["pnl"].sum())
        lpnl = float(abs(losses["pnl"].sum()))
        pf   = wpnl / lpnl if lpnl > 0 else float("inf")
        avg_win_pnl  = float(wins["pnl"].mean())  if len(wins) > 0 else 0.0
        avg_loss_pnl = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        expectancy   = wr * avg_win_pnl + (1 - wr) * avg_loss_pnl

        avg_mfe = float(grp["mfe"].mean()) if "mfe" in grp.columns else 0.0
        avg_mae = float(grp["mae"].mean()) if "mae" in grp.columns else 0.0
        mfe_mae = avg_mfe / (avg_mae + 1e-10)

        rows.append({
            "tf_score":        int(score),
            "count":           n,
            "win_rate":        round(wr, 4),
            "avg_pnl":         round(float(grp["pnl"].mean()), 2),
            "profit_factor":   round(pf, 3) if math.isfinite(pf) else 99.0,
            "avg_mfe":         round(avg_mfe, 6),
            "avg_mae":         round(avg_mae, 6),
            "mfe_mae_ratio":   round(mfe_mae, 3),
            "expectancy":      round(expectancy, 2),
        })

    return pd.DataFrame(rows).set_index("tf_score")


def tf_score_by_regime_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute win_rate for each (tf_score, regime) combination.
    Returns pivot table: rows=regime, cols=tf_score, values=win_rate.
    """
    df2 = df.copy()
    df2["win"] = (df2["pnl"] > 0).astype(float)
    pivot = df2.pivot_table(values="win", index="regime", columns="tf_score", aggfunc="mean")
    return pivot


def tf_score_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate and avg PnL by tf_score and hour of day."""
    if "hour_of_day" not in df.columns:
        df = df.copy()
        df["hour_of_day"] = pd.to_datetime(df["entry_time"]).dt.hour
    df2 = df.copy()
    df2["win"] = (df2["pnl"] > 0).astype(float)
    pivot = df2.pivot_table(
        values="win", index="hour_of_day", columns="tf_score", aggfunc="mean"
    )
    return pivot


def instrument_edge_by_tf_score(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate by (sym, tf_score) for top tf_scores (>=5)."""
    high_tf = df[df["tf_score"] >= 5]
    if high_tf.empty:
        return pd.DataFrame()
    high_tf = high_tf.copy()
    high_tf["win"] = (high_tf["pnl"] > 0).astype(float)
    pivot = high_tf.pivot_table(
        values="win", index="sym", columns="tf_score", aggfunc="mean"
    )
    return pivot


def find_edge_threshold(metrics_df: pd.DataFrame) -> Optional[int]:
    """
    Find the tf_score at which the edge becomes significant.
    Criterion: win_rate > 0.55 AND profit_factor > 1.5
    """
    for score in sorted(metrics_df.index):
        row = metrics_df.loc[score]
        if row["win_rate"] > 0.55 and row["profit_factor"] > 1.5:
            return int(score)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_tf_score_metrics(metrics_df: pd.DataFrame, fig, axes):
    """Bar charts: win_rate, profit_factor, avg_mfe/mae by tf_score."""
    scores = metrics_df.index.tolist()
    x = np.arange(len(scores))

    # Win rate
    ax = axes[0]
    bars = ax.bar(x, metrics_df["win_rate"], color=["red" if w < 0.5 else "lime" for w in metrics_df["win_rate"]])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="50%")
    ax.axhline(0.55, color="orange", linestyle=":", linewidth=0.8, label="55%")
    ax.set_title("Win Rate by TF Score", fontsize=10)
    ax.set_xlabel("TF Score"); ax.set_ylabel("Win Rate")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scores])
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Profit factor
    ax = axes[1]
    pf_vals = [min(v, 5.0) for v in metrics_df["profit_factor"]]
    ax.bar(x, pf_vals, color=["red" if v < 1.0 else "steelblue" for v in pf_vals])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="PF=1.0")
    ax.axhline(1.5, color="orange", linestyle=":", linewidth=0.8, label="PF=1.5")
    ax.set_title("Profit Factor by TF Score", fontsize=10)
    ax.set_xlabel("TF Score"); ax.set_ylabel("Profit Factor")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scores])
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # Average PnL
    ax = axes[2]
    avg_pnl = metrics_df["avg_pnl"].tolist()
    ax.bar(x, avg_pnl, color=["red" if v < 0 else "steelblue" for v in avg_pnl])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average PnL by TF Score", fontsize=10)
    ax.set_xlabel("TF Score"); ax.set_ylabel("Avg PnL ($)")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scores])
    ax.grid(axis="y", alpha=0.3)

    # MFE/MAE ratio
    ax = axes[3]
    ax.bar(x, metrics_df["mfe_mae_ratio"], color="teal", alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="MFE=MAE")
    ax.set_title("MFE/MAE Ratio by TF Score", fontsize=10)
    ax.set_xlabel("TF Score"); ax.set_ylabel("MFE/MAE")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scores])
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)


def plot_heatmap(data: pd.DataFrame, title: str, ax, cmap: str = "RdYlGn"):
    """Plot a heatmap with value annotations."""
    import matplotlib.pyplot as plt
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(data.columns))); ax.set_xticklabels([str(c) for c in data.columns], fontsize=7)
    ax.set_yticks(range(len(data.index)));   ax.set_yticklabels([str(r) for r in data.index], fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("03_tf_score_analysis.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    # Load or generate trades
    if DB_PATH.exists():
        print(f"\nLoading trades from {DB_PATH}...")
        df = load_trades_from_db(DB_PATH)
        if df.empty or "tf_score" not in df.columns or len(df) < 20:
            print("  Insufficient trades in DB, generating synthetic...")
            df = generate_synthetic_trades(500)
        else:
            print(f"  {len(df)} trades loaded")
    else:
        print(f"\nDB not found at {DB_PATH}. Generating synthetic trades...")
        df = generate_synthetic_trades(500)

    print(f"Total trades: {len(df)}")
    print(f"TF score distribution:\n{df['tf_score'].value_counts().sort_index().to_string()}")

    # Compute metrics
    print("\nComputing per-tf_score metrics...")
    metrics = compute_tf_score_metrics(df)
    print(metrics.to_string())

    # Find edge threshold
    threshold = find_edge_threshold(metrics)
    print(f"\nEdge threshold: tf_score >= {threshold}" if threshold else "\nNo clear edge threshold found")

    # High tf_score analysis
    print(f"\nHigh TF score (>=6) trades:")
    high_tf = df[df["tf_score"] >= 6]
    if len(high_tf) > 0:
        print(f"  Count: {len(high_tf)}")
        print(f"  Win rate: {(high_tf['pnl'] > 0).mean():.1%}")
        print(f"  Avg PnL: ${high_tf['pnl'].mean():,.0f}")

    # By regime heatmap
    print("\nTF Score × Regime heatmap (win rate):")
    heatmap_df = tf_score_by_regime_heatmap(df)
    if not heatmap_df.empty:
        print(heatmap_df.round(3).to_string())

    # By instrument
    print("\nInstrument edge at high TF score:")
    inst_edge = instrument_edge_by_tf_score(df)
    if not inst_edge.empty:
        print(inst_edge.round(3).to_string())

    # By hour of day
    print("\nTF Score × Hour of day (win rate):")
    hour_df = tf_score_by_hour(df)

    # Plotting
    if HAS_PLOT:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("TF Score Edge Analysis", fontsize=13, fontweight="bold")
        plot_tf_score_metrics(metrics, fig, axes[0])

        # Heatmap: regime × tf_score
        plot_heatmap(heatmap_df, "Win Rate: Regime × TF Score", axes[1, 0])

        # Heatmap: hour × tf_score (high tf only)
        high_tf_hours = hour_df[[c for c in hour_df.columns if c >= 5]] if not hour_df.empty else pd.DataFrame()
        plot_heatmap(high_tf_hours, "Win Rate: Hour × TF Score (>=5)", axes[1, 1])

        # Instrument edge
        plot_heatmap(inst_edge, "Win Rate by Instrument (TF >= 5)", axes[1, 2])

        # Cumulative PnL by tf_score
        ax_eq = axes[1, 3]
        for score in sorted(df["tf_score"].unique()):
            grp = df[df["tf_score"] == score].sort_values("entry_time")
            cumulative = grp["pnl"].cumsum().values
            ax_eq.plot(cumulative, label=f"TF={score}", linewidth=0.8, alpha=0.8)
        ax_eq.set_title("Cumulative PnL by TF Score", fontsize=9)
        ax_eq.set_xlabel("Trade #"); ax_eq.set_ylabel("Cumulative PnL ($)")
        ax_eq.legend(fontsize=6); ax_eq.grid(alpha=0.3)
        ax_eq.axhline(0, color="black", linewidth=0.7)

        plt.tight_layout()
        out = OUTPUTS / "tf_score_analysis.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nPlot → {out}")
        plt.close()

    # Save stats
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    stats = {
        "n_trades": len(df),
        "edge_threshold_tf_score": int(threshold) if threshold else None,
        "per_tf_score": _clean(metrics.reset_index().to_dict("records")),
        "regime_heatmap": _clean(heatmap_df.to_dict()) if not heatmap_df.empty else {},
    }
    out_json = OUTPUTS / "tf_score_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {out_json}")

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    for score in sorted(metrics.index):
        row = metrics.loc[score]
        edge = "✓ EDGE" if row["win_rate"] > 0.55 and row["profit_factor"] > 1.5 else "  "
        print(f"  TF={score}: WR={row['win_rate']:.1%} PF={row['profit_factor']:.2f} {edge}")


if __name__ == "__main__":
    main()
