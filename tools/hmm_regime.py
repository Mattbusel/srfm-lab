"""
hmm_regime.py — HMM-based market regime detection.

Fits a Gaussian HMM on price features, discovers hidden states
that correspond to BULL/BEAR/SIDEWAYS/HIGH_VOL regimes without
hand-crafted rules. More statistically rigorous than the manual
classifier in larsa-v5.

Features used:
- 1-bar return
- 5-bar rolling return
- 20-bar rolling volatility
- 20-bar rolling skewness
- Beta (|return|/cf) — SRFM-specific

Usage:
    python tools/hmm_regime.py --csv data/NDX_hourly_poly.csv --states 4 --plot
    python tools/hmm_regime.py --csv data/NDX_hourly_poly.csv --compare
"""

import argparse
import os
import sys
import csv
import json
import pickle
import math
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

# ---------------------------------------------------------------------------
# Data loading (mirrors arena_v2.load_ohlcv)
# ---------------------------------------------------------------------------

def load_ohlcv(path: str) -> List[dict]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            def g(*keys):
                for k in keys:
                    v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                    if v not in (None, "", "null", "None"):
                        try:
                            return float(v)
                        except Exception:
                            pass
                return None
            c = g("close", "Close")
            if c and c > 0:
                bars.append({
                    "date":   row.get("date") or row.get("Date") or "",
                    "open":   g("open", "Open") or c,
                    "high":   g("high", "High") or c,
                    "low":    g("low", "Low") or c,
                    "close":  c,
                    "volume": g("volume", "Volume") or 1000.0,
                })
    return bars


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rolling_std(arr, window):
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = float(np.std(arr[i - window + 1:i + 1]))
    return out


def _rolling_skew(arr, window):
    out = np.full(len(arr), 0.0)
    for i in range(window - 1, len(arr)):
        sl = arr[i - window + 1:i + 1]
        mu = np.mean(sl)
        std = np.std(sl)
        if std > 1e-12:
            out[i] = float(np.mean(((sl - mu) / std) ** 3))
    return out


def build_features(bars: List[dict], cf: float = 0.001) -> np.ndarray:
    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    n = len(closes)

    ret1 = np.zeros(n)
    ret1[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-12)

    # 5-bar rolling return
    ret5 = np.zeros(n)
    for i in range(5, n):
        ret5[i] = (closes[i] - closes[i - 5]) / (closes[i - 5] + 1e-12)

    # 20-bar rolling vol
    vol20 = _rolling_std(ret1, 20)
    vol20 = np.nan_to_num(vol20, nan=float(np.nanmean(np.abs(ret1))))

    # 20-bar rolling skewness
    skew20 = _rolling_skew(ret1, 20)

    # SRFM beta: |ret| / cf
    beta = np.abs(ret1) / (cf + 1e-12)
    beta = np.clip(beta, 0, 10)

    X = np.column_stack([ret1, ret5, vol20, skew20, beta])
    return X


# ---------------------------------------------------------------------------
# Hand-crafted regime emulation (simplified from RegimeDetector)
# ---------------------------------------------------------------------------

def _ema(arr, period):
    k = 2.0 / (period + 1)
    out = np.zeros(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def manual_regimes(bars: List[dict]) -> np.ndarray:
    """Approximate hand-crafted regime labels: 0=BULL 1=SIDEWAYS 2=BEAR 3=HIGH_VOL"""
    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    highs  = np.array([b["high"]  for b in bars], dtype=np.float64)
    lows   = np.array([b["low"]   for b in bars], dtype=np.float64)
    n = len(closes)

    ema12  = _ema(closes, 12)
    ema26  = _ema(closes, 26)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)

    # ATR
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[i] = atr[i-1] * 13/14 + tr / 14 if atr[i-1] > 0 else tr

    # ADX (simplified)
    atr50 = np.zeros(n)
    buf = []
    for i in range(n):
        buf.append(atr[i])
        if len(buf) > 50:
            buf.pop(0)
        atr50[i] = sum(buf) / len(buf)

    labels = np.zeros(n, dtype=int)
    for i in range(200, n):
        atr_ratio = atr[i] / (atr50[i] + 1e-9)
        if atr_ratio > 2.0:
            labels[i] = 3  # HIGH_VOL
        elif closes[i] < ema200[i] and closes[i] < ema50[i] and ema12[i] < ema26[i]:
            labels[i] = 2  # BEAR
        elif closes[i] > ema200[i] and closes[i] > ema50[i] and ema12[i] > ema26[i]:
            labels[i] = 0  # BULL
        else:
            labels[i] = 1  # SIDEWAYS

    return labels


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------

def fit_hmm(X: np.ndarray, n_states: int):
    """Fit GaussianHMM, fall back to KMeans if hmmlearn unavailable."""
    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X)
        states = model.predict(X)
        trans = model.transmat_
        means = model.means_
        return states, trans, means, "GaussianHMM", model
    except ImportError:
        pass

    # Fallback: KMeans clustering
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        states = km.fit_predict(X)
        means = km.cluster_centers_
        # Estimate rough transition matrix
        trans = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            trans[states[i], states[i+1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        trans = trans / np.where(row_sums > 0, row_sums, 1)
        return states, trans, means, "KMeans-fallback", km
    except ImportError:
        pass

    raise RuntimeError(
        "Neither hmmlearn nor sklearn is installed. "
        "Install with: pip install hmmlearn  or  pip install scikit-learn"
    )


# ---------------------------------------------------------------------------
# State labeling
# ---------------------------------------------------------------------------

LABEL_NAMES = ["BULL", "SIDEWAYS", "BEAR", "HIGH_VOL"]


def label_states(states: np.ndarray, returns: np.ndarray, vols: np.ndarray,
                 n_states: int) -> dict:
    """Map HMM state indices to regime names based on mean return and vol."""
    state_stats = {}
    for s in range(n_states):
        mask = states == s
        count = int(mask.sum())
        mr = float(returns[mask].mean()) if count > 0 else 0.0
        mv = float(vols[mask].mean()) if count > 0 else 0.0
        state_stats[s] = {"count": count, "mean_ret": mr, "mean_vol": mv}

    # Sort by mean return descending; top = BULL, bottom = BEAR
    sorted_by_ret = sorted(state_stats.items(), key=lambda x: x[1]["mean_ret"], reverse=True)

    # If 4 states: highest ret=BULL, lowest=BEAR, highest vol among remaining=HIGH_VOL, last=SIDEWAYS
    mapping = {}
    if n_states == 4:
        bull_s  = sorted_by_ret[0][0]
        bear_s  = sorted_by_ret[-1][0]
        middle  = [s for s, _ in sorted_by_ret[1:-1]]
        # Among middle 2, highest vol = HIGH_VOL
        if state_stats[middle[0]]["mean_vol"] >= state_stats[middle[1]]["mean_vol"]:
            hv_s   = middle[0]
            side_s = middle[1]
        else:
            hv_s   = middle[1]
            side_s = middle[0]
        mapping = {bull_s: "BULL", side_s: "SIDEWAYS", bear_s: "BEAR", hv_s: "HIGH_VOL"}
    elif n_states == 2:
        mapping = {sorted_by_ret[0][0]: "BULL", sorted_by_ret[1][0]: "BEAR"}
    elif n_states == 3:
        mapping = {
            sorted_by_ret[0][0]: "BULL",
            sorted_by_ret[1][0]: "SIDEWAYS",
            sorted_by_ret[2][0]: "BEAR",
        }
    else:
        for i, (s, _) in enumerate(sorted_by_ret):
            mapping[s] = LABEL_NAMES[min(i, 3)]

    return mapping, state_stats


# ---------------------------------------------------------------------------
# Sharpe backtest
# ---------------------------------------------------------------------------

def hmm_backtest_sharpe(bars, states, state_map):
    """Long in BULL, short in BEAR, flat otherwise."""
    closes = np.array([b["close"] for b in bars])
    n = len(closes)
    rets = []
    for i in range(1, n):
        r = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-12)
        label = state_map.get(int(states[i]), "SIDEWAYS")
        if label == "BULL":
            rets.append(r)
        elif label == "BEAR":
            rets.append(-r)
        else:
            rets.append(0.0)
    rets = np.array(rets)
    if rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(252 * 6.5))  # hourly bars


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def bar_chart(val, max_val, width=20):
    if max_val < 1e-12:
        return ""
    n = int(round(val / max_val * width))
    return "█" * max(0, n)


def compare_regimes(hmm_labels: np.ndarray, manual_labels: np.ndarray,
                    state_map: dict, bars: List[dict]) -> str:
    """Compute agreement and find major disagreement periods."""
    hmm_named   = np.array([state_map.get(int(s), "UNKNOWN") for s in hmm_labels])
    manual_named_map = {0: "BULL", 1: "SIDEWAYS", 2: "BEAR", 3: "HIGH_VOL"}
    manual_named = np.array([manual_named_map[int(m)] for m in manual_labels])

    agree = float((hmm_named == manual_named).mean()) * 100

    # Find top disagreement periods (runs of consecutive disagree)
    disagree_mask = hmm_named != manual_named
    lines = []
    i = 0
    periods = []
    while i < len(disagree_mask):
        if disagree_mask[i]:
            j = i
            while j < len(disagree_mask) and disagree_mask[j]:
                j += 1
            length = j - i
            if length >= 20:
                mid = (i + j) // 2
                date_str = bars[mid]["date"][:10] if bars[mid]["date"] else f"bar_{mid}"
                periods.append((length, date_str, hmm_named[mid], manual_named[mid]))
            i = j
        else:
            i += 1

    periods.sort(key=lambda x: -x[0])
    for length, date_str, h, m in periods[:3]:
        lines.append(f"  HMM sees {h:<9} where manual says {m:<9} ({date_str}, {length} bars)")

    return agree, lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HMM-based market regime detection for LARSA futures strategy."
    )
    parser.add_argument("--csv",     required=True, help="Path to OHLCV CSV")
    parser.add_argument("--states",  type=int, default=4, help="Number of HMM states (default 4)")
    parser.add_argument("--cf",      type=float, default=0.001, help="Curvature factor for SRFM beta")
    parser.add_argument("--plot",    action="store_true", help="Save regime plot (requires matplotlib)")
    parser.add_argument("--compare", action="store_true", help="Compare HMM vs hand-crafted regimes")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print(f"Loading {args.csv} ...")
    bars = load_ohlcv(args.csv)
    n = len(bars)
    print(f"  {n} bars loaded")

    print("Engineering features ...")
    X = build_features(bars, cf=args.cf)
    closes = np.array([b["close"] for b in bars])

    print(f"Fitting HMM ({args.states} states) ...")
    states, trans, means, method, model_obj = fit_hmm(X, args.states)

    ret1 = X[:, 0]
    vol20 = X[:, 2]
    state_map, state_stats = label_states(states, ret1, vol20, args.states)

    # Sharpe backtest
    sharpe = hmm_backtest_sharpe(bars, states, state_map)

    # Build output
    header = f"HMM REGIME DETECTION ({args.states} states, {n:,} bars) — {method}"
    sep    = "=" * len(header)

    lines = [header, sep]

    # State summary
    label_to_state = {v: k for k, v in state_map.items()}
    ordered_labels = ["BULL", "SIDEWAYS", "BEAR", "HIGH_VOL"]
    available = [l for l in ordered_labels if l in label_to_state]
    extra     = [l for l in state_map.values() if l not in available]
    for label in available + extra:
        s   = label_to_state[label]
        st  = state_stats[s]
        cnt = st["count"]
        pct = 100 * cnt / n
        mr  = st["mean_ret"]
        mv  = st["mean_vol"]
        lines.append(
            f"  State {s} -> {label:<9}: {cnt:>7,} bars ({pct:5.1f}%)  "
            f"mean_ret={mr:+.6f}  vol={mv:.6f}"
        )

    lines.append("")
    lines.append(f"  HMM Sharpe (BULL long / BEAR short): {sharpe:+.3f}")
    lines.append("")

    # Compare with hand-crafted
    if args.compare:
        manual = manual_regimes(bars)
        agree_pct, disagree_lines = compare_regimes(states, manual, state_map, bars)
        lines.append(f"Agreement with hand-crafted classifier: {agree_pct:.1f}%")
        if disagree_lines:
            lines.append("Key disagreements:")
            lines.extend(disagree_lines)
        lines.append("")

    # Transition matrix
    lines.append("Transition matrix:")
    all_labels = list(state_map.values())
    header_row = "       " + "  ".join(f"{l[:4]:<4}" for l in all_labels)
    lines.append(header_row)
    for i in range(args.states):
        row_label = state_map.get(i, f"S{i}")[:4]
        row_vals  = "  ".join(f"{trans[i, j]:.2f}" for j in range(args.states))
        lines.append(f"  {row_label:<4}   {row_vals}")

    output = "\n".join(lines)
    print("\n" + output)

    # Save markdown report
    md_path = "results/hmm_regime.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HMM Regime Detection Report\n\n```\n")
        f.write(output)
        f.write("\n```\n")
    print(f"\n  Saved report -> {md_path}")

    # Save model pickle
    pkl_path = "results/hmm_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":     model_obj,
            "method":    method,
            "n_states":  args.states,
            "state_map": state_map,
            "transmat":  trans,
            "means":     means,
            "cf":        args.cf,
        }, f)
    print(f"  Saved model  -> {pkl_path}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            colour_map = {"BULL": "#2ecc71", "SIDEWAYS": "#f39c12",
                          "BEAR": "#e74c3c", "HIGH_VOL": "#9b59b6"}
            colours = {s: colour_map.get(state_map.get(s, ""), "#aaaaaa")
                       for s in range(args.states)}

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(closes, lw=0.6, color="#2c3e50", label="Close")
            ax1.set_ylabel("Price")
            ax1.set_title(header)

            # Shade regime bands
            idx = np.arange(n)
            for s in range(args.states):
                mask = states == s
                ax2.fill_between(idx, 0, 1, where=mask,
                                  color=colours[s], alpha=0.7)

            patches = [mpatches.Patch(color=colour_map.get(state_map.get(s, ""), "#aaa"),
                                       label=state_map.get(s, f"S{s}"))
                       for s in range(args.states)]
            ax2.legend(handles=patches, loc="upper right", fontsize=8)
            ax2.set_ylabel("Regime")
            ax2.set_yticks([])

            plt.tight_layout()
            plot_path = "results/hmm_regime_plot.png"
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"  Saved plot   -> {plot_path}")
        except ImportError:
            print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
