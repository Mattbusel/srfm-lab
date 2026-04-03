"""
conv_predictor.py — Convergence event ML predictor.

Trains a classifier to predict: "will a convergence event occur
in the next N bars?"

Convergence = 2+ instruments BH active simultaneously.
This is the most important predictor — convergence events have
74.5% win rate and are 24.5x more profitable per trade than solo events.

Usage:
    python tools/conv_predictor.py --train --csv data/NDX_hourly_poly.csv
    python tools/conv_predictor.py --predict --bars-ahead 3

Classifier priority:
  1. lightgbm.LGBMClassifier
  2. sklearn.ensemble.GradientBoostingClassifier
  3. sklearn.linear_model.LogisticRegression
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
from typing import List, Optional, Tuple

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
# Inline indicators (self-contained; no dependency on lib/)
# ---------------------------------------------------------------------------

class _EMA:
    def __init__(self, p):
        self.k = 2.0 / (p + 1)
        self.v = None

    def update(self, x):
        self.v = x if self.v is None else x * self.k + self.v * (1 - self.k)
        return self.v


class _ATR:
    def __init__(self, p=14):
        self.p = p
        self._prev = None
        self._buf = []
        self.v = None

    def update(self, h, lo, c):
        tr = (h - lo) if self._prev is None else max(
            h - lo, abs(h - self._prev), abs(lo - self._prev)
        )
        self._prev = c
        if self.v is None:
            self._buf.append(tr)
            if len(self._buf) >= self.p:
                self.v = sum(self._buf) / len(self._buf)
        else:
            self.v = (self.v * (self.p - 1) + tr) / self.p
        return self.v or tr


class _BHSimulator:
    """
    Minimal Black Hole simulator to generate per-bar bh_mass and bh_active
    without depending on srfm_core internals.
    Replicates the key behaviour: tracks |return| accumulation above cf threshold.
    """

    def __init__(self, cf=0.001, form_thresh=1.5, collapse_thresh=1.0, decay=0.95):
        self.cf      = cf
        self.form    = form_thresh
        self.collapse = collapse_thresh
        self.decay   = decay
        self.bh_mass = 0.0
        self.active  = False
        self.ctl     = 0   # consecutive "TIMELIKE" bars (|ret| >= cf)

    def update(self, close: float, prev_close: float) -> bool:
        ret = abs((close - prev_close) / (prev_close + 1e-12))
        beta = ret / (self.cf + 1e-12)

        # Decay mass each bar
        self.bh_mass *= self.decay

        # Accumulate if above threshold
        if beta >= 1.0:
            self.bh_mass += beta * self.cf
            self.ctl += 1
        else:
            self.ctl = 0

        # State transitions
        if not self.active and self.bh_mass >= self.form:
            self.active = True
        elif self.active and self.bh_mass < self.collapse:
            self.active = False

        return self.active


# ---------------------------------------------------------------------------
# Per-instrument feature engineering
# ---------------------------------------------------------------------------

def compute_bar_features(
    bars: List[dict],
    cf: float = 0.001,
    regime_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns a 2D array of shape (n_bars, n_features).
    Features: bh_mass, bh_mass_velocity, ctl, beta_5ma, atr_ratio,
              regime_bull, regime_side, regime_bear, regime_hv.
    """
    n = len(bars)
    closes = np.array([b["close"] for b in bars])
    highs  = np.array([b["high"]  for b in bars])
    lows   = np.array([b["low"]   for b in bars])

    bh_sim = _BHSimulator(cf=cf)
    atr_ind = _ATR(14)
    atr_buf = []

    bh_masses  = np.zeros(n)
    bh_actives = np.zeros(n, dtype=bool)
    ctls       = np.zeros(n)
    betas      = np.zeros(n)
    atr_ratios = np.ones(n)

    for i in range(n):
        if i == 0:
            atr_v = atr_ind.update(highs[i], lows[i], closes[i])
            atr_buf.append(atr_v)
            continue

        ret_abs = abs((closes[i] - closes[i-1]) / (closes[i-1] + 1e-12))
        bh_active = bh_sim.update(closes[i], closes[i-1])

        atr_v = atr_ind.update(highs[i], lows[i], closes[i])
        atr_buf.append(atr_v)
        if len(atr_buf) > 50:
            atr_buf.pop(0)
        atr_avg = sum(atr_buf) / len(atr_buf)
        atr_ratio = atr_v / (atr_avg + 1e-12)

        bh_masses[i]  = bh_sim.bh_mass
        bh_actives[i] = bh_active
        ctls[i]       = bh_sim.ctl
        betas[i]      = ret_abs / (cf + 1e-12)
        atr_ratios[i] = atr_ratio

    # Derived: bh_mass velocity (delta over 5 bars)
    bh_vel = np.zeros(n)
    bh_vel[5:] = bh_masses[5:] - bh_masses[:-5]

    # 5-bar moving average of beta
    beta_5ma = np.zeros(n)
    for i in range(5, n):
        beta_5ma[i] = betas[i-4:i+1].mean()

    # Regime one-hot (0=BULL,1=SIDEWAYS,2=BEAR,3=HIGH_VOL)
    reg_bull = np.zeros(n)
    reg_side = np.zeros(n)
    reg_bear = np.zeros(n)
    reg_hv   = np.zeros(n)
    if regime_labels is not None:
        reg_bull[regime_labels == 0] = 1.0
        reg_side[regime_labels == 1] = 1.0
        reg_bear[regime_labels == 2] = 1.0
        reg_hv[regime_labels == 3]   = 1.0
    else:
        reg_bull[:] = 1.0  # default: treat as bull if unknown

    X = np.column_stack([
        bh_masses,
        bh_vel,
        ctls,
        beta_5ma,
        atr_ratios,
        reg_bull, reg_side, reg_bear, reg_hv,
    ])
    return X, bh_actives


# ---------------------------------------------------------------------------
# Multi-instrument convergence label construction
# ---------------------------------------------------------------------------

def build_dataset(
    csv_paths: dict,   # {"ES": path, "NQ": path, ...}
    bars_ahead: int = 5,
    cf_map: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Runs the BH simulator on each instrument, aligns by date/index,
    labels each bar 1 if a convergence event starts within `bars_ahead` bars.

    Returns: X (features for primary instrument), y (labels), dates
    """
    if cf_map is None:
        cf_map = {"ES": 0.001, "NQ": 0.0012, "YM": 0.0008}

    instrument_data = {}
    for inst, path in csv_paths.items():
        bars = load_ohlcv(path)
        cf = cf_map.get(inst, 0.001)
        X, bh_active = compute_bar_features(bars, cf=cf)
        instrument_data[inst] = {
            "bars":      bars,
            "X":         X,
            "bh_active": bh_active,
            "dates":     [b["date"] for b in bars],
        }

    # Use the first (primary) instrument as the reference timeline
    primary = list(instrument_data.keys())[0]
    n = len(instrument_data[primary]["bars"])
    dates = instrument_data[primary]["dates"]

    # Align bh_active arrays to primary timeline (by date if possible)
    aligned_active = {}
    for inst, data in instrument_data.items():
        if inst == primary:
            aligned_active[inst] = data["bh_active"]
            continue
        # Map by date index
        date_to_idx = {d: i for i, d in enumerate(data["dates"])}
        arr = np.zeros(n, dtype=bool)
        for i, d in enumerate(dates):
            j = date_to_idx.get(d)
            if j is not None:
                arr[i] = data["bh_active"][j]
        aligned_active[inst] = arr

    # Convergence: 2+ instruments BH active at the same bar
    active_matrix = np.column_stack(list(aligned_active.values()))  # (n, n_inst)
    conv_active   = active_matrix.sum(axis=1) >= 2                  # bool

    # Label: 1 if convergence event starts within next bars_ahead bars
    labels = np.zeros(n, dtype=int)
    for i in range(n - bars_ahead):
        if conv_active[i: i + bars_ahead].any():
            labels[i] = 1

    # Build cross-instrument correlation feature (20-bar rolling)
    insts = list(instrument_data.keys())
    corr_feature = np.zeros(n)
    if len(insts) >= 2:
        c0 = np.array([b["close"] for b in instrument_data[insts[0]]["bars"]])
        c1_raw = np.array([b["close"] for b in instrument_data[insts[1]]["bars"]])
        # Align c1 to primary timeline
        if len(c1_raw) == n:
            c1 = c1_raw
        else:
            c1 = np.full(n, np.nan)
            date_to_idx = {d: i for i, d in enumerate(instrument_data[insts[1]]["dates"])}
            for i, d in enumerate(dates):
                j = date_to_idx.get(d)
                if j is not None and j < len(c1_raw):
                    c1[i] = c1_raw[j]
            # Forward-fill NaNs
            last = c0[0]
            for i in range(n):
                if np.isnan(c1[i]):
                    c1[i] = last
                else:
                    last = c1[i]

        r0 = np.diff(np.log(c0 + 1e-9), prepend=0)
        r1 = np.diff(np.log(c1 + 1e-9), prepend=0)
        for i in range(20, n):
            sl0 = r0[i-19:i+1]
            sl1 = r1[i-19:i+1]
            if sl0.std() > 1e-12 and sl1.std() > 1e-12:
                corr_feature[i] = float(np.corrcoef(sl0, sl1)[0, 1])

    # Append correlation to primary features
    X_primary = instrument_data[primary]["X"]
    X_full = np.column_stack([X_primary, corr_feature])

    return X_full, labels, dates


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "bh_mass", "bh_mass_velocity", "ctl", "beta_5ma", "atr_ratio",
    "regime_bull", "regime_side", "regime_bear", "regime_hv", "correlation",
]


def get_classifier(n_pos: int, n_neg: int):
    scale = max(1, n_neg // max(n_pos, 1))
    name = "unknown"
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale,
            random_state=42,
            verbosity=-1,
        )
        name = "LightGBM"
        return clf, name
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
        name = "GradientBoostingClassifier"
        return clf, name
    except ImportError:
        pass

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )
        name = "LogisticRegression"
        return clf, name
    except ImportError:
        pass

    raise RuntimeError(
        "No classifier available. Install lightgbm or scikit-learn:\n"
        "  pip install lightgbm\n  pip install scikit-learn"
    )


def precision_recall_at_threshold(y_true, proba, threshold=None, top_pct=0.01):
    """Precision/recall at top X% of predictions or fixed threshold."""
    n = len(y_true)
    if threshold is None:
        k = max(1, int(n * top_pct))
        top_k = np.argsort(proba)[-k:]
        preds = np.zeros(n, dtype=int)
        preds[top_k] = 1
    else:
        preds = (proba >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def bar_chart(val, max_val, width=20):
    if max_val < 1e-12:
        return ""
    n = int(round(val / max_val * width))
    return "█" * max(0, n)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def cmd_train(args):
    os.makedirs("results", exist_ok=True)

    # Build CSV map — use available data files for ES, NQ, YM
    data_dir = args.data_dir or "data"
    csv_candidates = {
        "ES": [f"{data_dir}/ES_hourly_real.csv", f"{data_dir}/ES_hourly_rth.csv"],
        "NQ": [f"{data_dir}/NQ_hourly_real.csv", f"{data_dir}/NDX_hourly_poly.csv",
               f"{data_dir}/NQ_hourly_rth.csv"],
        "YM": [f"{data_dir}/YM_hourly_real.csv", f"{data_dir}/YM_hourly_rth.csv"],
    }

    if args.csv:
        # Single CSV provided: use it for the primary instrument only
        csv_paths = {"PRIMARY": args.csv}
    else:
        csv_paths = {}
        for inst, candidates in csv_candidates.items():
            for c in candidates:
                if os.path.exists(c):
                    csv_paths[inst] = c
                    break

    if not csv_paths:
        print("ERROR: No CSV files found. Provide --csv <path> or ensure data/ directory exists.")
        sys.exit(1)

    print(f"CONVERGENCE PREDICTOR — training")
    print(f"  Instruments : {list(csv_paths.keys())}")
    print(f"  Bars ahead  : {args.bars_ahead}")

    X, y, dates = build_dataset(csv_paths, bars_ahead=args.bars_ahead)
    n = len(y)

    # Drop first 200 warm-up bars
    WARMUP = 200
    X = X[WARMUP:]
    y = y[WARMUP:]
    dates = dates[WARMUP:]
    n = len(y)

    # Train/test split: 80% train, 20% test (time-ordered)
    split = int(n * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_test  = int(y_test.sum())
    n_neg_test  = int((y_test == 0).sum())

    print(f"\n  Training set : {len(y_train):,} bars")
    print(f"    Convergence events : {n_pos_train:,} ({100*n_pos_train/len(y_train):.2f}%)")
    print(f"    Non-convergence    : {n_neg_train:,} ({100*n_neg_train/len(y_train):.2f}%)")
    print(f"    Class imbalance    : scale_pos_weight={n_neg_train // max(n_pos_train, 1)}")

    clf, clf_name = get_classifier(n_pos_train, n_neg_train)
    print(f"\n  Fitting {clf_name} ...")
    clf.fit(X_train, y_train)

    # Evaluate on test set
    try:
        proba_test = clf.predict_proba(X_test)[:, 1]
    except AttributeError:
        proba_test = clf.decision_function(X_test)
        proba_test = 1 / (1 + np.exp(-proba_test))

    prec1, rec1, f1_1 = precision_recall_at_threshold(y_test, proba_test, top_pct=0.01)
    prec5, rec5, f1_5 = precision_recall_at_threshold(y_test, proba_test, top_pct=0.05)

    # Baseline: flag everything
    baseline_prec = n_pos_test / (n_pos_test + n_neg_test + 1e-12)

    output_lines = [
        f"CONVERGENCE PREDICTOR — {clf_name}",
        "=" * 50,
        f"Training on first 80% ({len(y_train):,} bars)",
        f"Testing  on last  20% ({len(y_test):,} bars)",
        "",
        "TRAINING SET:",
        f"  Convergence events : {n_pos_train:,} ({100*n_pos_train/len(y_train):.2f}%)",
        f"  Non-convergence    : {n_neg_train:,} ({100*n_neg_train/len(y_train):.2f}%)",
        f"  Class imbalance handled: scale_pos_weight={n_neg_train // max(n_pos_train, 1)}",
        "",
        "TEST SET PERFORMANCE:",
        f"  Precision@top1%  : {100*prec1:.1f}%  (of flagged events that ARE convergence)",
        f"  Recall@top1%     : {100*rec1:.1f}%  (fraction of convergence events caught)",
        f"  F1@top1%         : {f1_1:.3f}",
        f"  Precision@top5%  : {100*prec5:.1f}%",
        f"  Recall@top5%     : {100*rec5:.1f}%",
        f"  F1@top5%         : {f1_5:.3f}",
        "",
        f"  vs baseline (flag everything): precision={100*baseline_prec:.2f}%, recall=100%",
        "",
    ]

    # Feature importance
    try:
        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        else:
            imps = None

        if imps is not None:
            n_feats = min(len(FEATURE_NAMES), len(imps))
            pairs = sorted(zip(FEATURE_NAMES[:n_feats], imps[:n_feats]),
                           key=lambda x: -x[1])
            max_imp = pairs[0][1] if pairs else 1.0
            output_lines.append("FEATURE IMPORTANCE:")
            for fname, imp in pairs:
                bar = bar_chart(imp, max_imp, width=20)
                output_lines.append(f"  {fname:<22}  {bar:<20}  {imp:.3f}")
    except Exception:
        pass

    output = "\n".join(output_lines)
    print("\n" + output)

    # Save model
    pkl_path = "results/conv_predictor.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":       clf,
            "clf_name":    clf_name,
            "feature_names": FEATURE_NAMES,
            "bars_ahead":  args.bars_ahead,
            "csv_paths":   csv_paths,
        }, f)
    print(f"\n  Saved model -> {pkl_path}")

    # Save predictions on test set
    pred_path = "results/conv_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bar_index", "date", "probability", "label"])
        for i, (p, lbl, d) in enumerate(zip(proba_test, y_test, dates_test)):
            writer.writerow([split + WARMUP + i, d, f"{p:.6f}", int(lbl)])
    print(f"  Saved predictions -> {pred_path}")

    # Save markdown summary
    md_path = "results/conv_predictor.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Convergence Predictor Report\n\n```\n")
        f.write(output)
        f.write("\n```\n")
    print(f"  Saved report  -> {md_path}")


# ---------------------------------------------------------------------------
# Predict (load model + run on new bars)
# ---------------------------------------------------------------------------

def cmd_predict(args):
    pkl_path = "results/conv_predictor.pkl"
    if not os.path.exists(pkl_path):
        print(f"ERROR: Model not found at {pkl_path}. Run with --train first.")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    clf         = bundle["model"]
    clf_name    = bundle["clf_name"]
    bars_ahead  = bundle.get("bars_ahead", 5)
    csv_paths   = bundle.get("csv_paths", {})

    print(f"CONVERGENCE PREDICTOR — {clf_name}")
    print(f"  Predicting convergence within {bars_ahead} bars")

    X, y, dates = build_dataset(csv_paths, bars_ahead=bars_ahead)

    try:
        proba = clf.predict_proba(X)[:, 1]
    except AttributeError:
        proba = clf.decision_function(X)
        proba = 1 / (1 + np.exp(-proba))

    # Show top 10 most recent high-probability bars
    n = len(proba)
    top_k = 10
    top_idx = np.argsort(proba)[-top_k:][::-1]

    print(f"\n  Top {top_k} convergence predictions (most recent high-probability bars):")
    print(f"  {'Bar':>7}  {'Date':<24}  {'P(conv)':>8}  {'Actual':>8}")
    print(f"  {'-'*7}  {'-'*24}  {'-'*8}  {'-'*8}")
    for idx in top_idx:
        d = dates[idx][:24] if idx < len(dates) else ""
        actual = "YES" if idx < len(y) and y[idx] == 1 else "no"
        print(f"  {idx:>7}  {d:<24}  {proba[idx]:>8.4f}  {actual:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convergence event ML predictor for LARSA multi-instrument futures."
    )
    parser.add_argument("--train",      action="store_true", help="Train the predictor")
    parser.add_argument("--predict",    action="store_true", help="Run predictions using saved model")
    parser.add_argument("--csv",        help="Single CSV path (uses one instrument only)")
    parser.add_argument("--data-dir",   default="data", help="Directory containing instrument CSVs")
    parser.add_argument("--bars-ahead", type=int, default=5,
                        help="Predict convergence within N bars (default 5)")
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        sys.exit(0)

    if args.train:
        cmd_train(args)
    if args.predict:
        cmd_predict(args)


if __name__ == "__main__":
    main()
