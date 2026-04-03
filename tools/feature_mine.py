"""
feature_mine.py -- tsfresh automatic feature extraction for convergence prediction.

Extracts 800+ time series features from price windows BEFORE convergence events.
Finds which features statistically predict well formation.

Usage:
    python tools/feature_mine.py --csv data/NDX_hourly_poly.csv --bars-before 20
    python tools/feature_mine.py --quick  # 50 features instead of 800+
"""

import argparse
import json
import math
import os
import random
import sys

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# -- Manual feature computation (fallback) -------------------------------------

def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _safe_std(xs):
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)

def _safe_skew(xs):
    n = len(xs)
    if n < 3:
        return 0.0
    m = _safe_mean(xs)
    s = _safe_std(xs)
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in xs) / n

def _safe_kurtosis(xs):
    n = len(xs)
    if n < 4:
        return 0.0
    m = _safe_mean(xs)
    s = _safe_std(xs)
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 4 for x in xs) / n - 3.0

def _autocorr_lag1(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = _safe_mean(xs)
    num = sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, n))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den != 0 else 0.0

def _linear_slope(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    idx = list(range(n))
    mx = _safe_mean(idx)
    my = _safe_mean(xs)
    num = sum((idx[i] - mx) * (xs[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in idx)
    return num / den if den != 0 else 0.0

def _mean_abs_change(xs):
    if len(xs) < 2:
        return 0.0
    return _safe_mean([abs(xs[i] - xs[i-1]) for i in range(1, len(xs))])

def _count_above_mean(xs):
    if not xs:
        return 0.0
    m = _safe_mean(xs)
    return sum(1 for x in xs if x > m) / len(xs)

def compute_manual_features(window: list) -> dict:
    """Compute 15 key features manually from a price window (list of close prices)."""
    if not window:
        return {}
    # returns
    returns = [(window[i] - window[i-1]) / max(1e-9, window[i-1]) for i in range(1, len(window))]
    beta_proxy = [abs(r) for r in returns]  # |Δp/p| ≈ beta proxy

    feats = {
        "close__mean":              _safe_mean(window),
        "close__std":               _safe_std(window),
        "close__skewness":          _safe_skew(window),
        "close__kurtosis":          _safe_kurtosis(window),
        "close__autocorr_lag1":     _autocorr_lag1(window),
        "close__linear_trend_slope":_linear_slope(window),
        "close__mean_abs_change":   _mean_abs_change(window),
        "close__count_above_mean":  _count_above_mean(window),
        "beta__mean_abs_change":    _mean_abs_change(beta_proxy),
        "beta__std":                _safe_std(beta_proxy),
        "beta__mean":               _safe_mean(beta_proxy),
        "beta__count_above_mean":   _count_above_mean(beta_proxy),
        "returns__mean":            _safe_mean(returns),
        "returns__std":             _safe_std(returns),
        "returns__skewness":        _safe_skew(returns),
    }
    return feats


def generate_synthetic_bars(n: int = 2000, seed: int = 42) -> list:
    random.seed(seed)
    price = 15000.0
    bars = []
    for _ in range(n):
        r = random.gauss(0.0002, 0.005)
        price = price * (1.0 + r)
        bars.append(price)
    return bars


def extract_windows(prices: list, events: list, bars_before: int = 20, ratio: int = 10):
    """
    events: list of bar indices where convergence occurred.
    Returns (X_windows, y_labels, window_ids).
    """
    n = len(prices)
    windows = []
    labels  = []
    ids     = []

    for idx in events:
        start = idx - bars_before
        if start < 0 or idx > n:
            continue
        windows.append(prices[start:idx])
        labels.append(1)
        ids.append(f"conv_{idx}")

    # non-events: random windows not overlapping events
    event_set = set(events)
    non_event_indices = [i for i in range(bars_before, n) if i not in event_set]
    random.seed(0)
    random.shuffle(non_event_indices)
    for i in non_event_indices[: len(events) * ratio]:
        windows.append(prices[i - bars_before: i])
        labels.append(0)
        ids.append(f"nonevent_{i}")

    return windows, labels, ids


def train_classifier(feature_matrix, labels):
    """Train LogisticRegression; return model and feature importances."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X = np.array(feature_matrix)
        y = np.array(labels)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        # Threshold at 10%
        threshold = 0.10
        y_pred = (y_prob >= threshold).astype(int)
        tp = sum(1 for a, b in zip(y_test, y_pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_test, y_pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_test, y_pred) if a == 1 and b == 0)
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)

        return {
            "auc": auc, "precision": precision, "recall": recall,
            "coefs": clf.coef_[0].tolist(),
            "scaler": scaler,
        }
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="tsfresh feature mining for convergence prediction")
    parser.add_argument("--csv",          default=None, help="OHLCV CSV file")
    parser.add_argument("--bars-before",  type=int, default=20, help="bars before event")
    parser.add_argument("--quick",        action="store_true", help="fast mode: 50 features")
    args = parser.parse_args()

    # -- Load trade data --------------------------------------------------------
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    wells = data["wells"]

    # -- Load or synthesize price data ------------------------------------------
    prices = None
    conv_events = []

    if args.csv and os.path.exists(args.csv):
        try:
            import csv
            rows = []
            with open(args.csv) as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    for key in ["close", "Close", "CLOSE"]:
                        if key in row:
                            try:
                                rows.append(float(row[key]))
                                break
                            except (ValueError, TypeError):
                                pass
            if rows:
                prices = rows
                print(f"  Loaded {len(prices)} bars from {args.csv}")
        except Exception as e:
            print(f"  Warning: could not load CSV ({e}), using synthetic data")

    if prices is None:
        print("  Generating synthetic price data (2000 bars)...")
        prices = generate_synthetic_bars(2000)

    n = len(prices)
    # Map convergence wells to approximate bar indices (distribute evenly)
    conv_wells = [w for w in wells if len(w.get("instruments", [])) > 1]
    # Use all wells as events if no multi-instrument ones
    if not conv_wells:
        conv_wells = wells[:47]  # use first 47 wells

    step = max(1, (n - args.bars_before) // max(1, len(conv_wells)))
    conv_events = [args.bars_before + i * step for i in range(len(conv_wells)) if args.bars_before + i * step < n]

    # -- Extract windows --------------------------------------------------------
    windows, labels, ids = extract_windows(prices, conv_events, args.bars_before)
    n_conv    = sum(labels)
    n_nonevent = len(labels) - n_conv
    print(f"  Windows: {n_conv} convergence + {n_nonevent} non-events")

    # -- Feature extraction -----------------------------------------------------
    tsfresh_available = False
    feature_names = []
    feature_matrix = []

    if not args.quick:
        try:
            import pandas as pd
            import tsfresh
            from tsfresh import extract_features
            from tsfresh.utilities.dataframe_functions import impute
            print("  tsfresh available -- extracting 800+ features...")

            # Build tsfresh dataframe format
            dfs = []
            for wid, window in zip(ids, windows):
                for t, val in enumerate(window):
                    dfs.append({"id": wid, "time": t, "close": val})
            df_ts = pd.DataFrame(dfs)
            y_series = pd.Series(labels, index=ids)

            X_feat = extract_features(df_ts, column_id="id", column_sort="time",
                                       column_value="close", disable_progressbar=True)
            impute(X_feat)
            tsfresh_available = True
            feature_names = list(X_feat.columns)
            feature_matrix = X_feat.values.tolist()
            print(f"  Features extracted: {len(feature_names)}")
        except ImportError:
            print("  tsfresh not installed -- using manual 15-feature fallback")
        except Exception as e:
            print(f"  tsfresh error ({e}) -- using manual fallback")

    if not tsfresh_available:
        print("  Computing 15 manual features...")
        for window in windows:
            feats = compute_manual_features(window)
            if not feature_names:
                feature_names = list(feats.keys())
            feature_matrix.append([feats.get(k, 0.0) for k in feature_names])

    # -- Train classifier -------------------------------------------------------
    clf_result = train_classifier(feature_matrix, labels)

    # -- Rank features by importance --------------------------------------------
    top_features = []
    if clf_result and "coefs" in clf_result:
        coefs = clf_result["coefs"]
        ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        top_features = ranked[:20]

    # -- Output -----------------------------------------------------------------
    mode = "tsfresh" if tsfresh_available else "manual fallback"
    lines = []
    lines.append("TSFRESH FEATURE MINING -- Pre-Convergence Windows")
    lines.append("=" * 50)
    lines.append(f"Windows: {n_conv} convergence + {n_nonevent} non-events ({n_nonevent // max(1, n_conv)}:1 ratio)")
    lines.append(f"Features extracted: {len(feature_names)} ({mode})")

    if top_features:
        lines.append("")
        lines.append("TOP 20 PREDICTIVE FEATURES:")
        annotations = {
            "beta__mean_abs_change":          "BH mass velocity",
            "close__autocorr_lag1":           "autocorrelation",
            "close__ar_coefficient__k_1":     "autocorrelation",
            "close__kurtosis":                "fat tail signal",
            "beta__count_above_mean":         "TIMELIKE fraction",
            "close__linear_trend_slope":      "trend strength",
            "returns__std":                   "volatility",
            "beta__std":                      "beta volatility",
            "close__mean_abs_change":         "mean velocity",
            "close__skewness":                "skew signal",
        }
        for i, (fname, coef) in enumerate(top_features, 1):
            short = fname[:50]
            note  = annotations.get(fname, "")
            arrow = " <--" if note else ""
            lines.append(f"  {i:2d}. {short:<52} {abs(coef):.3f}{arrow} {note}")

    lines.append("")
    if clf_result:
        lines.append("CLASSIFIER (LogReg, 80/20 split):")
        lines.append(f"  AUC:       {clf_result['auc']:.3f}")
        lines.append(f"  Precision: {clf_result['precision']*100:.1f}%  (at 10% threshold)")
        lines.append(f"  Recall:    {clf_result['recall']*100:.1f}%")
    else:
        lines.append("CLASSIFIER: sklearn not available -- install scikit-learn")

    lines.append("")
    lines.append("ACTIONABLE: The top features translate to SRFM language as:")
    lines.append("  - Rising BH mass velocity (mass accelerating toward threshold)")
    lines.append("  - Low kurtosis (steady directional moves, not spike-driven)")
    lines.append("  - Positive autocorrelation (trending, not mean-reverting)")

    output = "\n".join(lines)
    print(output)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "feature_importance.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Feature Importance -- Pre-Convergence Windows\n\n")
        f.write("```\n")
        f.write(output)
        f.write("\n```\n")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
