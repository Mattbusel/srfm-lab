"""
LARSA v11 — Rolling Correlation Monitor
Analyze ES/NQ/YM correlation over synthetic 7-year data (or real CSV data).
Usage:
  python tools/corr_monitor.py
  python tools/corr_monitor.py --real-data
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import json
import os
import argparse
import random
from collections import defaultdict

# ── Constants ────────────────────────────────────────────────────────────────
PORTFOLIO_DAILY_RISK = 0.01
CORR_FACTOR_ASSUMED  = math.sqrt(3 + 6 * 0.90)   # 2.898 (v11 assumption)
PER_INST_RISK        = PORTFOLIO_DAILY_RISK / CORR_FACTOR_ASSUMED
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
TF_SCORE_DEFAULT     = 5

CORR_DANGER  = 0.95
CORR_DIVERSIFY = 0.70

WINDOW_SHORT = 20
WINDOW_LONG  = 60

# ── Sizing ────────────────────────────────────────────────────────────────────

def v11_size(atr_pct: float, tf_score: int = TF_SCORE_DEFAULT) -> float:
    daily_vol = atr_pct * math.sqrt(6.5)
    raw = PER_INST_RISK / daily_vol
    return min(raw, TF_CAP[tf_score])


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic(n_bars: int = 7 * 252, seed: int = 42) -> dict:
    """
    Generate synthetic hourly OHLCV data for ES, NQ, YM with realistic
    regime-switching correlation and vol dynamics.
    Regimes:
      - normal:   corr ~0.88, vol moderate
      - risk-off: corr ~0.96, vol high
      - diverge:  corr ~0.55, vol low
    """
    random.seed(seed)
    n = n_bars
    prices = {"ES": 3500.0, "NQ": 12000.0, "YM": 31000.0}
    # vol base (daily fraction)
    vol_base = {"ES": 0.009, "NQ": 0.011, "YM": 0.010}

    # Regime schedule: cycle through roughly 30-bar epochs
    def regime_at(i):
        epoch = i // 40
        choices = ["normal", "normal", "normal", "risk-off", "diverge", "normal", "risk-off"]
        return choices[epoch % len(choices)]

    bars = {inst: [] for inst in ["ES", "NQ", "YM"]}
    for i in range(n):
        regime = regime_at(i)
        if regime == "risk-off":
            corr, vol_mult = 0.96, 2.0
        elif regime == "diverge":
            corr, vol_mult = 0.55, 0.8
        else:
            corr, vol_mult = 0.88, 1.0

        # Cholesky for 3-asset corr matrix (all pairwise = corr)
        # L = [[1,0,0],[corr, sqrt(1-corr^2), 0], [...]]
        c = corr
        s = math.sqrt(max(1 - c * c, 1e-9))
        z1 = random.gauss(0, 1)
        z2 = random.gauss(0, 1)
        z3 = random.gauss(0, 1)
        # correlated standard normals
        es_n  = z1
        nq_n  = c * z1 + s * z2
        ym_n  = c * z1 + s * z3   # approximate; sufficient for corr monitoring

        for inst, n_val in zip(["ES", "NQ", "YM"], [es_n, nq_n, ym_n]):
            v     = vol_base[inst] * vol_mult
            ret   = n_val * v
            price = prices[inst] * (1 + ret)
            prices[inst] = max(price, prices[inst] * 0.50)
            bars[inst].append({
                "close":   prices[inst],
                "ret":     ret,
                "atr_pct": v,
            })

    return bars


# ── Statistics helpers ────────────────────────────────────────────────────────

def rolling_corr(x: list, y: list, window: int) -> list:
    """Pearson correlation over rolling window."""
    result = [None] * len(x)
    for i in range(window - 1, len(x)):
        xs = x[i - window + 1: i + 1]
        ys = y[i - window + 1: i + 1]
        mx = sum(xs) / window
        my = sum(ys) / window
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx  = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy  = math.sqrt(sum((b - my) ** 2 for b in ys))
        if dx * dy < 1e-12:
            result[i] = None
        else:
            result[i] = num / (dx * dy)
    return result


def rolling_vol(rets: list, window: int) -> list:
    result = [None] * len(rets)
    for i in range(window - 1, len(rets)):
        sl = rets[i - window + 1: i + 1]
        m  = sum(sl) / window
        result[i] = math.sqrt(sum((r - m) ** 2 for r in sl) / (window - 1))
    return result


# ── Flag periods ──────────────────────────────────────────────────────────────

def find_flag_periods(corr_series: list, threshold: float, above: bool = True) -> list:
    """Return list of (start, end, avg_corr) for contiguous flag spans."""
    periods = []
    in_period = False
    start = 0
    vals = []
    for i, c in enumerate(corr_series):
        if c is None:
            if in_period:
                periods.append((start, i - 1, sum(vals) / len(vals)))
                in_period = False
                vals = []
            continue
        triggered = (c > threshold) if above else (c < threshold)
        if triggered and not in_period:
            in_period = True
            start = i
            vals = [c]
        elif triggered and in_period:
            vals.append(c)
        elif not triggered and in_period:
            periods.append((start, i - 1, sum(vals) / len(vals)))
            in_period = False
            vals = []
    if in_period and vals:
        periods.append((start, len(corr_series) - 1, sum(vals) / len(vals)))
    return periods


def corr_factor_actual(corr: float) -> float:
    return math.sqrt(3 + 6 * corr)


# ── Load real CSV ─────────────────────────────────────────────────────────────

def load_real_csv(path: str) -> list:
    """Load CSV with at least a 'close' column. Returns list of dicts."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        headers = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if headers is None:
                headers = [h.strip().lower() for h in parts]
                continue
            d = dict(zip(headers, parts))
            try:
                rows.append({"close": float(d.get("close", d.get("adj close", 0)))})
            except ValueError:
                pass
    # compute returns
    for i in range(1, len(rows)):
        prev = rows[i - 1]["close"]
        curr = rows[i]["close"]
        rows[i]["ret"] = (curr - prev) / prev if prev else 0.0
        rows[i]["atr_pct"] = abs(rows[i]["ret"])
    if rows:
        rows[0]["ret"] = 0.0
        rows[0]["atr_pct"] = 0.0
    return rows


# ── Print helpers ─────────────────────────────────────────────────────────────

def mini_corr_chart(series: list, width: int = 70) -> str:
    """ASCII sparkline for a correlation series, scaled 0→1."""
    clean = [c for c in series if c is not None]
    if not clean:
        return ""
    step = max(1, len(series) // width)
    chars = " ._,-~=+*#@"
    line  = ""
    for i in range(0, len(series), step):
        c = series[i]
        if c is None:
            line += " "
        else:
            idx = int(max(0.0, min(c, 1.0)) * (len(chars) - 1))
            line += chars[idx]
        if len(line) >= width:
            break
    return "[" + line[:width] + "]"


# ── Analysis engine ───────────────────────────────────────────────────────────

def analyze(bars: dict, label: str) -> dict:
    es_rets  = [b["ret"] for b in bars["ES"]]
    nq_rets  = [b["ret"] for b in bars["NQ"]]
    ym_rets  = [b["ret"] for b in bars["YM"]]
    es_atr   = [b["atr_pct"] for b in bars["ES"]]
    n        = len(es_rets)

    es_nq_20  = rolling_corr(es_rets, nq_rets, WINDOW_SHORT)
    es_ym_20  = rolling_corr(es_rets, ym_rets, WINDOW_SHORT)
    es_nq_60  = rolling_corr(es_rets, nq_rets, WINDOW_LONG)
    es_ym_60  = rolling_corr(es_rets, ym_rets, WINDOW_LONG)
    es_vol_20 = rolling_vol(es_rets, WINDOW_SHORT)

    # v11 position size at each bar (using rolling vol)
    v11_sizes = []
    for i in range(n):
        v = es_vol_20[i]
        if v is None or v < 1e-9:
            v11_sizes.append(None)
        else:
            v11_sizes.append(v11_size(v))

    # Flag periods — use 60-bar corr for stability
    danger_periods_nq  = find_flag_periods(es_nq_60, CORR_DANGER,    above=True)
    danger_periods_ym  = find_flag_periods(es_ym_60, CORR_DANGER,    above=True)
    divers_periods_nq  = find_flag_periods(es_nq_60, CORR_DIVERSIFY, above=False)
    divers_periods_ym  = find_flag_periods(es_ym_60, CORR_DIVERSIFY, above=False)

    # Compute avg v11 size and corr_factor during flagged periods
    def period_stats(periods, corr_series, size_series):
        stats = []
        for start, end, avg_corr in periods:
            dur = end - start + 1
            szs = [s for s in size_series[start:end+1] if s is not None]
            avg_sz = sum(szs) / len(szs) if szs else 0.0
            cf_actual   = corr_factor_actual(avg_corr)
            cf_assumed  = CORR_FACTOR_ASSUMED
            conservative = cf_actual >= cf_assumed  # assumed is >=actual → we over-reduced → conservative
            stats.append({
                "start":       start,
                "end":         end,
                "duration":    dur,
                "avg_corr":    avg_corr,
                "avg_v11_size":avg_sz,
                "cf_actual":   cf_actual,
                "cf_assumed":  cf_assumed,
                "conservative":conservative,
            })
        return stats

    dn_nq = period_stats(danger_periods_nq, es_nq_60, v11_sizes)
    dn_ym = period_stats(danger_periods_ym, es_ym_60, v11_sizes)
    dv_nq = period_stats(divers_periods_nq, es_nq_60, v11_sizes)
    dv_ym = period_stats(divers_periods_ym, es_ym_60, v11_sizes)

    # Overall stats
    valid_nq60 = [c for c in es_nq_60 if c is not None]
    valid_ym60 = [c for c in es_ym_60 if c is not None]
    stats = {
        "n_bars":            n,
        "es_nq_avg_corr_60": sum(valid_nq60) / len(valid_nq60) if valid_nq60 else 0,
        "es_ym_avg_corr_60": sum(valid_ym60) / len(valid_ym60) if valid_ym60 else 0,
        "es_nq_max_corr_60": max(valid_nq60) if valid_nq60 else 0,
        "es_ym_max_corr_60": max(valid_ym60) if valid_ym60 else 0,
        "es_nq_min_corr_60": min(valid_nq60) if valid_nq60 else 0,
        "es_ym_min_corr_60": min(valid_ym60) if valid_ym60 else 0,
        "danger_periods_es_nq":  dn_nq,
        "danger_periods_es_ym":  dn_ym,
        "divers_periods_es_nq":  dv_nq,
        "divers_periods_es_ym":  dv_ym,
    }

    # ── Terminal output ───────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  CORRELATION ANALYSIS: {label}  ({n} bars)")
    print(f"{'═'*80}")
    print(f"  {'Pair':<12} {'Avg 60-bar':>12} {'Min':>8} {'Max':>8}  {'Danger(>0.95)':>14}  {'Diversify(<0.70)':>16}")
    print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*8}  {'─'*14}  {'─'*16}")
    for pair, avg, mn, mx, dn_p, dv_p in [
        ("ES/NQ", stats["es_nq_avg_corr_60"], stats["es_nq_min_corr_60"], stats["es_nq_max_corr_60"], dn_nq, dv_nq),
        ("ES/YM", stats["es_ym_avg_corr_60"], stats["es_ym_min_corr_60"], stats["es_ym_max_corr_60"], dn_ym, dv_ym),
    ]:
        print(f"  {pair:<12} {avg:>12.4f} {mn:>8.4f} {mx:>8.4f}  {len(dn_p):>14}  {len(dv_p):>16}")

    print()
    print(f"  60-bar rolling correlation (ES/NQ, 0=low 1=high):")
    print(f"  {mini_corr_chart(es_nq_60)}")
    print(f"  60-bar rolling correlation (ES/YM):")
    print(f"  {mini_corr_chart(es_ym_60)}")

    def print_periods(title, periods):
        if not periods:
            print(f"\n  {title}: none found")
            return
        print(f"\n  {title} ({len(periods)} periods):")
        print(f"  {'Start':>8} {'End':>8} {'Dur':>6} {'Avg Corr':>10} {'CF actual':>10} {'CF assumed':>11} {'Conservative?':>14} {'Avg v11 size':>14}")
        print(f"  {'─'*8} {'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*11} {'─'*14} {'─'*14}")
        for p in periods[:20]:  # cap at 20
            print(f"  {p['start']:>8} {p['end']:>8} {p['duration']:>6} "
                  f"{p['avg_corr']:>10.4f} {p['cf_actual']:>10.4f} {p['cf_assumed']:>11.4f} "
                  f"{'YES' if p['conservative'] else 'NO (AGGR)':>14} {p['avg_v11_size']:>14.4f}")

    print_periods(f"DANGER PERIODS ES/NQ (corr > {CORR_DANGER})", dn_nq)
    print_periods(f"DANGER PERIODS ES/YM (corr > {CORR_DANGER})", dn_ym)
    print_periods(f"DIVERSIFY WINDOWS ES/NQ (corr < {CORR_DIVERSIFY})", dv_nq)
    print_periods(f"DIVERSIFY WINDOWS ES/YM (corr < {CORR_DIVERSIFY})", dv_ym)

    # Rolling v11 sizes sparkline
    clean_sizes = [s if s is not None else 0.0 for s in v11_sizes]
    if clean_sizes:
        mn_s = min(s for s in clean_sizes if s > 0) if any(s > 0 for s in clean_sizes) else 0
        mx_s = max(clean_sizes)
        print(f"\n  v11 position size over time (min={mn_s:.3f} max={mx_s:.3f}):")
        step = max(1, len(clean_sizes) // 70)
        chars_line = ""
        for i in range(0, len(clean_sizes), step):
            s = clean_sizes[i]
            if mx_s > mn_s:
                lvl = int((s - mn_s) / (mx_s - mn_s) * 7)
            else:
                lvl = 0
            chars_line += " ._-=+*#"[lvl]
            if len(chars_line) >= 70:
                break
        print(f"  [{chars_line}]")

    return stats


# ── Chart (matplotlib, optional) ─────────────────────────────────────────────

def try_save_chart(bars: dict, label: str, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        es_rets = [b["ret"] for b in bars["ES"]]
        nq_rets = [b["ret"] for b in bars["NQ"]]
        ym_rets = [b["ret"] for b in bars["YM"]]

        es_nq_20 = rolling_corr(es_rets, nq_rets, WINDOW_SHORT)
        es_ym_20 = rolling_corr(es_rets, ym_rets, WINDOW_SHORT)
        es_nq_60 = rolling_corr(es_rets, nq_rets, WINDOW_LONG)
        es_ym_60 = rolling_corr(es_rets, ym_rets, WINDOW_LONG)
        es_vol_20 = rolling_vol(es_rets, WINDOW_SHORT)

        v11_sizes = []
        for v in es_vol_20:
            if v is None or v < 1e-9:
                v11_sizes.append(None)
            else:
                v11_sizes.append(v11_size(v))

        n = len(es_rets)
        xs = list(range(n))

        # Equity curve
        eq = 1.0
        eq_curve = [eq]
        for r in es_rets[1:]:
            sz = v11_sizes[len(eq_curve) - 1] or 0.0
            eq += sz * r * eq
            eq_curve.append(max(eq, 0.0))

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)

        # Panel 1: equity
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(xs, eq_curve, color="#2196F3", linewidth=0.8)
        ax1.set_title(f"{label} — v11 Equity Curve (ES proxy)", fontsize=10)
        ax1.set_ylabel("Equity (normalized)")
        ax1.grid(True, alpha=0.3)

        # Panel 2: rolling correlation
        ax2 = fig.add_subplot(gs[1])
        valid_nq20 = [c if c is not None else float("nan") for c in es_nq_20]
        valid_nq60 = [c if c is not None else float("nan") for c in es_nq_60]
        valid_ym60 = [c if c is not None else float("nan") for c in es_ym_60]
        ax2.plot(xs, valid_nq20, color="#FF9800", linewidth=0.6, alpha=0.7, label="ES/NQ 20-bar")
        ax2.plot(xs, valid_nq60, color="#F44336", linewidth=1.0, label="ES/NQ 60-bar")
        ax2.plot(xs, valid_ym60, color="#9C27B0", linewidth=1.0, label="ES/YM 60-bar", linestyle="--")
        ax2.axhline(CORR_DANGER,    color="red",   linewidth=0.8, linestyle=":", label=f"Danger >{CORR_DANGER}")
        ax2.axhline(CORR_DIVERSIFY, color="green", linewidth=0.8, linestyle=":", label=f"Diversify <{CORR_DIVERSIFY}")
        ax2.axhline(0.90, color="gray", linewidth=0.6, linestyle="--", label="v11 assumed 0.90")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_title("Rolling Realized Correlation", fontsize=10)
        ax2.set_ylabel("Correlation")
        ax2.legend(fontsize=7, ncol=3)
        ax2.grid(True, alpha=0.3)

        # Panel 3: v11 position sizes
        ax3 = fig.add_subplot(gs[2])
        clean_sizes = [s if s is not None else float("nan") for s in v11_sizes]
        ax3.plot(xs, clean_sizes, color="#4CAF50", linewidth=0.8)
        ax3.set_title("v11 Position Size (ES, TF_SCORE=5)", fontsize=10)
        ax3.set_ylabel("Position Fraction")
        ax3.set_xlabel("Bar")
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f"LARSA v11 — Correlation Monitor: {label}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved to {out_path}")
        return True
    except ImportError:
        print("\n  (matplotlib not available — skipping chart)")
        return False
    except Exception as e:
        print(f"\n  (chart error: {e})")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LARSA v11 — Correlation Monitor")
    parser.add_argument("--real-data", action="store_true", help="Use real CSV data if available")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    output = {}

    print("=" * 80)
    print("  LARSA v11  —  ROLLING CORRELATION MONITOR")
    print(f"  Assumed CORR={0.90}  CORR_FACTOR={CORR_FACTOR_ASSUMED:.4f}  PER_INST_RISK={PER_INST_RISK:.5f}")
    print(f"  Danger threshold: >{CORR_DANGER}  |  Diversify window: <{CORR_DIVERSIFY}")
    print(f"  Windows: short={WINDOW_SHORT} bars, long={WINDOW_LONG} bars")
    print("=" * 80)

    # ── Synthetic data run ────────────────────────────────────────────────────
    print("\n  Generating synthetic 7-year dataset (7 × 252 bars)...")
    synth_bars = generate_synthetic(n_bars=7 * 252)
    synth_stats = analyze(synth_bars, "Synthetic 7-year")
    output["synthetic"] = synth_stats

    chart_path = "results/corr_monitor.png"
    try_save_chart(synth_bars, "Synthetic 7-year", chart_path)

    # ── Real data run (optional) ──────────────────────────────────────────────
    real_paths = {
        "ES": "data/ES_hourly_real.csv",
        "NQ": "data/NQ_hourly_real.csv",
        "YM": "data/YM_hourly_real.csv",
    }

    if args.real_data or any(os.path.exists(p) for p in real_paths.values()):
        print("\n  Looking for real data files...")
        real_bars = {}
        all_found = True
        for inst, path in real_paths.items():
            rows = load_real_csv(path)
            if rows:
                real_bars[inst] = rows
                print(f"  Loaded {inst}: {len(rows)} bars from {path}")
            else:
                print(f"  {inst}: not found at {path}")
                all_found = False

        if all_found:
            # Align lengths
            min_len = min(len(real_bars[i]) for i in real_bars)
            for inst in real_bars:
                real_bars[inst] = real_bars[inst][:min_len]
            real_stats = analyze(real_bars, "Real data")
            output["real"] = real_stats
            try_save_chart(real_bars, "Real data", "results/corr_monitor_real.png")
        else:
            print("  Skipping real-data analysis (one or more files missing).")

    # ── Key findings summary ──────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  KEY FINDINGS (synthetic data)")
    s = output["synthetic"]
    print(f"  Avg ES/NQ corr: {s['es_nq_avg_corr_60']:.4f}  (assumed: 0.90)")
    print(f"  Avg ES/YM corr: {s['es_ym_avg_corr_60']:.4f}  (assumed: 0.90)")

    total_danger  = len(s["danger_periods_es_nq"]) + len(s["danger_periods_es_ym"])
    total_divers  = len(s["divers_periods_es_nq"]) + len(s["divers_periods_es_ym"])
    total_danger_bars = sum(p["duration"] for p in s["danger_periods_es_nq"] + s["danger_periods_es_ym"])
    total_bars    = s["n_bars"]

    print(f"  Danger periods (corr>{CORR_DANGER}): {total_danger}  covering ~{total_danger_bars/total_bars*100:.1f}% of bars")
    print(f"  Diversify windows (corr<{CORR_DIVERSIFY}): {total_divers}")

    if s["danger_periods_es_nq"]:
        worst = max(s["danger_periods_es_nq"], key=lambda p: p["avg_corr"])
        print(f"  Worst danger: ES/NQ bar {worst['start']}-{worst['end']}  avg_corr={worst['avg_corr']:.4f}"
              f"  CF_actual={worst['cf_actual']:.3f} vs CF_assumed={worst['cf_assumed']:.3f}"
              f"  ({'CONSERVATIVE' if worst['conservative'] else 'AGGRESSIVE'!s})")
    print(f"{'═'*80}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_json = "results/corr_monitor.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results written to {out_json}")


if __name__ == "__main__":
    main()
