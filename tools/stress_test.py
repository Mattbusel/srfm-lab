"""
LARSA v11 — Stress Test Tool
Replay 6 historical crisis scenarios through the v11 sizing engine.
Usage: python tools/stress_test.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import json
import os
import random

# ── Constants ────────────────────────────────────────────────────────────────
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004, "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012, "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,  "YM": 0.004},
}
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
PORTFOLIO_DAILY_RISK = 0.01
N_INST = 3
CORR = 0.90
CORR_FACTOR = math.sqrt(N_INST + N_INST * (N_INST - 1) * CORR)   # sqrt(3 + 6*0.90) = 2.898
PER_INST_RISK = PORTFOLIO_DAILY_RISK / CORR_FACTOR                 # ~0.003451

EQUITY = 10_000_000  # $10M

# ── Sizing helpers ────────────────────────────────────────────────────────────

def v11_size(atr: float, price: float, tf_score: int, direction: int = 1) -> float:
    daily_vol_pct = (atr / price) * math.sqrt(6.5)
    raw = PER_INST_RISK / daily_vol_pct
    return min(raw, TF_CAP[tf_score]) * direction


def v11_size_no_corr(atr: float, price: float, tf_score: int, direction: int = 1) -> float:
    """v11 without the correlation adjustment (uses full 1% per instrument)."""
    daily_vol_pct = (atr / price) * math.sqrt(6.5)
    raw = PORTFOLIO_DAILY_RISK / daily_vol_pct
    return min(raw, TF_CAP[tf_score]) * direction


def v9_size() -> float:
    """v9: fixed 0.65 regardless of vol."""
    return 0.65


def v10_size(atr: float, price: float) -> float:
    """v10: 1% per instrument, no corr adj, no TF cap."""
    daily_vol_pct = (atr / price) * math.sqrt(6.5)
    return PORTFOLIO_DAILY_RISK / daily_vol_pct


def dollar_loss(size_frac: float, daily_pct_move: float, equity: float) -> float:
    return abs(size_frac) * abs(daily_pct_move) * equity


# ── Crisis scenarios ──────────────────────────────────────────────────────────
# Each entry: name, description, normal_atr_pct, crisis_atr_mult, daily_pct_drop,
#             duration_days, instrument, price, tf_score

CRISES = [
    {
        "id":           1,
        "name":         "Volmageddon",
        "date":         "Feb 5 2018",
        "desc":         "VIX 11→50; ES -4% single day then rally",
        "instrument":   "ES",
        "price":        2700.0,
        "normal_atr":   0.008,      # 0.8% of price per bar
        "atr_mult":     6.0,
        "daily_drop":   0.040,
        "duration":     1,
        "tf_score":     5,
    },
    {
        "id":           2,
        "name":         "COVID Crash",
        "date":         "Mar 2020",
        "desc":         "S&P -34% over 23 days; avg -1.5%/day",
        "instrument":   "ES",
        "price":        3300.0,
        "normal_atr":   0.010,
        "atr_mult":     8.0,
        "daily_drop":   0.015,
        "duration":     23,
        "tf_score":     4,
    },
    {
        "id":           3,
        "name":         "2022 Rate Shock",
        "date":         "Jan–Dec 2022",
        "desc":         "S&P -27% over 9 months; avg -0.15%/day",
        "instrument":   "NQ",
        "price":        16000.0,
        "normal_atr":   0.012,
        "atr_mult":     2.5,
        "daily_drop":   0.0015,
        "duration":     180,
        "tf_score":     3,
    },
    {
        "id":           4,
        "name":         "Flash Crash",
        "date":         "May 6 2010",
        "desc":         "-9% in 30 min then recovered; ATR 20× on 1h bar",
        "instrument":   "ES",
        "price":        1130.0,
        "normal_atr":   0.009,
        "atr_mult":     20.0,
        "daily_drop":   0.090,
        "duration":     1,
        "tf_score":     6,
    },
    {
        "id":           5,
        "name":         "Aug 2015 China",
        "date":         "Aug 2015",
        "desc":         "-11% in 5 days; avg -2.2%/day",
        "instrument":   "YM",
        "price":        17500.0,
        "normal_atr":   0.011,
        "atr_mult":     5.0,
        "daily_drop":   0.022,
        "duration":     5,
        "tf_score":     5,
    },
    {
        "id":           6,
        "name":         "Dec 2018 Selloff",
        "date":         "Dec 2018",
        "desc":         "-20% in 3 weeks; avg -1.0%/day",
        "instrument":   "ES",
        "price":        2500.0,
        "normal_atr":   0.009,
        "atr_mult":     3.0,
        "daily_drop":   0.010,
        "duration":     21,
        "tf_score":     4,
    },
]

# ── Simulation helpers ────────────────────────────────────────────────────────

def simulate_episode(crisis: dict, equity_start: float = EQUITY) -> dict:
    """
    Simulate a 100-bar episode: 30 warmup + crisis + recovery to fill 100.
    Returns per-bar equity for v9, v10, v11 and margin-call counts.
    """
    warmup = 30
    crisis_bars = crisis["duration"]
    recovery = max(100 - warmup - crisis_bars, 0)

    price = crisis["price"]
    normal_atr_pct = crisis["normal_atr"]
    crisis_atr_pct = normal_atr_pct * crisis["atr_mult"]
    tf_score = crisis["tf_score"]

    random.seed(42)

    def run_version(sizer_fn):
        eq = equity_start
        eq_curve = [eq]
        margin_calls = 0
        for phase, n_bars in [("warmup", warmup), ("crisis", crisis_bars), ("recovery", recovery)]:
            for _ in range(n_bars):
                atr_pct = normal_atr_pct if phase in ("warmup", "recovery") else crisis_atr_pct
                atr = atr_pct * price
                size = sizer_fn(atr, price, tf_score)
                # daily return: crisis phase is adverse
                if phase == "crisis":
                    daily_ret = -(crisis["daily_drop"] + random.gauss(0, atr_pct * 0.2))
                elif phase == "warmup":
                    daily_ret = random.gauss(0.0002, atr_pct * 0.5)
                else:
                    daily_ret = random.gauss(0.0003, atr_pct * 0.4)
                pnl = size * daily_ret * eq
                eq += pnl
                if eq < 0:
                    margin_calls += 1
                    eq = 0.0
                eq_curve.append(eq)
        return eq_curve, margin_calls

    def v11_fn(atr, price, tf_score): return v11_size(atr, price, tf_score)
    def v11nc_fn(atr, price, tf_score): return v11_size_no_corr(atr, price, tf_score)
    def v9_fn(atr, price, tf_score):  return v9_size()
    def v10_fn(atr, price, tf_score): return v10_size(atr, price)

    v11_curve, v11_mc   = run_version(v11_fn)
    v11nc_curve, v11nc_mc = run_version(v11nc_fn)
    v9_curve,  v9_mc    = run_version(v9_fn)
    v10_curve, v10_mc   = run_version(v10_fn)

    return {
        "v11":   v11_curve,
        "v11nc": v11nc_curve,
        "v9":    v9_curve,
        "v10":   v10_curve,
        "v11_mc":   v11_mc,
        "v11nc_mc": v11nc_mc,
        "v9_mc":    v9_mc,
        "v10_mc":   v10_mc,
    }


# ── ASCII equity curve ─────────────────────────────────────────────────────────

def ascii_curve(series: list, width: int = 80, height: int = 8, label: str = "") -> str:
    if not series:
        return ""
    mn, mx = min(series), max(series)
    rng = mx - mn if mx != mn else 1.0
    lines = []
    for row in range(height - 1, -1, -1):
        threshold = mn + rng * row / (height - 1)
        line = ""
        # downsample to width
        step = max(1, len(series) / width)
        for col in range(width):
            idx = min(int(col * step), len(series) - 1)
            val = series[idx]
            line += "*" if val >= threshold else " "
        lines.append("|" + line)
    lines.append("+" + "-" * width)
    lines.append(f"  {label}  min=${mn/1e6:.2f}M  max=${mx/1e6:.2f}M")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  LARSA v11  —  STRESS TEST  (6 Historical Crises)")
    print(f"  Equity: ${EQUITY/1e6:.0f}M   CORR_FACTOR={CORR_FACTOR:.4f}   PER_INST_RISK={PER_INST_RISK:.5f}")
    print("=" * 80)

    results = {}

    for crisis in CRISES:
        inst   = crisis["instrument"]
        price  = crisis["price"]
        tf_sc  = crisis["tf_score"]
        natr   = crisis["normal_atr"] * price
        catr   = natr * crisis["atr_mult"]

        # Point-in-time sizing (at crisis ATR)
        sz_v11     = v11_size(catr, price, tf_sc)
        sz_v11nc   = v11_size_no_corr(catr, price, tf_sc)
        sz_v9      = v9_size()
        sz_v10     = v10_size(catr, price)

        # Max single-day dollar loss at $10M
        drop = crisis["daily_drop"]
        loss_v11   = dollar_loss(sz_v11,   drop, EQUITY)
        loss_v11nc = dollar_loss(sz_v11nc, drop, EQUITY)
        loss_v9    = dollar_loss(sz_v9,    drop, EQUITY)
        loss_v10   = dollar_loss(sz_v10,   drop, EQUITY)

        # Episode simulation
        ep = simulate_episode(crisis)

        print(f"\n{'─'*80}")
        print(f"  CRISIS {crisis['id']}: {crisis['name']}  ({crisis['date']})")
        print(f"  {crisis['desc']}")
        print(f"  Instrument: {inst}  Price: ${price:,.0f}  Normal ATR: {crisis['normal_atr']*100:.2f}%  Crisis ATR: {crisis['normal_atr']*crisis['atr_mult']*100:.2f}%  TF score: {tf_sc}")
        print()
        print(f"  {'Version':<16} {'Size (frac)':<14} {'Max 1-day Loss':>16} {'Margin Calls':>14}")
        print(f"  {'─'*16} {'─'*13} {'─'*16} {'─'*14}")
        for label, sz, loss, mc in [
            ("v11 (corr adj)",  sz_v11,   loss_v11,   ep["v11_mc"]),
            ("v11 (no corr)",   sz_v11nc, loss_v11nc, ep["v11nc_mc"]),
            ("v9  (fixed 0.65)",sz_v9,    loss_v9,    ep["v9_mc"]),
            ("v10 (1%/inst)",   sz_v10,   loss_v10,   ep["v10_mc"]),
        ]:
            print(f"  {label:<16} {sz:>12.4f}   ${loss:>13,.0f}   {mc:>12}")

        # ASCII equity curve (v11 vs v9)
        print()
        print("  Equity curve — v11 (with corr adj):")
        print(ascii_curve(ep["v11"], width=76, height=7, label="v11"))
        print()
        print("  Equity curve — v9 (fixed 0.65):")
        print(ascii_curve(ep["v9"], width=76, height=7, label="v9"))

        results[crisis["name"]] = {
            "crisis": crisis,
            "sizing": {
                "v11":   sz_v11,
                "v11nc": sz_v11nc,
                "v9":    sz_v9,
                "v10":   sz_v10,
            },
            "max_1day_loss_usd": {
                "v11":   loss_v11,
                "v11nc": loss_v11nc,
                "v9":    loss_v9,
                "v10":   loss_v10,
            },
            "margin_calls": {
                "v11":   ep["v11_mc"],
                "v11nc": ep["v11nc_mc"],
                "v9":    ep["v9_mc"],
                "v10":   ep["v10_mc"],
            },
            "equity_final": {
                "v11":   ep["v11"][-1],
                "v11nc": ep["v11nc"][-1],
                "v9":    ep["v9"][-1],
                "v10":   ep["v10"][-1],
            },
        }

    print(f"\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"  {'Crisis':<22} {'v11 loss':>12} {'v9 loss':>12} {'v10 loss':>12} {'v11 MC':>8} {'v9 MC':>8}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*8}")
    for c in CRISES:
        r = results[c["name"]]
        print(f"  {c['name']:<22} "
              f"${r['max_1day_loss_usd']['v11']:>10,.0f}  "
              f"${r['max_1day_loss_usd']['v9']:>10,.0f}  "
              f"${r['max_1day_loss_usd']['v10']:>10,.0f}  "
              f"{r['margin_calls']['v11']:>7}  "
              f"{r['margin_calls']['v9']:>7}")

    # Write JSON
    os.makedirs("results", exist_ok=True)
    out_path = "results/stress_test.json"
    # strip equity curves (too large) before serializing
    for k in results:
        pass  # we never stored the full curves in results dict
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results written to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
