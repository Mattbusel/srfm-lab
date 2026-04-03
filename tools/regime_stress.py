"""
regime_stress.py — Regime-conditioned performance analysis for LARSA v9 vs v11.

For each market regime (BULL, BEAR, SIDEWAYS, HIGH_VOL):
  - Average v11 position size
  - Worst single-bar loss
  - Win rate of signal direction vs next-bar outcome
  - v9 vs v11 sizing comparison
  - BH signal edge detection

Regime detection (no QC):
  BULL:      price > EMA(200) AND EMA(12) > EMA(26)
  BEAR:      price < EMA(200) AND EMA(12) < EMA(26)
  SIDEWAYS:  neither BULL/BEAR AND ADX < 18
  HIGH_VOL:  rolling 20-bar vol > 1.5× rolling 60-bar vol

Stress tests (pure regimes):
  1000-bar sustained BULL
  1000-bar sustained BEAR
  1000-bar HIGH_VOL (3× normal vol)
  1000-bar SIDEWAYS/choppy (mean-reverting, no trend)

Outputs:
    Terminal report
    results/regime_stress.json
    results/regime_stress.png

Usage:
    python tools/regime_stress.py
"""

import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}
TF_CAP        = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD      = 4
N_INST        = 3
BASE_CORR     = 0.90
PORTFOLIO_RISK = 0.01
START_EQUITY  = 1_000_000.0
SYMS          = ["ES", "NQ", "YM"]
ADX_THRESH    = 18.0
HIGH_VOL_MULT = 1.5

REGIMES = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]


# ── BH Instrument ────────────────────────────────────────────────────────────
class BHI:
    def __init__(self, sym, cf, res):
        self.sym = sym; self.cf = cf; self.res = res
        self.bh_mass = 0.0; self.bh_active = False; self.bh_dir = 0
        self.ctl = 0; self.bc = 0; self.prices = []
        self._wu = {"15m": 400, "1h": 120, "1d": 30}.get(res, 120)

    def update(self, c):
        self.bc += 1; self.prices.append(c)
        if len(self.prices) < 2:
            return
        beta = abs(c - self.prices[-2]) / (self.prices[-2] + 1e-9) / (self.cf + 1e-9)
        was = self.bh_active
        if beta < 1.0:
            self.ctl += 1
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * min(2.0, 1 + self.ctl * 0.1)
        else:
            self.ctl = 0; self.bh_mass *= 0.95
        self.bh_active = (self.bh_mass > (1.5 if not was else 1.0)) and self.ctl >= 3
        if not was and self.bh_active:
            lb = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if c > self.prices[-1 - lb] else -1
        elif was and not self.bh_active:
            self.bh_dir = 0
        if self.bc < self._wu:
            self.bh_active = False; self.bh_dir = 0

    def direction(self):
        if self.bh_dir:
            return self.bh_dir
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


class WilderATR:
    def __init__(self, p=14):
        self.p = p; self._trs = []; self._v = None

    def update(self, h, l, pc):
        tr = max(h - l, abs(h - pc), abs(l - pc))
        self._trs.append(tr)
        if len(self._trs) >= self.p:
            self._v = (np.mean(self._trs[-self.p:]) if self._v is None
                       else (self._v * (self.p - 1) + tr) / self.p)
        return self._v or 0.0

    @property
    def value(self):
        return self._v or 0.0


class MRI:
    def __init__(self, sym):
        self.sym = sym
        self.i15 = BHI(sym, CF["15m"][sym], "15m")
        self.i1h = BHI(sym, CF["1h"][sym],  "1h")
        self.i1d = BHI(sym, CF["1d"][sym],  "1d")
        self.atr = WilderATR(14)
        self.last_target = 0.0; self.bars_held = 0; self.pos_floor = 0.0

    def tf_score(self):
        return 4 * self.i1d.bh_active + 2 * self.i1h.bh_active + self.i15.bh_active

    def direction(self):
        if self.i1d.bh_active: return self.i1d.direction()
        if self.i1h.bh_active: return self.i1h.direction()
        if self.i15.bh_active: return self.i15.direction()
        return 0

    def update(self, o, h, l, c, pc):
        for sub in [o, (o + h) / 2, (l + c) / 2, c]:
            self.i15.update(sub)
        self.i1h.update(c)
        self.atr.update(h, l, pc)
        if self.i1h.bc % 6 == 0:
            self.i1d.update(c)


# ── Sizing ────────────────────────────────────────────────────────────────────
def size_v9(ceiling, direction):
    return ceiling * direction if direction else 0.0


def size_v11(ceiling, direction, atr, price, corr=BASE_CORR):
    if not direction or not ceiling:
        return 0.0
    cf = math.sqrt(N_INST + N_INST * (N_INST - 1) * corr)
    per = PORTFOLIO_RISK / cf
    if atr > 0 and price > 0:
        dv = (atr / price) * math.sqrt(6.5)
        raw = per / (dv + 1e-9)
        return min(raw, ceiling) * direction
    return ceiling * direction


# ── EMA helper ────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, period):
        self.p = period; self.v = None
        self.k = 2.0 / (period + 1)

    def update(self, x):
        self.v = x if self.v is None else self.v + self.k * (x - self.v)
        return self.v or 0.0


# ── ADX helper ────────────────────────────────────────────────────────────────
class ADX:
    def __init__(self, p=14):
        self.p = p
        self._pDM = 0.0; self._nDM = 0.0; self._TR = 0.0; self._ADX = 0.0
        self._bc = 0; self._prev_h = None; self._prev_l = None; self._prev_c = None

    def update(self, h, l, c):
        if self._prev_c is None:
            self._prev_h = h; self._prev_l = l; self._prev_c = c
            return 0.0
        ph = self._prev_h; pl = self._prev_l; pc = self._prev_c
        tr = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - ph, 0) if (h - ph) > (pl - l) else 0.0
        ndm = max(pl - l, 0) if (pl - l) > (h - ph) else 0.0
        k = 1.0 / self.p
        self._pDM = self._pDM * (1 - k) + pdm * k
        self._nDM = self._nDM * (1 - k) + ndm * k
        self._TR  = self._TR  * (1 - k) + tr  * k
        if self._TR > 0:
            pDI = 100 * self._pDM / self._TR
            nDI = 100 * self._nDM / self._TR
            denom = pDI + nDI
            dx = 100 * abs(pDI - nDI) / denom if denom > 0 else 0.0
            self._ADX = self._ADX * (1 - k) + dx * k
        self._prev_h = h; self._prev_l = l; self._prev_c = c
        self._bc += 1
        return self._ADX


# ── World generators ──────────────────────────────────────────────────────────
def gen_world_mixed(n=3000, seed=0, corr=0.90):
    """Mixed world for regime detection."""
    rng = np.random.default_rng(seed)
    vols_ann = np.array([0.15, 0.20, 0.14])
    vols = vols_ann / math.sqrt(252 * 6.5)
    C = np.array([[1, corr, corr], [corr, 1, corr], [corr, corr, 1]], dtype=float)
    L = np.linalg.cholesky(np.outer(vols, vols) * C)
    # alternating drift to create mixed regimes
    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    ps = {s: [v] for s, v in starts.items()}
    for i in range(n):
        # cycle: 600-bar segments with different drift
        seg = (i // 600) % 4
        drift = [0.0006, -0.0004, 0.00005, 0.0][seg]
        vol_mult = [1.0, 1.0, 0.4, 3.0][seg]
        z = L @ rng.standard_normal(3)
        for j, s in enumerate(SYMS):
            ret = drift + vol_mult * vols[j] * z[j] * math.sqrt(252 * 6.5)
            ps[s].append(ps[s][-1] * (1 + ret))
    result = {s: [] for s in SYMS}
    for s in SYMS:
        v = ps[s]
        for i in range(1, len(v)):
            hi = max(v[i], v[i - 1]) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(v[i], v[i - 1]) * (1 - abs(rng.normal(0, 0.002)))
            result[s].append({"o": v[i - 1], "h": hi, "l": lo, "c": v[i]})
    return result


def gen_pure_world(regime, n=1000, seed=99):
    """Pure regime world for stress test."""
    rng = np.random.default_rng(seed)
    if regime == "BULL":
        drift, vol_mult, mean_rev = 0.0007, 1.0, 0.0
    elif regime == "BEAR":
        drift, vol_mult, mean_rev = -0.0005, 1.0, 0.0
    elif regime == "HIGH_VOL":
        drift, vol_mult, mean_rev = 0.0001, 3.0, 0.0
    else:  # SIDEWAYS / choppy
        drift, vol_mult, mean_rev = 0.0, 0.3, 0.05

    vols = np.array([0.15, 0.20, 0.14]) / math.sqrt(252 * 6.5)
    corr = BASE_CORR
    C = np.array([[1, corr, corr], [corr, 1, corr], [corr, corr, 1]], dtype=float)
    L = np.linalg.cholesky(np.outer(vols, vols) * C)
    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    ps = {s: [v] for s, v in starts.items()}
    for i in range(n):
        z = L @ rng.standard_normal(3)
        for j, s in enumerate(SYMS):
            curr = ps[s][-1]
            ret = drift - mean_rev * (curr / starts[s] - 1) + vol_mult * vols[j] * z[j] * math.sqrt(252 * 6.5)
            ps[s].append(max(curr * (1 + ret), 1.0))
    result = {s: [] for s in SYMS}
    for s in SYMS:
        v = ps[s]
        for i in range(1, len(v)):
            hi = max(v[i], v[i - 1]) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(v[i], v[i - 1]) * (1 - abs(rng.normal(0, 0.002)))
            result[s].append({"o": v[i - 1], "h": hi, "l": lo, "c": v[i]})
    return result


# ── Regime labelling (on ES price series) ────────────────────────────────────
def label_regimes(bars_es):
    """Returns list of regime labels per bar."""
    ema12  = EMA(12); ema26 = EMA(26); ema200 = EMA(200)
    adx_ind = ADX(14)
    vol_buf_20  = []
    vol_buf_60  = []
    labels = []
    prev_c = bars_es[0]["c"]

    for b in bars_es:
        c = b["c"]; h = b["h"]; l = b["l"]
        e12  = ema12.update(c)
        e26  = ema26.update(c)
        e200 = ema200.update(c)
        adx_val = adx_ind.update(h, l, c)
        ret = abs(c - prev_c) / (prev_c + 1e-9)
        vol_buf_20.append(ret); vol_buf_60.append(ret)
        if len(vol_buf_20) > 20: vol_buf_20.pop(0)
        if len(vol_buf_60) > 60: vol_buf_60.pop(0)
        vol20 = np.std(vol_buf_20) if len(vol_buf_20) >= 2 else 0.0
        vol60 = np.std(vol_buf_60) if len(vol_buf_60) >= 2 else 0.0
        prev_c = c

        is_high_vol = (vol20 > HIGH_VOL_MULT * vol60) and vol60 > 0
        if is_high_vol:
            labels.append("HIGH_VOL")
        elif c > e200 and e12 > e26:
            labels.append("BULL")
        elif c < e200 and e12 < e26:
            labels.append("BEAR")
        elif adx_val < ADX_THRESH:
            labels.append("SIDEWAYS")
        else:
            labels.append("SIDEWAYS")  # default catch-all

    return labels


# ── Run one version through a world, collecting per-regime stats ─────────────
def run_version(bars, version, regime_labels):
    """
    Returns dict keyed by regime with lists of per-bar outcomes.
    Each outcome: (abs_position, bar_pnl_frac, signal_direction, next_bar_return)
    """
    insts   = {s: MRI(s) for s in SYMS}
    equity  = START_EQUITY
    positions = {s: 0.0 for s in SYMS}
    pc      = {s: bars[s][0]["c"] for s in SYMS}
    n       = min(len(bars[s]) for s in SYMS)
    regime_data = {r: [] for r in REGIMES}

    for i in range(n):
        # mark-to-market
        bar_pnl = 0.0
        for s in SYMS:
            b = bars[s][i]
            pos = positions[s]
            if abs(pos) > 1e-6 and pc[s] > 0:
                gain = pos * (b["c"] - pc[s]) / pc[s]
                equity += gain * equity
                bar_pnl += gain

        pr = {}
        for s in SYMS:
            b = bars[s][i]
            insts[s].update(b["o"], b["h"], b["l"], b["c"], pc[s])
            pc[s] = b["c"]; pr[s] = b["c"]

        equity = max(equity, 0.0)

        # sizes
        raw = {}
        for s in SYMS:
            inst = insts[s]
            tfs = inst.tf_score(); ceiling = TF_CAP[tfs]
            if tfs == 1 and abs(inst.last_target) < 0.01: ceiling = 0.0
            d = inst.direction(); a = inst.atr.value; p = pr[s]
            if ceiling == 0.0:
                tgt = 0.0
            elif version == "v9":
                tgt = size_v9(ceiling, d)
            else:
                tgt = size_v11(ceiling, d, a, p)
            raw[s] = tgt

        total = sum(abs(v) for v in raw.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        avg_pos = 0.0; avg_dir = 0
        for s in SYMS:
            inst = insts[s]
            tgt = raw[s] * scale
            if abs(tgt - inst.last_target) >= 0.02:
                inst.last_target = tgt
            positions[s] = inst.last_target
            avg_pos += abs(inst.last_target)
            avg_dir += inst.direction()

        # next-bar return for win rate calc
        if i + 1 < n:
            next_ret = (bars["ES"][i + 1]["c"] - pr["ES"]) / (pr["ES"] + 1e-9)
        else:
            next_ret = 0.0

        reg = regime_labels[i] if i < len(regime_labels) else "SIDEWAYS"
        regime_data[reg].append({
            "abs_pos":  avg_pos / N_INST,
            "bar_pnl":  bar_pnl,
            "signal_dir": 1 if avg_dir > 0 else (-1 if avg_dir < 0 else 0),
            "next_ret": next_ret,
        })

    return regime_data


# ── Summarise per-regime data ─────────────────────────────────────────────────
def summarise(regime_data):
    out = {}
    for reg, rows in regime_data.items():
        if not rows:
            out[reg] = {"n_bars": 0}
            continue
        positions = [r["abs_pos"] for r in rows]
        pnls      = [r["bar_pnl"] for r in rows]
        dirs      = [r["signal_dir"] for r in rows]
        nrets     = [r["next_ret"] for r in rows]
        # win rate: signal dir matches sign of next_bar return
        wins = sum(1 for d, nr in zip(dirs, nrets)
                   if d != 0 and math.copysign(1, d) == math.copysign(1, nr))
        n_dir = sum(1 for d in dirs if d != 0)
        # worst single-bar loss
        worst = min(pnls) if pnls else 0.0
        out[reg] = {
            "n_bars":       len(rows),
            "avg_abs_pos":  float(np.mean(positions)),
            "worst_bar_pnl":float(worst * 100),
            "win_rate":     float(wins / n_dir) if n_dir > 0 else 0.0,
            "n_trades":     n_dir,
            "bh_edge":      float(wins / n_dir - 0.5) if n_dir > 0 else 0.0,
        }
    return out


# ── Stress test single pure regime ───────────────────────────────────────────
def stress_one(regime, version, seed=99, n=1000):
    bars = gen_pure_world(regime, n=n, seed=seed)
    labels = [regime] * n  # all bars are this regime
    rd = run_version(bars, version, labels)
    rows = rd[regime]
    if not rows:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0}
    eq = START_EQUITY
    curve = [eq]
    for r in rows:
        eq *= (1 + r["bar_pnl"])
        eq = max(eq, 0.0)
        curve.append(eq)
    arr = np.array(curve)
    pk  = arr[0]; mx = 0.0
    for v in arr:
        pk = max(pk, v); mx = max(mx, (pk - v) / (pk + 1e-9))
    pnls = [r["bar_pnl"] for r in rows]
    dirs = [r["signal_dir"] for r in rows]
    nrets = [r["next_ret"] for r in rows]
    wins = sum(1 for d, nr in zip(dirs, nrets)
               if d != 0 and math.copysign(1, d) == math.copysign(1, nr))
    n_dir = sum(1 for d in dirs if d != 0)
    return {
        "return_pct":  float((arr[-1] / arr[0] - 1) * 100),
        "max_dd_pct":  float(mx * 100),
        "win_rate":    float(wins / n_dir) if n_dir > 0 else 0.0,
        "avg_abs_pos": float(np.mean([r["abs_pos"] for r in rows])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*65}")
    print("  LARSA v9 vs v11  —  Regime Stress Analysis")
    print(f"{'═'*65}\n")

    N_WORLDS = 5

    # ── Mixed-world regime breakdown ─────────────────────────────────────────
    print("  Running 5 mixed worlds for regime-conditioned analysis...")
    combined_v9  = {r: [] for r in REGIMES}
    combined_v11 = {r: [] for r in REGIMES}
    regime_bar_counts = {r: 0 for r in REGIMES}

    for seed in range(N_WORLDS):
        bars = gen_world_mixed(n=3000, seed=seed)
        labels = label_regimes(bars["ES"])
        for r in REGIMES:
            regime_bar_counts[r] += labels.count(r)
        rd_v9  = run_version(bars, "v9",  labels)
        rd_v11 = run_version(bars, "v11", labels)
        for r in REGIMES:
            combined_v9[r].extend(rd_v9[r])
            combined_v11[r].extend(rd_v11[r])

    summary_v9  = summarise(combined_v9)
    summary_v11 = summarise(combined_v11)

    print(f"\n{'─'*65}")
    print("  REGIME BREAKDOWN — across 5 mixed worlds (3000 bars each)")
    print(f"{'─'*65}")
    print(f"  {'Regime':<12} {'Bars':>6}  {'v11 avg|pos|':>13}  {'v9 avg|pos|':>12}  "
          f"{'v11 wr':>8}  {'v9 wr':>8}  {'v11 edge':>10}  {'worst bar%':>11}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*13}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*11}")

    for reg in REGIMES:
        s11 = summary_v11[reg]; s9 = summary_v9[reg]
        n   = s11.get("n_bars", 0)
        print(f"  {reg:<12} {n:>6}  "
              f"{s11.get('avg_abs_pos',0):>13.4f}  "
              f"{s9.get('avg_abs_pos',0):>12.4f}  "
              f"{s11.get('win_rate',0)*100:>7.1f}%  "
              f"{s9.get('win_rate',0)*100:>7.1f}%  "
              f"{s11.get('bh_edge',0)*100:>+9.1f}%  "
              f"{s11.get('worst_bar_pnl',0):>10.3f}%")

    # BH negative edge detection
    print(f"\n  BH signal edge analysis:")
    for reg in REGIMES:
        e = summary_v11[reg].get("bh_edge", 0)
        verdict = "NEGATIVE EDGE" if e < -0.02 else ("NEUTRAL" if abs(e) < 0.02 else "POSITIVE EDGE")
        print(f"    {reg:<12}: {e*100:+.1f}% edge  → {verdict}")

    # v9 vs v11 sizing comparison
    print(f"\n  v9 vs v11 sizing comparison:")
    for reg in REGIMES:
        s9 = summary_v9[reg]; s11 = summary_v11[reg]
        ratio = (s11.get("avg_abs_pos", 0) / (s9.get("avg_abs_pos", 1e-9)))
        better = "v11" if s11.get("win_rate", 0) >= s9.get("win_rate", 0) else "v9"
        print(f"    {reg:<12}: v11/v9 size ratio = {ratio:.2f}×,  "
              f"better win rate → {better}")

    # ── Pure regime stress tests ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PURE REGIME STRESS TESTS (1000 bars each)")
    print(f"{'─'*65}")
    print(f"  {'Regime':<12} {'Ver':>4}  {'Return%':>9}  {'MaxDD%':>8}  {'WinRate':>9}  {'Avg|Pos|':>10}")
    print(f"  {'─'*12}  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*10}")

    stress_results = {}
    for reg in REGIMES:
        r9  = stress_one(reg, "v9",  n=1000)
        r11 = stress_one(reg, "v11", n=1000)
        stress_results[reg] = {"v9": r9, "v11": r11}
        for ver, r in [("v9", r9), ("v11", r11)]:
            print(f"  {reg:<12} {ver:>4}  "
                  f"{r['return_pct']:>+8.1f}%  "
                  f"{r['max_dd_pct']:>7.1f}%  "
                  f"{r['win_rate']*100:>8.1f}%  "
                  f"{r['avg_abs_pos']:>10.4f}")

    # ── JSON output ──────────────────────────────────────────────────────────
    json_out = {
        "n_worlds": N_WORLDS,
        "regime_bar_counts": regime_bar_counts,
        "mixed_world_summary": {
            "v9":  {r: summary_v9[r]  for r in REGIMES},
            "v11": {r: summary_v11[r] for r in REGIMES},
        },
        "stress_tests": stress_results,
    }
    json_path = os.path.join(RESULTS, "regime_stress.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Saved → {json_path}")

    # ── Chart ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        colors = {"BULL": "#54a24b", "BEAR": "#e45756", "SIDEWAYS": "#79706e", "HIGH_VOL": "#f58518"}
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("LARSA v9 vs v11 — Regime Stress Analysis", fontsize=13, fontweight="bold")

        # Panel 1: Average absolute position by regime, v9 vs v11
        ax = axes[0, 0]
        x = np.arange(len(REGIMES)); w = 0.35
        vals9  = [summary_v9[r].get("avg_abs_pos", 0)  for r in REGIMES]
        vals11 = [summary_v11[r].get("avg_abs_pos", 0) for r in REGIMES]
        ax.bar(x - w/2, vals9,  w, label="v9",  color="#4c78a8", alpha=0.8)
        ax.bar(x + w/2, vals11, w, label="v11", color="#f58518", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REGIMES, fontsize=9)
        ax.set_ylabel("Avg |position| (frac of equity)")
        ax.set_title("Average Position Size by Regime")
        ax.legend()

        # Panel 2: Win rate by regime v9 vs v11
        ax = axes[0, 1]
        wr9  = [summary_v9[r].get("win_rate", 0) * 100  for r in REGIMES]
        wr11 = [summary_v11[r].get("win_rate", 0) * 100 for r in REGIMES]
        ax.bar(x - w/2, wr9,  w, label="v9",  color="#4c78a8", alpha=0.8)
        ax.bar(x + w/2, wr11, w, label="v11", color="#f58518", alpha=0.8)
        ax.axhline(50, color="red", ls="--", lw=1.2, label="50% (no edge)")
        ax.set_xticks(x); ax.set_xticklabels(REGIMES, fontsize=9)
        ax.set_ylabel("Win rate (%)")
        ax.set_title("Signal Win Rate by Regime")
        ax.legend()

        # Panel 3: Stress test returns
        ax = axes[1, 0]
        ret9  = [stress_results[r]["v9"]["return_pct"]  for r in REGIMES]
        ret11 = [stress_results[r]["v11"]["return_pct"] for r in REGIMES]
        bar9  = ax.bar(x - w/2, ret9,  w, label="v9",  color="#4c78a8", alpha=0.8)
        bar11 = ax.bar(x + w/2, ret11, w, label="v11", color="#f58518", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REGIMES, fontsize=9)
        ax.set_ylabel("Return % (1000-bar pure regime)")
        ax.set_title("Stress Test: Terminal Return by Regime")
        ax.legend()

        # Panel 4: Stress test max drawdowns
        ax = axes[1, 1]
        dd9  = [stress_results[r]["v9"]["max_dd_pct"]  for r in REGIMES]
        dd11 = [stress_results[r]["v11"]["max_dd_pct"] for r in REGIMES]
        ax.bar(x - w/2, dd9,  w, label="v9",  color="#4c78a8", alpha=0.8)
        ax.bar(x + w/2, dd11, w, label="v11", color="#f58518", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REGIMES, fontsize=9)
        ax.set_ylabel("Max drawdown (%)")
        ax.set_title("Stress Test: Max Drawdown by Regime")
        ax.legend()

        plt.tight_layout()
        png_path = os.path.join(RESULTS, "regime_stress.png")
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {png_path}")
    except ImportError:
        print("  matplotlib not available, skipping chart.")

    print(f"\n{'═'*65}")
    print("  Done.")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
