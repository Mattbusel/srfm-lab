"""
equity_paths.py — LARSA v11 Monte Carlo equity path simulation.

Runs 10,000 paths through a 3000-bar correlated ES/NQ/YM world.
Records peak equity, max drawdown, recovery time, terminal equity,
margin call events.  Builds full probability distributions.

Output:
    Terminal report
    results/equity_paths.json
    results/equity_paths.png  (4-panel chart)

Usage:
    python tools/equity_paths.py
    python tools/equity_paths.py --paths 1000
    python tools/equity_paths.py --paths 10000
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import time

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
CORR_FACTOR   = math.sqrt(N_INST + N_INST * (N_INST - 1) * BASE_CORR)
PER_INST_RISK  = PORTFOLIO_RISK / CORR_FACTOR
START_EQUITY  = 1_000_000.0
MARGIN_CALL_THRESH = 0.05   # equity < 5 % of initial = margin call / ruin
N_BARS        = 3000
SYMS          = ["ES", "NQ", "YM"]


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


# ── World generator ──────────────────────────────────────────────────────────
def gen_world(seed, n=N_BARS, corr=BASE_CORR, drift=0.0003):
    rng = np.random.default_rng(seed)
    vols = np.array([0.15, 0.20, 0.14]) / math.sqrt(252 * 6.5)
    C = np.array([[1, corr, corr], [corr, 1, corr], [corr, corr, 1]], dtype=float)
    L = np.linalg.cholesky(np.outer(vols, vols) * C)
    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    ps = {s: [v] for s, v in starts.items()}
    for _ in range(n):
        z = L @ rng.standard_normal(3)
        for j, s in enumerate(SYMS):
            ret = drift + vols[j] * z[j] * math.sqrt(252 * 6.5)
            ps[s].append(ps[s][-1] * (1 + ret))
    result = {s: [] for s in SYMS}
    for s in SYMS:
        v = ps[s]
        for i in range(1, len(v)):
            hi = max(v[i], v[i - 1]) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(v[i], v[i - 1]) * (1 - abs(rng.normal(0, 0.002)))
            result[s].append({"o": v[i - 1], "h": hi, "l": lo, "c": v[i]})
    return result


# ── Single path simulation ────────────────────────────────────────────────────
def run_path(seed):
    """Run one MC path.  Returns a dict of path statistics."""
    bars = gen_world(seed)
    insts = {s: MRI(s) for s in SYMS}
    equity = START_EQUITY
    peak = equity
    max_dd = 0.0
    margin_call = False
    in_dd = False
    dd_start_bar = 0
    recovery_bar = None
    recover_times = []
    equity_curve = [equity]
    pc = {s: bars[s][0]["c"] for s in SYMS}

    positions = {s: 0.0 for s in SYMS}

    for i in range(N_BARS):
        # mark-to-market
        for s in SYMS:
            b = bars[s][i]
            pos = positions[s]
            if abs(pos) > 1e-6 and pc[s] > 0:
                equity += pos * (b["c"] - pc[s]) / pc[s] * equity

        # update instruments
        pr = {}
        for s in SYMS:
            b = bars[s][i]
            insts[s].update(b["o"], b["h"], b["l"], b["c"], pc[s])
            pc[s] = b["c"]; pr[s] = b["c"]

        # margin call check
        if equity < START_EQUITY * MARGIN_CALL_THRESH:
            margin_call = True
            equity = max(equity, 0.0)
            for s in SYMS:
                positions[s] = 0.0
                insts[s].last_target = 0.0; insts[s].bars_held = 0
            equity_curve.append(equity)
            break

        # sizing targets
        raw = {}
        for s in SYMS:
            inst = insts[s]
            tfs = inst.tf_score(); ceiling = TF_CAP[tfs]
            if tfs == 1 and abs(inst.last_target) < 0.01:
                ceiling = 0.0
            if ceiling == 0.0:
                tgt = 0.0
            else:
                d = inst.direction(); p = pr[s]; a = inst.atr.value
                tgt = size_v11(ceiling, d, a, p)
                # pos_floor
                if tfs >= 6 and abs(tgt) > 0.15 and inst.i1h.ctl >= 5:
                    inst.pos_floor = max(inst.pos_floor, 0.70 * abs(tgt))
                if inst.pos_floor > 0 and tfs >= 4 and not abs(inst.last_target) < 0.001:
                    tgt = float(math.copysign(max(abs(tgt), inst.pos_floor), inst.last_target))
                    inst.pos_floor *= 0.95
                if tfs < 4 or abs(tgt) < 1e-9:
                    inst.pos_floor = 0.0
                if not inst.i1d.bh_active and not inst.i1h.bh_active:
                    inst.pos_floor = 0.0
                # hold gate
                if abs(inst.last_target) > 0.02:
                    inst.bars_held += 1
                is_rev = (abs(inst.last_target) > 0.001 and abs(tgt) > 0.001 and
                          math.copysign(1, tgt) != math.copysign(1, inst.last_target))
                if is_rev and inst.bars_held < MIN_HOLD:
                    tgt = inst.last_target
            raw[s] = tgt

        total = sum(abs(v) for v in raw.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        for s in SYMS:
            inst = insts[s]
            tgt = raw[s] * scale
            old = inst.last_target
            if abs(tgt - old) < 0.02:
                continue
            p = pr[s]
            fee = max(1, int(abs(tgt - old) * equity / (p * 50 + 1e-9))) * 4.0
            equity -= fee
            if abs(tgt) < 1e-9:
                inst.bars_held = 0
            elif math.copysign(1, tgt) != math.copysign(1, old):
                inst.bars_held = 0
            inst.last_target = tgt
            positions[s] = tgt

        equity = max(equity, 0.0)
        equity_curve.append(equity)

        # peak / drawdown tracking
        if equity > peak:
            peak = equity
            if in_dd:
                recover_times.append(i - dd_start_bar)
                in_dd = False
                recovery_bar = i
        else:
            dd = (peak - equity) / (peak + 1e-9)
            if dd > max_dd:
                max_dd = dd
            if dd > 0.02 and not in_dd:
                in_dd = True
                dd_start_bar = i

    terminal = equity_curve[-1]

    # time to recover from max drawdown (median of all recovery events)
    med_recover = float(np.median(recover_times)) if recover_times else float(N_BARS)

    return {
        "terminal": terminal,
        "peak": peak,
        "max_dd": max_dd,
        "recover_bars": med_recover,
        "margin_call": int(margin_call),
        "ruin": int(terminal < START_EQUITY * 0.5),
        # ruin_at_any_bar already captured via margin_call / terminal check
    }


def run_path_wrapper(args):
    seed = args
    try:
        return run_path(seed)
    except Exception as e:
        return {"terminal": START_EQUITY, "peak": START_EQUITY, "max_dd": 0.0,
                "recover_bars": 0.0, "margin_call": 0, "ruin": 0, "error": str(e)}


# ── Ruin-over-time helper ─────────────────────────────────────────────────────
def compute_ruin_curve(n_paths, seeds, n_bars=N_BARS):
    """Re-run paths and track equity_curve.  Only used for the ruin-over-time panel."""
    ruin_counts = np.zeros(n_bars + 1, dtype=int)
    sampled = min(n_paths, 500)   # cap at 500 for memory
    for seed in seeds[:sampled]:
        bars = gen_world(seed)
        insts = {s: MRI(s) for s in SYMS}
        equity = START_EQUITY
        peak = equity
        positions = {s: 0.0 for s in SYMS}
        pc = {s: bars[s][0]["c"] for s in SYMS}
        ruined = False
        for i in range(n_bars):
            for s in SYMS:
                b = bars[s][i]
                pos = positions[s]
                if abs(pos) > 1e-6 and pc[s] > 0:
                    equity += pos * (b["c"] - pc[s]) / pc[s] * equity
            pr = {}
            for s in SYMS:
                b = bars[s][i]
                insts[s].update(b["o"], b["h"], b["l"], b["c"], pc[s])
                pc[s] = b["c"]; pr[s] = b["c"]
            if equity < START_EQUITY * MARGIN_CALL_THRESH:
                ruined = True
            if ruined:
                ruin_counts[i] += 1
            else:
                equity = max(equity, 0.0)
                raw = {}
                for s in SYMS:
                    inst = insts[s]
                    tfs = inst.tf_score(); ceiling = TF_CAP[tfs]
                    if tfs == 1 and abs(inst.last_target) < 0.01: ceiling = 0.0
                    d = inst.direction(); p = pr[s]; a = inst.atr.value
                    tgt = size_v11(ceiling, d, a, p) if ceiling > 0 else 0.0
                    raw[s] = tgt
                total = sum(abs(v) for v in raw.values())
                scale = 1.0 / total if total > 1.0 else 1.0
                for s in SYMS:
                    inst = insts[s]
                    tgt = raw[s] * scale
                    if abs(tgt - inst.last_target) >= 0.02:
                        inst.last_target = tgt
                    positions[s] = inst.last_target
        if ruined:
            ruin_counts[n_bars] += 1
    return ruin_counts / max(sampled, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LARSA v11 Monte Carlo equity paths")
    parser.add_argument("--paths", type=int, default=10000, help="Number of MC paths")
    args = parser.parse_args()
    n_paths = args.paths

    print(f"\n{'═'*65}")
    print(f"  LARSA v11  —  Monte Carlo Equity Paths  ({n_paths:,} paths)")
    print(f"{'═'*65}")
    print(f"  Bars per path : {N_BARS:,}")
    print(f"  Start equity  : ${START_EQUITY:,.0f}")
    print(f"  Ruin threshold: {MARGIN_CALL_THRESH*100:.0f}% of start")
    print()

    seeds = list(range(n_paths))
    t0 = time.time()

    use_mp = n_paths >= 500
    if use_mp:
        try:
            cpus = max(1, multiprocessing.cpu_count() - 1)
            print(f"  Using multiprocessing ({cpus} workers)...")
            with multiprocessing.Pool(cpus) as pool:
                results = pool.map(run_path_wrapper, seeds, chunksize=50)
        except Exception as e:
            print(f"  Multiprocessing failed ({e}), falling back to serial.")
            results = [run_path_wrapper(s) for s in seeds]
    else:
        print("  Running serially...")
        results = [run_path_wrapper(s) for s in seeds]

    elapsed = time.time() - t0
    print(f"  Completed {n_paths:,} paths in {elapsed:.1f}s  ({elapsed/n_paths*1000:.1f}ms/path)\n")

    # ── Aggregate ────────────────────────────────────────────────────────────
    terminals    = np.array([r["terminal"]    for r in results])
    peaks        = np.array([r["peak"]        for r in results])
    max_dds      = np.array([r["max_dd"]      for r in results])
    recover_bars = np.array([r["recover_bars"] for r in results])
    margin_calls = np.array([r["margin_call"] for r in results])
    ruins        = np.array([r["ruin"]        for r in results])

    def pct(label, vals, mult=1.0):
        return f"  {label:<40s}: {np.percentile(vals*mult, 10):.2f} / {np.median(vals*mult):.2f} / {np.percentile(vals*mult, 90):.2f} / {np.percentile(vals*mult, 99):.2f}"

    mult_2x  = np.mean(terminals >= 2  * START_EQUITY)
    mult_5x  = np.mean(terminals >= 5  * START_EQUITY)
    mult_10x = np.mean(terminals >= 10 * START_EQUITY)
    mult_20x = np.mean(terminals >= 20 * START_EQUITY)
    ruin_risk = np.mean(ruins)
    mc_risk   = np.mean(margin_calls)
    ev_term   = float(np.mean(terminals))

    print(f"{'─'*65}")
    print("  PROBABILITY OF REACHING EQUITY MULTIPLES")
    print(f"{'─'*65}")
    print(f"  P(2×  = ${2*START_EQUITY/1e6:.0f}M+)  : {mult_2x*100:6.2f}%")
    print(f"  P(5×  = ${5*START_EQUITY/1e6:.0f}M+)  : {mult_5x*100:6.2f}%")
    print(f"  P(10× = ${10*START_EQUITY/1e6:.0f}M+) : {mult_10x*100:6.2f}%")
    print(f"  P(20× = ${20*START_EQUITY/1e6:.0f}M+) : {mult_20x*100:6.2f}%")
    print()
    print(f"{'─'*65}")
    print("  RISK METRICS")
    print(f"{'─'*65}")
    print(f"  Risk of ruin (terminal < 50% start) : {ruin_risk*100:6.2f}%")
    print(f"  Risk of margin call                 : {mc_risk*100:6.2f}%")
    print()
    print(f"{'─'*65}")
    print("  TERMINAL EQUITY — p10 / median / p90 / p99")
    print(f"{'─'*65}")
    print(f"  Raw ($)          : {np.percentile(terminals,10):>12,.0f} / {np.median(terminals):>12,.0f} / {np.percentile(terminals,90):>12,.0f} / {np.percentile(terminals,99):>12,.0f}")
    print(f"  Multiple of start: {np.percentile(terminals/START_EQUITY,10):>8.2f}× / {np.median(terminals/START_EQUITY):>8.2f}× / {np.percentile(terminals/START_EQUITY,90):>8.2f}× / {np.percentile(terminals/START_EQUITY,99):>8.2f}×")
    print(f"  Expected value   : ${ev_term:,.0f}  ({ev_term/START_EQUITY:.2f}× start)")
    print()
    print(f"{'─'*65}")
    print("  DRAWDOWN — p10 / median / p90 / p99")
    print(f"{'─'*65}")
    print(f"  Max drawdown (%): {np.percentile(max_dds*100,10):.1f} / {np.median(max_dds*100):.1f} / {np.percentile(max_dds*100,90):.1f} / {np.percentile(max_dds*100,99):.1f}")
    print(f"  Recovery (bars) : {np.percentile(recover_bars,10):.0f} / {np.median(recover_bars):.0f} / {np.percentile(recover_bars,90):.0f} / {np.percentile(recover_bars,99):.0f}")
    print()

    # ── JSON output ──────────────────────────────────────────────────────────
    out = {
        "n_paths": n_paths,
        "n_bars": N_BARS,
        "start_equity": START_EQUITY,
        "probability": {
            "2x": float(mult_2x),
            "5x": float(mult_5x),
            "10x": float(mult_10x),
            "20x": float(mult_20x),
        },
        "risk": {
            "ruin": float(ruin_risk),
            "margin_call": float(mc_risk),
        },
        "terminal_equity": {
            "p10": float(np.percentile(terminals, 10)),
            "median": float(np.median(terminals)),
            "p90": float(np.percentile(terminals, 90)),
            "p99": float(np.percentile(terminals, 99)),
            "expected_value": ev_term,
        },
        "max_drawdown_pct": {
            "p10": float(np.percentile(max_dds * 100, 10)),
            "median": float(np.median(max_dds * 100)),
            "p90": float(np.percentile(max_dds * 100, 90)),
            "p99": float(np.percentile(max_dds * 100, 99)),
        },
        "recovery_bars": {
            "p10": float(np.percentile(recover_bars, 10)),
            "median": float(np.median(recover_bars)),
            "p90": float(np.percentile(recover_bars, 90)),
        },
        "peak_equity": {
            "p10": float(np.percentile(peaks, 10)),
            "median": float(np.median(peaks)),
            "p90": float(np.percentile(peaks, 90)),
        },
    }
    json_path = os.path.join(RESULTS, "equity_paths.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {json_path}")

    # ── Charts ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"LARSA v11 — Monte Carlo Equity Paths  (n={n_paths:,}, {N_BARS} bars/path)",
                     fontsize=13, fontweight="bold")

        # Panel 1: Distribution of terminal equity (log scale)
        ax = axes[0, 0]
        valid = terminals[terminals > 0]
        ax.hist(np.log10(valid / START_EQUITY), bins=80, color="#4c78a8", edgecolor="none", alpha=0.85)
        ax.axvline(np.log10(np.median(valid / START_EQUITY)), color="orange", lw=2, label=f"median {np.median(valid/START_EQUITY):.2f}×")
        ax.axvline(0, color="red", lw=1.5, ls="--", label="start")
        ax.set_xlabel("Terminal equity (log₁₀ multiple of start)")
        ax.set_ylabel("Path count")
        ax.set_title("Distribution of Terminal Equity")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{10**x:.1f}×"))

        # Panel 2: Distribution of max drawdown
        ax = axes[0, 1]
        ax.hist(max_dds * 100, bins=80, color="#f58518", edgecolor="none", alpha=0.85)
        ax.axvline(np.median(max_dds * 100), color="blue", lw=2,
                   label=f"median {np.median(max_dds*100):.1f}%")
        ax.axvline(np.percentile(max_dds * 100, 90), color="red", lw=1.5, ls="--",
                   label=f"p90 {np.percentile(max_dds*100,90):.1f}%")
        ax.set_xlabel("Max drawdown (%)")
        ax.set_ylabel("Path count")
        ax.set_title("Distribution of Max Drawdown")
        ax.legend(fontsize=9)

        # Panel 3: Distribution of peak equity
        ax = axes[1, 0]
        ax.hist(peaks / START_EQUITY, bins=80, color="#54a24b", edgecolor="none", alpha=0.85)
        ax.axvline(np.median(peaks / START_EQUITY), color="orange", lw=2,
                   label=f"median {np.median(peaks/START_EQUITY):.2f}×")
        ax.set_xlabel("Peak equity (multiple of start)")
        ax.set_ylabel("Path count")
        ax.set_title("Distribution of Peak Equity Reached")
        ax.legend(fontsize=9)

        # Panel 4: Probability of ruin over time
        ax = axes[1, 1]
        print("  Computing ruin-over-time curve (sampled paths)...")
        ruin_curve = compute_ruin_curve(n_paths, seeds)
        ax.plot(ruin_curve * 100, color="#e45756", lw=1.5)
        ax.set_xlabel("Bar")
        ax.set_ylabel("% of paths in ruin")
        ax.set_title("Probability of Ruin Over Time")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        plt.tight_layout()
        png_path = os.path.join(RESULTS, "equity_paths.png")
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {png_path}")
    except ImportError:
        print("  matplotlib not available, skipping chart.")

    print(f"\n{'═'*65}")
    print("  Done.")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
